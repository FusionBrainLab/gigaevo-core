"""Engine-startup bridge between the legacy storage stack and the dataplane.

The Hydra config tree instantiates :class:`RedisProgramStorage` and
:class:`BanditModelRouter` independently of the coordinator, and each
constructor accepts an optional ``dataplane=`` parameter that — when
``None`` — keeps the legacy WATCH/MULTI/EXEC and in-process bandit
paths in place. Engine startup constructs a process-local
:class:`DataPlane`, primes it with the legacy program-state FSM table,
and rebinds the private ``dataplane`` / ``actor`` slots on the
already-instantiated storage and router so production flips to the new
substrate without touching the Hydra schema.

The rebinding is intentional: the alternative is teaching the Hydra
resolver to instantiate a coordinator before the dependent objects and
to inject it into their ``_args_``, which adds a config-schema surface
and a Hydra-cache lifetime risk (see :issue:`dataplane_gigaevo §10.4
#10`). Post-instantiation rebind keeps the wiring local to the
engine-entrypoint script and leaves the Hydra config free of
dataplane-specific keys.

A note on lifetimes: every call to :func:`build_dataplane` constructs a
new instance. This module does not cache. Two engine runs in the same
Python process (uncommon, but observed in notebook drivers and
integration test suites that call ``run_experiment`` twice) must
construct two independent dataplanes; a process-wide singleton would
share connection pool state across runs and turn a clean shutdown into
a use-after-shutdown on the second run.
"""

from __future__ import annotations

import os
import socket
from typing import TYPE_CHECKING
from uuid import uuid4

from loguru import logger

from gigaevo.dataplane.coordinator import DataPlane
from gigaevo.dataplane.ids import (
    ActorIdentity,
    CellKey,
    CounterKey,
    ProgramId,
    RunId,
    WorkerId,
)
from gigaevo.dataplane.permissions import Token, mint_root, mint_split

if TYPE_CHECKING:
    from gigaevo.database.redis_program_storage import RedisProgramStorage
    from gigaevo.evolution.engine.core import EvolutionEngine
    from gigaevo.evolution.storage.archive_storage import RedisArchiveStorage
    from gigaevo.llm.models import MultiModelRouter
    from gigaevo.prompts.fetcher import PromptFetcher
    from gigaevo.runner.dag_runner import DagRunner

__all__ = [
    "ENV_RUN_ID",
    "ENV_WORKER_ID",
    "EngineRoot",
    "build_dataplane",
    "build_engine_root",
    "build_actor_identity",
    "wire_storage",
    "wire_archive_storage",
    "wire_bandit_router",
    "wire_prompt_fetcher",
    "wire_dag_runner",
    "wire_evolution_engine",
]


#: Root subspace tags. Each one names a logically-disjoint Redis
#: key-space owned by this engine; ``mint_split`` from the matching
#: root token witnesses every per-call write into that space. The
#: literal values are namespaced under ``engine:`` so a caller-supplied
#: per-program / per-cell / per-counter tag cannot collide with a root
#: tag and silently claim the wrong subspace.
_PROGRAM_ROOT_TAG: str = "engine:program-root"
_CELL_ROOT_TAG: str = "engine:cell-root"
_COUNTER_ROOT_TAG: str = "engine:counter-root"


class EngineRoot:
    """Single-engine root permission witnesses for the three Redis subspaces.

    One engine instance owns exactly one of these. Each interior token
    witnesses the engine's exclusive write claim over the matching
    key-space (``program:*``, archive cells, CRDT counters). Per-call
    writes are minted by linear split from the matching root via
    :meth:`split_program_token` / :meth:`split_cell_token` /
    :meth:`split_counter_token`, so the per-call token is *traceable*
    to the engine root rather than minted ex nihilo at the call site.

    Two engines on the same Redis cluster MUST use disjoint key
    prefixes (set on :class:`DataPlane.key_prefix`); the linear tokens
    enforce single-writer within the prefix but cannot, by themselves,
    detect two engines minting orthogonal roots over the same prefix.

    Linearity-vs-longevity trade-off: the engine needs a long-lived
    witness over each subspace, but :func:`mint_split` consumes its
    parent. The class privately rotates the long-lived witness on each
    split — the public split helpers consume the current root and store
    the "next" sub-token from the split as the new root in one
    statement. Callers never see the rotation; they receive only the
    fresh per-call sub-token.

    The class is not a frozen dataclass because the internal rotation
    requires re-binding ``_program_root`` / ``_cell_root`` /
    ``_counter_root`` in place. The tokens themselves remain move-only
    (their move-only contract is enforced by :class:`Token`, not by the
    container) so an :class:`EngineRoot` instance cannot be silently
    duplicated via ``copy.deepcopy`` — the nested token's
    ``__deepcopy__`` raises and propagates out.
    """

    __slots__ = ("_program_root", "_cell_root", "_counter_root")

    def __init__(
        self,
        program_root: Token[ProgramId],
        cell_root: Token[CellKey],
        counter_root: Token[CounterKey],
    ) -> None:
        self._program_root: Token[ProgramId] = program_root
        self._cell_root: Token[CellKey] = cell_root
        self._counter_root: Token[CounterKey] = counter_root

    def split_program_token(self, program_id: ProgramId) -> Token[ProgramId]:
        """Derive a fresh per-program permission witness.

        Consumes the current program root via :func:`mint_split`,
        retains the "next" sub-token as the new long-lived root, and
        returns the per-call sub-token tagged with ``program_id`` so
        :meth:`DataPlane.transition_program_state` can verify
        ``token.consume() == program_id`` and reject mismatched routing
        deterministically.

        The rotation is in-place on the engine root; concurrent split
        calls within a single engine instance MUST be serialised by the
        caller (the dataplane is single-threaded asyncio, which it is).
        """
        next_root, per_call = mint_split(
            self._program_root,
            ProgramId(f"{_PROGRAM_ROOT_TAG}#next"),
            program_id,
        )
        self._program_root = next_root
        return per_call

    def split_cell_token(self, cell_key: CellKey) -> Token[CellKey]:
        """Derive a fresh per-cell permission witness; rotates the cell root."""
        next_root, per_call = mint_split(
            self._cell_root,
            CellKey(f"{_CELL_ROOT_TAG}#next"),
            cell_key,
        )
        self._cell_root = next_root
        return per_call

    def split_counter_token(self, counter_key: CounterKey) -> Token[CounterKey]:
        """Derive a fresh per-counter permission witness; rotates the counter root."""
        next_root, per_call = mint_split(
            self._counter_root,
            CounterKey(f"{_COUNTER_ROOT_TAG}#next"),
            counter_key,
        )
        self._counter_root = next_root
        return per_call


def build_engine_root() -> EngineRoot:
    """Mint the per-subspace root tokens for this engine instance.

    Called exactly once during engine startup, immediately after
    :func:`build_dataplane`. The returned :class:`EngineRoot` is
    threaded into the storage's constructor (or attached via
    :func:`wire_storage` together with the dataplane) so per-call
    writes derive their permission witnesses from this single origin
    rather than minting fresh roots ad-hoc.

    The three subspace tags are module-level constants; using literal
    strings rather than the inferred ``Token[ProgramId]`` phantom tag is
    intentional: the actual ``Tag`` value carried by the token is what
    :meth:`Token.consume` returns, and we want that value to be a
    stable, grep-able identifier rather than a per-program string that
    a later code reviewer might mistakenly assume to be a real id.
    """
    return EngineRoot(
        program_root=mint_root(ProgramId(_PROGRAM_ROOT_TAG)),
        cell_root=mint_root(CellKey(_CELL_ROOT_TAG)),
        counter_root=mint_root(CounterKey(_COUNTER_ROOT_TAG)),
    )


#: Environment variable used to override the auto-generated run identifier.
#: When set, the value is validated by :class:`ActorIdentity` and used
#: as-is so external orchestrators (k8s, slurm, CI) can pin a stable id.
ENV_RUN_ID = "GIGAEVO_DATAPLANE_RUN_ID"

#: Environment variable used to override the auto-generated worker
#: identifier. Defaults to ``{hostname}-{pid}`` so two engines on the
#: same host pick up distinct identities without explicit configuration.
ENV_WORKER_ID = "GIGAEVO_DATAPLANE_WORKER_ID"


async def build_dataplane(
    redis_url: str,
    *,
    key_prefix: str,
    max_connections: int = 64,
    socket_timeout_s: float = 30.0,
    socket_connect_timeout_s: float = 10.0,
) -> DataPlane:
    """Construct a started :class:`DataPlane`.

    Reuses the storage's Redis URL and key prefix so coordinator and
    legacy storage write into the same logical namespace. The caller
    owns the lifetime; :func:`DataPlane.shutdown` MUST be called in a
    ``finally`` block around the engine run-loop.

    :meth:`DataPlane.startup` loads the program-state FSM table with
    rows keyed under both the dp enum's uppercase form and the
    application-layer lowercase form, so a coordinator-routed call
    resolves the same row whichever vocabulary the persisted blob uses.
    """
    dp = DataPlane(
        redis_url,
        key_prefix=key_prefix,
        max_connections=max_connections,
        socket_timeout_s=socket_timeout_s,
        socket_connect_timeout_s=socket_connect_timeout_s,
    )
    await dp.startup()
    return dp


def build_actor_identity(
    *,
    run_id: str | None = None,
    worker_id: str | None = None,
) -> ActorIdentity:
    """Build an :class:`ActorIdentity` from explicit args or env / host fallbacks.

    Resolution order:

    1. Explicit ``run_id`` / ``worker_id`` arguments (typically the
       Hydra config's run identifier).
    2. ``GIGAEVO_DATAPLANE_RUN_ID`` / ``GIGAEVO_DATAPLANE_WORKER_ID``
       environment variables — useful when the engine is launched by
       an external orchestrator that already knows its identity.
    3. A fresh ULID-style run id (uuid4 hex without dashes) plus
       ``{hostname}-{pid}`` for the worker id.

    The identity flows into bandit CRDT counters as ``{run}:{worker}``;
    two engines that collide on this pair share a bandit counter cell.
    The host+pid worker default is collision-free in practice on a
    single host; cross-host collisions require an environment override.
    """
    resolved_run = run_id or os.environ.get(ENV_RUN_ID) or uuid4().hex
    resolved_worker = (
        worker_id
        or os.environ.get(ENV_WORKER_ID)
        or f"{socket.gethostname()}-{os.getpid()}"
    )
    actor = ActorIdentity(
        run_id=RunId(resolved_run),
        worker_id=WorkerId(resolved_worker),
    )
    logger.info(
        "DataPlane actor identity: run_id={} worker_id={}",
        actor.run_id,
        actor.worker_id,
    )
    return actor


def wire_storage(
    storage: RedisProgramStorage,
    dataplane: DataPlane,
    engine_root: EngineRoot | None = None,
) -> None:
    """Attach the coordinator and (optionally) the engine root to a storage.

    The storage constructor accepts ``dataplane=`` and ``engine_root=``;
    both end up on private attributes that the routing logic in
    :meth:`atomic_state_transition` and :meth:`fast_state_transition`
    reads. Post-Hydra rebinding is safe because:

    - the legacy path is the default and remains correct;
    - the dataplane path activates iff ``self._dataplane is not None``;
    - the engine-root path activates iff ``self._engine_root is not None``;
    - no concurrent caller can be inside a transition at startup
      (engine tasks haven't been started yet).

    Subsequent calls overwrite — the bridge is idempotent enough that
    a test fixture can rebind a coordinator without first nulling the
    previous one. Rebinding mid-run is undefined; the engine startup
    sequence places this call before ``engine.start()``.

    ``engine_root`` is optional so legacy entrypoints continue to wire
    storage with just a dataplane; when supplied, per-call FSM tokens
    derive from the engine root via linear split.
    """
    storage._dataplane = dataplane
    if engine_root is not None:
        storage._engine_root = engine_root


def wire_archive_storage(
    archive: RedisArchiveStorage,
    dataplane: DataPlane,
    engine_root: EngineRoot,
) -> bool:
    """Attach coordinator + engine root to a :class:`RedisArchiveStorage`.

    Returns ``True`` when the rebind happened, ``False`` for a non-
    archive input (the entrypoint can call this unconditionally). The
    archive's dataplane swap path then derives per-call cell tokens via
    :meth:`EngineRoot.split_cell_token` instead of minting an ad-hoc
    root, so every swap is structurally traceable to the engine's
    single cell-subspace origin.

    Idempotent within a run: re-attaching the same coordinator + engine
    root is a silent no-op. A different coordinator or engine root is
    not rejected because the archive does not track equality on either
    handle internally — the post-startup engine flow only ever wires
    once.
    """
    # Lazy import to keep the dataplane package self-contained.
    from gigaevo.evolution.storage.archive_storage import RedisArchiveStorage

    if not isinstance(archive, RedisArchiveStorage):
        return False
    archive._dataplane = dataplane
    archive._engine_root = engine_root
    logger.info(
        "RedisArchiveStorage wired to DataPlane: hash_key={}",
        archive._hash_key,
    )
    return True


def wire_bandit_router(
    router: MultiModelRouter,
    dataplane: DataPlane,
    actor: ActorIdentity,
    engine_root: EngineRoot | None = None,
) -> bool:
    """Attach coordinator + actor (+ engine root) to a bandit router, if present.

    Returns ``True`` when the router is a
    :class:`~gigaevo.llm.bandit.BanditModelRouter` and the rebind
    happened, ``False`` for the static :class:`MultiModelRouter` case
    (no bandit ledger to share). Non-bandit routers degrade silently
    so the entrypoint can call this unconditionally regardless of
    which LLM config the user selected.

    The bandit's internal :class:`SlidingWindowUCB1` validates that
    ``dataplane`` and ``actor`` are supplied together; this helper sets
    both atomically to keep that invariant intact.

    ``engine_root`` is optional: when supplied, every ledger write
    derives a per-call counter-token via
    :meth:`EngineRoot.split_counter_token`, satisfying the single-writer
    discipline the coordinator already enforces for FSM transitions and
    archive swaps. ``None`` keeps the legacy token-less ledger contract.
    """
    # Lazy import to avoid the dataplane package importing the llm
    # tree at module load — keeps the dataplane self-contained.
    from gigaevo.llm.bandit import BanditModelRouter, SlidingWindowUCB1

    if not isinstance(router, BanditModelRouter):
        return False
    bandit = router._bandit
    if not isinstance(bandit, SlidingWindowUCB1):
        # Defensive: a subclass could replace _bandit with another
        # adapter. Skip silently rather than corrupt that adapter's
        # internal state with attribute assignment it doesn't expect.
        return False
    bandit._dataplane = dataplane
    bandit._actor = actor
    if engine_root is not None:
        bandit._engine_root = engine_root
    logger.info(
        "BanditModelRouter wired to DataPlane: name={} arms={} actor={} engine_root={}",
        getattr(router, "_name", "<unnamed>"),
        bandit.arm_names,
        actor.pack(),
        "yes" if engine_root is not None else "no",
    )
    return True


def wire_prompt_fetcher(
    fetcher: PromptFetcher,
    main_dp: DataPlane,
    prompt_dp: DataPlane,
    actor: ActorIdentity,
) -> bool:
    """Attach engine-owned DataPlanes + actor to a prompt fetcher, if applicable.

    Returns ``True`` when the fetcher is a
    :class:`~gigaevo.prompts.fetcher.GigaEvoArchivePromptFetcher` and
    the rebind happened, ``False`` for the static
    :class:`~gigaevo.prompts.fetcher.FixedDirPromptFetcher` case (no
    co-evolution archive read and no per-prompt outcome ledger).
    Non-archive fetchers degrade silently so the entrypoint can call
    this unconditionally regardless of which prompt-fetcher config the
    user selected.

    The fetcher writes prompt-outcome stats to ``main_dp`` (same
    key-space the storage and bandit already own) and reads the
    co-evolution archive from ``prompt_dp`` (typically a different
    Redis DB under a different key prefix). Sharing ``main_dp`` with
    the rest of the engine eliminates the second connection pool the
    fetcher would otherwise build lazily in
    :meth:`GigaEvoArchivePromptFetcher.start`.

    ``actor`` is the same :class:`ActorIdentity` passed to
    :func:`wire_bandit_router`: one engine, one identity, multiple
    write surfaces. The fetcher writes under the
    ``prompt:trials:{id}`` counter namespace; the bandit writes under
    ``bandit:trials:{arm}``; identical ``(run_id, worker_id)`` keeps
    every per-engine sub-count addressable under one actor.

    Idempotency and conflict-detection are delegated to
    :meth:`GigaEvoArchivePromptFetcher.attach_dataplane`.
    """
    # Lazy import to avoid the dataplane package importing the prompts
    # tree at module load — keeps the dataplane self-contained.
    from gigaevo.prompts.fetcher import GigaEvoArchivePromptFetcher

    if not isinstance(fetcher, GigaEvoArchivePromptFetcher):
        return False
    fetcher.attach_dataplane(main_dp, prompt_dp, actor)
    return True


def wire_dag_runner(
    runner: DagRunner,
    dataplane: DataPlane,
    engine_root: EngineRoot,
) -> bool:
    """Attach coordinator + engine root to a :class:`DagRunner`, if applicable.

    Returns ``True`` when the rebind happened, ``False`` for a non-DagRunner
    input (the entrypoint can call this unconditionally regardless of the
    Hydra runner selection without inspecting the runtime type first).

    Idempotency: a second call with the exact same triple is a silent
    no-op. A second call with a different coordinator or engine root
    raises :class:`RuntimeError` — silent overwrite would corrupt the
    single-writer single-actor invariant that the per-call linear
    permission tokens derived from the engine root rely on.

    Must be called before :meth:`DagRunner.start`; the dataplane and
    engine-root references are read on every state-mutation path the
    runner originates, so a mid-run rebind is undefined.
    """
    from gigaevo.runner.dag_runner import DagRunner as _DagRunner

    if not isinstance(runner, _DagRunner):
        return False
    if runner._dataplane is dataplane and runner._engine_root is engine_root:
        return True
    if (runner._dataplane is not None or runner._engine_root is not None) and (
        runner._dataplane is not dataplane or runner._engine_root is not engine_root
    ):
        raise RuntimeError(
            "DagRunner already has a different DataPlane or EngineRoot "
            "attached; refusing to overwrite. Construct a fresh runner "
            "or detach the existing handles before reattaching."
        )
    runner._dataplane = dataplane
    runner._engine_root = engine_root
    logger.info("DagRunner wired to DataPlane: prefix={}", dataplane.key_prefix)
    return True


def wire_evolution_engine(
    engine: EvolutionEngine,
    dataplane: DataPlane,
    engine_root: EngineRoot,
) -> bool:
    """Attach coordinator + engine root to an :class:`EvolutionEngine`.

    Returns ``True`` when the rebind happened, ``False`` for a
    non-EvolutionEngine input. Same idempotency and conflict-rejection
    contract as :func:`wire_dag_runner`: identical re-attach is a
    silent no-op, conflicting re-attach raises.

    Must be called before :meth:`EvolutionEngine.start`; the engine
    reads the dataplane and engine-root references on every state-
    mutation path it originates.
    """
    from gigaevo.evolution.engine.core import EvolutionEngine as _EvolutionEngine

    if not isinstance(engine, _EvolutionEngine):
        return False
    if engine._dataplane is dataplane and engine._engine_root is engine_root:
        return True
    if (engine._dataplane is not None or engine._engine_root is not None) and (
        engine._dataplane is not dataplane or engine._engine_root is not engine_root
    ):
        raise RuntimeError(
            "EvolutionEngine already has a different DataPlane or "
            "EngineRoot attached; refusing to overwrite. Construct a "
            "fresh engine or detach the existing handles before reattaching."
        )
    engine._dataplane = dataplane
    engine._engine_root = engine_root
    logger.info("EvolutionEngine wired to DataPlane: prefix={}", dataplane.key_prefix)
    return True
