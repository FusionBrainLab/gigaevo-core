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
    from gigaevo.llm.models import MultiModelRouter

__all__ = [
    "ENV_RUN_ID",
    "ENV_WORKER_ID",
    "EngineRoot",
    "build_dataplane",
    "build_engine_root",
    "build_actor_identity",
    "wire_storage",
    "wire_bandit_router",
]


#: Root subspace tags. Each one names a logically-disjoint Redis
#: key-space owned by this engine; ``mint_split`` from the matching
#: root token witnesses every per-call write into that space. The
#: literal values are namespaced under ``engine:`` so a future
#: ``mint_combine`` (which uses caller-supplied tags) cannot accidentally
#: collide with a per-program / per-cell / per-counter tag.
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


def wire_bandit_router(
    router: MultiModelRouter,
    dataplane: DataPlane,
    actor: ActorIdentity,
) -> bool:
    """Attach coordinator + actor to a bandit router, if present.

    Returns ``True`` when the router is a
    :class:`~gigaevo.llm.bandit.BanditModelRouter` and the rebind
    happened, ``False`` for the static :class:`MultiModelRouter` case
    (no bandit ledger to share). Non-bandit routers degrade silently
    so the entrypoint can call this unconditionally regardless of
    which LLM config the user selected.

    The bandit's internal :class:`SlidingWindowUCB1` validates that
    ``dataplane`` and ``actor`` are supplied together; this helper sets
    both atomically to keep that invariant intact.
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
    logger.info(
        "BanditModelRouter wired to DataPlane: name={} arms={} actor={}",
        getattr(router, "_name", "<unnamed>"),
        bandit.arm_names,
        actor.pack(),
    )
    return True
