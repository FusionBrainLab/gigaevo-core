"""Engine-startup bridge wiring the dataplane into Hydra-built objects.

The Hydra config tree instantiates :class:`RedisProgramStorage`,
:class:`BanditModelRouter`, etc. independently of the coordinator. Each
constructor accepts ``dataplane=None`` and falls back to its own path.
This module constructs a process-local :class:`DataPlane`, primes it,
and rebinds the private ``dataplane`` / ``actor`` slots on the
already-instantiated objects so production flips to the new substrate
without touching the Hydra schema.

Post-instantiation rebind keeps the wiring local to the engine
entrypoint and leaves the Hydra config free of dataplane-specific keys.

Every call to :func:`build_dataplane` constructs a fresh instance; this
module does not cache. Two engine runs in the same Python process must
build two independent dataplanes — a process-wide singleton would share
pool state and turn shutdown into use-after-shutdown.
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


#: Root subspace tags. ``mint_split`` from the matching root token
#: witnesses every per-call write into that space. Namespaced under
#: ``engine:`` so a caller-supplied per-call tag cannot collide.
_PROGRAM_ROOT_TAG: str = "engine:program-root"
_CELL_ROOT_TAG: str = "engine:cell-root"
_COUNTER_ROOT_TAG: str = "engine:counter-root"


class EngineRoot:
    """Per-engine root permission witnesses for the three Redis subspaces.

    One engine owns exactly one instance. Each interior token witnesses
    the engine's exclusive write claim over the matching key-space
    (``program:*``, archive cells, CRDT counters). Per-call sub-tokens
    are derived by linear split from the matching root.

    Two engines on the same Redis cluster MUST use disjoint key prefixes
    — linear tokens enforce single-writer within a prefix but cannot
    detect orthogonal roots minted over the same one.

    :func:`mint_split` consumes its parent, but the engine needs a
    long-lived witness over each subspace; the split helpers atomically
    consume the current root and store the "next" sub-token as the new
    root, hiding the rotation from callers.

    Not a frozen dataclass because rotation rebinds the slots in place.
    The tokens themselves remain move-only via :class:`Token`, so the
    container cannot be silently duplicated via ``copy.deepcopy``.
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
        """Derive a fresh per-program permission witness; rotates the program root.

        The returned token is tagged with ``program_id`` so the coordinator
        can token-CAS against the routed key. Concurrent split calls
        within one engine MUST be caller-serialised (asyncio single-thread).
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
    :func:`build_dataplane`. The subspace tags are stable, grep-able
    string constants — using a per-id tag here would let a reviewer
    mistake the root for a per-call token.
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

    Coordinator and surrounding storage share the same Redis URL and key
    prefix so they write into the same logical namespace. The caller
    owns the lifetime; :func:`DataPlane.shutdown` MUST run in a
    ``finally`` block around the engine run-loop.
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

    1. Explicit ``run_id`` / ``worker_id`` arguments.
    2. ``GIGAEVO_DATAPLANE_RUN_ID`` / ``GIGAEVO_DATAPLANE_WORKER_ID`` env vars.
    3. ``uuid4().hex`` for run; ``{hostname}-{pid}`` for worker.

    The identity flows into bandit CRDT counters as ``{run}:{worker}``;
    two engines that collide on this pair share a counter cell.
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

    Post-Hydra rebinding is safe because the routing logic activates iff
    ``self._dataplane is not None``, the fallback path stays correct
    otherwise, and no engine task is in-flight at startup. Subsequent
    calls overwrite (idempotent for test fixtures); rebinding mid-run is
    undefined — call this before ``engine.start()``.
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

    Returns ``True`` on rebind, ``False`` for a non-archive input so the
    entrypoint can call unconditionally. The archive's swap path derives
    per-call cell tokens via :meth:`EngineRoot.split_cell_token`, so
    every swap is structurally traceable to the engine's cell root.
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
    """Attach coordinator + actor (+ engine root) to a bandit router, if applicable.

    Returns ``True`` on rebind, ``False`` for the static router case
    so the entrypoint can call unconditionally. The bandit's
    :class:`SlidingWindowUCB1` requires ``dataplane`` and ``actor``
    together; this helper sets both atomically. When ``engine_root`` is
    supplied, every ledger write derives a per-call counter token via
    :meth:`EngineRoot.split_counter_token`.
    """
    # Lazy import to keep the dataplane package self-contained.
    from gigaevo.llm.bandit import BanditModelRouter, SlidingWindowUCB1

    if not isinstance(router, BanditModelRouter):
        return False
    bandit = router._bandit
    if not isinstance(bandit, SlidingWindowUCB1):
        # A subclass could replace _bandit; skip silently rather than
        # poke attributes the adapter does not expect.
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

    Returns ``True`` on rebind, ``False`` for the static fetcher case so
    the entrypoint can call unconditionally. The fetcher writes prompt
    outcome stats to ``main_dp`` (same key-space as the rest of the
    engine) and reads the co-evolution archive from ``prompt_dp``
    (typically a different DB / prefix). Idempotency and conflict
    detection are delegated to
    :meth:`GigaEvoArchivePromptFetcher.attach_dataplane`.
    """
    # Lazy import to keep the dataplane package self-contained.
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

    Returns ``True`` on rebind, ``False`` for a non-DagRunner input.
    Idempotent for identical triples; a conflicting re-attach raises
    :class:`RuntimeError` because silent overwrite would corrupt the
    single-writer invariant the per-call linear tokens rely on. Must be
    called before :meth:`DagRunner.start`.
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

    Same contract as :func:`wire_dag_runner`: ``True`` on rebind,
    ``False`` for a non-engine input, idempotent for identical triples,
    raises :class:`RuntimeError` on conflicting re-attach. Must be
    called before :meth:`EvolutionEngine.start`.
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
