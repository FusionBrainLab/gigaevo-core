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
from gigaevo.dataplane.ids import ActorIdentity, RunId, WorkerId

if TYPE_CHECKING:
    from gigaevo.database.redis_program_storage import RedisProgramStorage
    from gigaevo.llm.models import MultiModelRouter

__all__ = [
    "ENV_RUN_ID",
    "ENV_WORKER_ID",
    "build_dataplane",
    "build_actor_identity",
    "wire_storage",
    "wire_bandit_router",
]


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
    """Construct a started :class:`DataPlane` and load the legacy FSM table.

    Reuses the storage's Redis URL and key prefix so coordinator and
    legacy storage write into the same logical namespace. The caller
    owns the lifetime; :func:`DataPlane.shutdown` MUST be called in a
    ``finally`` block around the engine run-loop.

    The FSM table loaded here matches the lowercase
    :class:`~gigaevo.programs.program_state.ProgramState` vocabulary
    persisted in legacy program blobs. The coordinator ships an
    uppercase default that does not match the on-disk format; without
    this reload every coordinator-routed transition would reject as
    ``unknown source state``.
    """
    dp = DataPlane(
        redis_url,
        key_prefix=key_prefix,
        max_connections=max_connections,
        socket_timeout_s=socket_timeout_s,
        socket_connect_timeout_s=socket_connect_timeout_s,
    )
    await dp.startup()
    # Lazy import so the dataplane package keeps no static dependency
    # on the legacy storage module — the bridge is one-directional.
    from gigaevo.database.redis_program_storage import load_legacy_program_fsm

    await load_legacy_program_fsm(dp)
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
) -> None:
    """Attach the coordinator to an already-instantiated storage.

    The storage constructor accepts ``dataplane=`` and stores it in
    ``self._dataplane``; routing logic in :meth:`atomic_state_transition`
    and :meth:`fast_state_transition` reads that attribute. Post-Hydra
    rebinding is safe because:

    - the legacy path is the default and remains correct;
    - the dataplane path activates iff ``self._dataplane is not None``;
    - no concurrent caller can be inside a transition at startup
      (engine tasks haven't been started yet).

    Subsequent calls overwrite — the bridge is idempotent enough that
    a test fixture can rebind a coordinator without first nulling the
    previous one. Rebinding mid-run is undefined; the engine startup
    sequence places this call before ``engine.start()``.
    """
    storage._dataplane = dataplane


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
