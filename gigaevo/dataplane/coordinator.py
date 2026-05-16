"""The :class:`DataPlane` — sole public surface for every Redis interaction.

This module ships as a *shell* with explicit lifecycle methods and
per-resource method stubs that raise :class:`NotImplementedError`. The
public contract — argument shapes, return shapes, error variants — is
fully expressed in the type system today so call sites can be staged
against the real signatures while the method bodies are still landing.

The coordinator owns:

    - one :class:`RedisConnection` (the connection pool)
    - one :class:`LuaRegistry` (script SHAs)
    - the FSM tables loaded into Redis via
      :func:`gigaevo.dataplane.transitions.load_fsm_table`
    - the engine-root :class:`Token`\\ s (split per subspace)

The FSM transition tables are NOT copied into the coordinator — the
single source of truth lives in :mod:`gigaevo.dataplane.transitions` and
the coordinator references those module-level constants by name during
:meth:`startup`. A copy would create a second place that drifts; a
property accessor would imply the coordinator owned mutable state. The
tables are intentionally immutable singletons.

Outside this module, no other code should import ``redis`` or
``redis.asyncio`` directly — the ruff config enforces this (see
``lints.toml``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Literal

from loguru import logger

from .connection import RedisConnection
from .crash import OneShotFlag
from .errors import DataPlaneError, NotStartedError, ShutdownError, StaleReadError
from .ids import (
    ActorIdentity,
    CellKey,
    CounterKey,
    KeyPrefix,
    LeaseToken,
    ProgramId,
    ScriptName,
    make_script_name,
)
from .models import Err, Ok, Result, Versioned
from .permissions import Token
from .scripts import LuaRegistry, load_lua_source
from .transitions import (
    CLAIM_STATE_TRANSITIONS,
    LOCK_STATE_TRANSITIONS,
    PROGRAM_STATE_TRANSITIONS,
    ProgramState,
    load_fsm_table,
)

_DEFAULT_KEY_PREFIX: Final[str] = "gigaevo"


# ── built-in script names ──────────────────────────────────────────────
#
# Each per-resource Lua script registers under a stable name. The names
# are fixed module constants so callers (and tests) refer to one symbol
# instead of a magic string repeated at every site.

_SCRIPT_COUNTER_INC: Final[ScriptName] = make_script_name("counter_inc")


# ── public contract dataclasses ─────────────────────────────────────────
#
# Every method on :class:`DataPlane` returns a typed shape. The
# dataclasses below are the public contract: callers can write the
# pattern-match arms today even though the method bodies still raise
# ``NotImplementedError``. Each is frozen + slotted so the wire / cache
# representation cannot drift.


@dataclass(slots=True, frozen=True)
class ProgramPatch:
    """Field-level patch applied during a program FSM transition.

    A ``ProgramPatch`` is a JSON-serialisable mapping that the server
    merges into the persisted program blob *before* it advances the
    state. The frozen-dataclass shape rather than a bare ``dict`` exists
    so a) the caller cannot mutate the patch after construction (which
    would race the in-flight Lua call), b) future fields (e.g. a JSON
    pointer / a content-hash floor) can be added without rewriting every
    call site, and c) tests can assert against an exact value.

    The ``fields`` payload is constrained to JSON-serialisable values at
    the codec boundary; this dataclass does not re-validate the shape.
    """

    fields: dict[str, object] = field(default_factory=dict)


# A placeholder for the future :class:`gigaevo.programs.program.Program`
# concrete type. The coordinator's Lua scripts treat the blob opaquely:
# they merge ``ProgramPatch.fields`` into the persisted JSON and advance
# the state. The Python-side shape is whatever the application layer
# decodes the blob into. When the Program module lands its concrete
# dataclass it will replace this alias in one edit.
type ProgramSnapshot = dict[str, object]


@dataclass(slots=True, frozen=True)
class BatchTransitionItem:
    """One element of a multi-program atomic transition batch.

    The server applies the whole batch as a single Lua call: either
    every transition succeeds or none do. Each item carries its own
    token because the underlying subspace is per-program — a single
    coordinator-root token cannot serve a multi-aggregate write.
    """

    program_id: ProgramId
    token: Token[ProgramId]
    expected_from: ProgramState | None
    to: ProgramState
    patch: ProgramPatch | None = None


@dataclass(slots=True, frozen=True)
class BatchTransitionOutcome:
    """Per-item outcome of a batch transition.

    ``items`` is the same length as the input batch and in the same
    order. Each entry is the post-transition :class:`Versioned` snapshot
    for the program; a per-item ``Err`` cannot occur because the batch
    is atomic at the server, so a partial failure surfaces as an
    :class:`Err` on the outer :data:`Result` instead.
    """

    items: tuple[Versioned[ProgramSnapshot], ...]


@dataclass(slots=True, frozen=True)
class InstanceLease:
    """A live, TTL-bounded lease over a key prefix.

    Returned from :meth:`DataPlane.acquire_instance_lock`. The
    ``token`` field is the opaque server-minted lease id; renew /
    release calls present the same token so the server can token-CAS
    and surface :class:`LockLost` if a different holder has taken over.

    ``expires_at_monotonic`` is captured at acquisition time from
    ``time.monotonic()`` plus ``ttl_s`` so a watchdog can detect a
    near-expiry condition without a Redis round-trip. ``flag`` is a
    one-shot signal owned by the holder; the recovery path (Sentinel
    failover, lease lost) sets it.
    """

    token: LeaseToken
    key: str
    ttl_s: float
    expires_at_monotonic: float
    flag: OneShotFlag = field(compare=False, repr=False)


@dataclass(slots=True, frozen=True)
class EliteInserted:
    """The cell was empty; the candidate became the elite."""

    kind: Literal["inserted"] = "inserted"


@dataclass(slots=True, frozen=True)
class EliteSwapped:
    """The candidate displaced the prior elite. ``displaced_id`` carries the loser."""

    displaced_id: ProgramId
    kind: Literal["swapped"] = "swapped"


@dataclass(slots=True, frozen=True)
class EliteRejected:
    """The candidate lost the comparison. ``occupant_id`` carries the survivor."""

    occupant_id: ProgramId
    kind: Literal["rejected"] = "rejected"


type EliteSwapOutcome = EliteInserted | EliteSwapped | EliteRejected
"""Discriminated outcome of :meth:`DataPlane.try_replace_elite`.

Pattern-match on the variant::

    match outcome:
        case EliteInserted():
            ...
        case EliteSwapped(displaced_id=loser_id):
            ...
        case EliteRejected(occupant_id=winner_id):
            ...

The ``kind`` Literal field exists so a JSON / msgpack codec can
round-trip the variant tag without runtime reflection on the dataclass
class.
"""


class DataPlane:
    """The single object that talks to Redis.

    Constructed once per process. Wired through Hydra config (engine
    startup) or directly in tests. Other modules that previously imported
    ``redis-py`` are progressively migrated to call this object's
    methods instead of building their own connection.
    """

    def __init__(
        self,
        redis_url: str,
        *,
        key_prefix: str = _DEFAULT_KEY_PREFIX,
        max_connections: int = 64,
        socket_timeout_s: float = 30.0,
        socket_connect_timeout_s: float = 10.0,
    ) -> None:
        self._connection = RedisConnection(
            redis_url,
            key_prefix=key_prefix,
            max_connections=max_connections,
            socket_timeout_s=socket_timeout_s,
            socket_connect_timeout_s=socket_connect_timeout_s,
        )
        self._lua: LuaRegistry | None = None
        self._started: bool = False

    # ── lifecycle ────────────────────────────────────────────────────

    async def startup(self) -> None:
        """Initialise the connection pool, load scripts, prime FSM tables.

        Idempotent: re-calling on a started coordinator is a no-op. On
        partial failure (e.g. FSM table load fails) the connection pool
        is closed and the lua-registry handle is dropped before the
        exception propagates so the coordinator is left in the
        pre-startup state — the caller can retry without leaking
        sockets or holding a half-initialised registry.
        """
        if self._started:
            return
        await self._connection.startup()
        try:
            lua = LuaRegistry(self._connection.pool)
            self._register_builtin_scripts(lua)
            await lua.load_all()
            self._lua = lua
            await load_fsm_table(
                self._connection.pool,
                key_prefix=self._connection.key_prefix,
                name="program_state",
                table=PROGRAM_STATE_TRANSITIONS,
            )
            await load_fsm_table(
                self._connection.pool,
                key_prefix=self._connection.key_prefix,
                name="claim_state",
                table=CLAIM_STATE_TRANSITIONS,
            )
            await load_fsm_table(
                self._connection.pool,
                key_prefix=self._connection.key_prefix,
                name="lock_state",
                table=LOCK_STATE_TRANSITIONS,
            )
        except Exception:
            # Roll back the pool AND the registry handle so the
            # coordinator is observably in its pre-startup state. A
            # later script-load attempt after a successful pool-startup
            # leaves ``self._lua`` non-None on its own, which would lie
            # about progress; the explicit reset to None below is
            # required for the invariant
            # ``self._started == (self._lua is not None)``.
            await self._connection.shutdown()
            self._lua = None
            raise
        self._started = True
        logger.info("DataPlane started: prefix={}", self._connection.key_prefix)

    async def shutdown(self) -> None:
        """Release background tasks, close the connection pool.

        Idempotent. Wraps any internal failure in :class:`ShutdownError`
        so callers see a single typed error class on the way out.
        """
        if not self._started:
            return
        try:
            # TODO: cancel registered renewers + drain in-flight sagas as
            # those become real in their respective follow-up changes.
            await self._connection.shutdown()
        except Exception as exc:
            raise ShutdownError(reason=repr(exc)) from exc
        finally:
            self._lua = None
            self._started = False

    @property
    def started(self) -> bool:
        return self._started

    @property
    def key_prefix(self) -> str:
        return self._connection.key_prefix

    # ── internals ────────────────────────────────────────────────────

    def _require_started(self, method: str) -> LuaRegistry:
        """Return the LuaRegistry or raise :class:`NotStartedError`.

        Every state-access method body that lands in subsequent changes
        must call this guard first; the ``method`` argument flows into
        the error's diagnostic field so a misbehaving caller is
        identified without a traceback walk.
        """
        lua = self._lua
        if lua is None:
            raise NotStartedError(method=method)
        return lua

    def _require_lua(self) -> LuaRegistry:
        """Deprecated internal alias for :meth:`_require_started`.

        Retained because the method bodies that have not yet landed
        reference this name. New code uses :meth:`_require_started` with
        an explicit method label.
        """
        return self._require_started("_require_lua")

    # ── program FSM ──────────────────────────────────────────────────

    async def transition_program_state(
        self,
        program_id: ProgramId,
        *,
        token: Token[ProgramId],
        expected_from: ProgramState | None,
        to: ProgramState,
        patch: ProgramPatch | None = None,
        deadline_monotonic: float | None = None,
    ) -> Result[Versioned[ProgramSnapshot], DataPlaneError]:
        """Validated FSM transition for a program; one atomic Lua call.

        ``token`` is consumed on entry; the caller must mint a fresh
        token (via :func:`mint_split` from the engine-root token) per
        call. ``expected_from=None`` means any current state is
        acceptable; the FSM table still validates ``(from, to)``.
        ``patch`` is a typed :class:`ProgramPatch` merged into the
        persisted program blob before the state advance.
        """
        # TODO: call self._require_started("transition_program_state")
        # and replace the NotImplementedError raise with the Lua call.
        _ = (self, program_id, token, expected_from, to, patch, deadline_monotonic)
        raise NotImplementedError("transition_program_state")

    async def transition_program_state_batch(
        self,
        items: tuple[BatchTransitionItem, ...],
        *,
        deadline_monotonic: float | None = None,
    ) -> Result[BatchTransitionOutcome, DataPlaneError]:
        """Apply a batch of program-FSM transitions atomically.

        Either every item in ``items`` succeeds or none do. A partial
        failure surfaces as an :class:`Err` on the outer
        :data:`Result`; the :class:`BatchTransitionOutcome` is only
        produced on full-batch success. The order of
        ``BatchTransitionOutcome.items`` matches ``items``.

        Each item carries its own token; the batch helper does not
        accept a single combined token because the underlying subspaces
        are per-program.
        """
        # TODO: call self._require_started("transition_program_state_batch")
        # and dispatch a single Lua MULTI / EVALSHA call.
        _ = (self, items, deadline_monotonic)
        raise NotImplementedError("transition_program_state_batch")

    async def read_program(
        self,
        program_id: ProgramId,
        *,
        min_epoch: int = 0,
        min_generation: int = 0,
    ) -> Result[Versioned[ProgramSnapshot] | None, DataPlaneError]:
        """Versioned program read with optional freshness floor.

        Returns ``Ok(None)`` if the program is unknown (not yet
        written); ``Ok(Versioned(...))`` if it exists; ``Err(...)`` for
        infra failures or a freshness-floor violation.
        """
        # TODO: call self._require_started("read_program") and dispatch
        # the read Lua / HGET-pipeline.
        _ = (self, program_id, min_epoch, min_generation)
        raise NotImplementedError("read_program")

    # ── instance lock ────────────────────────────────────────────────

    async def acquire_instance_lock(
        self,
        prefix: KeyPrefix,
        *,
        ttl_s: float,
        deadline_monotonic: float | None = None,
    ) -> Result[InstanceLease, DataPlaneError]:
        """Acquire a TTL-bounded lease over a key prefix.

        Returns a typed :class:`InstanceLease` (carried via :class:`Ok`)
        or a typed :class:`gigaevo.dataplane.errors.LockHeld` on the
        contention path. The lease's :class:`OneShotFlag` is owned by
        this coordinator; a Sentinel failover or detected lock-lost
        condition raises the flag so the caller's hot path can short-
        circuit to recovery without a blocking renewal call.
        """
        # TODO: call self._require_started("acquire_instance_lock")
        # before the SETNX-with-token + EXPIRE Lua call.
        _ = (self, prefix, ttl_s, deadline_monotonic)
        raise NotImplementedError("acquire_instance_lock")

    async def renew_instance_lock(
        self,
        lease: InstanceLease,
        *,
        ttl_s: float,
    ) -> Result[InstanceLease, DataPlaneError]:
        """Token-CAS EXPIRE; fails with :class:`LockLost` on token mismatch.

        On success returns a fresh :class:`InstanceLease` with the
        updated ``ttl_s`` / ``expires_at_monotonic`` — the input lease
        becomes logically stale and should not be reused for the
        renew / release call pair after this point.
        """
        # TODO: call self._require_started("renew_instance_lock") and
        # token-CAS the EXPIRE.
        _ = (self, lease, ttl_s)
        raise NotImplementedError("renew_instance_lock")

    async def release_instance_lock(self, lease: InstanceLease) -> None:
        """Token-CAS DEL; no-op if the lease was already released or lost."""
        # TODO: call self._require_started("release_instance_lock") and
        # token-CAS the DEL.
        _ = (self, lease)
        raise NotImplementedError("release_instance_lock")

    # ── archive ──────────────────────────────────────────────────────

    async def try_replace_elite(
        self,
        cell: CellKey,
        candidate_id: ProgramId,
        *,
        token: Token[CellKey],
        candidate_score: float,
        tiebreak_bit: int,
        deadline_monotonic: float | None = None,
    ) -> Result[EliteSwapOutcome, DataPlaneError]:
        """Atomic archive cell swap.

        Comparison reduces to ``(candidate_score, tiebreak_bit)`` —
        both computed Python-side from the caller's ``is_better``
        predicate before the Lua call. The Lua script atomically reads
        the occupant's score, compares, and either swaps, inserts (if
        the cell was empty), or rejects. The discriminated
        :data:`EliteSwapOutcome` carries the loser / survivor id so the
        caller can route follow-up work (e.g. discarding the displaced
        program) without a second read.
        """
        # TODO: call self._require_started("try_replace_elite") before
        # the compare-and-swap Lua call.
        _ = (
            self,
            cell,
            candidate_id,
            token,
            candidate_score,
            tiebreak_bit,
            deadline_monotonic,
        )
        raise NotImplementedError("try_replace_elite")

    # ── CRDT counters ────────────────────────────────────────────────

    async def crdt_inc(
        self,
        key: CounterKey,
        *,
        actor: ActorIdentity,
        delta: int = 1,
        deadline_monotonic: float | None = None,
    ) -> Result[int, DataPlaneError]:
        """Increment a per-actor G-counter; consensus value is sum across actors.

        Returns the post-increment per-actor value (not the cross-actor
        sum). Use :meth:`crdt_read` for the consensus sum.

        ``delta`` may be negative; the per-actor sub-count is signed. The
        G-counter merge invariant only requires that *each actor* writes
        monotonically — callers that decrement should hold a token for
        that actor's subspace.
        """
        _ = deadline_monotonic  # deadline propagation lands with the connection-pool wait-timeout
        lua = self._require_lua()
        counts_key, gen_key, epoch_key = self._counter_keys(key)
        try:
            raw = await lua.evalsha(
                _SCRIPT_COUNTER_INC,
                keys=[counts_key, gen_key, epoch_key],
                args=[actor.pack(), int(delta)],
            )
        except DataPlaneError as exc:
            return Err(exc)
        # Lua returns ASCII-encoded ints under decode_responses=True.
        new_count = int(raw[0])
        return Ok(new_count)

    async def crdt_read(
        self,
        key: CounterKey,
        *,
        min_epoch: int = 0,
        min_generation: int = 0,
    ) -> Result[Versioned[int], DataPlaneError]:
        """Read a G-counter as a :class:`Versioned` sum across all actors.

        Per-actor reads are not exposed: a G-counter's only meaningful
        observable is the cross-actor sum. The freshness floor is the
        usual ``(min_epoch, min_generation)`` pair; reads below either
        floor return :class:`StaleReadError`.

        Three Redis commands run as a single non-transactional pipeline
        (one round-trip). The G-counter's eventual-consistency model
        does not require the three reads to be atomic — a HGETALL that
        misses the last increment merely yields a slightly-older
        ``Versioned`` whose epoch / generation reflect that staleness,
        which is exactly what the freshness floor catches.
        """
        self._require_lua()
        counts_key, gen_key, epoch_key = self._counter_keys(key)
        redis = self._connection.pool
        pipe = redis.pipeline(transaction=False)
        pipe.hgetall(counts_key)
        pipe.get(gen_key)
        pipe.get(epoch_key)
        counts_map, gen_raw, epoch_raw = await pipe.execute()  # type: ignore[misc]
        generation = int(gen_raw) if gen_raw is not None else 0
        epoch = int(epoch_raw) if epoch_raw is not None else 0
        total = sum(int(v) for v in counts_map.values()) if counts_map else 0
        versioned = Versioned(value=total, epoch=epoch, generation=generation)
        if not versioned.is_at_least(min_epoch, min_generation):
            return Err(
                StaleReadError(
                    observed_epoch=epoch,
                    observed_generation=generation,
                    min_epoch=min_epoch,
                    min_generation=min_generation,
                )
            )
        return Ok(versioned)

    # ── built-in-script bookkeeping ──────────────────────────────────

    def _register_builtin_scripts(self, lua: LuaRegistry) -> None:
        """Load every per-resource Lua source into the registry.

        Called once during :meth:`startup` before :meth:`LuaRegistry.load_all`
        so the SHA cache is primed in one round-trip. Each script lives
        as a ``.lua`` file under ``gigaevo/dataplane/scripts/``; the name
        used here matches the file stem.
        """
        lua.register(_SCRIPT_COUNTER_INC, load_lua_source(_SCRIPT_COUNTER_INC))

    def _counter_keys(self, key: CounterKey) -> tuple[str, str, str]:
        """Resolve the three Redis keys backing a CRDT counter.

        Returns ``(counts_hash, generation_counter, epoch_counter)``.
        The epoch counter is shared across every coordinator op (any
        write bumps it); the generation counter is per-counter so two
        independent counters do not race on each other's freshness
        witness.
        """
        prefix = self._connection.key_prefix
        return (
            f"{prefix}:{key}:counts",
            f"{prefix}:{key}:gen",
            f"{prefix}:ts",
        )


__all__ = [
    "BatchTransitionItem",
    "BatchTransitionOutcome",
    "DataPlane",
    "EliteInserted",
    "EliteRejected",
    "EliteSwapOutcome",
    "EliteSwapped",
    "InstanceLease",
    "ProgramPatch",
    "ProgramSnapshot",
]
