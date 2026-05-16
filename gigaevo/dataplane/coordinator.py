"""The :class:`DataPlane` — sole public surface for every Redis interaction.

This module ships as a *shell* with explicit lifecycle methods and
per-resource method stubs that raise :class:`NotImplementedError`. Each
follow-up change fills in one method group (program FSM, instance
lock, archive cell swap, CRDT counters, transaction saga).

The coordinator owns:

    - one :class:`RedisConnection` (the connection pool)
    - one :class:`LuaRegistry` (script SHAs)
    - the FSM tables loaded into Redis via
      :func:`gigaevo.dataplane.transitions.load_fsm_table`
    - the engine-root :class:`Token`\\ s (split per subspace)

Outside this module, no other code should import ``redis`` or
``redis.asyncio`` directly — the ruff config enforces this (see
``lints.toml``).
"""

from __future__ import annotations

from typing import Any, Final

from loguru import logger

from .connection import RedisConnection
from .errors import NotStartedError, ShutdownError
from .ids import (
    CellKey,
    CounterKey,
    KeyPrefix,
    ProgramId,
)
from .models import Result, Versioned
from .permissions import Token
from .scripts import LuaRegistry
from .transitions import (
    CLAIM_STATE_TRANSITIONS,
    LOCK_STATE_TRANSITIONS,
    PROGRAM_STATE_TRANSITIONS,
    ProgramState,
    load_fsm_table,
)

_DEFAULT_KEY_PREFIX: Final[str] = "gigaevo"


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
        is closed before the exception propagates so callers can retry
        without leaking sockets.
        """
        if self._started:
            return
        await self._connection.startup()
        try:
            lua = LuaRegistry(self._connection.pool)
            # Per-resource scripts register themselves before load_all.
            await lua.load_all()
            self._lua = lua
            await load_fsm_table(
                self._connection.pool,
                key_prefix=self._connection.key_prefix,
                name="program_state",
                table=PROGRAM_STATE_TRANSITIONS,  # type: ignore[arg-type]
            )
            await load_fsm_table(
                self._connection.pool,
                key_prefix=self._connection.key_prefix,
                name="claim_state",
                table=CLAIM_STATE_TRANSITIONS,  # type: ignore[arg-type]
            )
            await load_fsm_table(
                self._connection.pool,
                key_prefix=self._connection.key_prefix,
                name="lock_state",
                table=LOCK_STATE_TRANSITIONS,  # type: ignore[arg-type]
            )
        except Exception:
            # Roll back the pool so the caller is left with a clean state.
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

    def _require_lua(self) -> LuaRegistry:
        """Return the LuaRegistry or raise :class:`NotStartedError`."""
        lua = self._lua
        if lua is None:
            raise NotStartedError(method="_require_lua")
        return lua

    # ── program FSM ──────────────────────────────────────────────────

    async def transition_program_state(
        self,
        program_id: ProgramId,
        *,
        token: Token[ProgramId],
        expected_from: ProgramState | None,
        to: ProgramState,
        patch: dict[str, Any] | None = None,
        deadline_monotonic: float | None = None,
    ) -> Result[Versioned[Any], Any]:
        """Validated FSM transition for a program; one atomic Lua call.

        ``token`` is consumed on entry; the caller must mint a fresh
        token (via :func:`mint_split` from the engine-root token) per
        call. ``expected_from=None`` means any current state is
        acceptable; the FSM table still validates ``(from, to)``.
        ``patch`` is a dict of fields to merge into the persisted
        program blob before the state advance.
        """
        _ = (self, program_id, token, expected_from, to, patch, deadline_monotonic)
        raise NotImplementedError("transition_program_state")

    async def read_program(
        self,
        program_id: ProgramId,
        *,
        min_epoch: int = 0,
        min_generation: int = 0,
    ) -> Result[Versioned[Any] | None, Any]:
        """Versioned program read with optional freshness floor."""
        _ = (self, program_id, min_epoch, min_generation)
        raise NotImplementedError("read_program")

    # ── instance lock ────────────────────────────────────────────────

    async def acquire_instance_lock(
        self,
        prefix: KeyPrefix,
        *,
        ttl_s: float,
        deadline_monotonic: float | None = None,
    ) -> Result[Any, Any]:
        """Acquire a TTL-bounded lease over a key prefix.

        Returns a typed ``InstanceLease`` (carried via :class:`Ok`) or a
        typed :class:`LockHeld` on the failure path.
        """
        _ = (self, prefix, ttl_s, deadline_monotonic)
        raise NotImplementedError("acquire_instance_lock")

    async def renew_instance_lock(
        self,
        lease: Any,
        *,
        ttl_s: float,
    ) -> Result[Any, Any]:
        """Token-CAS EXPIRE; fails with :class:`LockLost` on token mismatch."""
        _ = (self, lease, ttl_s)
        raise NotImplementedError("renew_instance_lock")

    async def release_instance_lock(self, lease: Any) -> None:
        """Token-CAS DEL; no-op if the lease was already released or lost."""
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
    ) -> Result[Any, Any]:
        """Atomic archive cell swap.

        Comparison reduces to ``(candidate_score, tiebreak_bit)`` —
        both computed Python-side from the caller's ``is_better``
        predicate before the Lua call. The Lua script atomically reads
        the occupant's score, compares, and either swaps or rejects.
        """
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
        actor: str,
        delta: int = 1,
        deadline_monotonic: float | None = None,
    ) -> Result[int, Any]:
        """Increment a per-actor G-counter; consensus value is sum across actors."""
        _ = (self, key, actor, delta, deadline_monotonic)
        raise NotImplementedError("crdt_inc")

    async def crdt_read(
        self,
        key: CounterKey,
        *,
        min_epoch: int = 0,
        min_generation: int = 0,
    ) -> Result[Versioned[int], Any]:
        """Read a G-counter as a :class:`Versioned` sum across all actors."""
        _ = (self, key, min_epoch, min_generation)
        raise NotImplementedError("crdt_read")


__all__ = ["DataPlane"]
