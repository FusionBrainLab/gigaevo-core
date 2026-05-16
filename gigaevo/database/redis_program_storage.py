from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable
import gc
from itertools import islice
from types import TracebackType
from typing import TYPE_CHECKING, Any, TypeVar, cast

from loguru import logger
from redis import asyncio as aioredis
from redis.exceptions import WatchError

from gigaevo.database.merge_strategies import resolve_merge_strategy
from gigaevo.database.program_storage import ProgramStorage
from gigaevo.database.redis import (
    RedisConnection,
    RedisInstanceLock,
    RedisMetricsCollector,
    RedisProgramKeys,
    RedisProgramStorageConfig,
)
from gigaevo.exceptions import StorageError
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState, validate_transition
from gigaevo.utils.json import dumps as _dumps
from gigaevo.utils.json import loads as _loads
from gigaevo.utils.trackers.base import LogWriter

if TYPE_CHECKING:
    from gigaevo.dataplane import DataPlane

T = TypeVar("T")

__all__ = [
    "RedisProgramStorage",
    "RedisProgramStorageConfig",
    "load_legacy_program_fsm",
]

# Constants
MGET_CHUNK_SIZE = 1024
SCAN_BATCH_SIZE = 1000
STREAM_MAX_LEN = 10_000


class RedisProgramStorage(ProgramStorage):
    """Redis-backed program storage with distributed locking and metrics.

    Optional ``dataplane`` parameter enables FSM-validated, single-Lua-call
    state transitions in :meth:`atomic_state_transition` and
    :meth:`fast_state_transition`. When ``dataplane`` is ``None`` the
    legacy WATCH/MULTI/EXEC path runs unchanged — fakeredis-backed
    integration tests, read-only analytics callers, and any pre-existing
    deployment that has not yet wired a coordinator continue to work.

    The dataplane path expects the coordinator's
    ``program_state`` FSM table to be loaded with the gigaevo lowercase
    state values rather than the dataplane's default uppercase enum (the
    persisted blob format predates the coordinator and uses lowercase).
    :func:`load_legacy_program_fsm` reloads the table; engine startup
    calls it once after :meth:`DataPlane.startup`. Per-call linear
    permission tokens are minted via :func:`mint_root` inside the
    routing path; threading a long-lived engine-root token through the
    storage stack is a follow-up.
    """

    def __init__(
        self,
        config: RedisProgramStorageConfig,
        writer: LogWriter | None = None,
        *,
        dataplane: DataPlane | None = None,
    ):
        super().__init__()
        self.config = config
        self._merge = resolve_merge_strategy(config.merge_strategy)  # type: ignore[arg-type]

        # Composed components
        self._conn = RedisConnection(config.to_connection_config())
        self._keys = RedisProgramKeys(config.to_key_config())
        self._lock = RedisInstanceLock(self._conn, self._keys, config.to_lock_config())
        self._metrics = RedisMetricsCollector(
            self._conn, self._keys, writer, config.metrics_interval
        )
        self._dataplane = dataplane

    # --------------------- Context Manager ---------------------

    async def __aenter__(self) -> RedisProgramStorage:
        """Acquire instance lock and start metrics collection."""
        if not self.config.read_only:
            await self._lock.acquire()
        # Ensure connection is established
        await self._conn.get()
        self._metrics.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Release resources."""
        await self.close()

    # --------------------- Helpers ---------------------

    async def with_redis(
        self, name: str, fn: Callable[[aioredis.Redis], Awaitable[T]]
    ) -> T:
        """Execute Redis operation. Compatibility shim for external code."""
        return await self._conn.execute(name, fn)

    def _check_write_allowed(self, operation: str) -> None:
        """Raise error if write operation is attempted in read-only mode."""
        if self.config.read_only:
            raise StorageError(
                f"Cannot perform '{operation}' in read-only mode. "
                f"Create storage without read_only=True for write operations."
            )

    _T = TypeVar("_T")

    @staticmethod
    def _chunks(items: Iterable[Any], n: int) -> Iterable[list[Any]]:
        it = iter(items)
        while batch := list(islice(it, n)):
            yield batch

    # ``Program`` Pydantic fields that must remain dicts after a lua
    # round-trip. Lua / cjson cannot distinguish an empty object ``{}``
    # from an empty array ``[]``: ``cjson.encode({})`` emits ``[]`` on
    # both inputs, so any empty dict field on the persisted blob comes
    # back as a list after the coordinator's ``transition_state.lua``
    # re-encodes the merged payload. Normalising those fields back to
    # ``{}`` on read keeps the Program model schema satisfied without
    # forcing the caller to remember the wire-format quirk.
    _DICT_FIELDS: frozenset[str] = frozenset({"stage_results", "metrics", "metadata"})

    @classmethod
    def _safe_deserialize(
        cls,
        raw: str,
        ctx: str,
        *,
        exclude: frozenset[str] | None = None,
    ) -> Program | None:
        try:
            data = _loads(raw)
            # The dataplane's ``transition_state.lua`` stamps the
            # post-transition blob with an ``epoch`` field equal to the
            # bumped global counter (``{prefix}:ts``). The legacy storage
            # uses the same counter under the name ``atomic_counter`` on
            # the :class:`Program` Pydantic model, which forbids extra
            # fields. Promote ``epoch`` into ``atomic_counter`` when the
            # explicit ``atomic_counter`` is absent so callers see a
            # uniform field name regardless of which write path produced
            # the blob.
            if isinstance(data, dict):
                if "epoch" in data:
                    epoch = data.pop("epoch")
                    if "atomic_counter" not in data and isinstance(epoch, int):
                        data["atomic_counter"] = epoch
                for fname in cls._DICT_FIELDS:
                    if isinstance(data.get(fname), list) and not data[fname]:
                        data[fname] = {}
            return Program.from_dict(data, exclude=exclude)
        except Exception as e:
            logger.warning("[RedisProgramStorage] Corrupt data in {}: {}", ctx, e)
            return None

    async def _mget_by_keys(
        self,
        r: aioredis.Redis,
        keys: list[str],
        ctx: str,
        *,
        exclude: frozenset[str] | None = None,
    ) -> list[Program]:
        out: list[Program] = []
        for batch in self._chunks(keys, MGET_CHUNK_SIZE):
            blobs = await r.mget(*batch)
            for raw in blobs:
                if raw:
                    p = self._safe_deserialize(raw, ctx, exclude=exclude)
                    if p is not None:
                        out.append(p)
        return out

    # --------------------- CRUD Operations ---------------------

    async def add(self, program: Program) -> None:
        """Add a new program. If program exists, cleans up old status set first."""
        self._check_write_allowed("add")

        async def _add(r: aioredis.Redis) -> None:
            key = self._keys.program(program.id)
            new_status = program.state.value

            # Check if program already exists and get old status
            existing_raw = await r.get(key)
            old_status: str | None = None
            if existing_raw:
                existing = self._safe_deserialize(existing_raw, "add/get")
                if existing:
                    old_status = existing.state.value

            counter = await r.incr(self._keys.timestamp())
            data = program.to_dict()
            data["atomic_counter"] = int(counter)

            pipe = r.pipeline(transaction=False)
            pipe.set(key, _dumps(data))

            # Clean up old status set if different
            if old_status and old_status != new_status:
                pipe.srem(self._keys.status_set(old_status), program.id)

            pipe.sadd(self._keys.status_set(new_status), program.id)
            pipe.xadd(
                self._keys.status_stream(),
                {"id": program.id, "status": new_status, "event": "created"},
                maxlen=STREAM_MAX_LEN,
                approximate=True,
            )
            await pipe.execute()

        await self._conn.execute("add", _add)

    async def get(self, program_id: str) -> Program | None:
        async def _get(r: aioredis.Redis) -> Program | None:
            raw = await r.get(self._keys.program(program_id))
            return self._safe_deserialize(raw, f"get:{program_id}") if raw else None

        return await self._conn.execute("get", _get)

    async def update(self, program: Program) -> None:
        self._check_write_allowed("update")

        async def _update(r: aioredis.Redis) -> None:
            key = self._keys.program(program.id)
            retries = 0
            while True:
                try:
                    async with r.pipeline(transaction=True) as pipe:
                        await pipe.watch(key)
                        existing_raw = await pipe.get(key)
                        existing = (
                            self._safe_deserialize(existing_raw, "update/get")
                            if existing_raw
                            else None
                        )
                        counter = await r.incr(self._keys.timestamp())
                        merged = self._merge(existing, program)
                        data = merged.to_dict()
                        data["atomic_counter"] = int(counter)
                        pipe.multi()
                        pipe.set(key, _dumps(data))
                        await pipe.execute()
                        break
                except WatchError:
                    retries += 1
                    if retries > 1:
                        await asyncio.sleep(min(0.001 * (2 ** (retries - 2)), 0.032))
                    continue

        await self._conn.execute("update", _update)

    async def write_exclusive(self, program: Program) -> None:
        """Fast write: 2 RT (INCR + SET) instead of 4 RT (WATCH + GET + INCR + MULTI/SET/EXEC).

        Safe only when the caller holds exclusive ownership of this program
        (i.e., during DAG execution where asyncio.Lock + RedisInstanceLock
        prevent concurrent writes).
        """
        self._check_write_allowed("write_exclusive")

        async def _write(r: aioredis.Redis) -> None:
            key = self._keys.program(program.id)
            counter = await r.incr(self._keys.timestamp())
            data = program.to_dict()
            data["atomic_counter"] = int(counter)
            await r.set(key, _dumps(data))

        await self._conn.execute("write_exclusive", _write)

    async def remove(self, program_id: str) -> None:
        """Remove a program and clean up its status set entry."""
        self._check_write_allowed("remove")

        async def _del(r: aioredis.Redis) -> None:
            key = self._keys.program(program_id)

            # Get program to find its status
            existing_raw = await r.get(key)
            old_status: str | None = None
            if existing_raw:
                existing = self._safe_deserialize(existing_raw, "remove/get")
                if existing:
                    old_status = existing.state.value

            pipe = r.pipeline(transaction=False)
            pipe.delete(key)

            # Clean up status set
            if old_status:
                pipe.srem(self._keys.status_set(old_status), program_id)

            await pipe.execute()

        await self._conn.execute("remove", _del)

    async def exists(self, program_id: str) -> bool:
        async def _exists(r: aioredis.Redis) -> bool:
            return bool(await r.exists(self._keys.program(program_id)))

        return await self._conn.execute("exists", _exists)

    async def mget(
        self,
        program_ids: list[str],
        *,
        exclude: frozenset[str] | None = None,
    ) -> list[Program]:
        if not program_ids:
            return []

        async def _mget(r: aioredis.Redis) -> list[Program]:
            keys = [self._keys.program(pid) for pid in program_ids]
            return await self._mget_by_keys(r, keys, "mget", exclude=exclude)

        return await self._conn.execute("mget", _mget)

    async def _all_program_ids_from_sets(self, r: aioredis.Redis) -> list[str]:
        """Get all program IDs via SUNION of status sets.

        Faster than SCAN because it reads pre-indexed sets directly instead of
        iterating the entire keyspace with pattern matching.  Every program is
        in exactly one status set (invariant maintained by add/transition ops).
        """
        set_keys = [self._keys.status_set(s.value) for s in ProgramState]
        return list(await r.sunion(*set_keys))  # type: ignore[misc,arg-type]

    async def size(self) -> int:
        """Count programs by summing SCARD across status sets (O(1) per set)."""

        async def _size(r: aioredis.Redis) -> int:
            total = 0
            for s in ProgramState:
                total += await r.scard(self._keys.status_set(s.value))
            return total

        return await self._conn.execute("size", _size)

    async def get_all(self, *, exclude: frozenset[str] | None = None) -> list[Program]:
        """Get all programs using SCAN + chunked MGET.

        Args:
            exclude: Optional set of field names to skip during deserialization
                (passed through to :meth:`Program.from_dict`). Excluded fields
                get their Pydantic defaults. Example: ``exclude=frozenset({"stage_results"})``
                saves ~33% deserialization cost for analytics callers.
        """

        async def _scan_then_mget(r: aioredis.Redis) -> list[Program]:
            keys: list[str] = []
            async for key in r.scan_iter(
                match=self._keys.program_pattern(), count=SCAN_BATCH_SIZE
            ):
                keys.append(key)
            if not keys:
                return []
            return await self._mget_by_keys(r, keys, "get_all", exclude=exclude)

        return await self._conn.execute("get_all", _scan_then_mget)

    async def get_all_program_ids(self) -> list[str]:
        """Return program IDs (not full Redis keys) via status-set SUNION."""

        async def _get_all_ids(r: aioredis.Redis) -> list[str]:
            return await self._all_program_ids_from_sets(r)

        return await self._conn.execute("get_all_program_ids", _get_all_ids)

    async def has_data(self) -> bool:
        """Check if database has any programs (fast: SCARD on status sets)."""

        async def _check(r: aioredis.Redis) -> bool:
            for s in ProgramState:
                if await r.scard(self._keys.status_set(s.value)) > 0:
                    return True
            return False

        return await self._conn.execute("has_data", _check)

    # --------------------- Status Operations ---------------------

    async def transition_status(
        self, program_id: str, old: str | None, new: str
    ) -> None:
        self._check_write_allowed("transition_status")

        async def _tx(r: aioredis.Redis) -> None:
            pipe = r.pipeline(transaction=False)
            if old:
                pipe.srem(self._keys.status_set(old), program_id)
            pipe.sadd(self._keys.status_set(new), program_id)
            await pipe.execute()

        await self._conn.execute("transition_status", _tx)

    async def publish_status_event(
        self, status: str, program_id: str, extra: dict[str, Any] | None = None
    ) -> None:
        self._check_write_allowed("publish_status_event")

        async def _event(r: aioredis.Redis) -> None:
            data: dict[str, Any] = {
                "id": program_id,
                "status": status,
                **(extra or {}),
            }
            await r.xadd(
                self._keys.status_stream(),
                data,  # type: ignore[arg-type]
                maxlen=STREAM_MAX_LEN,
                approximate=True,
            )

        await self._conn.execute("publish_status_event", _event)

    async def get_all_by_status(
        self, status: str, *, exclude: frozenset[str] | None = None
    ) -> list[Program]:
        ids = await self._ids_for_status(status)
        if not ids:
            return []

        async def _by_status(r: aioredis.Redis) -> list[Program]:
            keys = [self._keys.program(pid) for pid in ids]
            programs = await self._mget_by_keys(
                r, keys, f"get_all_by_status:{status}", exclude=exclude
            )
            return [p for p in programs if p.state.value == status]

        return await self._conn.execute("get_all_by_status", _by_status)

    async def count_by_status(self, status: str) -> int:
        """Return count of programs with the given status (without fetching data)."""

        async def _count(r: aioredis.Redis) -> int:
            return await r.scard(self._keys.status_set(status))

        return await self._conn.execute("count_by_status", _count)

    async def get_ids_by_status(self, status: str) -> list[str]:
        """Return IDs of programs with the given status (no full fetch)."""
        return await self._ids_for_status(status)

    async def _ids_for_status(self, status: str) -> list[str]:
        async def _members(r: aioredis.Redis) -> list[str]:
            return list(await r.smembers(self._keys.status_set(status)))

        return await self._conn.execute("_ids_for_status", _members)

    async def _transition_via_dataplane(
        self,
        program: Program,
        old_state: str | None,
        new_state: str,
        *,
        method: str,
    ) -> None:
        """Route a single-program FSM transition through the coordinator.

        Builds a :class:`ProgramPatch` from the in-memory :class:`Program`
        (excluding the reserved ``state`` / ``id`` / ``epoch`` fields the
        Lua script owns), mints a per-call linear permission token, and
        invokes :meth:`DataPlane.transition_program_state`. The Lua
        script does the FSM legality check, idempotency dedup, blob
        merge, status-set update, and audit-stream append atomically.

        On success, the post-transition blob's epoch is mirrored into
        ``program.atomic_counter`` so callers that read it as a per-program
        revision (merge tiebreaker) observe a value consistent with the
        persisted blob.

        Failure modes:
            * ``Err(TransitionError(kind="illegal"))`` — the FSM table
              rejected ``(from, to)``; surfaces as :class:`StorageError`
              with ``illegal_transition`` in the message so call sites
              that previously got a silent state-machine drift now get
              a typed, observable failure (bug class #14).
            * ``Err(TransitionError(kind="stale"))`` — the blob does not
              exist or ``expected_from`` mismatched the observed state;
              the caller's pre-image is out of date.
            * Other ``Err`` variants surface as :class:`StorageError`
              with the kind and detail attached.

        The minted token is consumed inside :meth:`DataPlane.transition_program_state`;
        a fresh one is required per call. Threading a long-lived
        engine-root token through the storage stack so successive calls
        derive per-program sub-tokens by linear split rather than
        re-minting is a follow-up that lands together with the engine
        lifecycle migration.
        """
        from gigaevo.dataplane.ids import ProgramId as _DpProgramId
        from gigaevo.dataplane.models import Err
        from gigaevo.dataplane.permissions import mint_root

        dp = self._dataplane
        assert dp is not None  # narrowing for the type checker

        patch_fields = program.to_dict()
        # ``state`` / ``id`` / ``epoch`` are reserved by the Lua script
        # itself. ``atomic_counter`` is dropped here so the post-lua blob
        # carries only the coordinator's ``epoch`` field; the read path
        # in :meth:`_safe_deserialize` renames ``epoch`` to
        # ``atomic_counter`` when the latter is absent, keeping the
        # Program model schema consistent. Sending both would let the
        # caller-supplied (pre-INCR) value masquerade as the post-INCR
        # counter and break the monotonicity invariant downstream merge
        # tiebreakers rely on.
        for reserved in ("state", "id", "epoch", "atomic_counter"):
            patch_fields.pop(reserved, None)

        from gigaevo.dataplane.coordinator import ProgramPatch as _ProgramPatch

        program_id = _DpProgramId(program.id)
        token = mint_root(program_id)
        expected_from = ProgramState(old_state) if old_state else None
        target = ProgramState(new_state)

        # The dataplane's ``ProgramState`` enum uses uppercase values
        # while the persisted blob format uses lowercase; we pass the
        # gigaevo :class:`ProgramState` (lowercase ``.value``) directly
        # to the coordinator wrapper, which forwards ``.value`` verbatim
        # to the Lua script. The coordinator's FSM table must be loaded
        # with lowercase keys for the membership check to succeed —
        # see :func:`load_legacy_program_fsm`.
        result = await dp.transition_program_state(
            program_id,
            token=token,  # type: ignore[arg-type]
            expected_from=cast(Any, expected_from),
            to=cast(Any, target),
            patch=_ProgramPatch(fields=patch_fields),
        )

        if isinstance(result, Err):
            err = result.error
            kind = getattr(err, "kind", "unknown")
            detail = getattr(err, "detail", repr(err))
            raise StorageError(
                f"{method}: dataplane transition rejected "
                f"({kind}): {old_state!r} -> {new_state!r}: {detail}"
            )

        # Mirror the persisted counter into the in-memory program so
        # subsequent merge-tiebreak comparisons see the post-transition
        # value, matching the legacy-path invariant where the storage
        # bumps ``atomic_counter`` before returning to the caller.
        post_blob = result.value.value
        if isinstance(post_blob, dict) and "epoch" in post_blob:
            epoch_val = post_blob["epoch"]
            if isinstance(epoch_val, int):
                program.atomic_counter = epoch_val
        elif result.value.epoch:
            program.atomic_counter = result.value.epoch
        # Keep the in-memory state in sync with the persisted blob.
        # The Pydantic validator allows the same-state assignment as
        # a no-op; mismatches between the caller's requested state
        # and the persisted state are already represented by the
        # ``Err`` branch above (stale / illegal), so reaching here means
        # ``program.state`` should equal ``target``.
        if program.state != target:
            program.state = target

    async def atomic_state_transition(
        self, program: Program, old_state: str | None, new_state: str
    ) -> None:
        self._check_write_allowed("atomic_state_transition")

        if self._dataplane is not None:
            await self._transition_via_dataplane(
                program, old_state, new_state, method="atomic_state_transition"
            )
            return

        async def _atomic(r: aioredis.Redis) -> None:
            key = self._keys.program(program.id)
            retries = 0

            while True:
                try:
                    async with r.pipeline(transaction=True) as pipe:
                        await pipe.watch(key)

                        existing_raw = await pipe.get(key)
                        existing = (
                            self._safe_deserialize(existing_raw, "atomic_transition")
                            if existing_raw
                            else None
                        )

                        counter = await r.incr(self._keys.timestamp())

                        base = self._merge(existing, program) if existing else program
                        data = base.to_dict()
                        data["atomic_counter"] = int(counter)

                        # Use the MERGED state for status set operations, not
                        # the caller's new_state. This prevents dual-set
                        # membership when a concurrent transition (e.g. DISCARD
                        # by _maintain) wins the merge over the caller's
                        # requested state (e.g. DONE by _execute_dag).
                        actual_state = base.state.value

                        # Collect stale status sets to clean up
                        sets_to_remove: set[str] = set()
                        if old_state:
                            sets_to_remove.add(old_state)
                        if existing:
                            sets_to_remove.add(existing.state.value)
                        # Don't remove from the target set
                        sets_to_remove.discard(actual_state)

                        pipe.multi()
                        pipe.set(key, _dumps(data))

                        for s in sets_to_remove:
                            pipe.srem(self._keys.status_set(s), program.id)
                        pipe.sadd(self._keys.status_set(actual_state), program.id)

                        pipe.xadd(
                            self._keys.status_stream(),
                            {
                                "id": program.id,
                                "status": actual_state,
                                "event": "transition",
                            },
                            maxlen=STREAM_MAX_LEN,
                            approximate=True,
                        )

                        await pipe.execute()
                        break

                except WatchError:
                    retries += 1
                    if retries > 1:
                        await asyncio.sleep(min(0.001 * (2 ** (retries - 2)), 0.032))
                    logger.debug(
                        "[RedisProgramStorage] Concurrent modification for {}, retrying (attempt {})",
                        program.id,
                        retries,
                    )
                    continue

        await self._conn.execute("atomic_state_transition", _atomic)

    async def fast_state_transition(
        self, program: Program, old_state: str, new_state: str
    ) -> None:
        """Fast state transition: 2 RT (INCR + pipeline) instead of ~5 RT.

        Safe only when the caller holds exclusive single-process ownership
        (e.g., asyncio.Lock in ProgramStateManager). Does NOT provide cross-process
        safety — assumes each program is processed by exactly one engine instance.
        Unlike atomic_state_transition, does not WATCH/GET/MERGE.
        """
        self._check_write_allowed("fast_state_transition")

        if self._dataplane is not None:
            await self._transition_via_dataplane(
                program, old_state, new_state, method="fast_state_transition"
            )
            return

        async def _fast(r: aioredis.Redis) -> None:
            key = self._keys.program(program.id)
            counter = await r.incr(self._keys.timestamp())
            data = program.to_dict()
            data["atomic_counter"] = int(counter)

            pipe = r.pipeline(transaction=False)
            pipe.set(key, _dumps(data))
            if old_state != new_state:
                pipe.srem(self._keys.status_set(old_state), program.id)
            pipe.sadd(self._keys.status_set(new_state), program.id)
            pipe.xadd(
                self._keys.status_stream(),
                {"id": program.id, "status": new_state, "event": "transition"},
                maxlen=STREAM_MAX_LEN,
                approximate=True,
            )
            await pipe.execute()

        await self._conn.execute("fast_state_transition", _fast)

    async def batch_transition_state(
        self,
        programs: list[Program],
        old_state: str,
        new_state: str,
    ) -> int:
        """Batch-transition programs between states using pipelined ops.

        Much faster than individual atomic_state_transition calls for large
        batches (e.g., refresh phase with 5000 programs). Assumes exclusive
        ownership — no WATCH/MERGE needed.

        Returns the number of programs transitioned.
        """
        self._check_write_allowed("batch_transition_state")
        if not programs:
            return 0

        old_enum = ProgramState(old_state)
        new_enum = ProgramState(new_state)
        # The (old, new) transition pair is the same for every item in
        # the batch; validate once before walking the programs rather
        # than on each iteration.
        validate_transition(old_enum, new_enum)

        async def _batch(r: aioredis.Redis) -> int:
            old_set_key = self._keys.status_set(old_state)
            new_set_key = self._keys.status_set(new_state)
            stream_key = self._keys.status_stream()
            ts_key = self._keys.timestamp()

            count = 0
            for chunk in self._chunks(programs, MGET_CHUNK_SIZE):
                n = len(chunk)
                # Reserve N counters in one call
                end_counter = await r.incrby(ts_key, n)
                start_counter = end_counter - n + 1

                pipe = r.pipeline(transaction=False)
                chunk_ids = []
                for i, prog in enumerate(chunk):
                    # Per-program FSM check: the caller asserts the
                    # whole batch shares ``old_state`` as its precondition;
                    # surface a programming error here instead of letting
                    # a divergent in-memory state slip into the persisted
                    # blob.
                    validate_transition(prog.state, new_enum)
                    prog.state = new_enum
                    data = prog.to_dict()
                    data["atomic_counter"] = int(start_counter + i)

                    pipe.set(self._keys.program(prog.id), _dumps(data))
                    chunk_ids.append(prog.id)

                # Bulk SREM/SADD: one command per chunk instead of per program
                pipe.srem(old_set_key, *chunk_ids)
                pipe.sadd(new_set_key, *chunk_ids)
                pipe.xadd(
                    stream_key,
                    {"id": "batch", "status": new_state, "event": "batch_transition"},
                    maxlen=STREAM_MAX_LEN,
                    approximate=True,
                )
                await pipe.execute()
                count += n

            return count

        return await self._conn.execute("batch_transition_state", _batch)

    async def batch_transition_by_ids(
        self,
        program_ids: list[str],
        old_state: str,
        new_state: str,
    ) -> int:
        """Batch-transition programs by ID using raw JSON patching.

        Much faster than batch_transition_state for large batches because it
        skips Pydantic deserialization/reserialization entirely. Reads raw JSON
        blobs, patches the ``state`` field in-place, and writes back.

        Only transitions programs whose current state matches ``old_state``.
        Returns the number of programs actually transitioned.
        """
        self._check_write_allowed("batch_transition_by_ids")
        if not program_ids:
            return 0

        validate_transition(ProgramState(old_state), ProgramState(new_state))

        async def _batch_raw(r: aioredis.Redis) -> int:
            old_set_key = self._keys.status_set(old_state)
            new_set_key = self._keys.status_set(new_state)
            stream_key = self._keys.status_stream()
            ts_key = self._keys.timestamp()

            count = 0
            for id_chunk in self._chunks(program_ids, MGET_CHUNK_SIZE):
                keys = [self._keys.program(pid) for pid in id_chunk]
                blobs = await r.mget(*keys)

                # Filter: only patch programs that exist and are in old_state
                to_patch: list[tuple[str, str, dict]] = []  # (key, pid, parsed)
                for key, pid, raw in zip(keys, id_chunk, blobs):
                    if not raw:
                        continue
                    parsed = _loads(raw)
                    if parsed.get("state") == old_state:
                        to_patch.append((key, pid, parsed))

                if not to_patch:
                    continue

                n = len(to_patch)
                end_counter = await r.incrby(ts_key, n)
                start_counter = end_counter - n + 1

                pipe = r.pipeline(transaction=False)
                patch_ids = []
                for i, (key, pid, parsed) in enumerate(to_patch):
                    parsed["state"] = new_state
                    parsed["atomic_counter"] = int(start_counter + i)
                    pipe.set(key, _dumps(parsed))
                    patch_ids.append(pid)

                # Bulk SREM/SADD: one command per chunk instead of per program
                pipe.srem(old_set_key, *patch_ids)
                pipe.sadd(new_set_key, *patch_ids)
                pipe.xadd(
                    stream_key,
                    {"id": "batch", "status": new_state, "event": "batch_transition"},
                    maxlen=STREAM_MAX_LEN,
                    approximate=True,
                )
                await pipe.execute()
                count += n

            return count

        return await self._conn.execute("batch_transition_by_ids", _batch_raw)

    async def remove_ids_from_status_set(self, status: str, ids: list[str]) -> None:
        """Remove specific IDs from a status set using SREM."""
        if not ids:
            return
        self._check_write_allowed("remove_ids_from_status_set")

        async def _srem(r: aioredis.Redis) -> None:
            await r.srem(self._keys.status_set(status), *ids)

        await self._conn.execute("remove_ids_from_status_set", _srem)

    async def batch_move_status_sets(
        self,
        program_ids: list[str],
        from_status: str,
        to_status: str,
    ) -> None:
        """Move IDs between status sets WITHOUT modifying program data blobs.

        Much faster than batch_transition_by_ids for transitions to terminal
        states (e.g. DISCARDED) because it skips MGET/parse/patch/serialize.
        Only does SREM + SADD + XADD in a single pipeline.
        """
        if not program_ids:
            return
        self._check_write_allowed("batch_move_status_sets")

        async def _move(r: aioredis.Redis) -> None:
            from_key = self._keys.status_set(from_status)
            to_key = self._keys.status_set(to_status)
            stream_key = self._keys.status_stream()

            pipe = r.pipeline(transaction=False)
            pipe.srem(from_key, *program_ids)
            pipe.sadd(to_key, *program_ids)
            pipe.xadd(
                stream_key,
                {
                    "id": "batch_move",
                    "status": to_status,
                    "event": "batch_move_status",
                },
                maxlen=STREAM_MAX_LEN,
                approximate=True,
            )
            await pipe.execute()

        await self._conn.execute("batch_move_status_sets", _move)

    # --------------------- Run State (resume support) ---------------------

    async def save_run_state(self, field: str, value: int) -> None:
        """Persist a named integer counter into the run-state hash."""
        self._check_write_allowed("save_run_state")

        async def _set(r: aioredis.Redis) -> None:
            await r.hset(self._keys.run_state(), field, str(value))

        await self._conn.execute("save_run_state", _set)

    async def load_run_state(self, field: str) -> int | None:
        """Load a previously saved integer counter. Returns None if not found."""

        async def _get(r: aioredis.Redis) -> str | None:
            return await r.hget(self._keys.run_state(), field)

        raw = await self._conn.execute("load_run_state", _get)
        return int(raw) if raw is not None else None

    async def recover_stranded_programs(self) -> int:
        """Reset all RUNNING programs to QUEUED after a crash/kill.

        Uses write_exclusive (no merge) because the caller has exclusive access
        during startup, and merge_states(RUNNING, QUEUED) would wrongly keep RUNNING.
        Returns the number of programs recovered.
        """
        ids = await self.get_ids_by_status(ProgramState.RUNNING.value)
        if not ids:
            return 0

        recovered = 0
        for pid in ids:
            prog = await self.get(pid)
            if prog is None:
                # Dangling entry in status set — clean it up
                async def _clean(r: aioredis.Redis, _pid: str = pid) -> None:
                    await r.srem(
                        self._keys.status_set(ProgramState.RUNNING.value), _pid
                    )

                await self._conn.execute("recover_stranded_clean", _clean)
                continue

            # Crash recovery is a one-way reset of mid-flight RUNNING
            # programs back to the front of the queue; the gigaevo
            # forward FSM does not include RUNNING → QUEUED because
            # under normal evolution every RUNNING program either
            # completes (DONE) or is discarded. Assert the precondition
            # before the bypass so a future caller pointing this code at
            # a program in any other state surfaces the misuse instead
            # of silently corrupting the FSM invariant.
            assert prog.state == ProgramState.RUNNING, (
                f"recover_stranded_programs: expected RUNNING, "
                f"got {prog.state.value} for {prog.id}"
            )
            prog.state = ProgramState.QUEUED
            await self.write_exclusive(prog)

            async def _move(r: aioredis.Redis, _pid: str = pid) -> None:
                pipe = r.pipeline(transaction=False)
                pipe.srem(self._keys.status_set(ProgramState.RUNNING.value), _pid)
                pipe.sadd(self._keys.status_set(ProgramState.QUEUED.value), _pid)
                await pipe.execute()

            await self._conn.execute("recover_stranded_move", _move)
            recovered += 1

        logger.info(
            "[RedisProgramStorage] Recovered {} stranded RUNNING → QUEUED", recovered
        )
        return recovered

    # --------------------- Activity Monitoring ---------------------

    async def wait_for_activity(self, timeout: float) -> None:
        """Block on stream read; exits quickly during shutdown."""
        if self._conn.is_closing:
            return

        poll_ms = max(1, int(timeout * 1000))
        try:
            r = await self._conn.get()
            await r.xread({self._keys.status_stream(): "$"}, block=poll_ms, count=1)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.debug("[RedisProgramStorage] wait_for_activity fallback: {}", e)
            await asyncio.sleep(timeout)

    # --------------------- Admin Operations ---------------------

    async def flushdb(self) -> None:
        self._check_write_allowed("flushdb")

        async def _flush(r: aioredis.Redis) -> None:
            await r.flushdb()

        await self._conn.execute("flushdb", _flush)

    # --------------------- Instance Locking (delegates) ---------------------

    async def acquire_instance_lock(self) -> bool:
        """Acquire exclusive lock to prevent multiple instances."""
        if self.config.read_only:
            logger.info(
                "[RedisProgramStorage] Skipping instance lock (read-only mode) "
                "for prefix '{}'",
                self._keys.prefix,
            )
            return True
        return await self._lock.acquire()

    async def release_instance_lock(self) -> None:
        """Release the instance lock."""
        if self.config.read_only:
            return
        await self._lock.release()

    async def renew_instance_lock(self) -> bool:
        """Renew the instance lock to prevent expiry."""
        if self.config.read_only:
            return True
        return await self._lock.renew()

    # --------------------- Shutdown ---------------------

    async def close(self) -> None:
        """Close all resources gracefully."""
        # Release lock first
        if not self.config.read_only:
            await self._lock.release()

        # Stop metrics collection
        await self._metrics.stop()

        # Close connection
        await self._conn.close()

        gc.collect()

    def __repr__(self) -> str:
        return (
            f"<RedisProgramStorage "
            f"prefix={self._keys.prefix!r} "
            f"connected={self._conn.is_connected} "
            f"read_only={self.config.read_only}>"
        )


async def load_legacy_program_fsm(dataplane: DataPlane) -> None:
    """Re-load the coordinator's program-state FSM with lowercase values.

    The dataplane ships an uppercase ``ProgramState`` enum whose values
    drive both the in-Redis FSM hash and the wire protocol of
    :meth:`DataPlane.transition_program_state`. The legacy program blob
    format predates the coordinator and persists ``state`` as the
    lowercase :class:`gigaevo.programs.program_state.ProgramState` value
    (``"queued"`` / ``"running"`` / ``"done"`` / ``"discarded"``).

    Routing the storage through the coordinator while keeping the
    persisted blob format unchanged requires the FSM hash to be keyed
    on the lowercase values so the Lua script's ``HGET FSM[from]``
    membership check matches what it reads from the blob. This helper
    reloads ``{key_prefix}:fsm:program_state`` from the gigaevo enum
    after ``await dataplane.startup()`` so the coordinator's transition
    semantics line up with the application-layer state vocabulary.

    Idempotent: callers may invoke this once per process at startup or
    after every coordinator restart. The reload is a single DEL + HSET
    pipeline.
    """
    from gigaevo.dataplane.transitions import load_fsm_table

    legacy_table: dict[ProgramState, set[ProgramState]] = {
        ProgramState.QUEUED: {
            ProgramState.QUEUED,
            ProgramState.RUNNING,
            ProgramState.DISCARDED,
        },
        ProgramState.RUNNING: {
            ProgramState.RUNNING,
            ProgramState.DONE,
            ProgramState.DISCARDED,
        },
        ProgramState.DONE: {
            ProgramState.DONE,
            ProgramState.QUEUED,
            ProgramState.DISCARDED,
        },
        ProgramState.DISCARDED: {
            ProgramState.DISCARDED,
        },
    }
    await load_fsm_table(
        dataplane._connection.pool,  # type: ignore[arg-type]  # private but stable API
        key_prefix=dataplane.key_prefix,
        name="program_state",
        table=legacy_table,
    )
