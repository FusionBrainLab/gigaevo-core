from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable
import gc
from itertools import islice
from types import TracebackType
from typing import Any, TypeVar

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
from gigaevo.utils.json import dumps as _dumps
from gigaevo.utils.json import loads as _loads
from gigaevo.utils.trackers.base import LogWriter

T = TypeVar("T")

__all__ = ["RedisProgramStorageConfig", "RedisProgramStorage"]

# Constants
MGET_CHUNK_SIZE = 1024
SCAN_BATCH_SIZE = 1000
STREAM_MAX_LEN = 10_000


class RedisProgramStorage(ProgramStorage):
    """Redis-backed program storage with distributed locking and metrics."""

    def __init__(
        self, config: RedisProgramStorageConfig, writer: LogWriter | None = None
    ):
        self.config = config
        self._merge = resolve_merge_strategy(config.merge_strategy)

        # Composed components
        self._conn = RedisConnection(config.to_connection_config())
        self._keys = RedisProgramKeys(config.to_key_config())
        self._lock = RedisInstanceLock(self._conn, self._keys, config.to_lock_config())
        self._metrics = RedisMetricsCollector(
            self._conn, self._keys, writer, config.metrics_interval
        )

    # --------------------- Context Manager ---------------------

    async def __aenter__(self) -> "RedisProgramStorage":
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

    @staticmethod
    def _chunks(items: Iterable[str], n: int) -> Iterable[list[str]]:
        it = iter(items)
        while batch := list(islice(it, n)):
            yield batch

    @staticmethod
    def _safe_deserialize(raw: str, ctx: str) -> Program | None:
        try:
            return Program.from_dict(_loads(raw))
        except Exception as e:
            logger.warning("[RedisProgramStorage] Corrupt data in {}: {}", ctx, e)
            return None

    async def _mget_by_keys(
        self, r: aioredis.Redis, keys: list[str], ctx: str
    ) -> list[Program]:
        out: list[Program] = []
        for batch in self._chunks(keys, MGET_CHUNK_SIZE):
            blobs = await r.mget(*batch)
            for raw in blobs:
                if raw:
                    p = self._safe_deserialize(raw, ctx)
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
            new_program = program.model_copy(
                update={"atomic_counter": counter}, deep=True
            )

            pipe = r.pipeline(transaction=False)
            pipe.set(key, _dumps(new_program.to_dict()))

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
                        merged = self._merge(existing, program).model_copy(
                            update={"atomic_counter": int(counter)}
                        )
                        pipe.multi()
                        pipe.set(key, _dumps(merged.to_dict()))
                        await pipe.execute()
                        break
                except WatchError:
                    continue

        await self._conn.execute("update", _update)

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

    async def mget(self, program_ids: list[str]) -> list[Program]:
        if not program_ids:
            return []

        async def _mget(r: aioredis.Redis) -> list[Program]:
            keys = [self._keys.program(pid) for pid in program_ids]
            return await self._mget_by_keys(r, keys, "mget")

        return await self._conn.execute("mget", _mget)

    async def size(self) -> int:
        """Count programs using SCAN (non-blocking)."""

        async def _size(r: aioredis.Redis) -> int:
            count = 0
            async for _ in r.scan_iter(
                match=self._keys.program_pattern(), count=SCAN_BATCH_SIZE
            ):
                count += 1
            return count

        return await self._conn.execute("size", _size)

    async def get_all(self) -> list[Program]:
        """Get all programs using SCAN + chunked MGET."""

        async def _scan_then_mget(r: aioredis.Redis) -> list[Program]:
            keys: list[str] = []
            async for key in r.scan_iter(
                match=self._keys.program_pattern(), count=SCAN_BATCH_SIZE
            ):
                keys.append(key)
            if not keys:
                return []
            return await self._mget_by_keys(r, keys, "get_all")

        return await self._conn.execute("get_all", _scan_then_mget)

    async def get_all_program_ids(self) -> list[str]:
        """Return program IDs (not full Redis keys) using SCAN."""

        async def _get_all_ids(r: aioredis.Redis) -> list[str]:
            ids: list[str] = []
            async for key in r.scan_iter(
                match=self._keys.program_pattern(), count=SCAN_BATCH_SIZE
            ):
                ids.append(key.split(":")[-1])
            return ids

        return await self._conn.execute("get_all_program_ids", _get_all_ids)

    async def has_data(self) -> bool:
        """Check if database has any programs."""

        async def _check(r: aioredis.Redis) -> bool:
            async for _ in r.scan_iter(match=self._keys.program_pattern(), count=1):
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
            data = {"id": program_id, "status": status, **(extra or {})}
            await r.xadd(
                self._keys.status_stream(),
                data,
                maxlen=STREAM_MAX_LEN,
                approximate=True,
            )

        await self._conn.execute("publish_status_event", _event)

    async def get_all_by_status(self, status: str) -> list[Program]:
        ids = await self._ids_for_status(status)
        if not ids:
            return []

        async def _by_status(r: aioredis.Redis) -> list[Program]:
            keys = [self._keys.program(pid) for pid in ids]
            programs = await self._mget_by_keys(r, keys, f"get_all_by_status:{status}")
            return [p for p in programs if p.state.value == status]

        return await self._conn.execute("get_all_by_status", _by_status)

    async def count_by_status(self, status: str) -> int:
        """Return count of programs with the given status (without fetching data)."""

        async def _count(r: aioredis.Redis) -> int:
            return await r.scard(self._keys.status_set(status))

        return await self._conn.execute("count_by_status", _count)

    async def _ids_for_status(self, status: str) -> list[str]:
        async def _members(r: aioredis.Redis) -> list[str]:
            return list(await r.smembers(self._keys.status_set(status)))

        return await self._conn.execute("_ids_for_status", _members)

    async def atomic_state_transition(
        self, program: Program, old_state: str | None, new_state: str
    ) -> None:
        self._check_write_allowed("atomic_state_transition")

        async def _atomic(r: aioredis.Redis) -> None:
            key = self._keys.program(program.id)

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
                        updated = program.model_copy(
                            update={"atomic_counter": int(counter)}, deep=True
                        )

                        if existing:
                            updated = self._merge(existing, updated).model_copy(
                                update={"atomic_counter": int(counter)}, deep=False
                            )

                        pipe.multi()
                        pipe.set(key, _dumps(updated.to_dict()))

                        if old_state:
                            pipe.srem(self._keys.status_set(old_state), program.id)
                        pipe.sadd(self._keys.status_set(new_state), program.id)

                        pipe.xadd(
                            self._keys.status_stream(),
                            {
                                "id": program.id,
                                "status": new_state,
                                "event": "transition",
                            },
                            maxlen=STREAM_MAX_LEN,
                            approximate=True,
                        )

                        await pipe.execute()
                        break

                except WatchError:
                    logger.debug(
                        "[RedisProgramStorage] Concurrent modification for {}, retrying",
                        program.id,
                    )
                    continue

        await self._conn.execute("atomic_state_transition", _atomic)

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
