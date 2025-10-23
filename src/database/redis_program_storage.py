"""Redis-backed :class:`ProgramStorage` implementation.

Separated from *program_storage.py* to keep concerns isolated and allow the
abstract interface to stay lightweight.
"""

from __future__ import annotations

import asyncio
from typing import (
    Any,
    Awaitable,
    Callable,
    Iterable,
    TypeVar,
)
from itertools import islice
from loguru import logger
from pydantic import AnyUrl, BaseModel, Field
from redis import asyncio as aioredis
from redis.exceptions import WatchError
from src.database.merge_strategies import resolve_merge_strategy
from src.database.program_storage import ProgramStorage
from src.exceptions import StorageError
from src.programs.program import Program
from src.utils.json import dumps as _dumps
from src.utils.json import loads as _loads

__all__ = [
    "RedisProgramStorageConfig",
    "RedisProgramStorage",
]

T = TypeVar("T")


class RedisProgramStorageConfig(BaseModel):
    """Minimal, predictable Redis settings (unchanged key schema)."""
    redis_url: AnyUrl = Field(default="redis://localhost:6379/0")
    key_prefix: str = Field(default="metaevolve")

    program_key_tpl: str = Field(default="{prefix}:program:{pid}")
    status_stream_tpl: str = Field(default="{prefix}:status_events")
    status_set_tpl: str = Field(default="{prefix}:status:{status}")

    # Behavior
    max_retries: int = Field(default=5, ge=1)
    retry_delay: float = Field(default=0.2, ge=0.0)
    max_connections: int = Field(default=100, ge=10)
    connection_pool_timeout: float = Field(default=60.0, ge=1.0)
    health_check_interval: int = Field(default=180, ge=1)

    merge_strategy: str | Callable[[Program | None, Program], Program] = Field(default="additive")

    model_config = {"arbitrary_types_allowed": True, "extra": "forbid"}


# --------------------------- Storage ---------------------------

class RedisProgramStorage(ProgramStorage):

    _MGET_CHUNK: int = 1024

    def __init__(self, config: RedisProgramStorageConfig):
        self.config = config
        self._merge = resolve_merge_strategy(self.config.merge_strategy)
        self._redis: aioredis.Redis | None = None
        self._lock = asyncio.Lock()

    def _k_program(self, pid: str) -> str:
        return self.config.program_key_tpl.format(prefix=self.config.key_prefix, pid=pid)

    def _k_stream(self) -> str:
        return self.config.status_stream_tpl.format(prefix=self.config.key_prefix)

    def _k_status(self, status: str) -> str:
        return self.config.status_set_tpl.format(prefix=self.config.key_prefix, status=status)

    def _k_ts(self) -> str:
        return f"{self.config.key_prefix}:ts"

    async def _conn(self) -> aioredis.Redis:
        if self._redis is not None:
            return self._redis
        async with self._lock:
            if self._redis is None:
                r = aioredis.from_url(
                    str(self.config.redis_url),
                    decode_responses=True,
                    max_connections=self.config.max_connections,
                    health_check_interval=self.config.health_check_interval,
                    socket_connect_timeout=self.config.connection_pool_timeout,
                    socket_timeout=self.config.connection_pool_timeout,
                    retry_on_timeout=True,
                )
                await r.ping()
                logger.debug("[RedisProgramStorage] connected {}", self.config.redis_url)
                self._redis = r
        return self._redis

    async def _with_redis(self, name: str, fn: Callable[[aioredis.Redis], Awaitable[T]]) -> T:
        delay = self.config.retry_delay
        for attempt in range(1, self.config.max_retries + 1):
            try:
                return await fn(await self._conn())
            except Exception as e:
                if attempt == self.config.max_retries:
                    # Keep the log calm; raise a clear error.
                    logger.debug("[RedisProgramStorage] {} failed: {}", name, e)
                    raise StorageError(f"Redis op {name} failed: {e}") from e
                await asyncio.sleep(min(delay, 1.0))
                delay *= 2

    @staticmethod
    def _chunks(items: Iterable[str], n: int) -> Iterable[list[str]]:
        it = iter(items)
        while (batch := list(islice(it, n))):
            yield batch

    @staticmethod
    def _safe_deserialize(raw: str, ctx: str) -> Program | None:
        try:
            return Program.from_dict(_loads(raw))
        except Exception as e:
            # Soft log; skip the broken record.
            logger.debug("[RedisProgramStorage] bad JSON in {}: {}", ctx, e)
            return None

    async def _mget_by_keys(self, r: aioredis.Redis, keys: list[str], ctx: str) -> list[Program]:
        out: list[Program] = []
        for batch in self._chunks(keys, self._MGET_CHUNK):
            blobs = await r.mget(*batch)
            for raw in blobs:
                if raw:
                    p = self._safe_deserialize(raw, ctx)
                    if p is not None:
                        out.append(p)
        return out

    async def add(self, program: Program) -> None:
        async def _add(r: aioredis.Redis):
            key = self._k_program(program.id)
            status = program.state.value
            pipe = r.pipeline(transaction=False)
            counter = await r.incr(self._k_ts())
            new_program = program.model_copy(update={"atomic_counter": counter}, deep=True)
            pipe.set(key, _dumps(new_program.to_dict()))
            pipe.sadd(self._k_status(status), program.id)
            pipe.xadd(self._k_stream(), {"id": program.id, "status": status, "event": "created"},
                    maxlen=10_000, approximate=True)
            await pipe.execute()
        await self._with_redis("add", _add)

    async def update(self, program: Program):
        async def _update(r: aioredis.Redis):
            key = self._k_program(program.id)
            while True:
                try:
                    async with r.pipeline(transaction=True) as pipe:
                        await pipe.watch(key)
                        existing_raw = await pipe.get(key)
                        existing = self._safe_deserialize(existing_raw, "update/get") if existing_raw else None
                        counter = await r.incr(self._k_ts())
                        new_program = program.model_copy(update={"atomic_counter": int(counter)}, deep=True)
                        merged = self._merge(existing, new_program)
                        merged = merged.model_copy(update={"atomic_counter": int(counter)}, deep=True)
                        pipe.multi()  # enter transaction
                        pipe.set(key, _dumps(merged.to_dict()))
                        await pipe.execute()
                        break
                except WatchError:
                    continue
        await self._with_redis("update", _update)

    async def get(self, program_id: str) -> Program | None:
        async def _get(r: aioredis.Redis):
            raw = await r.get(self._k_program(program_id))
            return self._safe_deserialize(raw, f"get:{program_id}") if raw else None
        return await self._with_redis("get", _get)

    async def exists(self, program_id: str) -> bool:
        async def _exists(r: aioredis.Redis):
            return bool(await r.exists(self._k_program(program_id)))
        return await self._with_redis("exists", _exists)

    async def remove(self, program_id: str):
        async def _del(r: aioredis.Redis):
            await r.delete(self._k_program(program_id))
        await self._with_redis("remove", _del)

    async def transition_status(self, program_id: str, old: str | None, new: str) -> None:
        async def _tx(r: aioredis.Redis):
            pipe = r.pipeline(transaction=False)
            if old:
                pipe.srem(self._k_status(old), program_id)
            pipe.sadd(self._k_status(new), program_id)
            await pipe.execute()
        await self._with_redis("transition_status", _tx)

    async def publish_status_event(
        self,
        status: str,
        program_id: str,
        extra: dict[str, Any] | None = None,
    ) -> None:
        async def _event(r: aioredis.Redis):
            data = {"id": program_id, "status": status, **(extra or {})}
            pipe = r.pipeline(transaction=False)
            pipe.xadd(self._k_stream(), data, maxlen=10_000, approximate=True)
            pipe.sadd(self._k_status(status), program_id)
            await pipe.execute()
        await self._with_redis("publish_status_event", _event)

    async def _ids_for_status(self, status: str) -> list[str]:
        async def _members(r: aioredis.Redis):
            return list(await r.smembers(self._k_status(status)))
        return await self._with_redis("_ids_for_status", _members)

    async def get_all_by_status(self, status: str) -> list[Program]:
        ids = await self._ids_for_status(status)
        if not ids:
            return []

        async def _by_status(r: aioredis.Redis):
            keys = [self._k_program(pid) for pid in ids]
            programs = await self._mget_by_keys(r, keys, f"get_all_by_status:{status}")
            # If the set drifted, keep only exact matches.
            return [p for p in programs if p.state.value == status]

        return await self._with_redis("get_all_by_status", _by_status)

    async def mget(self, program_ids: list[str]) -> list[Program]:
        if not program_ids:
            return []

        async def _mget(r: aioredis.Redis):
            keys = [self._k_program(pid) for pid in program_ids]
            return await self._mget_by_keys(r, keys, "mget")

        return await self._with_redis("mget", _mget)

    async def get_all(self) -> list[Program]:
        """SCAN keys then fetch values via chunked MGET."""
        async def _scan_then_mget(r: aioredis.Redis):
            match = self._k_program("*")
            cursor = 0
            keys: list[str] = []
            while True:
                cursor, batch = await r.scan(cursor=cursor, match=match, count=1000)
                if batch:
                    keys.extend(batch)
                if cursor == 0:
                    break
            if not keys:
                return []
            return await self._mget_by_keys(r, keys, "get_all")

        return await self._with_redis("get_all", _scan_then_mget)

    async def wait_for_activity(self, timeout: float):
        """Wait using Redis stream; fall back to sleep on errors/timeouts."""
        poll_ms = max(1, int(timeout * 1000))
        try:
            redis = await self._conn()
            stream = self._k_stream()
            _ = await redis.xread({stream: "$"}, block=poll_ms, count=1)
            return
        except Exception as e:
            logger.debug("[RedisProgramStorage] wait_for_activity fallback: {}", e)
            await asyncio.sleep(timeout)

    async def flushdb(self):
        async def _flush(r: aioredis.Redis):
            await r.flushdb()
        await self._with_redis("flushdb", _flush)

    async def close(self):
        if self._redis is not None:
            await self._redis.close()
            self._redis = None