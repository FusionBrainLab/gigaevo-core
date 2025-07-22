"""Redis-backed :class:`ProgramStorage` implementation.

Separated from *program_storage.py* to keep concerns isolated and allow the
abstract interface to stay lightweight.
"""

from __future__ import annotations

import asyncio
import hashlib
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    TypeVar,
    Union,
)

from loguru import logger
from pydantic import AnyUrl, BaseModel, Field
from redis import asyncio as aioredis

from src.database.merge_strategies import resolve_merge_strategy
from src.database.program_storage import ProgramStorage
from src.exceptions import StorageError, ensure_not_none
from src.programs.program import Program
from src.utils.error_handling import (
    RetryConfig,
    resilient_operation,
)
from src.utils.json import dumps as _dumps
from src.utils.json import loads as _loads

__all__ = [
    "RedisProgramStorageConfig",
    "RedisProgramStorage",
]

T = TypeVar("T")


class RedisProgramStorageConfig(BaseModel):
    """Configuration for :class:`RedisProgramStorage`."""

    redis_url: AnyUrl = Field(
        default="redis://localhost:6379/0", description="Redis connection URL"
    )
    key_prefix: str = Field(
        default="metaevolve", description="Namespace prefix"
    )
    program_key_tpl: str = Field(
        default="{prefix}:program:{pid}",
        description="Template for individual program keys",
    )
    status_stream_tpl: str = Field(
        default="{prefix}:status_events",
        description="Redis stream for program events",
    )
    status_set_tpl: str = Field(
        default="{prefix}:status:{status}",
        description="Redis set for tracking program IDs by status",
    )

    max_retries: int = Field(default=5, ge=1)
    retry_delay: float = Field(default=0.2, ge=0.0)
    max_connections: int = Field(default=100, ge=10)
    connection_pool_timeout: float = Field(default=60.0, ge=1.0)
    health_check_interval: int = Field(default=180, ge=1)

    merge_strategy: Union[
        str, Callable[[Optional[Program], Program], Program]
    ] = Field(
        default="additive", description="Merge policy for concurrent updates"
    )

    completion_statuses: set[str] = Field(
        default_factory=lambda: {"dag_processing_completed"}
    )
    discard_statuses: set[str] = Field(default_factory=lambda: {"discarded"})

    model_config = {"arbitrary_types_allowed": True, "extra": "forbid"}


class RedisProgramStorage(ProgramStorage):
    """Redis-based :class:`ProgramStorage` using flexible, template-driven key layout."""

    def __init__(self, config: RedisProgramStorageConfig | None = None):
        self.config = config or RedisProgramStorageConfig()
        self._merge_strategy = resolve_merge_strategy(
            self.config.merge_strategy
        )
        self._redis: Optional[aioredis.Redis] = None
        self._lock = asyncio.Lock()

    def _prog_key(self, pid: str) -> str:
        return self.config.program_key_tpl.format(
            prefix=self.config.key_prefix, pid=pid
        )

    def _stream_key(self) -> str:
        return self.config.status_stream_tpl.format(
            prefix=self.config.key_prefix
        )

    def _status_set_key(self, status: str) -> str:
        return self.config.status_set_tpl.format(
            prefix=self.config.key_prefix, status=status
        )

    def hash_code(self, code: str) -> str:
        return hashlib.sha256(code.encode("utf-8")).hexdigest()

    async def _conn(self) -> aioredis.Redis:
        if self._redis is not None:
            return self._redis

        async with self._lock:
            if self._redis is None:
                redis = aioredis.from_url(
                    str(self.config.redis_url),
                    decode_responses=True,
                    max_connections=self.config.max_connections,
                    health_check_interval=self.config.health_check_interval,
                    socket_connect_timeout=self.config.connection_pool_timeout,
                    socket_timeout=self.config.connection_pool_timeout,
                    retry_on_timeout=True,
                )

                # Test connection
                try:
                    await redis.ping()
                    logger.info(
                        f"[RedisProgramStorage] Connected to Redis at {self.config.redis_url} with {self.config.max_connections} max connections"
                    )
                except Exception as e:
                    logger.error(
                        f"[RedisProgramStorage] Failed to connect to Redis: {e}"
                    )
                    raise

                self._redis = redis
            return self._redis

    async def _execute(
        self, name: str, fn: Callable[[aioredis.Redis], Awaitable[T]]
    ) -> T:
        for attempt in range(self.config.max_retries):
            try:
                redis = await self._conn()
                return await fn(redis)
            except Exception as e:
                error_msg = str(e).lower()
                if "too many connections" in error_msg:
                    logger.warning(
                        f"[RedisProgramStorage] Connection pool exhausted for operation {name} (attempt {attempt + 1}/{self.config.max_retries})"
                    )
                    # Add exponential backoff for connection pool exhaustion
                    await asyncio.sleep(
                        self.config.retry_delay * (2**attempt) + 0.1
                    )
                elif "connection" in error_msg or "timeout" in error_msg:
                    logger.warning(
                        f"[RedisProgramStorage] Connection issue for operation {name} (attempt {attempt + 1}/{self.config.max_retries}): {e}"
                    )
                    await asyncio.sleep(self.config.retry_delay * (2**attempt))
                else:
                    logger.error(
                        f"[RedisProgramStorage] Non-connection error for operation {name}: {e}"
                    )
                    if attempt == self.config.max_retries - 1:
                        raise StorageError(
                            f"Redis op {name} failed: {e}"
                        ) from e
                    await asyncio.sleep(self.config.retry_delay)

        raise StorageError(
            f"Redis op {name} failed after {self.config.max_retries} attempts"
        )

    @resilient_operation(
        retry_config=RetryConfig(max_attempts=5, base_delay=0.2),
        circuit_breaker_name="redis_write",
        operation_name="redis_add",
    )
    async def add(self, program: Program) -> None:
        ensure_not_none(program, "program")

        async def _add(redis):
            await redis.set(
                self._prog_key(program.id), _dumps(program.to_dict())
            )

        await self._execute("add", _add)

    @resilient_operation(
        retry_config=RetryConfig(max_attempts=5, base_delay=0.2),
        circuit_breaker_name="redis_write",
        operation_name="redis_update",
    )
    async def update(self, program: Program) -> None:
        ensure_not_none(program, "program")

        async def _merge(redis):
            key = self._prog_key(program.id)
            existing_raw = await redis.get(key)
            existing_prog = (
                Program.from_dict(_loads(existing_raw))
                if existing_raw
                else None
            )
            merged = self._merge_strategy(existing_prog, program)
            await redis.set(key, _dumps(merged.to_dict()))

        await self._execute("update", _merge)

    @resilient_operation(
        retry_config=RetryConfig(max_attempts=3, base_delay=0.1),
        circuit_breaker_name="redis_get",
        operation_name="redis_get",
    )
    async def get(self, program_id: str) -> Optional[Program]:
        async def _get(redis):
            raw = await redis.get(self._prog_key(program_id))
            if raw:
                try:
                    return Program.from_dict(_loads(raw))
                except Exception as e:
                    logger.error(
                        f"Failed to deserialize program {program_id}: {e}"
                    )
            return None

        return await self._execute("get", _get)

    async def exists(self, program_id: str) -> bool:
        async def _exists(redis):
            return bool(await redis.exists(self._prog_key(program_id)))

        return await self._execute("exists", _exists)

    async def remove(self, program_id: str) -> None:
        async def _del(redis):
            await redis.delete(self._prog_key(program_id))

        await self._execute("remove", _del)

    async def publish_status_event(
        self,
        status: str,
        program_id: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        async def _pipe(redis):
            data = {"id": program_id, "status": status, **(extra or {})}
            pipe = redis.pipeline()
            try:
                pipe.xadd(
                    self._stream_key(), data, maxlen=10_000, approximate=True
                )
            except TypeError:
                pipe.xadd(self._stream_key(), data)
            pipe.sadd(self._status_set_key(status), program_id)
            await pipe.execute()

        await self._execute("publish_status_event", _pipe)

    async def _get_ids_for_status(self, status: str) -> List[str]:
        async def _members(redis):
            return await redis.smembers(self._status_set_key(status))

        return await self._execute("_get_ids_for_status", _members)

    async def get_all_by_status(self, status: str) -> List[Program]:
        if status == "fresh":
            all_programs = await self.get_all()
            return [p for p in all_programs if p.state == status]

        ids = await self._get_ids_for_status(status)
        if not ids:
            return []

        async def _mget(redis):
            keys = [self._prog_key(pid) for pid in ids]
            pipe = redis.pipeline()
            for k in keys:
                pipe.get(k)
            blobs = await pipe.execute()

            programs = []
            for raw in blobs:
                if raw:
                    try:
                        program = Program.from_dict(_loads(raw))
                        if program.state == status:
                            programs.append(program)
                    except Exception as exc:
                        logger.error(
                            "[RedisProgramStorage] Deserialization error: %s",
                            exc,
                        )
            return programs

        return await self._execute("get_all_by_status", _mget)

    async def mget(self, program_ids: List[str]) -> List[Program]:
        if not program_ids:
            return []

        async def _mget(redis):
            keys = [self._prog_key(pid) for pid in program_ids]
            blobs = await redis.mget(*keys)

            programs = []
            for raw in blobs:
                if raw:
                    try:
                        programs.append(Program.from_dict(_loads(raw)))
                    except Exception as exc:
                        logger.error(
                            f"[RedisProgramStorage] Deserialization error in mget: {exc}"
                        )
            return programs

        return await self._execute("mget", _mget)

    async def get_all(self) -> List[Program]:
        async def _get_all(redis):
            keys = await redis.keys(self._prog_key("*"))

            if not keys:
                return []

            # Use pipeline for batch operations instead of concurrent individual gets
            pipe = redis.pipeline()
            for key in keys:
                pipe.get(key)

            blobs = await pipe.execute()

            programs = []
            for raw in blobs:
                if raw:
                    try:
                        programs.append(Program.from_dict(_loads(raw)))
                    except Exception as exc:
                        logger.error(
                            f"[RedisProgramStorage] Deserialization error in get_all: {exc}"
                        )

            return programs

        return await self._execute("get_all", _get_all)

    async def get_connection_info(self) -> Dict[str, Any]:
        """Get connection pool information for monitoring."""
        try:
            redis = await self._conn()
            if hasattr(redis, "connection_pool"):
                pool = redis.connection_pool
                return {
                    "max_connections": getattr(
                        pool, "max_connections", "unknown"
                    ),
                    "created_connections": getattr(
                        pool, "created_connections", "unknown"
                    ),
                    "available_connections": getattr(
                        pool, "available_connections", "unknown"
                    ),
                    "in_use_connections": getattr(
                        pool, "in_use_connections", "unknown"
                    ),
                }
            return {"status": "connection_pool_not_available"}
        except Exception as e:
            return {"error": str(e)}

    async def close(self):
        await (await self._conn()).close()
