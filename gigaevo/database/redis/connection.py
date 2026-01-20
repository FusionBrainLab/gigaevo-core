from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

from loguru import logger

from gigaevo.database.redis.config import RedisConnectionConfig
from gigaevo.exceptions import StorageError
from redis import asyncio as aioredis

T = TypeVar("T")


class RedisConnection:
    """Manages Redis connection with retry logic and graceful shutdown."""

    def __init__(self, config: RedisConnectionConfig):
        self.config = config
        self._redis: aioredis.Redis | None = None
        self._lock = asyncio.Lock()
        self._closing = False

    @property
    def is_connected(self) -> bool:
        return self._redis is not None

    @property
    def is_closing(self) -> bool:
        return self._closing

    async def get(self) -> aioredis.Redis:
        """Get Redis connection, creating one if needed."""
        if self._closing:
            raise StorageError("RedisConnection is closing; cannot get connection.")
        if self._redis is not None:
            return self._redis

        async with self._lock:
            if self._redis is None:
                if self._closing:
                    raise StorageError(
                        "RedisConnection is closing; cannot get connection."
                    )
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
                logger.debug("[RedisConnection] Connected to {}", self.config.redis_url)
                self._redis = r

        return self._redis

    async def execute(
        self, name: str, fn: Callable[[aioredis.Redis], Awaitable[T]]
    ) -> T:
        """Execute a Redis operation with retry logic."""
        if self._closing:
            raise StorageError(f"Redis op {name} refused: connection is closing.")

        delay = self.config.retry_delay
        for attempt in range(1, self.config.max_retries + 1):
            try:
                return await fn(await self.get())
            except Exception as e:
                if attempt == self.config.max_retries or self._closing:
                    logger.debug("[RedisConnection] {} failed: {}", name, e)
                    raise StorageError(f"Redis op {name} failed: {e}") from e
                await asyncio.sleep(min(delay, 1.0))
                delay *= 2

        raise StorageError(
            f"Redis op {name} failed after {self.config.max_retries} attempts"
        )

    async def close(self) -> None:
        """Close the Redis connection."""
        self._closing = True

        r, self._redis = self._redis, None
        if r is not None:
            try:
                await r.aclose()
            except Exception:
                pass
            try:
                await r.connection_pool.disconnect(inuse_connections=True)
            except Exception:
                pass

        await asyncio.sleep(0)  # Yield to event loop
