from __future__ import annotations

import asyncio
import contextlib
import os
import socket
import time
import uuid

from loguru import logger

from gigaevo.database.redis.config import RedisLockConfig
from gigaevo.database.redis.connection import RedisConnection
from gigaevo.database.redis.keys import RedisProgramKeys
from gigaevo.exceptions import StorageError
from redis import asyncio as aioredis


class RedisInstanceLock:
    """Distributed instance lock with auto-renewal.

    Prevents multiple instances from using the same Redis prefix.
    """

    def __init__(
        self,
        connection: RedisConnection,
        keys: RedisProgramKeys,
        config: RedisLockConfig,
    ):
        self._conn = connection
        self._keys = keys
        self._config = config

        self._token: str | None = None
        self._renewal_task: asyncio.Task[None] | None = None
        self._instance_id = (
            f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
        )

    @property
    def is_held(self) -> bool:
        return self._token is not None

    @property
    def instance_id(self) -> str:
        return self._instance_id

    async def acquire(self) -> bool:
        """Acquire the instance lock."""

        async def _acquire(r: aioredis.Redis) -> bool:
            lock_key = self._keys.instance_lock()
            lock_value = f"{self._instance_id}:{time.time()}"

            acquired = await r.set(
                lock_key, lock_value, nx=True, ex=self._config.lock_expiry_secs
            )

            if not acquired:
                existing = await r.get(lock_key)
                raise StorageError(
                    f"Cannot start: another instance is using Redis prefix '{self._keys.prefix}'. "
                    f"Lock held by: {existing}. "
                    f"If this is a stale lock from a crashed instance, "
                    f"manually delete Redis key: {lock_key}"
                )

            logger.info(
                "[RedisInstanceLock] Acquired exclusive lock for prefix '{}'",
                self._keys.prefix,
            )
            self._token = lock_value
            return True

        result = await self._conn.execute("acquire_instance_lock", _acquire)

        # Start renewal task
        self._renewal_task = asyncio.create_task(self._renew_periodically())
        return result

    async def release(self) -> None:
        """Release the instance lock."""
        # Stop renewal task first
        if self._renewal_task:
            self._renewal_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._renewal_task
            self._renewal_task = None

        if not self._token:
            return

        async def _release(r: aioredis.Redis) -> None:
            lock_key = self._keys.instance_lock()
            current = await r.get(lock_key)
            if current and current.startswith(self._instance_id):
                await r.delete(lock_key)
                logger.info(
                    "[RedisInstanceLock] Released lock for prefix '{}'",
                    self._keys.prefix,
                )
            self._token = None

        try:
            await self._conn.execute("release_instance_lock", _release)
        except Exception as e:
            logger.warning("[RedisInstanceLock] Failed to release lock: {}", e)

    async def renew(self) -> bool:
        """Renew the instance lock."""
        if not self._token:
            return False

        async def _renew(r: aioredis.Redis) -> bool:
            lock_key = self._keys.instance_lock()
            current = await r.get(lock_key)

            if not current or not current.startswith(self._instance_id):
                logger.error(
                    "[RedisInstanceLock] Lost lock! Another instance may have taken over."
                )
                return False

            lock_value = f"{self._instance_id}:{time.time()}"
            await r.set(lock_key, lock_value, ex=self._config.lock_expiry_secs)
            self._token = lock_value
            return True

        try:
            return await self._conn.execute("renew_instance_lock", _renew)
        except Exception as e:
            logger.error("[RedisInstanceLock] Failed to renew lock: {}", e)
            return False

    async def _renew_periodically(self) -> None:
        """Background task to renew the lock periodically."""
        while not self._conn.is_closing:
            try:
                await asyncio.sleep(self._config.lock_renewal_secs)

                if self._conn.is_closing:
                    break

                if not await self.renew():
                    logger.critical(
                        "[RedisInstanceLock] Failed to renew lock! "
                        "Another instance may be using the same prefix. STOPPING."
                    )
                    break
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("[RedisInstanceLock] Renewal error: {}", e)
