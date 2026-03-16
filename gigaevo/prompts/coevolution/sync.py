"""Synchronization hook for prompt co-evolution.

MainRunSyncHook blocks the prompt run's engine until the main run advances
by at least one generation. This prevents the lightweight prompt run from
racing far ahead of the expensive main run.
"""

from __future__ import annotations

import asyncio
import time

from loguru import logger


class MainRunSyncHook:
    """Pre-step hook that blocks until the main run advances by 1 generation.

    Polls the main run's ``engine:total_generations`` counter in Redis and
    waits until it exceeds the value seen on the previous call.

    Args:
        host: Redis host of the main run
        port: Redis port
        db: Redis DB of the main run
        prefix: Key prefix of the main run (e.g. "chains/hotpotqa")
        timeout: Maximum seconds to wait before proceeding anyway
        poll_interval: Seconds between polls
    """

    def __init__(
        self,
        host: str,
        port: int,
        db: int,
        prefix: str,
        timeout: float = 7200.0,
        poll_interval: float = 5.0,
    ):
        self._host = host
        self._port = port
        self._db = db
        self._prefix = prefix
        self._timeout = timeout
        self._poll_interval = poll_interval
        self._last_main_gen: int = -1
        self._redis: object | None = None

    def _get_redis(self) -> object:
        if self._redis is None:
            from redis import asyncio as aioredis

            self._redis = aioredis.Redis(
                host=self._host,
                port=self._port,
                db=self._db,
                decode_responses=True,
            )
        return self._redis

    async def __call__(self) -> None:
        """Poll until the main run's generation counter advances."""
        r = self._get_redis()
        key = f"{self._prefix}:run_state"
        field = "engine:total_generations"
        start = time.monotonic()

        while True:
            raw = await r.hget(key, field)  # type: ignore[attr-defined]
            current_gen = int(raw) if raw else 0

            if current_gen > self._last_main_gen:
                logger.debug(
                    "[MainRunSyncHook] Main run at gen {} (was {})",
                    current_gen,
                    self._last_main_gen,
                )
                self._last_main_gen = current_gen
                return

            elapsed = time.monotonic() - start
            if elapsed > self._timeout:
                logger.warning(
                    "[MainRunSyncHook] Timeout after {:.0f}s waiting for main gen > {} "
                    "(current={}), proceeding",
                    elapsed,
                    self._last_main_gen,
                    current_gen,
                )
                return

            await asyncio.sleep(self._poll_interval)
