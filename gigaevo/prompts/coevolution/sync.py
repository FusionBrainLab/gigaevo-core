"""Synchronization hook for prompt co-evolution.

MainRunSyncHook blocks the prompt run's engine until the main run(s) advance
by at least one generation. This prevents the lightweight prompt run from
racing far ahead of the expensive main run(s).

Supports 1-to-many coupling: waits until the minimum generation across all
tracked main runs exceeds the last-seen value.
"""

from __future__ import annotations

import asyncio
import time

from loguru import logger


class MainRunSyncHook:
    """Pre-step hook that blocks until main run(s) advance by 1 generation.

    Polls each main run's ``engine:total_generations`` counter in Redis and
    waits until the minimum across all sources exceeds the previous value.

    Supports both single-source (backwards compat) and multi-source configs.

    Args:
        host: Redis host
        port: Redis port
        db: Redis DB of a single main run (backwards compat)
        prefix: Key prefix of a single main run (backwards compat)
        sources: List of {"db": int, "prefix": str} for multi-source sync.
            If provided, ``db`` and ``prefix`` are ignored.
        timeout: Maximum seconds to wait before proceeding anyway
        poll_interval: Seconds between polls
    """

    def __init__(
        self,
        host: str,
        port: int,
        db: int | None = None,
        prefix: str | None = None,
        sources: list[dict[str, int | str]] | None = None,
        timeout: float = 7200.0,
        poll_interval: float = 5.0,
    ):
        self._host = host
        self._port = port
        self._timeout = timeout
        self._poll_interval = poll_interval
        self._last_main_gen: int = -1

        # Build list of (db, prefix) sources
        if sources:
            self._sources = [(int(s["db"]), str(s["prefix"])) for s in sources]
        elif db is not None and prefix is not None:
            self._sources = [(db, prefix)]
        else:
            raise ValueError(
                "MainRunSyncHook requires either (db, prefix) "
                "or sources=[{db, prefix}, ...]"
            )

        self._redis_clients: dict[int, object] = {}

    def _get_redis(self, db: int) -> object:
        if db not in self._redis_clients:
            from redis import asyncio as aioredis

            self._redis_clients[db] = aioredis.Redis(
                host=self._host,
                port=self._port,
                db=db,
                decode_responses=True,
            )
        return self._redis_clients[db]

    async def _get_min_gen(self) -> int:
        """Read the minimum generation across all tracked main runs."""
        gens = []
        for db, prefix in self._sources:
            try:
                r = self._get_redis(db)
                key = f"{prefix}:run_state"
                raw = await r.hget(key, "engine:total_generations")  # type: ignore[attr-defined]
                gens.append(int(raw) if raw else 0)
            except Exception as exc:
                logger.warning(
                    "[MainRunSyncHook] Error reading gen from db={}: {}", db, exc
                )
                gens.append(0)
        return min(gens) if gens else 0

    async def __call__(self) -> None:
        """Poll until the minimum main run generation advances."""
        start = time.monotonic()

        while True:
            min_gen = await self._get_min_gen()

            if min_gen > self._last_main_gen:
                logger.debug(
                    "[MainRunSyncHook] Main runs min gen {} (was {}, {} sources)",
                    min_gen,
                    self._last_main_gen,
                    len(self._sources),
                )
                self._last_main_gen = min_gen
                return

            elapsed = time.monotonic() - start
            if elapsed > self._timeout:
                logger.warning(
                    "[MainRunSyncHook] Timeout after {:.0f}s waiting for min gen > {} "
                    "(current min={}), proceeding",
                    elapsed,
                    self._last_main_gen,
                    min_gen,
                )
                return

            await asyncio.sleep(self._poll_interval)
