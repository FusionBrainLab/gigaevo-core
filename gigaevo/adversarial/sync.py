"""Progress-based synchronization hook for adversarial co-evolution.

ProgressBasedSyncHook blocks a population's engine until the opponent
population(s) have processed a minimum number of additional programs.
Unlike MainRunSyncHook (which waits for generation boundaries), this
hook operates on continuous program-count progress — designed for
SteadyStateEvolutionEngine where "generation" is just an epoch boundary.

Supports asymmetric update ratios via ``sync_every_n_epochs``:
set to K for K:1 updates (this population runs K epochs per sync).
Inspired by GAN training routines (critic iterations > generator).
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from loguru import logger
from redis import asyncio as aioredis


class ProgressBasedSyncHook:
    """Pre-step hook that blocks until opponent processes ``min_delta`` more programs.

    Polls each opponent run's ``engine:programs_processed`` counter in Redis and
    waits until the minimum across all sources advances by at least ``min_delta``.

    Supports K:1 asymmetric update ratios via ``sync_every_n_epochs``: when set
    to K > 1, the hook is a no-op for K-1 out of every K calls.

    Args:
        host: Redis host
        port: Redis port
        sources: List of {"db": int, "prefix": str} — opponent run(s).
            Must be non-empty.
        min_delta: Minimum programs opponent must process between syncs (default: 10).
            MUST be <= max_mutations_per_generation to avoid deadlock when both
            populations wait for each other to advance.
        sync_every_n_epochs: Only sync every N epochs (default: 1).
            Set to K for K:1 asymmetric updates.
        timeout: Maximum seconds to wait before proceeding anyway (default: 7200)
        poll_interval: Seconds between polls (default: 5.0)
    """

    def __init__(
        self,
        host: str,
        port: int,
        sources: list[dict[str, int | str]],
        min_delta: int = 10,
        sync_every_n_epochs: int = 1,
        timeout: float = 7200.0,
        poll_interval: float = 5.0,
        **kwargs: Any,  # absorb extra keys from Hydra config inheritance
    ):
        if not sources:
            raise ValueError("ProgressBasedSyncHook requires at least one source")

        self._host = host
        self._port = port
        self._sources = [(int(s["db"]), str(s["prefix"])) for s in sources]
        self._min_delta = min_delta
        self._sync_every_n_epochs = max(1, sync_every_n_epochs)
        self._timeout = timeout
        self._poll_interval = poll_interval
        self._last_progress: int = -1  # sentinel: not yet initialized
        self._epoch_call_count: int = 0
        self._redis_clients: dict[int, aioredis.Redis] = {}  # type: ignore[type-arg]

        sources_desc = ", ".join(f"db={db} prefix={pfx!r}" for db, pfx in self._sources)
        logger.info(
            "[ProgressBasedSyncHook] Init | sources=[{}] min_delta={} "
            "sync_every={}epochs timeout={}s poll={}s",
            sources_desc,
            self._min_delta,
            self._sync_every_n_epochs,
            self._timeout,
            self._poll_interval,
        )

    def _get_redis(self, db: int) -> aioredis.Redis:  # type: ignore[type-arg]
        if db not in self._redis_clients:
            self._redis_clients[db] = aioredis.Redis(
                host=self._host,
                port=self._port,
                db=db,
                decode_responses=True,
            )
        return self._redis_clients[db]

    async def _get_min_progress(self) -> int:
        """Read the minimum programs_processed across all tracked opponent runs."""
        values = []
        for db, prefix in self._sources:
            try:
                r = self._get_redis(db)
                key = f"{prefix}:run_state"
                raw = await r.hget(key, "engine:programs_processed")
                values.append(int(raw) if raw else 0)
            except Exception as exc:
                logger.warning(
                    "[ProgressBasedSyncHook] Error reading progress from db={}: {}",
                    db,
                    exc,
                )
                values.append(0)
        return min(values) if values else 0

    async def __call__(self) -> None:
        """Block until opponent has processed min_delta more programs.

        On the first call, records the baseline progress and returns immediately.
        Subsequent calls block until ``min_progress >= last_progress + min_delta``.
        Respects ``sync_every_n_epochs``: skips K-1 out of every K calls.
        """
        # Asymmetric ratio: skip this epoch if not a sync epoch
        self._epoch_call_count += 1
        if self._epoch_call_count < self._sync_every_n_epochs:
            return
        self._epoch_call_count = 0

        # First sync: record baseline, don't block
        if self._last_progress < 0:
            self._last_progress = await self._get_min_progress()
            logger.info(
                "[ProgressBasedSyncHook] Baseline recorded: progress={}",
                self._last_progress,
            )
            return

        target = self._last_progress + self._min_delta
        start = time.monotonic()
        last_progress_log = start

        while True:
            min_progress = await self._get_min_progress()

            if min_progress >= target:
                elapsed = time.monotonic() - start
                logger.info(
                    "[ProgressBasedSyncHook] Opponent advanced to {} "
                    "(was {}, target={}, waited {:.1f}s, {} sources)",
                    min_progress,
                    self._last_progress,
                    target,
                    elapsed,
                    len(self._sources),
                )
                self._last_progress = min_progress
                return

            elapsed = time.monotonic() - start
            if elapsed > self._timeout:
                logger.warning(
                    "[ProgressBasedSyncHook] Timeout after {:.0f}s waiting for "
                    "progress >= {} (current min={}), proceeding",
                    elapsed,
                    target,
                    min_progress,
                )
                # Reset baseline to current reality so the NEXT epoch doesn't
                # also wait the full timeout with a stale target.
                self._last_progress = min_progress
                return

            now = time.monotonic()
            if (now - last_progress_log) >= 60.0:
                logger.info(
                    "[ProgressBasedSyncHook] Waiting {:.0f}s for progress >= {} "
                    "(current min={})",
                    elapsed,
                    target,
                    min_progress,
                )
                last_progress_log = now

            await asyncio.sleep(self._poll_interval)
