"""Synchronization hook for prompt co-evolution.

MainRunSyncHook blocks the prompt run's engine until the main run(s) advance
by at least one *processed program*. This prevents the lightweight prompt run
from racing far ahead of the expensive main run(s).

Supports 1-to-many coupling: waits until the minimum ``programs_processed``
across all tracked main runs exceeds the last-seen value.
"""

from __future__ import annotations

import asyncio
import time

from loguru import logger
from redis import asyncio as aioredis

from gigaevo.evolution.engine.snapshot import ENGINE_SNAPSHOT_KEY, EngineSnapshot


class MainRunSyncHook:
    """Pre-step hook that blocks until main run(s) advance by 1 processed program.

    Polls each main run's ``programs_processed`` field from the
    ``engine:snapshot`` JSON blob in Redis and waits until the minimum across
    all sources exceeds the previous value.

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
        self._last_main_progress: int = -1

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

        self._redis_clients: dict[int, aioredis.Redis] = {}  # type: ignore[type-arg]

        sources_desc = ", ".join(f"db={db} prefix={pfx!r}" for db, pfx in self._sources)
        logger.info(
            "[MainRunSyncHook] Init | sources=[{}] timeout={}s poll={}s",
            sources_desc,
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
        """Read the minimum ``programs_processed`` across all tracked main runs.

        Reads the ``engine:snapshot`` JSON blob from each source's run-state
        hash. Missing snapshot or corrupt JSON -> 0 (matches
        :func:`gigaevo.evolution.engine.snapshot.load_engine_snapshot` fallback
        semantics). We read across DBs directly via aioredis here because the
        snapshot helper takes a single storage abstraction and this hook must
        poll foreign-run prefixes.
        """
        progresses = []
        for db, prefix in self._sources:
            try:
                r = self._get_redis(db)
                key = f"{prefix}:run_state"
                raw = await r.hget(key, ENGINE_SNAPSHOT_KEY)
                if raw is None:
                    progresses.append(0)
                    continue
                try:
                    snap = EngineSnapshot.model_validate_json(raw)
                except Exception as exc:
                    logger.warning(
                        "[MainRunSyncHook] engine:snapshot JSON corrupt on "
                        "db={} prefix={!r} ({}); treating as 0",
                        db,
                        prefix,
                        exc,
                    )
                    progresses.append(0)
                    continue
                progresses.append(snap.programs_processed)
            except Exception as exc:
                logger.warning(
                    "[MainRunSyncHook] Error reading programs_processed from db={}: {}",
                    db,
                    exc,
                )
                progresses.append(0)
        return min(progresses) if progresses else 0

    async def __call__(self) -> None:
        """Poll until the minimum main run programs_processed advances."""
        start = time.monotonic()
        last_progress_log = start

        while True:
            min_progress = await self._get_min_progress()

            if min_progress > self._last_main_progress:
                elapsed = time.monotonic() - start
                logger.info(
                    "[MainRunSyncHook] Main runs advanced to programs_processed={} "
                    "(was {}, waited {:.1f}s, {} sources)",
                    min_progress,
                    self._last_main_progress,
                    elapsed,
                    len(self._sources),
                )
                self._last_main_progress = min_progress
                return

            elapsed = time.monotonic() - start
            if elapsed > self._timeout:
                logger.warning(
                    "[MainRunSyncHook] Timeout after {:.0f}s waiting for "
                    "programs_processed > {} (current min={}), proceeding",
                    elapsed,
                    self._last_main_progress,
                    min_progress,
                )
                return

            now = time.monotonic()
            if (now - last_progress_log) >= 60.0:
                logger.info(
                    "[MainRunSyncHook] Waiting {:.0f}s for programs_processed > {} "
                    "(current min={})",
                    elapsed,
                    self._last_main_progress,
                    min_progress,
                )
                last_progress_log = now

            await asyncio.sleep(self._poll_interval)
