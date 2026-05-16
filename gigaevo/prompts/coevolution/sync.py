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

from gigaevo.dataplane import DataPlane, Err


class MainRunSyncHook:
    """Pre-step hook that blocks until main run(s) advance by 1 generation.

    Polls each main run's ``engine:total_generations`` counter — persisted
    by :class:`~gigaevo.database.redis_program_storage.RedisProgramStorage`
    inside the ``{prefix}:run_state`` hash — through
    :meth:`DataPlane.raw_hash_get`. Waits until the minimum across all
    sources exceeds the previous value.

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
        dataplanes: Optional list of pre-wired :class:`DataPlane` handles
            (one per source). When ``None`` the hook lazily constructs its
            own; lifecycle is managed via :meth:`close`.
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
        *,
        dataplanes: list[DataPlane] | None = None,
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

        if dataplanes is not None:
            if len(dataplanes) != len(self._sources):
                raise ValueError(
                    "MainRunSyncHook: dataplanes length "
                    f"({len(dataplanes)}) must match sources length "
                    f"({len(self._sources)})"
                )
            self._dataplanes: list[DataPlane | None] = list(dataplanes)
            self._dp_owned: list[bool] = [False] * len(dataplanes)
        else:
            self._dataplanes = [None] * len(self._sources)
            self._dp_owned = [False] * len(self._sources)

        sources_desc = ", ".join(f"db={db} prefix={pfx!r}" for db, pfx in self._sources)
        logger.info(
            "[MainRunSyncHook] Init | sources=[{}] timeout={}s poll={}s",
            sources_desc,
            self._timeout,
            self._poll_interval,
        )

    async def _get_dataplane(self, idx: int) -> DataPlane:
        """Resolve (and lazily construct) the DataPlane for source ``idx``."""
        dp = self._dataplanes[idx]
        if dp is not None:
            return dp
        db, prefix = self._sources[idx]
        url = f"redis://{self._host}:{self._port}/{db}"
        dp = DataPlane(url, key_prefix=prefix)
        await dp.startup()
        self._dataplanes[idx] = dp
        self._dp_owned[idx] = True
        return dp

    async def close(self) -> None:
        """Tear down any DataPlane handles the hook constructed itself."""
        for i, dp in enumerate(self._dataplanes):
            if dp is None or not self._dp_owned[i]:
                continue
            try:
                await dp.shutdown()
            except Exception as exc:  # noqa: BLE001 - shutdown best-effort
                logger.warning(
                    "[MainRunSyncHook] DataPlane[{}] shutdown failed: {}", i, exc
                )
            self._dataplanes[i] = None
            self._dp_owned[i] = False

    async def _get_min_gen(self) -> int:
        """Read the minimum generation across all tracked main runs."""
        gens: list[int] = []
        for idx, (db, prefix) in enumerate(self._sources):
            try:
                dp = await self._get_dataplane(idx)
                key = f"{prefix}:run_state"
                result = await dp.raw_hash_get(key, "engine:total_generations")
                if isinstance(result, Err) or result.value is None:
                    gens.append(0)
                else:
                    try:
                        gens.append(int(result.value))
                    except (TypeError, ValueError):
                        gens.append(0)
            except Exception as exc:  # noqa: BLE001 - read boundary
                logger.warning(
                    "[MainRunSyncHook] Error reading gen from db={}: {}", db, exc
                )
                gens.append(0)
        return min(gens) if gens else 0

    async def __call__(self) -> None:
        """Poll until the minimum main run generation advances."""
        start = time.monotonic()
        last_progress_log = start

        while True:
            min_gen = await self._get_min_gen()

            if min_gen > self._last_main_gen:
                elapsed = time.monotonic() - start
                logger.info(
                    "[MainRunSyncHook] Main runs advanced to gen {} "
                    "(was {}, waited {:.1f}s, {} sources)",
                    min_gen,
                    self._last_main_gen,
                    elapsed,
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

            now = time.monotonic()
            if (now - last_progress_log) >= 60.0:
                logger.info(
                    "[MainRunSyncHook] Waiting {:.0f}s for min gen > {} (current min={})",
                    elapsed,
                    self._last_main_gen,
                    min_gen,
                )
                last_progress_log = now

            await asyncio.sleep(self._poll_interval)
