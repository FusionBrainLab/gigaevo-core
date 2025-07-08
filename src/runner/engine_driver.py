from __future__ import annotations

"""Thin asynchronous wrapper around :class:`EvolutionEngine`.

Provides start/stop/pause/resume helpers and exposes the background task so that
callers can await completion or inspect state without poking into the engine's
internals.
"""

import asyncio
import contextlib
from typing import Optional

from loguru import logger

from src.evolution.engine import EvolutionEngine

__all__ = ["EngineDriver"]


class EngineDriver:
    """Manage the lifecycle of a single :class:`EvolutionEngine`."""

    def __init__(self, engine: EvolutionEngine) -> None:
        self._engine = engine
        self._task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Launch the engine's ``run`` coroutine in the background (idempotent)."""
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._engine.run(), name="evolution-engine")
            logger.info("[EngineDriver] Evolution engine started")

    async def stop(self) -> None:
        """Request graceful shutdown and await completion."""
        self._engine.stop()
        if self._task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            logger.info("[EngineDriver] Evolution engine stopped")

    # ------------------------------------------------------------------
    # Control delegations
    # ------------------------------------------------------------------

    def pause(self) -> None:
        self._engine.pause()

    def resume(self) -> None:
        self._engine.resume()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def is_running(self) -> bool:
        return self._engine.is_running()

    async def get_status(self):
        return await self._engine.get_status()

    # ------------------------------------------------------------------
    # Property helpers
    # ------------------------------------------------------------------

    @property
    def task(self) -> Optional[asyncio.Task]:
        return self._task 

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    async def _monitor_metrics(self):
        """Poll engine metrics periodically and update Prometheus counters."""
        from src.runner.metrics import MetricsService

        prev_gen = 0
        try:
            while self._engine.is_running():
                gen = self._engine.metrics.total_generations
                if gen > prev_gen:
                    MetricsService.inc_generation(gen - prev_gen)
                    prev_gen = gen
                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            pass 