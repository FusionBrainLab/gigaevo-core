from __future__ import annotations

"""Utilities for CPU-bound Stage implementations.

`WorkerPoolStage` provides a reusable thread-pool so subclasses can implement a
synchronous :py:meth:`_work` function instead of juggling ``asyncio.to_thread``
or executors themselves.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import os

from loguru import logger

from src.programs.stages.base import Stage
from src.programs.stages.state import ProgramStageResult

__all__ = ["WorkerPoolStage"]


class WorkerPoolStage(Stage):
    """Base class for stages that spend most of their time in CPU-bound Python.

    Subclasses MUST implement :py:meth:`_work` (regular function) and MAY
    override :py:meth:`_after_work` for post-processing within the event loop.
    """

    _executor: ThreadPoolExecutor | None = None

    # ------------------------------------------------------------------
    # Executor management
    # ------------------------------------------------------------------

    @classmethod
    def _get_executor(cls) -> ThreadPoolExecutor:
        if cls._executor is None:
            cls._executor = ThreadPoolExecutor(
                max_workers=max(4, (os.cpu_count() or 4) * 2),
                thread_name_prefix="metaevolve-stage",
            )
            logger.debug(
                f"[WorkerPoolStage] Created shared ThreadPoolExecutor with {cls._executor._max_workers} workers"  # type: ignore[attr-defined]
            )
        return cls._executor

    # ------------------------------------------------------------------
    # Stage implementation
    # ------------------------------------------------------------------

    async def _execute_stage(  # noqa: D401 – override
        self, program: "Program", started_at: datetime
    ) -> ProgramStageResult:
        """Off-load the synchronous :py:meth:`_work` method to the shared pool."""
        loop = asyncio.get_running_loop()
        result: ProgramStageResult = await loop.run_in_executor(
            self._get_executor(), self._work, program, started_at
        )
        # Optional async post-processing hook
        if hasattr(self, "_after_work"):
            result = await self._after_work(program, result)  # type: ignore[arg-type]
        return result

    # ------------------------------------------------------------------
    # Methods for subclasses
    # ------------------------------------------------------------------

    def _work(  # noqa: D401 – to be overridden
        self, program: "Program", started_at: datetime
    ) -> ProgramStageResult:
        """Pure Python implementation executed inside a thread.

        Must return a :class:`~src.programs.stages.state.ProgramStageResult`.
        """
        raise NotImplementedError

    async def _after_work(  # noqa: D401 – optional
        self, program: "Program", result: ProgramStageResult
    ) -> ProgramStageResult:
        """Optional async post-processing step executed back in the event loop."""
        return result
