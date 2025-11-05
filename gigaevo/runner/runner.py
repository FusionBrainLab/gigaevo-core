from __future__ import annotations

import asyncio
import contextlib

from loguru import logger

from gigaevo.evolution.engine import EvolutionEngine
from gigaevo.runner.dag_runner import DagRunner


class EvolutionRunner:
    def __init__(
        self,
        *,
        evolution_engine: EvolutionEngine,
        dag_runner: DagRunner,
    ) -> None:
        self.engine = evolution_engine
        self.dag_runner = dag_runner
        self._running = False
        self._stopping = False
        self._bg_task: asyncio.Task | None = None

    # ---------------- Supervision ----------------

    async def run(self) -> None:
        if self._running:
            logger.warning("[EvolutionRunner] already running")
            return
        self._running = True
        logger.info("[EvolutionRunner] starting")

        # start components (create their internal tasks)
        self.engine.start()
        self.dag_runner.start()

        tasks = [
            t
            for t in (self.engine.task, self.dag_runner.task)
            if isinstance(t, asyncio.Task)
        ]
        if not tasks:
            logger.error("[EvolutionRunner] no tasks to supervise")
            self._running = False
            return

        try:
            # Wait until either component task finishes (unexpected) or we're cancelled
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )

            # Log outcomes of the one(s) that finished
            for t in done:
                name = self._task_name(t)
                with contextlib.suppress(asyncio.CancelledError):
                    try:
                        t.result()
                        logger.warning(
                            "background task '{}' finished unexpectedly", name
                        )
                    except Exception as e:
                        logger.error(
                            "background task '{}' failed: {}", name, e, exc_info=True
                        )

            # If the first finished, we stop both components and then drain pending
            await self._stop_components()
            for t in pending:
                await self._cancel_task(t)

        except asyncio.CancelledError:
            # Ctrl+C / external cancel -> graceful shutdown
            await self._stop_components()
            raise
        except Exception as e:
            logger.error("[EvolutionRunner] supervisor error: {}", e, exc_info=True)
            await self._stop_components()
        finally:
            self._running = False
            logger.info("[EvolutionRunner] stopped")

    async def stop(self) -> None:
        """External graceful stop (awaits component shutdown and the bg run task)."""
        if self._stopping:
            return
        self._stopping = True
        logger.info("[EvolutionRunner] stopping")

        await self._stop_components()

        # if running in background context manager
        if self._bg_task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await self._bg_task
            self._bg_task = None

        self._stopping = False
        self._running = False
        logger.info("[EvolutionRunner] shutdown complete")

    # ---------------- Internals ----------------

    async def _stop_components(self) -> None:
        """Tell components to stop (they cancel+await their own tasks), then ensure dangling handles are cancelled."""
        # Ask components to stop themselves (must be awaitable)
        with contextlib.suppress(Exception):
            await self.engine.stop()
        with contextlib.suppress(Exception):
            await self.dag_runner.stop()

        # As an extra safety net, cancel any leftover task handles if still present
        await self._cancel_task(self.engine.task)
        await self._cancel_task(self.dag_runner.task)

    @staticmethod
    async def _cancel_task(task: asyncio.Task | None) -> None:
        if not task or task.done():
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    @staticmethod
    def _task_name(task: asyncio.Task) -> str:
        with contextlib.suppress(Exception):
            return task.get_name()
        return "task"

    async def __aenter__(self):
        self._bg_task = asyncio.create_task(self.run(), name="runner-bg")
        await asyncio.sleep(0)  # yield to let it schedule children
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.stop()
