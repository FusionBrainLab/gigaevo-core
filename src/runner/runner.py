from __future__ import annotations

import asyncio
import contextlib
import os
from datetime import datetime, timezone

from loguru import logger
from pydantic import BaseModel, Field, computed_field, field_validator

from src.database.program_storage import ProgramStorage
from src.database.state_manager import ProgramStateManager
from src.evolution.engine import EvolutionEngine
from src.runner.dag_blueprint import DAGBlueprint
from src.runner.dag_runner import DagRunner
from src.runner.evolution_runner import EvolutionRunner


class RunnerConfig(BaseModel):
    poll_interval: float = Field(default=0.5, gt=0, le=60.0)
    max_concurrent_dags: int = Field(default=8, gt=0, le=1000)
    log_interval: int = Field(default=10, gt=0, le=10000)
    dag_timeout: float = Field(default=2400, gt=0, le=3600.0)

    @field_validator("poll_interval")
    @classmethod
    def _validate_poll_interval(cls, v: float) -> float:
        if v < 0.01:
            raise ValueError("poll_interval must be >= 0.01s")
        if v > 30.0:
            logger.debug("Large poll_interval ({}s) may slow responsiveness", v)
        return v

    @field_validator("max_concurrent_dags")
    @classmethod
    def _validate_concurrency(cls, v: int) -> int:
        cpu = os.cpu_count() or 4
        if v > cpu * 4:
            logger.warning("max_concurrent_dags ({}) > 4x CPU count ({})", v, cpu)
        return v


class RunnerMetrics(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    loop_iterations: int = 0
    dag_runs_started: int = 0
    dag_runs_completed: int = 0
    dag_errors: int = 0
    lock: asyncio.Lock = Field(default_factory=asyncio.Lock, repr=False, exclude=True)

    @computed_field
    @property
    def uptime_seconds(self) -> int:
        return int((datetime.now(timezone.utc) - self.started_at).total_seconds())

    @computed_field
    @property
    def dag_runs_active(self) -> int:
        return max(0, self.dag_runs_started - self.dag_runs_completed - self.dag_errors)

    @computed_field
    @property
    def success_rate(self) -> float:
        finished = self.dag_runs_completed + self.dag_errors
        return 1.0 if finished == 0 else self.dag_runs_completed / finished

    @computed_field
    @property
    def average_iterations_per_second(self) -> float:
        return (
            0.0
            if self.uptime_seconds == 0
            else self.loop_iterations / self.uptime_seconds
        )

    def to_dict(self) -> dict[str, int | float | str]:
        return {
            "uptime_seconds": self.uptime_seconds,
            "loop_iterations": self.loop_iterations,
            "dag_runs_started": self.dag_runs_started,
            "dag_runs_completed": self.dag_runs_completed,
            "dag_runs_active": self.dag_runs_active,
            "dag_errors": self.dag_errors,
            "success_rate": round(self.success_rate, 3),
            "avg_iterations_per_sec": round(self.average_iterations_per_second, 2),
            "started_at": self.started_at.isoformat(),
        }

    async def increment_loop_iterations(self) -> None:
        async with self.lock:
            self.loop_iterations += 1

    async def increment_dag_runs_started(self) -> None:
        async with self.lock:
            self.dag_runs_started += 1

    async def increment_dag_runs_completed(self) -> None:
        async with self.lock:
            self.dag_runs_completed += 1

    async def increment_dag_errors(self) -> None:
        async with self.lock:
            self.dag_errors += 1


class RunnerManager:
    def __init__(
        self,
        *,
        engine: EvolutionEngine,
        dag_blueprint: DAGBlueprint,
        storage: ProgramStorage,
        config: RunnerConfig,
    ) -> None:
        self.storage = storage
        self.engine = engine
        self.dag_blueprint = dag_blueprint
        self.config = config
        self.metrics = RunnerMetrics()

        self.state_manager = ProgramStateManager(self.storage)
        self.engine_driver = EvolutionRunner(self.engine)
        self.dag_runner = DagRunner(
            self.storage,
            self.dag_blueprint,
            self.state_manager,
            self.metrics,
            self.config,
        )

        self._running = False
        self._stopping = False
        self._bg_task: asyncio.Task | None = None

        logger.info(
            "[RunnerManager] init (poll_interval={:.2f}s, max_concurrent_dags={})",
            self.config.poll_interval,
            self.config.max_concurrent_dags,
        )

    async def run(self) -> None:
        if self._running:
            logger.warning("[RunnerManager] already running")
            return
        self._running = True
        logger.info("[RunnerManager] starting")

        self.engine_driver.start()
        self.dag_runner.start()

        tasks = [
            t
            for t in (self.engine_driver.task, self.dag_runner.task)
            if isinstance(t, asyncio.Task)
        ]
        if not tasks:
            logger.error("[RunnerManager] no tasks to supervise")
            self._running = False
            return

        try:
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )

            for t in done:
                name = self._task_name(t)
                try:
                    t.result()
                    logger.warning("background task '{}' finished unexpectedly", name)
                except Exception as e:
                    logger.error(
                        "background task '{}' failed: {}", name, e, exc_info=True
                    )

            for t in pending:
                await self._cancel_task(t)

        except asyncio.CancelledError:
            logger.info("[RunnerManager] run() cancelled")
        except Exception as e:
            logger.error("[RunnerManager] supervisor error: {}", e, exc_info=True)
        finally:
            for t in [
                t
                for t in (self.engine_driver.task, self.dag_runner.task)
                if isinstance(t, asyncio.Task)
            ]:
                await self._cancel_task(t)
            self._running = False
            logger.info("[RunnerManager] stopped")

    async def stop(self) -> None:
        if not self._running or self._stopping:
            return
        self._stopping = True
        logger.info("[RunnerManager] stopping")

        await self.engine_driver.stop()
        await self.dag_runner.stop()

        if self._bg_task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await self._bg_task
            self._bg_task = None

        self._stopping = False
        self._running = False
        logger.info("[RunnerManager] shutdown complete")

    def pause_engine(self) -> None:
        self.engine_driver.pause()

    def resume_engine(self) -> None:
        self.engine_driver.resume()

    def get_metrics(self) -> dict[str, int | float | str]:
        return self.metrics.to_dict()

    async def get_status(self) -> dict[str, object]:
        status: dict[str, object] = {
            "running": self._running,
            "engine_running": self.engine_driver.is_running(),
            "active_dag_count": self.dag_runner.active_count(),
            **self.get_metrics(),
        }
        try:
            status["engine_status"] = await self.engine_driver.get_status()
        except Exception as e:
            status["engine_status"] = {"error": str(e)}
        return status

    @staticmethod
    async def _cancel_task(task: asyncio.Task | None) -> None:
        if not task or task.done():
            return
        try:
            task.cancel()
            await task
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning("error cancelling task {}: {}", task.get_name(), e)

    @staticmethod
    def _task_name(task: asyncio.Task) -> str:
        try:
            return task.get_name()
        except Exception:
            return "task"

    async def __aenter__(self):
        self._bg_task = asyncio.create_task(self.run(), name="runner-bg")
        await asyncio.sleep(0)
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.stop()
