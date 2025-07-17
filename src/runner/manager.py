# RunnerManager: Combined orchestrator for EvolutionEngine and DAG execution.
# Provides lifecycle management, metrics collection, and graceful shutdown
# for distributed evolutionary computation workloads.

from __future__ import annotations

import asyncio
import contextlib
from datetime import datetime, timezone
from typing import Dict, List, Optional

from loguru import logger
from pydantic import BaseModel, Field, computed_field, field_validator

from src.database.program_storage import ProgramStorage
from src.evolution.engine import EvolutionEngine
from src.programs.automata import ExecutionOrderDependency
from src.runner.dag_spec import DAGSpec
from src.programs.stages.base import Stage
from src.programs.state_manager import ProgramStateManager
from src.runner.factories import DagFactory
from src.runner.scheduler import DagScheduler
from src.runner.engine_driver import EngineDriver
from src.runner.metrics import MetricsService

__all__ = [
    "RunnerConfig",
    "RunnerMetrics",
    "RunnerManager",
]


class RunnerConfig(BaseModel):
    """Configuration options controlling concurrent execution of DAGs and evolutionary engine."""

    poll_interval: float = Field(
        default=0.5,
        gt=0,
        le=60.0,
        description="How often to poll Redis for new programs (seconds)",
    )

    max_concurrent_dags: int = Field(
        default=8,
        gt=0,
        le=1000,
        description="Maximum number of programs to execute concurrently",
    )

    log_interval: int = Field(
        default=10,
        gt=0,
        le=10000,
        description="Log stats every N polling iterations",
    )

    prometheus_port: int | None = Field(
        default=None,
        ge=1024,
        le=65535,
        description="If set, Runner exports Prometheus metrics on this port.",
    )

    dag_timeout: float = Field(
        default=2400,  # Increased from 900 to 2400 (40 minutes) to match DAG timeout
        gt=0,
        le=3600.0,  # Max 60 minutes
        description="How long to wait for a DAG to complete (seconds). Programs stuck for 2x this time get discarded.",
    )

    @field_validator("poll_interval")
    @classmethod
    def validate_poll_interval_reasonable(cls, v):  # noqa: D401 – pydantic naming
        if v < 0.01:
            raise ValueError("poll_interval too small, minimum 0.01s recommended")
        if v > 30.0:
            logger.debug(f"Large poll_interval ({v}s) may cause slow response times")
        return v

    @field_validator("max_concurrent_dags")
    @classmethod
    def validate_concurrency_limits(cls, v):  # noqa: D401
        import os

        cpu_count = os.cpu_count() or 4
        if v > cpu_count * 4:
            logger.warning(
                f"max_concurrent_dags ({v}) exceeds 4x CPU count ({cpu_count})"
            )
        return v


class RunnerMetrics(BaseModel):
    """Light-weight runtime metrics for the combined evolution & DAG runner."""
    
    model_config = {'arbitrary_types_allowed': True}

    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    loop_iterations: int = 0
    dag_runs_started: int = 0
    dag_runs_completed: int = 0
    dag_errors: int = 0
    lock: asyncio.Lock = Field(default_factory=asyncio.Lock, repr=False, exclude=True)

    @computed_field
    @property
    def uptime_seconds(self) -> int:  # noqa: D401
        return int((datetime.now(timezone.utc) - self.started_at).total_seconds())

    @computed_field
    @property
    def dag_runs_active(self) -> int:  # noqa: D401
        return max(0, self.dag_runs_started - self.dag_runs_completed - self.dag_errors)

    @computed_field
    @property
    def success_rate(self) -> float:  # noqa: D401
        total_finished = self.dag_runs_completed + self.dag_errors
        return 1.0 if total_finished == 0 else self.dag_runs_completed / total_finished

    @computed_field
    @property
    def average_iterations_per_second(self) -> float:  # noqa: D401
        uptime = self.uptime_seconds
        return 0.0 if uptime == 0 else self.loop_iterations / uptime

    def to_dict(self) -> Dict[str, int | float | str]:
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

    async def increment_loop_iterations(self):
        async with self.lock:
            self.loop_iterations += 1

    async def increment_dag_runs_started(self):
        async with self.lock:
            self.dag_runs_started += 1

    async def increment_dag_runs_completed(self):
        async with self.lock:
            self.dag_runs_completed += 1

    async def increment_dag_errors(self):
        async with self.lock:
            self.dag_errors += 1


class RunnerManager:
    """Combined orchestrator for EvolutionEngine and asynchronous DAG execution.

    This class orchestrates the execution of the evolution engine alongside 
    asynchronous DAG processing using a dedicated DagScheduler for improved 
    separation of concerns and better resource management.
    """

    # ---------------------------------------------------------------------
    # Constructor & basic state
    # ---------------------------------------------------------------------

    def __init__(
        self,
        *,  # force keyword-only for clarity going forward
        engine: EvolutionEngine,
        dag_spec: DAGSpec,
        storage: ProgramStorage | None = None,
        config: Optional[RunnerConfig] = None,
    ) -> None:
        if storage is None:
            raise ValueError("A 'storage' instance is required.")

        self.storage = storage
        self.engine = engine
        self._dag_spec = dag_spec   
        self._dag_factory = DagFactory(
            dag_spec=self._dag_spec,
        )
        self.config = config or RunnerConfig()
        self.metrics = RunnerMetrics()

        # Metrics exporter
        MetricsService.init(self.config.prometheus_port)

        # Helper for persisting stage updates
        self._state_manager = ProgramStateManager(self.storage)

        self._engine_driver = EngineDriver(engine)
        self._scheduler: DagScheduler = DagScheduler(
            self.storage,
            self._dag_factory,
            self._state_manager,
            self.metrics,
            self.config,
        )
        self._running = False
        self._stopping = False

        # Background task when used as async context manager
        self._bg_task: Optional[asyncio.Task] = None
        self._prom_task: Optional[asyncio.Task] = None

        logger.info(
            "[RunnerManager] Created (poll_interval={:.2f}s, max_concurrent_dags={})".format(
                self.config.poll_interval, self.config.max_concurrent_dags
            )
        )

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Start the evolution engine and the DAG scheduler."""
        if self._running:
            logger.warning("[RunnerManager] Already running – ignoring run()")
            return

        self._running = True
        logger.info("[RunnerManager] Starting …")

        # start background components
        self._engine_driver.start()
        # start scheduler run-loop
        self._scheduler.start()

        # start periodic metrics export
        self._prom_task = asyncio.create_task(self._export_metrics_loop(), name="prom-export")

        # Filter out None tasks for robustness
        tasks = [task for task in [
            self._engine_driver.task,
            self._scheduler._task,
            self._prom_task,
        ] if task is not None]
        
        if not tasks:
            logger.error("[RunnerManager] No valid tasks to run!")
            return
        
        try:
            # Wait for any task to complete or fail
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            
            # Check if any task finished unexpectedly
            for task in done:
                task_name = task.get_name() if hasattr(task, 'get_name') else 'unknown'
                try:
                    task.result()  # This will raise an exception if the task failed
                    logger.warning(f"Task '{task_name}' finished unexpectedly (this usually indicates an error)")
                except Exception as e:
                    logger.error(f"Task '{task_name}' failed with error: {e}", exc_info=True)

            # Gracefully shutdown remaining tasks
            if pending:
                logger.info("Initiating graceful shutdown of remaining tasks...")
                for task in pending:
                    await self._cancel_task(task)

        except asyncio.CancelledError:
            logger.info("RunnerManager run() was cancelled.")
        except Exception as e:
            logger.error(f"Unexpected error in RunnerManager.run(): {e}", exc_info=True)
        finally:
            # Ensure all tasks are cleaned up
            for task in tasks:
                await self._cancel_task(task)

    async def stop(self) -> None:
        """Request graceful shutdown and wait for completion."""
        if not self._running or self._stopping:
            return

        self._stopping = True
        logger.info("[RunnerManager] Stopping …")

        if self._scheduler._task:
            self._scheduler._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._scheduler._task

        # Stop engine
        await self._engine_driver.stop()

        self._running = False
        logger.info("[RunnerManager] Shutdown complete")

        await self._scheduler.stop()

        if self._prom_task is not None:
            self._prom_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._prom_task

        if self._bg_task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await self._bg_task
            self._bg_task = None

    # Pause / resume simply forward to engine for now
    def pause_engine(self):
        self._engine_driver.pause()

    def resume_engine(self):
        self._engine_driver.resume()

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def get_metrics(self):
        return self.metrics.to_dict()

    async def get_status(self):
        status = {
            "running": self._running,
            "engine_running": self._engine_driver.is_running(),
            "active_dag_count": self._scheduler.active_count(),
            "metrics": self.get_metrics(),
        }
        try:
            status["engine_status"] = await self._engine_driver.get_status()
        except Exception as e:  # noqa: BLE001 – defensive
            status["engine_status"] = {"error": str(e)}
        return status

    # ------------------------------------------------------------------
    # Async helpers
    # ------------------------------------------------------------------

    @staticmethod
    async def _cancel_task(task: Optional[asyncio.Task]) -> None:
        if task is None or task.done():
            return
        try:
            task.cancel()
            await task
        except asyncio.CancelledError:
            pass  # Expected
        except Exception as e:
            logger.warning(f"Error while cancelling task {task.get_name()}: {e}")

    # ------------------------------------------------------------------
    # Async context-manager helpers
    # ------------------------------------------------------------------

    async def __aenter__(self):
        """Start the runner as a context manager."""
        self._bg_task = asyncio.create_task(self.run(), name="runner-bg")
        # Small sleep ensures tasks actually start, helpful for tests
        await asyncio.sleep(0)
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.stop()
        if self._bg_task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await self._bg_task
            self._bg_task = None

    # ------------------------------------------------------------------
    # Internal metrics export loop
    # ------------------------------------------------------------------

    async def _export_metrics_loop(self):
        try:
            while self._running and not self._stopping:
                MetricsService.tick_uptime()
                MetricsService.export_dict("engine", self.engine.metrics.to_dict())

                strategy = getattr(self.engine, "strategy", None)
                if strategy is not None and hasattr(strategy, "metrics"):
                    try:
                        MetricsService.export_dict("strategy", strategy.metrics.to_dict())
                    except Exception:  # pragma: no cover – defensive
                        pass

                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            pass 