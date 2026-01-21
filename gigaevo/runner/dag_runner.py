from __future__ import annotations

import asyncio
import contextlib
import ctypes
from datetime import datetime, timezone
import gc
import os
import time
from typing import Any, NamedTuple

from loguru import logger
from pydantic import BaseModel, Field, computed_field, field_validator

from gigaevo.database.program_storage import ProgramStorage
from gigaevo.database.state_manager import ProgramStateManager
from gigaevo.programs.dag.dag import DAG
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState
from gigaevo.runner.dag_blueprint import DAGBlueprint
from gigaevo.utils.metrics_collector import start_metrics_collector
from gigaevo.utils.trackers.base import LogWriter


class TaskInfo(NamedTuple):
    task: asyncio.Task
    program_id: str
    started_at: float


class DagRunnerMetrics(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    loop_iterations: int = 0
    dag_runs_started: int = 0
    dag_runs_completed: int = 0
    dag_errors: int = 0
    dag_timeouts: int = 0
    orphaned_programs_discarded: int = 0
    dag_build_failures: int = 0
    state_update_failures: int = 0

    @computed_field
    @property
    def uptime_seconds(self) -> int:
        return int((datetime.now(timezone.utc) - self.started_at).total_seconds())

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

    def increment_loop_iterations(self) -> None:
        self.loop_iterations += 1

    def increment_dag_runs_started(self) -> None:
        self.dag_runs_started += 1

    def increment_dag_runs_completed(self) -> None:
        self.dag_runs_completed += 1

    def increment_dag_errors(self) -> None:
        self.dag_errors += 1

    def record_timeout(self) -> None:
        self.dag_timeouts += 1
        self.dag_errors += 1

    def record_orphaned(self) -> None:
        self.orphaned_programs_discarded += 1
        self.dag_errors += 1

    def record_build_failure(self) -> None:
        self.dag_build_failures += 1
        self.dag_errors += 1

    def record_state_update_failure(self) -> None:
        self.state_update_failures += 1
        self.dag_errors += 1


class DagRunnerConfig(BaseModel):
    poll_interval: float = Field(
        default=0.5,
        gt=0,
        le=60.0,
        description="Interval in seconds to poll for new programs",
    )
    max_concurrent_dags: int = Field(
        default=8,
        gt=0,
        le=1000,
        description="Maximum number of DAGs to run concurrently",
    )
    metrics_collection_interval: float = Field(
        default=1.0, gt=0, description="Interval in seconds for metrics collection"
    )
    dag_timeout: float = Field(
        default=3600, gt=0, le=3600.0, description="Timeout for DAG execution"
    )

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


class DagRunner:
    def __init__(
        self,
        storage: ProgramStorage,
        dag_blueprint: DAGBlueprint,
        config: DagRunnerConfig,
        writer: LogWriter,
    ) -> None:
        self._storage = storage
        self._dag_blueprint = dag_blueprint
        self._state_manager = ProgramStateManager(storage)
        self._metrics = DagRunnerMetrics()
        self._config = config
        self._writer = writer.bind(path=["dag_runner"])

        self._active: dict[str, TaskInfo] = {}
        self._sema = asyncio.Semaphore(self._config.max_concurrent_dags)

        self._task: asyncio.Task | None = None
        self._stopping = False

        # async metrics collector task (no threads)
        self._metrics_collector_task: asyncio.Task | None = None

    @property
    def task(self) -> asyncio.Task | None:
        return self._task

    def start(self) -> None:
        if self._task:
            logger.warning("[DagRunner] already running")
            return

        self._task = asyncio.create_task(self._run(), name="dag-scheduler")
        self._stopping = False

        async def _collect_metrics() -> dict[str, Any]:
            metrics_dict = self._metrics.model_dump(mode="json")
            metrics_dict["dag_active_count"] = float(self.active_count())
            return metrics_dict

        self._metrics_collector_task = start_metrics_collector(
            writer=self._writer,
            collect_fn=_collect_metrics,
            interval=self._config.metrics_collection_interval,
            stop_flag=lambda: self._stopping,
            task_name="dag-metrics-collector",
        )

    async def stop(self) -> None:
        self._stopping = True

        # cancel scheduler loop
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

        # cancel all active DAG tasks
        for info in list(self._active.values()):
            await self._cancel_task(info)
        self._active.clear()

        # cancel metrics collector task
        if self._metrics_collector_task:
            self._metrics_collector_task.cancel()
            self._metrics_collector_task = None

        await self._storage.close()

    def active_count(self) -> int:
        return sum(1 for info in self._active.values() if not info.task.done())

    async def _run(self) -> None:
        logger.info("[DagScheduler] start")
        try:
            while not self._stopping:
                try:
                    self._metrics.increment_loop_iterations()

                    # timeouts + harvest finished/failed tasks
                    await self._maintain()

                    # start new DAGs up to capacity
                    await self._launch()

                    # storage-side wait (stream or sleep)
                    await self._storage.wait_for_activity(self._config.poll_interval)

                except asyncio.CancelledError:
                    # allow clean shutdown; propagate to caller
                    raise
                except Exception:
                    # don’t kill the scheduler on a transient failure in one loop tick
                    logger.exception("[DagScheduler] iteration failed")
                    await asyncio.sleep(0.1)

        except asyncio.CancelledError:
            logger.debug("[DagScheduler] cancelled")
            raise
        finally:
            logger.info("[DagScheduler] stopped")

    async def _maintain(self) -> None:
        now = time.monotonic()
        finished: list[TaskInfo] = []
        timed_out: list[TaskInfo] = []

        for info in list(self._active.values()):
            if info.task.done():
                finished.append(info)
            elif (now - info.started_at) > self._config.dag_timeout:
                timed_out.append(info)

        for info in timed_out:
            await self._cancel_task(info)
            self._active.pop(info.program_id, None)
            try:
                prog = await self._storage.get(info.program_id)
                if prog:
                    await self._state_manager.set_program_state(
                        prog, ProgramState.DISCARDED
                    )
                self._metrics.record_timeout()
                logger.error("[DagScheduler] program {} timed out", info.program_id)
            except Exception as e:
                logger.error(
                    "[DagScheduler] discard after timeout failed for {}: {}",
                    info.program_id,
                    e,
                )

        for info in finished:
            self._active.pop(info.program_id, None)
            try:
                info.task.result()
                self._metrics.increment_dag_runs_completed()
                logger.debug(
                    "[DagScheduler] harvested completed task for program {}",
                    info.program_id,
                )
            except Exception as e:
                self._metrics.increment_dag_errors()
                logger.error("[DagScheduler] program {} failed: {}", info.program_id, e)
            finally:
                del info

        if finished or timed_out:
            gc.collect()
            try:
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except Exception:
                pass

            logger.debug(
                "[DagScheduler] Cleaned up {} finished + {} timed out tasks, forced GC",
                len(finished),
                len(timed_out),
            )

    async def _launch(self) -> None:
        try:
            fresh = await self._storage.get_all_by_status(ProgramState.FRESH.value)
            processing = await self._storage.get_all_by_status(
                ProgramState.DAG_PROCESSING_STARTED.value
            )
        except Exception as e:
            logger.error("[DagScheduler] fetch-by-status failed: {}", e)
            return

        # clean up orphaned PROCESSING programs (no task tracked)
        for p in processing:
            if p.id not in self._active:
                try:
                    await self._state_manager.set_program_state(
                        p, ProgramState.DISCARDED
                    )
                    self._metrics.record_orphaned()
                    logger.warning("[DagScheduler] orphaned program {} discarded", p.id)
                except Exception as se:
                    logger.error(
                        "[DagScheduler] orphan discard failed for {}: {}", p.id, se
                    )

        capacity = self._config.max_concurrent_dags - len(self._active)
        if capacity <= 0:
            return

        # start DAGs for fresh programs up to capacity
        for program in fresh:
            if capacity <= 0:
                break
            if program.id in self._active:
                continue

            try:
                dag: DAG = self._dag_blueprint.build(
                    self._state_manager, writer=self._writer
                )
            except Exception as e:
                import traceback

                logger.error(
                    "[DagScheduler] DAG build failed for {}: {}", program.id, e
                )
                logger.error("[DagScheduler] Traceback:\n{}", traceback.format_exc())
                try:
                    await self._state_manager.set_program_state(
                        program, ProgramState.DISCARDED
                    )
                    self._metrics.record_build_failure()
                except Exception as se:
                    logger.error(
                        "[DagScheduler] state update failed for {}: {}", program.id, se
                    )
                    self._metrics.record_state_update_failure()
                continue

            async def _run_one(prog: Program = program, dag_inst: DAG = dag) -> None:
                async with self._sema:
                    await self._execute_dag(dag_inst, prog)

            task = asyncio.create_task(_run_one(), name=f"dag-{program.id[:8]}")
            self._active[program.id] = TaskInfo(task, program.id, time.monotonic())
            capacity -= 1

            try:
                await self._state_manager.set_program_state(
                    program, ProgramState.DAG_PROCESSING_STARTED
                )
                self._metrics.increment_dag_runs_started()
                logger.info("[DagScheduler] launched {}", program.id)
            except Exception as e:
                logger.error(
                    "[DagScheduler] mark-started failed for {}: {}", program.id, e
                )
                task.cancel()
                self._active.pop(program.id, None)

    async def _execute_dag(self, dag: DAG, program: Program) -> None:
        ok = True
        try:
            await dag.run(program)
        except Exception as exc:
            ok = False
            logger.error("[DagScheduler] DAG run failed for {}: {}", program.id, exc)
        finally:
            dag.automata.topology.nodes.clear()
            dag.automata.topology = None
            dag.automata = None
            dag.state_manager = None
            dag._writer = None
            dag._stage_sema = None

        try:
            new_state = (
                ProgramState.DAG_PROCESSING_COMPLETED if ok else ProgramState.DISCARDED
            )
            await self._state_manager.set_program_state(program, new_state)

            # Log completion immediately when state is updated
            if ok:
                logger.debug(
                    "[DagScheduler] DAG completed for {} (state updated)", program.id
                )
        except Exception as se:
            self._metrics.record_state_update_failure()
            logger.error(
                "[DagScheduler] state update failed for {}: {}", program.id, se
            )

    async def _cancel_task(self, info: TaskInfo) -> None:
        if info.task.done():
            return
        info.task.cancel()
        with contextlib.suppress(asyncio.CancelledError, asyncio.TimeoutError):
            await asyncio.wait_for(info.task, timeout=2.0)
