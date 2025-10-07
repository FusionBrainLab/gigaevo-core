from __future__ import annotations

import asyncio
import contextlib
import time
from typing import TYPE_CHECKING, NamedTuple

from loguru import logger

from src.database.program_storage import ProgramStorage
from src.programs.dag import DAG
from src.programs.program import Program
from src.programs.program_state import ProgramState
from src.programs.state_manager import ProgramStateManager
from src.runner.dag_blueprint import DAGBlueprint

if TYPE_CHECKING:
    from .manager import RunnerConfig, RunnerMetrics

__all__ = ["DagScheduler"]


class TaskInfo(NamedTuple):
    task: asyncio.Task
    program_id: str
    started_at: float  # monotonic seconds


class DagScheduler:
    def __init__(
        self,
        storage: ProgramStorage,
        dag_blueprint: DAGBlueprint,
        state_manager: ProgramStateManager,
        metrics: RunnerMetrics,
        config: RunnerConfig,
    ) -> None:
        self._storage = storage
        self._dag_blueprint = dag_blueprint
        self._state_manager = state_manager
        self._metrics = metrics
        self._config = config

        self._active: dict[str, TaskInfo] = {}
        self._sema = asyncio.Semaphore(self._config.max_concurrent_dags)
        self._task: asyncio.Task | None = None
        self._stopping = False

    @property
    def task(self) -> asyncio.Task | None:
        return self._task

    def start(self) -> None:
        if self._task:
            return
        self._task = asyncio.create_task(self._run(), name="dag-scheduler")

    async def stop(self) -> None:
        self._stopping = True
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

        for info in list(self._active.values()):
            await self._cancel_task(info)
        self._active.clear()

    def active_count(self) -> int:
        return len(self._active)

    async def _run(self) -> None:
        try:
            while not self._stopping:
                await self._metrics.increment_loop_iterations()

                await self._maintain()   # timeouts + harvest
                await self._launch()     # start new DAGs up to capacity

                if self._metrics.loop_iterations % self._config.log_interval == 0:
                    logger.info(
                        "[DagScheduler] active={} metrics={}",
                        len(self._active),
                        self._metrics.to_dict(),
                    )

                await self._storage.wait_for_activity(self._config.poll_interval)

        except asyncio.CancelledError:
            logger.debug("[DagScheduler] cancelled")
        except Exception as exc:
            logger.exception("[DagScheduler] unhandled exception: {}", exc)
            raise


    async def _maintain(self) -> None:
        now = time.monotonic()
        finished: list[TaskInfo] = []
        timed_out: list[TaskInfo] = []

        for pid, info in list(self._active.items()):
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
                    await self._state_manager.set_program_state(prog, ProgramState.DISCARDED)
                await self._metrics.increment_dag_errors()
                logger.error("[DagScheduler] program {} timed out", info.program_id)
            except Exception as e:
                logger.error("[DagScheduler] discard after timeout failed for {}: {}", info.program_id, e)

        for info in finished:
            self._active.pop(info.program_id, None)
            try:
                info.task.result()
                await self._metrics.increment_dag_runs_completed()
                logger.debug("[DagScheduler] program {} completed", info.program_id)
            except Exception as e:
                await self._metrics.increment_dag_errors()
                logger.error("[DagScheduler] program {} failed: {}", info.program_id, e)

    # -------------------- launching --------------------

    async def _launch(self) -> None:
        try:
            fresh = await self._storage.get_all_by_status(ProgramState.FRESH.value)
            processing = await self._storage.get_all_by_status(ProgramState.DAG_PROCESSING_STARTED.value)
        except Exception as e:
            logger.error("[DagScheduler] fetch-by-status failed: {}", e)
            return

        for p in processing:
            if p.id not in self._active:
                try:
                    await self._state_manager.set_program_state(p, ProgramState.DISCARDED)
                    await self._metrics.increment_dag_errors()
                    logger.warning("[DagScheduler] orphaned program {} discarded", p.id)
                except Exception as se:
                    logger.error("[DagScheduler] orphan discard failed for {}: {}", p.id, se)

        capacity = self._config.max_concurrent_dags - len(self._active)
        if capacity <= 0:
            return

        for program in fresh:
            if capacity <= 0:
                break
            if program.id in self._active:
                continue

            try:
                dag: DAG = self._dag_blueprint.build(self._state_manager)
            except Exception as e:
                logger.error("[DagScheduler] DAG build failed for {}: {}", program.id, e)
                try:
                    await self._state_manager.set_program_state(program, ProgramState.DISCARDED)
                    await self._metrics.increment_dag_errors()
                except Exception as se:
                    logger.error("[DagScheduler] state update failed for {}: {}", program.id, se)
                continue

            async def _run_one(prog: Program = program, dag_inst: DAG = dag) -> None:
                async with self._sema:
                    await self._execute_dag(dag_inst, prog)

            task = asyncio.create_task(_run_one(), name=f"dag-{program.id[:8]}")
            self._active[program.id] = TaskInfo(task, program.id, time.monotonic())
            capacity -= 1
            await self._metrics.increment_dag_runs_started()

            try:
                await self._state_manager.set_program_state(program, ProgramState.DAG_PROCESSING_STARTED)
                logger.info("[DagScheduler] launched {}", program.id)
            except Exception as e:
                logger.error("[DagScheduler] mark-started failed for {}: {}", program.id, e)
                task.cancel()
                self._active.pop(program.id, None)


    async def _execute_dag(self, dag: DAG, program: Program) -> None:
        ok = True
        try:
            await dag.run(program)
        except Exception as exc:
            ok = False
            logger.error("[DagScheduler] DAG run failed for {}: {}", program.id, exc)

        try:
            new_state = ProgramState.DAG_PROCESSING_COMPLETED if ok else ProgramState.DISCARDED
            await self._state_manager.set_program_state(program, new_state)
        except Exception as se:
            await self._metrics.increment_dag_errors()
            logger.error("[DagScheduler] state update failed for {}: {}", program.id, se)

    async def _cancel_task(self, info: TaskInfo) -> None:
        if info.task.done():
            return
        info.task.cancel()
        with contextlib.suppress(asyncio.CancelledError, asyncio.TimeoutError):
            await asyncio.wait_for(info.task, timeout=2.0)
