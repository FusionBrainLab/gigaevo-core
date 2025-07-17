from __future__ import annotations

import asyncio
import contextlib
import time
from typing import Dict, TYPE_CHECKING, NamedTuple

from loguru import logger

from src.database.program_storage import ProgramStorage
from src.programs.dag import DAG
from src.programs.program import Program
from src.programs.state_manager import ProgramStateManager
from src.programs.program_state import ProgramState

from .factories import DagFactory
from .metrics import MetricsService

if TYPE_CHECKING:
    from .manager import RunnerMetrics, RunnerConfig

__all__ = ["DagScheduler"]


class TaskInfo(NamedTuple):
    task: asyncio.Task
    program_id: str
    start_time: float


class DagScheduler:
    def __init__(
        self,
        storage: ProgramStorage,
        dag_factory: DagFactory,
        state_manager: ProgramStateManager,
        metrics: RunnerMetrics,
        config: RunnerConfig,
    ) -> None:
        self._storage = storage
        self._factory = dag_factory
        self._state_manager = state_manager
        self._metrics = metrics
        self._config = config

        self._active_tasks: Dict[str, TaskInfo] = {}
        self._sema = asyncio.Semaphore(self._config.max_concurrent_dags)
        self._task: asyncio.Task | None = None
        self._stopping = False

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="dag-scheduler")

    async def stop(self) -> None:
        self._stopping = True
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        for task_info in list(self._active_tasks.values()):
            await self._cancel_task_safely(task_info)
        self._active_tasks.clear()

    def active_count(self) -> int:
        return len(self._active_tasks)

    async def _run(self) -> None:
        try:
            while not self._stopping:
                await self._metrics.increment_loop_iterations()
                await self._cleanup_timed_out_tasks()
                await self._launch_missing_dags()
                await self._cleanup_finished_dags()

                if self._metrics.loop_iterations % self._config.log_interval == 0:
                    logger.info(
                        f"[DagScheduler] {self._metrics.to_dict()} (active={len(self._active_tasks)})"
                    )

                await self._wait_for_trigger()
                MetricsService.tick_uptime()
        except asyncio.CancelledError:
            logger.debug("[DagScheduler] Cancelled")
        except Exception as exc:
            logger.exception(f"[DagScheduler] Unhandled exception: {exc}")
            raise

    async def _cleanup_timed_out_tasks(self):
        current_time = time.time()
        timed_out_tasks = []
        
        for program_id, task_info in self._active_tasks.items():
            elapsed = current_time - task_info.start_time
            if elapsed > self._config.dag_timeout:
                timed_out_tasks.append((program_id, task_info, elapsed))
        
        for program_id, task_info, elapsed in timed_out_tasks:
            logger.error(
                f"[DagScheduler] Task for program {program_id} exceeded timeout "
                f"({elapsed:.1f}s > {self._config.dag_timeout}s) - force cancelling and discarding"
            )
            
            await self._cancel_task_safely(task_info)
            self._active_tasks.pop(program_id, None)
            
            try:
                program = await self._storage.get(program_id)
                if program:
                    await self._state_manager.set_program_state(program, ProgramState.DISCARDED)
                    logger.info(f"[DagScheduler] Program {program_id} marked as discarded due to timeout")
            except Exception as e:
                logger.error(f"[DagScheduler] Failed to mark timed-out program {program_id} as discarded: {e}")
            
            await self._metrics.increment_dag_errors()
            MetricsService.inc_dag_error()

    async def _launch_missing_dags(self):
        try:
            all_programs = await self._storage.get_all()
            candidates = [p for p in all_programs if p.state == ProgramState.FRESH]
            orphaned = [
                p for p in all_programs 
                if p.state == ProgramState.DAG_PROCESSING_STARTED and p.id not in self._active_tasks
            ]
            
            if orphaned:
                logger.warning(
                    f"[DagScheduler] Found {len(orphaned)} orphaned DAG_PROCESSING_STARTED programs. "
                    f"Cleaning them up now."
                )
                
                for orphaned_program in orphaned:
                    try:
                        await self._state_manager.set_program_state(orphaned_program, ProgramState.DISCARDED)
                        logger.info(f"[DagScheduler] Orphaned program {orphaned_program.id} marked as discarded")
                        await self._metrics.increment_dag_errors()
                        MetricsService.inc_dag_error()
                    except Exception as e:
                        logger.error(f"[DagScheduler] Failed to mark orphaned program {orphaned_program.id} as discarded: {e}")
                
        except Exception as e:
            logger.error(f"[DagScheduler] Failed to fetch programs: {e}")
            return

        for program in candidates:
            if program.id in self._active_tasks:
                continue
                
            if len(self._active_tasks) >= self._config.max_concurrent_dags:
                break

            try:
                dag: DAG = self._factory.create(self._state_manager)
            except Exception as e:
                logger.error(f"[DagScheduler] Failed to create DAG for program {program.id}: {e}")
                try:
                    await self._state_manager.set_program_state(program, ProgramState.DISCARDED)
                    logger.info(f"[DagScheduler] Program {program.id} marked as discarded due to DAG creation failure")
                except Exception as state_error:
                    logger.error(f"[DagScheduler] Failed to mark program {program.id} as discarded: {state_error}")
                continue

            async def _run_with_guaranteed_cleanup(prog: Program = program, dag_inst: DAG = dag):
                async with self._sema:
                    await self._execute_dag_with_guaranteed_state_update(dag_inst, prog)

            start_time = time.time()
            task = asyncio.create_task(_run_with_guaranteed_cleanup(), name=f"dag-{program.id[:8]}")
            
            self._active_tasks[program.id] = TaskInfo(task, program.id, start_time)
            
            await self._metrics.increment_dag_runs_started()
            MetricsService.inc_dag_started()
            
            try:
                await self._state_manager.set_program_state(program, ProgramState.DAG_PROCESSING_STARTED)
                logger.info(f"[DagScheduler] Launched DAG for program {program.id}")
            except Exception as e:
                logger.error(f"[DagScheduler] Failed to mark program {program.id} as started: {e}")
                task.cancel()
                self._active_tasks.pop(program.id, None)

    async def _cleanup_finished_dags(self):
        finished_task_infos = {}
        for pid, task_info in list(self._active_tasks.items()):
            if task_info.task.done():
                finished_task_infos[pid] = self._active_tasks.pop(pid)
        
        if finished_task_infos:
            for pid, task_info in finished_task_infos.items():
                try:
                    task_info.task.result()
                    await self._metrics.increment_dag_runs_completed()
                    MetricsService.inc_dag_completed()
                    logger.debug(f"[DagScheduler] DAG for program {pid} completed successfully")
                except Exception as e:
                    await self._metrics.increment_dag_errors()
                    MetricsService.inc_dag_error()
                    logger.error(f"[DagScheduler] DAG for program {pid} failed: {e}")

    async def _execute_dag_with_guaranteed_state_update(self, dag: DAG, program: Program):
        success = False
        try:
            await dag.run(program)
            success = True
        except Exception as exc:
            logger.error(f"[DagScheduler] DAG execution failed for program {program.id}: {exc}")
            success = False
        
        try:
            if success:
                await self._state_manager.set_program_state(program, ProgramState.DAG_PROCESSING_COMPLETED)
                logger.debug(f"[DagScheduler] Program {program.id} completed successfully")
            else:
                await self._state_manager.set_program_state(program, ProgramState.DISCARDED)
                logger.info(f"[DagScheduler] Program {program.id} marked as discarded due to DAG failure")
        except Exception as state_error:
            logger.critical(
                f"[DagScheduler] CRITICAL: Failed to update state for program {program.id}: {state_error}. "
                f"Program may become orphaned!"
            )
            raise RuntimeError(f"Critical state update failure for program {program.id}: {state_error}")

    async def _wait_for_trigger(self):
        poll_ms = int(self._config.poll_interval * 1000)
        if hasattr(self._storage, "_stream_key") and hasattr(self._storage, "_conn"):
            try:
                redis = await self._storage._conn()
                if hasattr(redis, "xread"):
                    stream_key = self._storage._stream_key()
                    await redis.xread({stream_key: "$"}, block=poll_ms, count=1)
                    return
            except asyncio.TimeoutError:
                logger.debug("Redis xread timed out, falling back to sleep")
            except Exception as e:
                logger.debug(f"Redis xread failed, falling back to sleep: {e}")

        await asyncio.sleep(self._config.poll_interval)

    async def _cancel_task_safely(self, task_info: TaskInfo):
        if task_info.task.done():
            return
            
        task_info.task.cancel()
        
        try:
            await asyncio.wait_for(task_info.task, timeout=5.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
        except Exception as e:
            logger.warning(f"[DagScheduler] Unexpected error during task cancellation: {e}")
        
        try:
            program = await self._storage.get(task_info.program_id)
            if program and program.state != ProgramState.DISCARDED:
                await self._state_manager.set_program_state(program, ProgramState.DISCARDED)
                logger.info(f"[DagScheduler] Program {task_info.program_id} marked as discarded after task cancellation")
        except Exception as e:
            logger.error(f"[DagScheduler] Failed to mark cancelled program {task_info.program_id} as discarded: {e}")

 