from __future__ import annotations

"""Asynchronous scheduler that monitors storage for programs requiring DAG
execution and runs them with concurrency limits.

This provides separation of concerns, allowing the RunnerManager to focus 
on high-level lifecycle orchestration while this scheduler handles the 
detailed DAG execution workflow.
"""

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

if TYPE_CHECKING:  # pragma: no cover
    from .manager import RunnerMetrics, RunnerConfig

__all__ = ["DagScheduler"]


class TaskInfo(NamedTuple):
    """Information about a running DAG task."""
    task: asyncio.Task
    program_id: str
    start_time: float


class DagScheduler:
    """Background task that launches DAGs while respecting concurrency limits."""

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

        # Enhanced task tracking with start times
        self._active_tasks: Dict[str, TaskInfo] = {}
        self._sema = asyncio.Semaphore(self._config.max_concurrent_dags)

        self._task: asyncio.Task | None = None
        self._stopping = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="dag-scheduler")

    async def stop(self) -> None:
        self._stopping = True
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        # cancel any running DAG tasks
        for task_info in list(self._active_tasks.values()):
            await self._cancel_task_safely(task_info)
        self._active_tasks.clear()

    def active_count(self) -> int:
        return len(self._active_tasks)

    # ------------------------------------------------------------------
    # Internal run-loop
    # ------------------------------------------------------------------

    async def _run(self) -> None:  # noqa: C901 – inherently complex scheduling logic
        try:
            while not self._stopping:
                await self._metrics.increment_loop_iterations()

                # Critical: Check for timed-out tasks FIRST to prevent accumulation
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
        except Exception as exc:  # pragma: no cover
            logger.exception(f"[DagScheduler] Unhandled exception: {exc}")
            raise

    # ------------------------------------------------------------------
    # Core helpers with robust timeout handling
    # ------------------------------------------------------------------

    async def _cleanup_timed_out_tasks(self):
        """Actively monitor and cleanup tasks that have exceeded the timeout."""
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
            
            # Cancel the task first, then remove from tracking to avoid race conditions
            await self._cancel_task_safely(task_info)
            
            # Remove from active tracking after cancellation
            self._active_tasks.pop(program_id, None)
            
            # Critical: Mark program as discarded to prevent retries
            try:
                program = await self._storage.get(program_id)
                if program:
                    await self._state_manager.set_program_state(program, ProgramState.DISCARDED)
                    logger.info(f"[DagScheduler] Program {program_id} marked as discarded due to timeout")
            except Exception as e:
                logger.error(f"[DagScheduler] Failed to mark timed-out program {program_id} as discarded: {e}")
            
            # Update metrics
            await self._metrics.increment_dag_errors()
            MetricsService.inc_dag_error()

    async def _launch_missing_dags(self):
        try:
            # Get programs needing DAG processing
            all_programs = await self._storage.get_all()
            
            # Only process FRESH programs - don't retry DAG_PROCESSING_STARTED ones
            # This prevents infinite retries of potentially problematic programs
            candidates = [
                p for p in all_programs 
                if p.state == ProgramState.FRESH
            ]
            
            # Log any orphaned programs but don't retry them automatically
            orphaned = [
                p for p in all_programs 
                if p.state == ProgramState.DAG_PROCESSING_STARTED and p.id not in self._active_tasks
            ]
            
            if orphaned:
                logger.warning(
                    f"[DagScheduler] Found {len(orphaned)} orphaned DAG_PROCESSING_STARTED programs. "
                    f"These will be cleaned up by timeout mechanism if stuck."
                )
                
        except Exception as e:  # pragma: no cover
            logger.error(f"[DagScheduler] Failed to fetch programs: {e}")
            return

        for program in candidates:
            # Skip if already active (shouldn't happen with FRESH-only filtering)
            if program.id in self._active_tasks:
                continue
                
            # Skip if semaphore is full
            if len(self._active_tasks) >= self._config.max_concurrent_dags:
                break

            # Create DAG with error handling
            try:
                dag: DAG = self._factory.create(self._state_manager)
            except Exception as e:
                logger.error(f"[DagScheduler] Failed to create DAG for program {program.id}: {e}")
                # Mark program as discarded since we can't process it
                try:
                    await self._state_manager.set_program_state(program, ProgramState.DISCARDED)
                    logger.info(f"[DagScheduler] Program {program.id} marked as discarded due to DAG creation failure")
                except Exception as state_error:
                    logger.error(f"[DagScheduler] Failed to mark program {program.id} as discarded: {state_error}")
                continue

            # Create wrapper that ensures state is always updated
            async def _run_with_guaranteed_cleanup(prog: Program = program, dag_inst: DAG = dag):
                async with self._sema:
                    await self._execute_dag_with_guaranteed_state_update(dag_inst, prog)

            # Record start time and create task
            start_time = time.time()
            task = asyncio.create_task(_run_with_guaranteed_cleanup(), name=f"dag-{program.id[:8]}")
            
            # Track task with timing info
            self._active_tasks[program.id] = TaskInfo(task, program.id, start_time)
            
            await self._metrics.increment_dag_runs_started()
            MetricsService.inc_dag_started()
            
            # Mark as started with error handling
            try:
                await self._state_manager.set_program_state(program, ProgramState.DAG_PROCESSING_STARTED)
                logger.info(f"[DagScheduler] Launched DAG for program {program.id}")
            except Exception as e:
                # If we can't mark as started, cancel the task and remove from tracking
                logger.error(f"[DagScheduler] Failed to mark program {program.id} as started: {e}")
                task.cancel()
                self._active_tasks.pop(program.id, None)
                # Don't increment metrics since we're not actually running

    async def _cleanup_finished_dags(self):
        """Clean up completed DAG tasks."""
        finished_task_infos = {}
        for pid, task_info in list(self._active_tasks.items()):
            if task_info.task.done():
                finished_task_infos[pid] = self._active_tasks.pop(pid)
        
        # Process all finished tasks
        if finished_task_infos:
            for pid, task_info in finished_task_infos.items():
                try:
                    task_info.task.result()  # This will raise if the task failed
                    await self._metrics.increment_dag_runs_completed()
                    MetricsService.inc_dag_completed()
                    logger.debug(f"[DagScheduler] DAG for program {pid} completed successfully")
                except Exception as e:
                    await self._metrics.increment_dag_errors()
                    MetricsService.inc_dag_error()
                    logger.error(f"[DagScheduler] DAG for program {pid} failed: {e}")

    async def _execute_dag_with_guaranteed_state_update(self, dag: DAG, program: Program):
        """Execute a DAG with guaranteed state update regardless of how it fails."""
        success = False
        try:
            # Run the actual DAG
            await dag.run(program)
            success = True
            
        except Exception as exc:
            # Log the error but don't re-raise yet - we need to update state first
            logger.error(f"[DagScheduler] DAG execution failed for program {program.id}: {exc}")
            success = False
        
        # CRITICAL: Always update program state regardless of success/failure
        try:
            if success:
                await self._state_manager.set_program_state(program, ProgramState.DAG_PROCESSING_COMPLETED)
                logger.debug(f"[DagScheduler] Program {program.id} completed successfully")
            else:
                await self._state_manager.set_program_state(program, ProgramState.DISCARDED)
                logger.info(f"[DagScheduler] Program {program.id} marked as discarded due to DAG failure")
        except Exception as state_error:
            # If we can't update state, that's a critical error
            logger.critical(
                f"[DagScheduler] CRITICAL: Failed to update state for program {program.id}: {state_error}. "
                f"Program may become orphaned!"
            )
            # Re-raise the state error since this is critical - we can't leave programs in an inconsistent state
            raise RuntimeError(f"Critical state update failure for program {program.id}: {state_error}")
        
        # Note: We don't re-raise the original DAG execution exception here because:
        # 1. The program state has been correctly updated (success -> COMPLETED, failure -> DISCARDED)
        # 2. Re-raising would cause unnecessary error logging in _cleanup_finished_dags()
        # 3. The task completion/failure will be properly tracked by the scheduler

    # ------------------------------------------------------------------
    # Trigger wait helper (copied from old logic)
    # ------------------------------------------------------------------

    async def _wait_for_trigger(self):
        poll_ms = int(self._config.poll_interval * 1000)
        if hasattr(self._storage, "_stream_key") and hasattr(self._storage, "_conn"):
            try:
                redis = await self._storage._conn()  # type: ignore[attr-defined]
                if hasattr(redis, "xread"):
                    stream_key = self._storage._stream_key()  # type: ignore[attr-defined]
                    await redis.xread({stream_key: "$"}, block=poll_ms, count=1)
                    return
            except asyncio.TimeoutError:
                # Normal timeout - just continue to fallback sleep
                logger.debug("Redis xread timed out, falling back to sleep")
            except Exception as e:
                # Any other error with Redis
                logger.debug(f"Redis xread failed, falling back to sleep: {e}")

        await asyncio.sleep(self._config.poll_interval)

    # ------------------------------------------------------------------
    # util
    # ------------------------------------------------------------------

    async def _cancel_task_safely(self, task_info: TaskInfo):
        """Safely cancel a task and ensure the program is marked as discarded."""
        if task_info.task.done():
            return
            
        # Cancel the task
        task_info.task.cancel()
        
        # Wait for cancellation with timeout
        try:
            await asyncio.wait_for(task_info.task, timeout=5.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass  # Expected
        except Exception as e:
            logger.warning(f"[DagScheduler] Unexpected error during task cancellation: {e}")
        
        # Ensure program is marked as discarded
        try:
            program = await self._storage.get(task_info.program_id)
            if program and program.state != ProgramState.DISCARDED:
                await self._state_manager.set_program_state(program, ProgramState.DISCARDED)
                logger.info(f"[DagScheduler] Program {task_info.program_id} marked as discarded after task cancellation")
        except Exception as e:
            logger.error(f"[DagScheduler] Failed to mark cancelled program {task_info.program_id} as discarded: {e}")

 