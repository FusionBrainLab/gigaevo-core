from __future__ import annotations

import asyncio
from asyncio import CancelledError
from datetime import datetime, timezone
import time
from typing import cast

from loguru import logger

from gigaevo.database.state_manager import ProgramStateManager
from gigaevo.programs.core_types import (
    ProgramStageResult,
    StageError,
    StageState,
)
from gigaevo.programs.dag.automata import (
    DAGAutomata,
    DataFlowEdge,
    ExecutionOrderDependency,
)
from gigaevo.programs.program import Program
from gigaevo.programs.stages.base import Stage
from gigaevo.utils.trackers.base import LogWriter

DEFAULT_STALL_GRACE_SECONDS = 120.0
DEFAULT_STAGE_TIMEOUT = 90.0


class DAG:
    """
    Minimal DAG runner (new Stage API):
      - Delegates scheduling/validation/cache rules to DAGAutomata.
      - Launches only the stages Automata says are ready.
      - Applies Automata's auto-skip decisions.
      - Passes only COMPLETED producer outputs as inputs.
      - Enforces dag_timeout.
      - Emits blocker diagnostics if stalled (no progress).
    """

    def __init__(
        self,
        nodes: dict[str, Stage],
        data_flow_edges: list[DataFlowEdge],
        execution_order_deps: dict[str, list[ExecutionOrderDependency]] | None,
        state_manager: ProgramStateManager,
        *,
        max_parallel_stages: int = 8,
        dag_timeout: float | None = 2400.0,
        writer: LogWriter,
    ) -> None:
        self.automata = DAGAutomata.build(nodes, data_flow_edges, execution_order_deps)
        self.state_manager = state_manager
        self._stage_sema = asyncio.Semaphore(max(1, max_parallel_stages))
        self.dag_timeout = dag_timeout
        max_stage_timeout = max((s.timeout for s in nodes.values()))
        self.stall_grace_seconds = max(
            DEFAULT_STALL_GRACE_SECONDS, max_stage_timeout * 1.5
        )
        self._previous_launched_hash = None
        self._writer: LogWriter = writer.bind(path=["dag", "internals"])

    async def run(self, program: Program) -> None:
        pid = self._pid(program)
        logger.debug("[DAG][{}] Run started", pid)

        try:
            if self.dag_timeout is not None:
                await asyncio.wait_for(
                    self._run_internal(program), timeout=self.dag_timeout
                )
            else:
                await self._run_internal(program)

            self._writer.scalar("dag_timeout", 0)
        except asyncio.TimeoutError:
            logger.error("[DAG][{}] DAG run timed out after {}s", pid, self.dag_timeout)
            self._writer.scalar("dag_timeout", 1)
            raise

    def _pid(self, program: Program) -> str:
        return program.id[:8]

    def _canonical_stage_name(self, stage_name: str) -> str:
        return self.automata.topology.nodes[stage_name].stage_name  # type: ignore

    async def _run_internal(self, program: Program) -> None:
        pid = self._pid(program)

        # Initialize all stages to PENDING
        for name in self.automata.topology.nodes.keys():
            program.stage_results.setdefault(
                name, ProgramStageResult(status=StageState.PENDING)
            )

        # Persist initial state (PENDING stages) for monitoring/crash recovery
        # Note: Further snapshots are NOT needed because update_stage_result()
        # persists the ENTIRE program object after each stage completes,
        # including any changes to metrics, metadata, etc.
        await self.state_manager.update_program(program)

        running: set[str] = set()
        launched_this_run: set[str] = set()
        finished_this_run: set[str] = set()
        pending_tasks: dict[asyncio.Task, str] = {}

        tick = 0
        last_progress_ts = time.time()
        stalled_reported = False

        while True:
            tick += 1

            tuple_to_hash = tuple(
                sorted(list(running))
                + sorted(list(launched_this_run))
                + sorted(list(finished_this_run))
            )
            if tuple_to_hash != self._previous_launched_hash:
                self._previous_launched_hash = tuple_to_hash
                logger.debug(
                    "[DAG][{}] Running={} Launched={} Finished={}",
                    pid,
                    sorted(list(running)),
                    sorted(list(launched_this_run)),
                    sorted(list(finished_this_run)),
                )

            # 1) Auto-skips
            to_skip = self.automata.get_stages_to_skip(
                program, running, launched_this_run, finished_this_run
            )
            skip_progress = False
            for stage_name in to_skip:
                res = program.stage_results.get(stage_name)

                # don't re-skip RUNNING or FINAL; allow overwrite if None or PENDING
                if res is not None and res.status not in (StageState.PENDING,):
                    logger.debug(
                        "[DAG][{}] '{}' already finalized/running as {}; not re-skipping",
                        pid,
                        stage_name,
                        res.status.name,
                    )
                    continue

                now_ts = datetime.now(timezone.utc)
                skip_result = ProgramStageResult(
                    status=StageState.SKIPPED,
                    error=StageError(
                        type="AutoSkip",
                        message="Automata decided to skip stage due to contradictions or policy.",
                        stage=self._canonical_stage_name(stage_name),
                    ),
                    started_at=now_ts,
                    finished_at=now_ts,
                )
                await self._persist_stage_result(program, stage_name, skip_result)
                finished_this_run.add(stage_name)
                launched_this_run.add(stage_name)
                skip_progress = True
                logger.info("[DAG][{}] Stage '{}' AUTO-SKIPPED.", pid, stage_name)

            if to_skip and not skip_progress and not running:
                blockers = self.automata.summarize_blockers_for_log(
                    program, running, launched_this_run, finished_this_run
                )
                msg = (
                    f"[DAG][{pid}] DEADLOCK: Automata requested skips={sorted(to_skip)} "
                    f"but none could be applied (states not PENDING). Blockers:\n{blockers}"
                )
                logger.error(msg)
                raise RuntimeError(msg)

            # 2) Ready set
            ready, newly_cached = self.automata.get_ready_stages(
                program, running, launched_this_run, finished_this_run
            )
            finished_this_run.update(newly_cached)
            if newly_cached:
                logger.info(
                    "[DAG][{}] Stages CACHED (skipped execution): {}",
                    pid,
                    sorted(list(newly_cached)),
                )

            # 3) Launch ready
            new_tasks_map = await self._launch_ready(program, ready)
            if new_tasks_map:
                running.update(new_tasks_map.keys())
                launched_this_run.update(new_tasks_map.keys())
                # Track tasks
                for name, task in new_tasks_map.items():
                    pending_tasks[task] = name

            # 4) Progress accounting (skips, launches, or cache hits)
            if skip_progress or new_tasks_map or newly_cached:
                last_progress_ts = time.time()
                stalled_reported = False

            # 5) Termination check
            if (
                not pending_tasks
                and not to_skip
                and not new_tasks_map
                and not newly_cached
            ):
                # Are there unresolved stages left (neither done nor skipped)?
                all_names = set(self.automata.topology.nodes.keys())
                done, skipped = self.automata._compute_done_sets(
                    program, finished_this_run
                )
                unresolved = sorted(list(all_names - done - skipped))
                if unresolved:
                    blockers = self.automata.summarize_blockers_for_log(
                        program, running, launched_this_run, finished_this_run
                    )
                    logger.warning(
                        "[DAG][{}] No ready stages, nothing running, but unresolved stages remain: {}\nBlockers:\n{}",
                        pid,
                        unresolved,
                        blockers,
                    )
                else:
                    logger.info("[DAG][{}] Idle & no pending work — terminating.", pid)
                break

            # 6) Wait for tasks
            collected_any = False
            if pending_tasks:
                # Wait for at least one task to complete, or timeout for watchdog/skips
                # Using a short timeout allows us to re-check for skips or stalls periodically
                wait_timeout = 1.0
                done_tasks, _ = await asyncio.wait(
                    pending_tasks.keys(),
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=wait_timeout,
                )

                for task in done_tasks:
                    stage_name = pending_tasks.pop(task)
                    await self._process_finished_task(
                        program, stage_name, task, running, finished_this_run
                    )
                    collected_any = True

            if collected_any:
                last_progress_ts = time.time()
                stalled_reported = False

            # 7) Stall watchdog (no progress while there is pending work)
            now = time.time()
            if (
                now - last_progress_ts
            ) > self.stall_grace_seconds and not stalled_reported:
                stalled_reported = True
                blockers = self.automata.summarize_blockers_for_log(
                    program, running, launched_this_run, finished_this_run
                )
                logger.warning(
                    "[DAG][{}] STALLED (no progress for {}s). Diagnostics:\n{}",
                    pid,
                    self.stall_grace_seconds,
                    blockers,
                )

            # Yield to avoid tight loop if we didn't wait on tasks (e.g. just processed skips)
            if (
                not collected_any
                and not new_tasks_map
                and not skip_progress
                and not newly_cached
                and pending_tasks
            ):
                # If we have pending tasks but didn't collect any (timeout), we are fine.
                # The asyncio.wait timeout acts as our yield.
                pass
            elif (
                not collected_any
                and not new_tasks_map
                and not skip_progress
                and not newly_cached
            ):
                # Should mostly be covered by the wait, but if we have no tasks running
                # and are just looping (e.g. waiting for something else? unlikely here), sleep briefly.
                await asyncio.sleep(0.005)

    async def _launch_ready(
        self, program: Program, ready: set[str]
    ) -> dict[str, asyncio.Task]:
        pid = self._pid(program)
        tasks: dict[str, asyncio.Task] = {}
        if not ready:
            return tasks

        now_ts = datetime.now(timezone.utc)
        for name in sorted(list(ready)):
            await self.state_manager.mark_stage_running(
                program, name, started_at=now_ts
            )
            logger.info("[DAG][{}] Stage '{}' STARTED.", pid, name)

            async def _run_stage(stage_name=name):
                async with self._stage_sema:
                    named_inputs = self.automata.build_named_inputs(program, stage_name)
                    stage: Stage = self.automata.topology.nodes[stage_name]
                    stage.attach_inputs(named_inputs)
                    return await stage.execute(program)

            tasks[name] = asyncio.create_task(_run_stage(), name=f"stage-{name[:16]}")

        logger.debug("[DAG][{}] Launched stages: {}", pid, sorted(list(tasks.keys())))
        return tasks

    async def _process_finished_task(
        self,
        program: Program,
        stage_name: str,
        task: asyncio.Task,
        running: set[str],
        finished_this_run: set[str],
    ) -> None:
        pid = self._pid(program)

        # We get the result (or exception) from the task
        try:
            outcome = task.result()
        except CancelledError:
            outcome = CancelledError()
        except Exception as e:
            outcome = e

        logger.debug(
            "[DAG][{}] Collected result for '{}': type={}",
            pid,
            stage_name,
            type(outcome).__name__ if outcome is not None else "None",
        )

        running.discard(stage_name)
        now = datetime.now(timezone.utc)
        started_at = program.stage_results[stage_name].started_at or now

        result: ProgramStageResult
        if isinstance(outcome, Exception):
            if isinstance(outcome, CancelledError):
                result = ProgramStageResult(
                    status=StageState.CANCELLED,
                    error=StageError(
                        type="Cancelled",
                        message="Stage task was cancelled.",
                        stage=self._canonical_stage_name(stage_name),
                    ),
                    started_at=started_at,
                    finished_at=now,
                )
                logger.warning("[DAG][{}] Stage '{}' CANCELLED.", pid, stage_name)
            else:
                result = ProgramStageResult(
                    status=StageState.FAILED,
                    error=StageError.from_exception(
                        outcome, stage=self._canonical_stage_name(stage_name)
                    ),
                    started_at=started_at,
                    finished_at=now,
                )
        else:
            result = cast(ProgramStageResult, outcome)

        if result.status == StageState.FAILED and result.error is not None:
            logger.exception(
                "[DAG][{}] Stage '{}' FAILED with exception.\n### ERROR SUMMARY ###:\n{}",
                pid,
                stage_name,
                result.error.pretty(include_traceback=True),
            )

        await self._persist_stage_result(program, stage_name, result)

        finished_this_run.add(stage_name)
        logger.info(
            "[DAG][{}] Stage '{}' FINALIZED as {}.",
            pid,
            stage_name,
            result.status.name,
        )

    async def _persist_stage_result(
        self, program: Program, stage_name: str, result: ProgramStageResult
    ) -> None:
        await self._write_stage_status(stage_name, result)
        await self.state_manager.update_stage_result(program, stage_name, result)

    async def _write_stage_status(
        self, stage_name: str, result: ProgramStageResult
    ) -> None:
        self._writer.scalar(
            "stage_success",
            int(result.status == StageState.COMPLETED),
            path=[stage_name],
        )
        self._writer.scalar(
            "stage_failure", int(result.status == StageState.FAILED), path=[stage_name]
        )
        self._writer.scalar(
            "stage_skipped", int(result.status == StageState.SKIPPED), path=[stage_name]
        )
        self._writer.scalar(
            "stage_cancelled",
            int(result.status == StageState.CANCELLED),
            path=[stage_name],
        )
        self._writer.scalar(
            "stage_duration", float(result.duration_seconds() or 0.0), path=[stage_name]
        )
