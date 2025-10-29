from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Dict, Set, cast

from loguru import logger

from src.database.state_manager import ProgramStateManager
from src.programs.core_types import (
    FINAL_STATES,
    ProgramStageResult,
    StageError,
    StageState,
)
from src.programs.dag.automata import (
    DAGAutomata,
    DataFlowEdge,
    ExecutionOrderDependency,
)
from src.programs.program import Program
from src.programs.stages.base import Stage


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
        nodes: Dict[str, Stage],
        data_flow_edges: list[DataFlowEdge],
        execution_order_deps: dict[str, list[ExecutionOrderDependency]] | None,
        state_manager: ProgramStateManager,
        *,
        max_parallel_stages: int = 8,
        dag_timeout: float | None = 2400.0,
        stall_grace_seconds: float = 30.0,
    ) -> None:
        self.automata = DAGAutomata.build(nodes, data_flow_edges, execution_order_deps)
        self.state_manager = state_manager
        self._stage_sema = asyncio.Semaphore(max(1, max_parallel_stages))
        self.dag_timeout = dag_timeout
        self.stall_grace_seconds = stall_grace_seconds
        self._previous_launched_hash = None

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
        except asyncio.TimeoutError:
            logger.error("[DAG][{}] DAG run timed out after {}s", pid, self.dag_timeout)
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

        # Persist the initial snapshot
        await self._persist_program_snapshot(program)

        running: set[str] = set()
        launched_this_run: set[str] = set()
        finished_this_run: set[str] = set()

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
            ready = self.automata.get_ready_stages(
                program, running, launched_this_run, finished_this_run
            )

            # 3) Launch ready
            tasks = await self._launch_ready(program, ready)
            if tasks:
                running.update(tasks.keys())
                launched_this_run.update(tasks.keys())

            # 4) Collect
            collected_any = False
            if tasks:
                await self._collect(program, tasks, running, finished_this_run)
                collected_any = True

            # 5) Progress accounting
            if skip_progress or tasks or collected_any:
                last_progress_ts = time.time()
                stalled_reported = False

            # 6) Termination / stall detection
            if not tasks and not running and not to_skip:
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

            # Yield to avoid tight loop when nothing progressed
            if not tasks and not skip_progress and running:
                await asyncio.sleep(0.005)

        await self._persist_program_snapshot(program)

    async def _launch_ready(
        self, program: Program, ready: Set[str]
    ) -> Dict[str, asyncio.Task]:
        pid = self._pid(program)
        tasks: Dict[str, asyncio.Task] = {}
        if not ready:
            logger.debug("[DAG][{}] No ready stages to launch.", pid)
            return tasks

        now_ts = datetime.now(timezone.utc)
        for name in sorted(list(ready)):
            await self.state_manager.mark_stage_running(
                program, name, started_at=now_ts
            )
            logger.info("[DAG][{}] Stage '{}' STARTED.", pid, name)

            async def _run_stage(stage_name=name):
                async with self._stage_sema:
                    try:
                        logger.debug(
                            "[DAG][{}] Preparing inputs for '{}'...", pid, stage_name
                        )
                        named_inputs = self.automata.build_named_inputs(
                            program, stage_name
                        )
                        # Attach the inputs to the stage
                        stage: Stage = self.automata.topology.nodes[stage_name]
                        stage.attach_inputs(named_inputs)
                        result = await stage.execute(program)
                        await self._persist_stage_result(program, stage_name, result)
                        logger.debug(
                            "[DAG][{}] Stage '{}' finished with status {}",
                            pid,
                            stage_name,
                            result.status.name,
                        )
                        return result

                    except Exception as exc:
                        now = datetime.now(timezone.utc)
                        logger.exception(
                            "[DAG][{}] Unhandled exception in stage '{}' wrapper",
                            pid,
                            stage_name,
                        )
                        return ProgramStageResult(
                            status=StageState.FAILED,
                            error=StageError.from_exception(
                                exc, stage=self._canonical_stage_name(stage_name)
                            ),
                            started_at=now,
                            finished_at=now,
                        )

            tasks[name] = asyncio.create_task(_run_stage(), name=f"stage-{name[:16]}")

        logger.debug("[DAG][{}] Launched stages: {}", pid, sorted(list(tasks.keys())))
        return tasks

    async def _collect(
        self,
        program: Program,
        tasks: dict[str, asyncio.Task],
        running: set[str],
        finished_this_run: set[str],
    ) -> None:
        pid = self._pid(program)
        logger.debug("[DAG][{}] Collecting {} stage result(s)...", pid, len(tasks))
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        # Debug: log the types of results we get
        for stage_name, outcome in zip(tasks.keys(), results):
            logger.debug(
                "[DAG][{}] Collected result for '{}': type={}",
                pid,
                stage_name,
                type(outcome).__name__ if outcome is not None else "None",
            )

        for stage_name, outcome in zip(tasks.keys(), results):
            running.discard(stage_name)
            now = datetime.now(timezone.utc)

            if isinstance(outcome, Exception):
                result = ProgramStageResult(
                    status=StageState.FAILED,
                    error=StageError.from_exception(
                        outcome, stage=self._canonical_stage_name(stage_name)
                    ),
                    started_at=now,
                    finished_at=now,
                )
                logger.error(
                    "[DAG][{}] Stage '{}' FAILED with raised exception: {}",
                    pid,
                    stage_name,
                    outcome,
                )
            else:
                result = cast(ProgramStageResult, outcome)

            await self._persist_stage_result(program, stage_name, result)
            await self._persist_program_snapshot(program)

            if result.status in FINAL_STATES:
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
        await self.state_manager.update_stage_result(program, stage_name, result)

    async def _persist_program_snapshot(self, program: Program) -> None:
        await self.state_manager.storage.update(program)
