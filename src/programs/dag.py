from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Set, cast

from loguru import logger

from src.programs.program import Program, ProgramStageResult, StageState
from src.programs.state_manager import ProgramStateManager
from src.programs.utils import build_stage_result
from src.programs.automata import DAGAutomata, FINAL_STATES
from src.programs.automata import DataFlowEdge, ExecutionOrderDependency
from src.programs.stages.base import Stage


class DAG:
    """
    Minimal DAG runner:
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

    # ------------------------ public ------------------------

    async def run(self, program: Program) -> None:
        pid = self._pid(program)
        logger.debug("[DAG][{}] Run started", pid)

        try:
            if self.dag_timeout is not None:
                await asyncio.wait_for(self._run_internal(program), timeout=self.dag_timeout)
            else:
                await self._run_internal(program)
        except asyncio.TimeoutError:
            logger.error("[DAG][{}] DAG run timed out after {}s", pid, self.dag_timeout)
            raise

    # ------------------------ internals ------------------------

    def _pid(self, program: Program) -> str:
        try:
            return getattr(program, "id", "<?>")[:8]
        except Exception:
            return "<?>"

    async def _run_internal(self, program: Program) -> None:
        pid = self._pid(program)

        running: Set[str] = set()
        launched_this_run: Set[str] = set()
        finished_this_run: Set[str] = set()

        tick = 0
        last_progress_ts = time.time()
        stalled_reported = False

        while True:
            tick += 1
            logger.debug(
                "[DAG][{}] Tick {} | running={} launched={} finished={}",
                pid, tick, sorted(list(running)), sorted(list(launched_this_run)), sorted(list(finished_this_run))
            )

            # 1) Auto-skip (exec contradictions)
            to_skip = self.automata.get_stages_to_skip(program, running, launched_this_run, finished_this_run)
            skip_progress = False
            for stage_name in to_skip:
                if stage_name in program.stage_results:
                    logger.debug("[DAG][{}] '{}' already has result; not re-skipping", pid, stage_name)
                    continue
                skip_result = self.automata.create_skip_result(stage_name, program)
                await self._persist_stage_result(program, stage_name, skip_result)
                finished_this_run.add(stage_name)
                launched_this_run.add(stage_name)
                skip_progress = True
                logger.info("[DAG][{}] Stage '{}' AUTO-SKIPPED.", pid, stage_name)

            # 2) Ready set
            ready = self.automata.get_ready_stages(program, running, launched_this_run, finished_this_run)

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
                all_names = set(self.automata.topology.nodes.keys())  # type: ignore
                done, skipped = self.automata._compute_done_sets(program, finished_this_run)
                unresolved = sorted(list(all_names - done - skipped))
                if unresolved:
                    blockers = self.automata.summarize_blockers_for_log(
                        program, running, launched_this_run, finished_this_run
                    )
                    logger.warning(
                        "[DAG][{}] No ready stages, nothing running, but unresolved stages remain: {}\nBlockers:\n{}",
                        pid, unresolved, blockers
                    )
                else:
                    logger.info("[DAG][{}] Idle & no pending work — terminating.", pid)
                break

            # 7) Stall watchdog (no progress while there is pending work)
            now = time.time()
            if (now - last_progress_ts) > self.stall_grace_seconds and not stalled_reported:
                stalled_reported = True
                blockers = self.automata.summarize_blockers_for_log(
                    program, running, launched_this_run, finished_this_run
                )
                logger.warning(
                    "[DAG][{}] STALLED (no progress for {}s). Diagnostics:\n{}",
                    pid, self.stall_grace_seconds, blockers
                )

            # Yield to avoid tight loop when nothing progressed
            if not tasks and not skip_progress and running:
                await asyncio.sleep(0.005)

        # Final snapshot (best effort)
        await self._persist_program_snapshot(program)

    async def _launch_ready(self, program: Program, ready: Set[str]) -> Dict[str, asyncio.Task]:
        pid = self._pid(program)
        tasks: Dict[str, asyncio.Task] = {}
        if not ready:
            logger.debug("[DAG][{}] No ready stages to launch.", pid)
            return tasks

        now_ts = datetime.now(timezone.utc)
        for name in sorted(list(ready)):
            await self.state_manager.mark_stage_running(program, name, started_at=now_ts)
            logger.info("[DAG][{}] Stage '{}' STARTED.", pid, name)

            async def _run_stage(stage_name=name):
                async with self._stage_sema:
                    try:
                        logger.debug("[DAG][{}] Preparing inputs for '{}'...", pid, stage_name)
                        named_inputs = self.automata.build_named_inputs(program, stage_name)

                        # Defensive mandatory check at runtime
                        stage = self.automata.topology.nodes[stage_name]  # type: ignore
                        missing_mandatory = [
                            inp for inp in stage.__class__.mandatory_inputs() if inp not in named_inputs
                        ]
                        if missing_mandatory:
                            logger.warning(
                                "[DAG][{}] Stage '{}' SKIPPED at runtime — missing mandatory inputs: {}",
                                pid, stage_name, missing_mandatory
                            )
                            return build_stage_result(
                                status=StageState.SKIPPED,
                                started_at=now_ts,
                                error=f"Missing mandatory inputs: {missing_mandatory} | available={list(named_inputs.keys())}",
                                stage_name=stage_name,
                            )

                        logger.debug(
                            "[DAG][{}] Running stage '{}' with inputs: {}",
                            pid, stage_name, sorted(list(named_inputs.keys()))
                        )
                        stage.set_named_inputs(named_inputs)
                        result = await stage.run(program)
                        logger.debug(
                            "[DAG][{}] Stage '{}' finished with status {}",
                            pid, stage_name, getattr(result, "status", "<?>")
                        )
                        return result
                    except Exception as exc:
                        logger.exception("[DAG][{}] Unhandled exception in stage '{}'", pid, stage_name)
                        return ProgramStageResult(
                            status=StageState.FAILED,
                            error=f"Unhandled exception in stage '{stage_name}': {exc}",
                            started_at=now_ts,
                            finished_at=datetime.now(timezone.utc),
                        )

            tasks[name] = asyncio.create_task(_run_stage(), name=f"stage-{name[:16]}")

        logger.debug("[DAG][{}] Launched stages: {}", pid, sorted(list(tasks.keys())))
        return tasks

    async def _collect(
        self,
        program: Program,
        tasks: Dict[str, asyncio.Task],
        running: Set[str],
        finished_this_run: Set[str],
    ) -> None:
        pid = self._pid(program)
        logger.debug("[DAG][{}] Collecting {} stage result(s)...", pid, len(tasks))
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        for stage_name, outcome in zip(tasks.keys(), results):
            running.discard(stage_name)
            now = datetime.now(timezone.utc)

            if isinstance(outcome, Exception):
                result = ProgramStageResult(
                    status=StageState.FAILED,
                    error=f"Unhandled exception in stage '{stage_name}': {outcome}",
                    started_at=now,
                    finished_at=now,
                )
                logger.error("[DAG][{}] Stage '{}' FAILED with raised exception: {}", pid, stage_name, outcome)
            else:
                result = cast(ProgramStageResult, outcome)

            await self._persist_stage_result(program, stage_name, result)
            await self._persist_program_snapshot(program)

            if result.status in FINAL_STATES:
                finished_this_run.add(stage_name)
                logger.info("[DAG][{}] Stage '{}' FINALIZED as {}.", pid, stage_name, result.status.name)

    async def _persist_stage_result(self, program: Program, stage_name: str, result: ProgramStageResult) -> None:
        pid = self._pid(program)
        try:
            await self.state_manager.update_stage_result(program, stage_name, result)
            logger.debug("[DAG][{}] Persisted stage result for '{}' (status={})", pid, stage_name, result.status.name)
        except Exception as e:
            logger.error("[DAG][{}] Failed to persist stage result for '{}': {}", pid, stage_name, e)

    async def _persist_program_snapshot(self, program: Program) -> None:
        pid = self._pid(program)
        try:
            logger.debug("[DAG][{}] Persisting program snapshot (id={}).", pid, getattr(program, "id", "?"))
            await self.state_manager.storage.update(program)
        except Exception as e:
            logger.error("[DAG][{}] Failed to persist program snapshot: {}", pid, e)
