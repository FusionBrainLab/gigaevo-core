import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from loguru import logger

from src.programs.automata import DAGAutomata, ExecutionOrderDependency
from src.programs.program import (
    Program,
    ProgramStageResult,
    StageState,
)
from src.programs.state_manager import ProgramStateManager
from src.programs.stages.base import Stage
from src.programs.program_state import ProgramState
from src.programs.utils import format_error_for_llm

FINAL_STATES = {
    StageState.COMPLETED,
    StageState.SKIPPED,
    StageState.CANCELLED,
    StageState.FAILED,
}

def all_stages_finalized(program: Program, all_stage_names: Set[str]) -> bool:
    for stage in all_stage_names:
        res = program.stage_results.get(stage)
        if res is None or res.status not in FINAL_STATES:
            logger.debug(f"[DAG] Stage '{stage}' not finalized (status={res.status if res else 'missing'})")
            return False
    return True

class DAG:
    def __init__(
        self,
        nodes: Dict[str, Stage],
        edges: Dict[str, List[str]],
        state_manager: ProgramStateManager,
        *,
        entry_points: Optional[List[str]] = None,
        execution_order_deps: Optional[
            Dict[str, List[ExecutionOrderDependency]]
        ] = None,
        max_parallel_stages: int = 8,
        dag_timeout: Optional[float] = 2400.0,
    ):
        self.nodes = nodes
        self.edges = edges
        self.state_manager = state_manager
        self.entry_points = entry_points or list(nodes.keys())
        self.dag_timeout = dag_timeout
        self.automata = DAGAutomata()
        self._setup_automata(execution_order_deps or {})
        self._validate_no_cycles()
        self._validate_dependencies()
        self._stage_sema = asyncio.Semaphore(max(1, max_parallel_stages))

    def _setup_automata(self, execution_order_deps: Dict[str, List[ExecutionOrderDependency]]) -> None:
        for src, dsts in self.edges.items():
            for dst in dsts:
                self.automata.add_regular_dependency(dst, src)
        for stage_name, deps in execution_order_deps.items():
            for dep in deps:
                self.automata.add_execution_order_dependency(stage_name, dep)

    def _validate_dependencies(self) -> None:
        errors = self.automata.validate_dependencies(set(self.nodes.keys()))
        if errors:
            raise ValueError(f"Invalid dependencies found: {'; '.join(errors)}")

    async def run(self, program: Program, timeout: Optional[float] = None) -> None:
        try:
            await asyncio.wait_for(self._run_internal(program), timeout=timeout or self.dag_timeout)
        except asyncio.TimeoutError:
            logger.error(f"[DAG] DAG run for program {program.id} timed out.")
            await self.state_manager.set_program_state(program, ProgramState.DAG_PROCESSING_COMPLETED)

    async def _run_internal(self, program: Program) -> None:
        logger.info(f"[DAG] Starting DAG run for program {program.id}")
        ready, running = set(self.entry_points), set()
        done, skipped = set(program.stage_results.keys()), set()

        while True:
            try:
                latest = await self.state_manager.storage.get(program.id)
                if latest:
                    program = latest
            except Exception as e:
                logger.warning(f"[DAG] Failed to refresh program {program.id} from Redis: {e}")

            # Refresh done/skipped state
            done = {
                name for name, result in program.stage_results.items()
                if result.status in FINAL_STATES
            }
            skipped = {
                name for name, result in program.stage_results.items()
                if result.status == StageState.SKIPPED
            }

            # Compute ready and to_skip *before launching*
            ready = self.automata.get_ready_stages(program, set(self.nodes.keys()), done, running, skipped)
            to_skip = self.automata.get_stages_to_skip(program, set(self.nodes.keys()), done, running, skipped)

            for stage_name in to_skip:
                if stage_name not in program.stage_results:
                    skip_result = self.automata.create_skip_result(stage_name, program)
                    await self.state_manager.update_stage_result(program, stage_name, skip_result)
                    skipped.add(stage_name)
                    logger.warning(f"[DAG] Program {program.id}: Stage '{stage_name}' skipped due to failed dependency.")

            tasks = await self._launch_ready_stages(program, ready)
            running.update(ready)
            ready.clear()

            if tasks:
                results = await self._collect_results(program, tasks, done, running)

            if not ready and not running and all_stages_finalized(program, set(self.nodes.keys())):
                # Fallback: force-skip unresolved
                for stage_name in self.nodes:
                    if stage_name not in program.stage_results:
                        logger.warning(f"[DAG] Program {program.id}: Forcing skip of unresolved stage '{stage_name}'")
                        skip_result = self.automata.create_skip_result(stage_name, program)
                        await self.state_manager.update_stage_result(program, stage_name, skip_result)
                logger.info(f"[DAG] Program {program.id}: All stages finalized — terminating DAG.")
                break

        all_succeeded = all(
            result.is_completed() for result in program.stage_results.values()
        ) and len(program.stage_results) == len(self.nodes)

        summary = self._generate_completion_summary(program, all_succeeded)
        logger.info(f"[DAG] {summary}")

        if not all_succeeded:
            analysis = self._analyze_completion_failures(program)
            logger.info(f"[DAG] Failure Analysis: {analysis}")

        await self.state_manager.set_program_state(program, ProgramState.DAG_PROCESSING_COMPLETED)

    async def _launch_ready_stages(self, program: Program, ready: Set[str]) -> Dict[str, asyncio.Task]:
        tasks = {}
        now_ts = datetime.now(timezone.utc)
        for name in ready:
            await self.state_manager.mark_stage_running(program, name, started_at=now_ts)
            logger.info(f"[DAG] Program {program.id}: Stage '{name}' started.")

            async def _run_stage(stage_name=name):
                async with self._stage_sema:
                    return await self.nodes[stage_name].run(program)

            tasks[name] = asyncio.create_task(_run_stage(), name=f"stage-{name[:8]}")
        return tasks

    async def _collect_results(self, program: Program, tasks: Dict[str, asyncio.Task], done: Set[str], running: Set[str]) -> Dict[str, ProgramStageResult]:
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        outcomes = {}
        for name, result in zip(tasks.keys(), results):
            running.discard(name)
            done.add(name)
            finished_at = datetime.now(timezone.utc)

            if isinstance(result, Exception):
                stage_result = ProgramStageResult(
                    status=StageState.FAILED,
                    error=format_error_for_llm(error=result, stage_name=name, context="Exception escaped from stage execution"),
                    started_at=program.stage_results.get(name, ProgramStageResult()).started_at or finished_at,
                    finished_at=finished_at,
                )
                logger.error(f"[DAG] Program {program.id}: Stage '{name}' failed with unexpected exception: {result}")
            else:
                stage_result = result
                if stage_result.is_failed():
                    logger.error(f"[DAG] Program {program.id}: Stage '{name}' failed: {stage_result.error}")
                else:
                    logger.info(f"[DAG] Program {program.id}: Stage '{name}' completed successfully.")

            program.stage_results[name] = stage_result
            outcomes[name] = stage_result

        try:
            await self.state_manager.storage.update(program)
        except Exception as e:
            logger.error(f"[DAG] Failed to batch-update program {program.id}: {e}")

        return outcomes

    def _generate_completion_summary(self, program: Program, all_succeeded: bool) -> str:
        status = "SUCCESS" if all_succeeded else "PARTIAL"
        completed = sum(1 for r in program.stage_results.values() if r.is_completed())
        failed = sum(1 for r in program.stage_results.values() if r.is_failed())
        skipped = sum(1 for r in program.stage_results.values() if r.is_skipped())
        total = len(self.nodes)
        summary = f"Program {program.id} DAG execution {status}: {completed}/{total} completed"
        if failed:
            summary += f", {failed} failed"
        if skipped:
            summary += f", {skipped} skipped"
        metrics_count = len(program.metrics) if program.metrics else 0
        summary += f" | Contract Status: metrics={metrics_count}"
        if not program.metrics:
            summary += " ⚠️ (may be rejected by evolution engine)"
        else:
            summary += " ✅"
        return summary

    def _analyze_completion_failures(self, program: Program) -> str:
        analysis = []
        failed_stages = [s for s, r in program.stage_results.items() if r.is_failed()]
        skipped_stages = [s for s, r in program.stage_results.items() if r.is_skipped()]
        if failed_stages:
            analysis.append(f"Failed stages: {failed_stages}")
            critical = [s for s in failed_stages if any(x in s.lower() for x in ['validate', 'metric'])]
            if critical:
                analysis.append(f"⚠️ Critical failures: {critical}")
        if skipped_stages:
            analysis.append(f"Skipped stages: {skipped_stages}")
        if not program.metrics:
            analysis.append("💡 No metrics extracted")
        return " | ".join(analysis) if analysis else "No specific failure patterns identified"

    def get_stage_dependencies(self, stage_name: str) -> Dict[str, any]:
        return self.automata.get_stage_dependencies(stage_name)

    def _validate_no_cycles(self):
        visited, stack = set(), set()
        def visit(node: str):
            if node in stack:
                raise ValueError(f"Cycle detected in DAG at stage: {node}")
            if node in visited:
                return
            visited.add(node)
            stack.add(node)
            for neighbor in self.edges.get(node, []):
                visit(neighbor)
            stack.remove(node)
        for node in self.nodes:
            visit(node)

    def to_dot(self) -> str:
        lines = ["digraph DAG {"]
        for src, dsts in self.edges.items():
            for dst in dsts:
                lines.append(f'  "{src}" -> "{dst}";')
        lines.append("}")
        return "\n".join(lines)
