import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set
from collections import defaultdict
import networkx as nx

from loguru import logger

from src.programs.automata import DAGAutomata, ExecutionOrderDependency
from src.programs.program import (
    Program,
    ProgramStageResult,
    StageState,
)
from src.programs.stages.base import Stage
from src.programs.state_manager import ProgramStateManager
from src.programs.utils import format_error_for_llm

FINAL_STATES = {
    StageState.COMPLETED,
    StageState.SKIPPED,
    StageState.CANCELLED,
    StageState.FAILED,
}


class DAG:
    def __init__(
        self,
        nodes: Dict[str, Stage],
        edges: Dict[str, List[str]],
        state_manager: ProgramStateManager,
        *,
        execution_order_deps: Optional[
            Dict[str, List[ExecutionOrderDependency]]
        ] = None,
        max_parallel_stages: int = 8,
        dag_timeout: Optional[float] = 2400.0,
    ):
        self.nodes = nodes
        self.edges = edges
        self.state_manager = state_manager
        self.dag_timeout = dag_timeout
        self._stage_names: Set[str] = set(self.nodes.keys())
        self.automata = DAGAutomata()
        # Validate configuration early (nodes/edges/entry points/exec deps/types)
        self._validate_graph_integrity(execution_order_deps or {})
        # Build transition rules
        self._setup_automata(execution_order_deps or {})
        self._validate_acyclic_graph()
        self._validate_dependencies()
        self._stage_sema = asyncio.Semaphore(max(1, max_parallel_stages))

    def _validate_graph_integrity(
        self, execution_order_deps: Dict[str, List[ExecutionOrderDependency]]
    ) -> None:
        # Validate node types
        bad_nodes = [name for name, obj in self.nodes.items() if not isinstance(obj, Stage)]
        if bad_nodes:
            raise ValueError(
                f"Non-Stage objects registered as nodes: {', '.join(sorted(bad_nodes))}"
            )

        stage_names = self._stage_names

        # Validate edges sources and destinations
        unknown_sources = [src for src in self.edges.keys() if src not in stage_names]
        if unknown_sources:
            raise ValueError(
                f"Edges contain unknown source stage(s): {', '.join(sorted(set(unknown_sources)))}"
            )
        unknown_dests = [dst for dsts in self.edges.values() for dst in dsts if dst not in stage_names]
        if unknown_dests:
            raise ValueError(
                f"Edges reference unknown destination stage(s): {', '.join(sorted(set(unknown_dests)))}"
            )

        # Validate execution order dependencies
        if execution_order_deps:
            unknown_exec_keys = [k for k in execution_order_deps.keys() if k not in stage_names]
            if unknown_exec_keys:
                raise ValueError(
                    f"Execution-order deps contain unknown target stage(s): {', '.join(sorted(set(unknown_exec_keys)))}"
                )
            unknown_exec_targets = [
                dep.stage_name
                for deps in execution_order_deps.values()
                for dep in deps
                if dep.stage_name not in stage_names
            ]
            if unknown_exec_targets:
                raise ValueError(
                    f"Execution-order deps reference unknown stage(s): {', '.join(sorted(set(unknown_exec_targets)))}"
                )

        # Sanity check timeouts and parallelism
        if self.dag_timeout is not None and self.dag_timeout <= 0:
            raise ValueError("dag_timeout must be positive or None")
        # max_parallel_stages validated in __init__ when creating semaphore

    def _setup_automata(
        self, execution_order_deps: Dict[str, List[ExecutionOrderDependency]]
    ) -> None:
        for src, dsts in self.edges.items():
            for dst in dsts:
                self.automata.add_regular_dependency(dst, src)
        for stage_name, deps in execution_order_deps.items():
            for dep in deps:
                self.automata.add_execution_order_dependency(stage_name, dep)

    def _validate_dependencies(self) -> None:
        errors = self.automata.validate_dependencies(self._stage_names)
        # Ensure all transition rule targets exist among nodes
        invalid_rule_targets = [
            stage_name
            for stage_name in self.automata.transition_rules.keys()
            if stage_name not in self.nodes
        ]
        if invalid_rule_targets:
            errors.append(
                "Transition rules defined for non-existent stage(s): "
                + ", ".join(sorted(invalid_rule_targets))
            )
        if errors:
            raise ValueError(f"Invalid dependencies found: {'; '.join(errors)}")

    async def run(
        self, program: Program, timeout: Optional[float] = None
    ) -> None:
        """Run the DAG for a given program, respecting an optional timeout.

        Raises asyncio.TimeoutError to let the caller decide lifecycle updates.
        """
        try:
            await asyncio.wait_for(
                self._run_internal(program), timeout=timeout or self.dag_timeout
            )
        except asyncio.TimeoutError:
            logger.error(f"[DAG] DAG run for program {program.id} timed out.")
            raise

    async def _run_internal(self, program: Program) -> None:
        """Main scheduling loop: computes ready stages, launches tasks, and collects results."""
        logger.info(f"[DAG] Starting DAG run for program {program.id}")
        running = set()
        done, skipped = set(program.stage_results.keys()), set()

        def _compute_done_skipped() -> None:
            nonlocal done, skipped
            done = {
                name
                for name, result in program.stage_results.items()
                if result.status in FINAL_STATES
            }
            skipped = {
                name
                for name, result in program.stage_results.items()
                if result.status == StageState.SKIPPED
            }

        while True:
            try:
                latest = await self.state_manager.storage.get(program.id)
                if latest:
                    program = latest
            except Exception as e:
                logger.warning(
                    f"[DAG] Failed to refresh program {program.id} from Redis: {e}"
                )

            # Refresh done/skipped state
            _compute_done_skipped()

            # Compute ready and to_skip
            ready = self.automata.get_ready_stages(
                program, self._stage_names, done, running, skipped
            )
            to_skip = self.automata.get_stages_to_skip(
                program, self._stage_names, done, running, skipped
            )

            progress_made = False
            for stage_name in to_skip:
                if stage_name not in program.stage_results:
                    skip_result = self.automata.create_skip_result(
                        stage_name, program
                    )
                    await self.state_manager.update_stage_result(
                        program, stage_name, skip_result
                    )
                    skipped.add(stage_name)
                    logger.warning(
                        f"[DAG] Program {program.id}: Stage '{stage_name}' skipped due to failed dependency."
                    )
                    progress_made = True

            # Recompute ready after applying skips, as they can unlock stages
            if to_skip:
                _compute_done_skipped()
                ready = self.automata.get_ready_stages(
                    program, self._stage_names, done, running, skipped
                )

            tasks = await self._launch_ready_stages(program, ready)
            if tasks:
                running.update(ready)
                progress_made = progress_made or bool(ready)

            if tasks:
                await self._collect_results(
                    program, tasks, done, running
                )

            if not ready and not running:
                # If nothing can progress and no progress was made this iteration, force-skip unresolved
                unresolved = [
                    stage_name
                    for stage_name in self.nodes
                    if stage_name not in program.stage_results
                ]
                if unresolved and not progress_made:
                    for stage_name in unresolved:
                        logger.warning(
                            f"[DAG] Program {program.id}: Forcing skip of unresolved stage '{stage_name}'"
                        )
                        skip_result = self.automata.create_skip_result(
                            stage_name, program
                        )
                        await self.state_manager.update_stage_result(
                            program, stage_name, skip_result
                        )
                logger.info(
                    f"[DAG] Program {program.id}: All stages finalized — terminating DAG."
                )
                break

        all_succeeded = all(
            result.is_completed() for result in program.stage_results.values()
        ) and len(program.stage_results) == len(self.nodes)

        summary = self._generate_completion_summary(program, all_succeeded)
        logger.info(f"[DAG] {summary}")

        if not all_succeeded:
            analysis = self._analyze_completion_failures(program)
            logger.info(f"[DAG] Failure Analysis: {analysis}")

        # Do not change Program.state here; caller (scheduler) is responsible

    async def _launch_ready_stages(
        self, program: Program, ready: Set[str]
    ) -> Dict[str, asyncio.Task]:
        tasks = {}
        now_ts = datetime.now(timezone.utc)
        for name in ready:
            await self.state_manager.mark_stage_running(
                program, name, started_at=now_ts
            )
            logger.info(f"[DAG] Program {program.id}: Stage '{name}' started.")

            async def _run_stage(stage_name=name, prog=program):
                async with self._stage_sema:
                    return await self.nodes[stage_name].run(prog)

            tasks[name] = asyncio.create_task(
                _run_stage(), name=f"stage-{name[:8]}"
            )
        return tasks

    async def _collect_results(
        self,
        program: Program,
        tasks: Dict[str, asyncio.Task],
        done: Set[str],
        running: Set[str],
    ) -> Dict[str, ProgramStageResult]:
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        outcomes = {}
        for name, result in zip(tasks.keys(), results):
            running.discard(name)
            done.add(name)
            finished_at = datetime.now(timezone.utc)

            if isinstance(result, Exception):
                stage_result = ProgramStageResult(
                    status=StageState.FAILED,
                    error=format_error_for_llm(
                        error=result,
                        stage_name=name,
                        context="Exception escaped from stage execution",
                    ),
                    started_at=program.stage_results.get(
                        name, ProgramStageResult()
                    ).started_at
                    or finished_at,
                    finished_at=finished_at,
                )
                logger.error(
                    f"[DAG] Program {program.id}: Stage '{name}' failed with unexpected exception: {result}"
                )
            else:
                stage_result = result
                if stage_result.is_failed():
                    logger.error(
                        f"[DAG] Program {program.id}: Stage '{name}' failed: {stage_result.error}"
                    )
                else:
                    logger.info(
                        f"[DAG] Program {program.id}: Stage '{name}' completed successfully."
                    )

            program.stage_results[name] = stage_result
            outcomes[name] = stage_result

        try:
            await self.state_manager.storage.update(program)
        except Exception as e:
            logger.error(
                f"[DAG] Failed to batch-update program {program.id}: {e}"
            )

        return outcomes

    def _generate_completion_summary(
        self, program: Program, all_succeeded: bool
    ) -> str:
        status = "SUCCESS" if all_succeeded else "PARTIAL"
        completed = sum(
            1 for r in program.stage_results.values() if r.is_completed()
        )
        failed = sum(1 for r in program.stage_results.values() if r.is_failed())
        skipped = sum(
            1 for r in program.stage_results.values() if r.is_skipped()
        )
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
        failed_stages = [
            s for s, r in program.stage_results.items() if r.is_failed()
        ]
        skipped_stages = [
            s for s, r in program.stage_results.items() if r.is_skipped()
        ]
        if failed_stages:
            analysis.append(f"Failed stages: {failed_stages}")
            critical = [
                s
                for s in failed_stages
                if any(x in s.lower() for x in ["validate", "metric"])
            ]
            if critical:
                analysis.append(f"⚠️ Critical failures: {critical}")
        if skipped_stages:
            analysis.append(f"Skipped stages: {skipped_stages}")
        if not program.metrics:
            analysis.append("💡 No metrics extracted")
        return (
            " | ".join(analysis)
            if analysis
            else "No specific failure patterns identified"
        )

    def _validate_acyclic_graph(self) -> None:
        """Validate there are no cycles in the combined graph (regular + exec-order deps)."""
        G = self._build_combined_graph()
        if not nx.is_directed_acyclic_graph(G):
            try:
                cycle_edges = nx.find_cycle(G, orientation="original")
                cycle_nodes = [cycle_edges[0][0]] + [v for (_, v, *_) in cycle_edges]
                cycle_desc = " -> ".join(cycle_nodes)
            except Exception:
                cycle_desc = "(could not extract cycle nodes)"
            raise ValueError(
                f"Cycle detected in DAG (including exec-order deps): {cycle_desc}"
            )

    def _build_combined_graph(self) -> nx.DiGraph:
        """Build a combined NetworkX DiGraph of regular and exec-order dependencies."""
        G = nx.DiGraph()
        G.add_nodes_from(self.nodes.keys())
        for src, dsts in self.edges.items():
            for dst in dsts:
                G.add_edge(src, dst)
        for stage_name, rule in self.automata.transition_rules.items():
            for dep in rule.execution_order_dependencies:
                G.add_edge(dep.stage_name, stage_name)
        return G

    def _build_adjacency(self) -> Dict[str, List[str]]:
        """Return adjacency list for combined graph using defaultdict(list)."""
        adjacency = defaultdict(list)
        for name in self.nodes.keys():
            _ = adjacency[name]
        for src, dsts in self.edges.items():
            adjacency[src].extend(dsts)
        for stage_name, rule in self.automata.transition_rules.items():
            for dep in rule.execution_order_dependencies:
                adjacency[dep.stage_name].append(stage_name)
        return dict(adjacency)

    def get_topological_order(self) -> List[str]:
        """Return a topological order of the combined graph for debugging/metrics."""
        G = self._build_combined_graph()
        return list(nx.topological_sort(G))

