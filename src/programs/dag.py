"""DAG runner: execution-order vs dataflow (mandatory/optional inputs)

Concepts
- Execution-order dependencies: control when a stage is allowed to start.
  They do not carry data. Use for sequencing (e.g., always_after, on_success).
- Regular edges: carry data only. They never gate readiness or auto-skip.
  The runner collects predecessor outputs strictly in incoming-edge order.

Interplay
- Readiness is decided using execution-order deps only.
- At launch, the runner waits until all predecessors reach a final state and
  collects all successful predecessor outputs in edge order.
- Each stage declares mandatory_inputs() and optional_inputs() as lists of input names.
  - If any mandatory inputs are missing, the stage is skipped.
  - Otherwise, the stage receives all mandatory inputs and any available optional inputs.

Notes
- This replaces earlier join-policy/bounds; the model is simpler and explicit.
- Edges remain data-only; sequencing is expressed via execution-order deps.
"""

import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set
from collections import defaultdict
import networkx as nx

from loguru import logger

from src.programs.automata import DAGAutomata, ExecutionOrderDependency, DataFlowEdge
from src.programs.program import (
    Program,
    ProgramStageResult,
    StageState,
)
from src.programs.stages.base import Stage
from src.programs.state_manager import ProgramStateManager
from src.programs.utils import format_error_for_llm, build_stage_result

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
        data_flow_edges: List[DataFlowEdge],
        state_manager: ProgramStateManager,
        *,
        execution_order_deps: Optional[
            Dict[str, List[ExecutionOrderDependency]]
        ] = None,
        max_parallel_stages: int = 8,
        dag_timeout: Optional[float] = 2400.0,
    ):
        self.nodes = nodes
        self.data_flow_edges = data_flow_edges
        self.state_manager = state_manager
        self.dag_timeout = dag_timeout
        self._stage_names: Set[str] = set(self.nodes.keys())
        self.automata = DAGAutomata()
        
        # Build data flow mapping for semantic input names
        self._data_flow_map = {}
        for edge in data_flow_edges:
            # Map: (source_stage, destination_stage) -> input_name
            self._data_flow_map[(edge.source_stage, edge.destination_stage)] = edge.input_name
        # Validate configuration early (nodes/edges/entry points/exec deps/types)
        self._validate_graph_integrity(execution_order_deps or {})
        # Build transition rules
        self._setup_automata(execution_order_deps or {})
        self._validate_acyclic_graph()
        self._validate_dependencies()
        self._validate_input_topology_consistency()
        self._stage_sema = asyncio.Semaphore(max(1, max_parallel_stages))

    def _validate_graph_integrity(
        self, execution_order_deps: Dict[str, List[ExecutionOrderDependency]]
    ) -> None:
        # Validate node types
        bad_nodes = [
            name
            for name, obj in self.nodes.items()
            if not isinstance(obj, Stage)
        ]
        if bad_nodes:
            raise ValueError(
                f"Non-Stage objects registered as nodes: {', '.join(sorted(bad_nodes))}"
            )

        stage_names = self._stage_names

        # Validate data flow edges sources and destinations
        unknown_sources = [
            edge.source_stage for edge in self.data_flow_edges if edge.source_stage not in stage_names
        ]
        if unknown_sources:
            raise ValueError(
                f"Data flow edges contain unknown source stage(s): {', '.join(sorted(set(unknown_sources)))}"
            )
        unknown_dests = [
            edge.destination_stage for edge in self.data_flow_edges if edge.destination_stage not in stage_names
        ]
        if unknown_dests:
            raise ValueError(
                f"Data flow edges reference unknown destination stage(s): {', '.join(sorted(set(unknown_dests)))}"
            )

        # Validate execution order dependencies
        if execution_order_deps:
            unknown_exec_keys = [
                k for k in execution_order_deps.keys() if k not in stage_names
            ]
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
        # Regular edges do not gate readiness; only execution-order deps do.
        for stage_name, deps in execution_order_deps.items():
            for dep in deps:
                self.automata.add_execution_order_dependency(stage_name, dep)

    def _validate_dependencies(self) -> None:
        errors = self.automata.validate_dependencies(self._stage_names)
        # Ensure all transition rule targets exist among nodes
        invalid_rule_targets = [
            stage_name
            for stage_name in dict(self.automata.transition_rules).keys()
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
            # Additional rule: wait until all regular predecessors finished (any final state)
            if ready:
                gated = set()
                for stage_name in list(ready):
                    preds = self._get_predecessors_in_order(stage_name)
                    if preds:
                        all_final = True
                        for p in preds:
                            res = program.stage_results.get(p)
                            if res is None or res.status not in FINAL_STATES:
                                all_final = False
                                break
                        if not all_final:
                            gated.add(stage_name)
                ready -= gated
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
                await self._collect_results(program, tasks, done, running)

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

    def _get_predecessors_in_order(self, stage_name: str) -> List[str]:
        """Get predecessors based on data flow edges."""
        preds = []
        for edge in self.data_flow_edges:
            if edge.destination_stage == stage_name:
                preds.append(edge.source_stage)
        return preds

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
                    stage = self.nodes[stage_name]
                    preds = self._get_predecessors_in_order(stage_name)
                    # Select successful predecessors and enforce mandatory kinds
                    successful_pred_names: List[str] = []
                    for p in preds:
                        res = prog.stage_results.get(p)
                        if (
                            res
                            and res.status == StageState.COMPLETED
                            and res.output is not None
                        ):
                            successful_pred_names.append(p)

                    # Build named inputs from data flow edges
                    named_inputs = {}
                    for edge in self.data_flow_edges:
                        if edge.destination_stage == stage_name:
                            # Check if the source stage completed successfully
                            source_result = prog.stage_results.get(edge.source_stage)
                            if (source_result and 
                                source_result.status == StageState.COMPLETED and 
                                source_result.output is not None):
                                named_inputs[edge.input_name] = source_result.output
                    
                    stage.set_named_inputs(named_inputs)
                    # Validate mandatory inputs are present
                    mandatory_inputs = stage.__class__.mandatory_inputs()
                    missing_mandatory = [name for name in mandatory_inputs if name not in named_inputs]
                    if missing_mandatory:
                        return build_stage_result(
                            status=StageState.SKIPPED,
                            started_at=now_ts,
                            error=f"Missing mandatory inputs: {missing_mandatory}",
                            stage_name=stage_name,
                        )
                    return await stage.run(prog)

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
                cycle_nodes = [cycle_edges[0][0]] + [
                    v for (_, v, *_) in cycle_edges
                ]
                cycle_desc = " -> ".join(cycle_nodes)
            except Exception:
                cycle_desc = "(could not extract cycle nodes)"
            raise ValueError(
                f"Cycle detected in DAG (including exec-order deps): {cycle_desc}"
            )

    def _build_combined_graph(self) -> nx.DiGraph:
        """Build a combined NetworkX DiGraph of data flow and exec-order dependencies."""
        G = nx.DiGraph()
        G.add_nodes_from(self.nodes.keys())
        
        # Add data flow edges
        for edge in self.data_flow_edges:
            G.add_edge(edge.source_stage, edge.destination_stage)
        
        # Add execution order dependencies
        rules = self.automata.transition_rules
        for stage_name in list(rules):
            rule = rules[stage_name] if stage_name in rules else None
            if not rule:
                continue
            for dep in rule.execution_order_dependencies:
                G.add_edge(dep.stage_name, stage_name)
        
        return G

    def _build_adjacency(self) -> Dict[str, List[str]]:
        """Return adjacency list for combined graph using defaultdict(list)."""
        adjacency = defaultdict(list)
        for name in list(self.nodes.keys()):
            _ = adjacency[name]
        
        # Add data flow edges
        for edge in self.data_flow_edges:
            adjacency[edge.source_stage].append(edge.destination_stage)
        
        # Add execution order dependencies
        rules = self.automata.transition_rules
        for stage_name in list(rules):
            rule = rules[stage_name] if stage_name in rules else None
            if not rule:
                continue
            for dep in rule.execution_order_dependencies:
                adjacency[dep.stage_name].append(stage_name)
        return dict(adjacency)


    def _validate_input_topology_consistency(self) -> None:
        """Ensure stage input declarations are consistent with data flow topology.
        
        Validates that each stage's mandatory inputs are provided by data flow edges,
        and that provided inputs match declared input names.
        """
        errors: List[str] = []
        
        for stage_name, stage in self.nodes.items():
            # Find incoming data flow edges
            incoming_edges = [
                edge for edge in self.data_flow_edges 
                if edge.destination_stage == stage_name
            ]
            incoming_count = len(incoming_edges)
            
            mandatory_inputs = stage.__class__.mandatory_inputs()
            optional_inputs = stage.__class__.optional_inputs()
            all_declared_inputs = set(mandatory_inputs + optional_inputs)
            
            # Error: Stage declares no inputs but has incoming data flow edges
            if not all_declared_inputs and incoming_count > 0:
                errors.append(
                    f"Stage '{stage_name}' declares no inputs but has {incoming_count} incoming data flow edges. "
                    f"Either update the stage to accept inputs or remove the unnecessary edges."
                )
            
            # Build what input names will be provided
            provided_input_names = set()
            for edge in incoming_edges:
                provided_input_names.add(edge.input_name)
            
            # Check for missing mandatory inputs
            missing_mandatory = set(mandatory_inputs) - provided_input_names
            if missing_mandatory:
                errors.append(
                    f"Stage '{stage_name}' requires mandatory inputs {missing_mandatory} but they are not provided by data flow edges. "
                    f"Provided inputs: {provided_input_names}"
                )
            
            # Check for unexpected inputs (not declared by stage)
            unexpected_inputs = provided_input_names - all_declared_inputs
            if unexpected_inputs:
                errors.append(
                    f"Stage '{stage_name}' will receive unexpected inputs {unexpected_inputs} not declared in mandatory_inputs() or optional_inputs(). "
                    f"Declared inputs: {all_declared_inputs}, Provided inputs: {provided_input_names}"
                )
        
        # Fail fast on errors (actual constraint violations)
        if errors:
            raise ValueError(f"Input/topology validation failed: {'; '.join(errors)}")
        

    def get_topological_order(self) -> List[str]:
        """Return a topological order of the combined graph for debugging/metrics."""
        G = self._build_combined_graph()
        return list(nx.topological_sort(G))
