from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Literal, Optional, Set, Tuple, cast

import networkx as nx
from loguru import logger
from pydantic import BaseModel, Field, ConfigDict

from src.programs.program import Program, ProgramStageResult, StageState
from src.programs.stages.base import Stage
from src.programs.utils import format_error_for_llm

__all__ = [
    "DataFlowEdge",
    "ExecutionOrderDependency",
    "StageTransitionRule",
    "DAGAutomata",
    "FINAL_STATES",
]

# Final states that indicate a stage has completed execution (in any terminal sense)
FINAL_STATES: Set[StageState] = {
    StageState.COMPLETED,
    StageState.FAILED,
    StageState.CANCELLED,
    StageState.SKIPPED,
}

# --------------------------- Models ---------------------------


class DataFlowEdge(BaseModel):
    """Represents a data flow connection between stages with semantic input naming."""
    source_stage: str = Field(..., description="Name of the source stage that produces data")
    destination_stage: str = Field(..., description="Name of the destination stage that consumes data")
    input_name: str = Field(..., description="Semantic name for this input in the destination stage")

    @classmethod
    def create(cls, source: str, destination: str, input_name: str) -> "DataFlowEdge":
        return cls(source_stage=source, destination_stage=destination, input_name=input_name)


class ExecutionOrderDependency(BaseModel):
    stage_name: str = Field(..., description="Name of the stage this dependency refers to")
    condition: Literal["success", "failure", "always"] = Field(
        ..., description="When this dependency is considered satisfied"
    )

    def _satisfied_by_status(self, status: StageState) -> bool:
        if self.condition == "always":
            return status in FINAL_STATES
        if self.condition == "success":
            return status == StageState.COMPLETED
        if self.condition == "failure":
            return status in (StageState.FAILED, StageState.CANCELLED, StageState.SKIPPED)
        return False

    def is_satisfied_historically(self, result: Optional[ProgramStageResult]) -> bool:
        if result is None or result.status in (StageState.PENDING, StageState.RUNNING):
            return False
        return self._satisfied_by_status(result.status)

    @classmethod
    def on_success(cls, stage_name: str) -> "ExecutionOrderDependency":
        return cls(stage_name=stage_name, condition="success")

    @classmethod
    def on_failure(cls, stage_name: str) -> "ExecutionOrderDependency":
        return cls(stage_name=stage_name, condition="failure")

    @classmethod
    def always_after(cls, stage_name: str) -> "ExecutionOrderDependency":
        return cls(stage_name=stage_name, condition="always")


class StageTransitionRule(BaseModel):
    stage_name: str = Field(...)
    execution_order_dependencies: List[ExecutionOrderDependency] = Field(default_factory=list)
    model_config = ConfigDict(arbitrary_types_allowed=True)


@dataclass(frozen=True)
class _Topology:
    nodes: Dict[str, Stage]
    edges: List[DataFlowEdge]
    incoming_by_dest: Dict[str, List[DataFlowEdge]]
    preds_by_dest: Dict[str, List[str]]
    exec_rules: Dict[str, StageTransitionRule]

    def is_cacheable(self, stage_name: str) -> bool:
        return self.nodes[stage_name].cacheable

    def declared_inputs(self, stage_name: str) -> Tuple[Set[str], Set[str]]:
        st = self.nodes[stage_name].__class__
        return set(st.mandatory_inputs()), set(st.optional_inputs())


# --------------------------- Automata ---------------------------


class DAGAutomata(BaseModel):
    transition_rules: Dict[str, StageTransitionRule] = Field(default_factory=dict)
    topology: Optional[_Topology] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # --------------- factory & validation ---------------

    @classmethod
    def build(
        cls,
        nodes: Dict[str, Stage],
        data_flow_edges: List[DataFlowEdge],
        execution_order_deps: Optional[Dict[str, List[ExecutionOrderDependency]]] = None,
    ) -> "DAGAutomata":
        rules: Dict[str, StageTransitionRule] = {}

        # Validate node types
        bad_nodes = [k for k, v in nodes.items() if not isinstance(v, Stage)]
        if bad_nodes:
            raise ValueError(f"Non-Stage objects registered as nodes: {', '.join(sorted(bad_nodes))}")

        # Build incoming maps
        incoming_by_dest: Dict[str, List[DataFlowEdge]] = {}
        for e in data_flow_edges:
            if e.source_stage not in nodes:
                raise ValueError(f"Data flow edge references unknown source '{e.source_stage}'")
            if e.destination_stage not in nodes:
                raise ValueError(f"Data flow edge references unknown destination '{e.destination_stage}'")
            incoming_by_dest.setdefault(e.destination_stage, []).append(e)

        preds_by_dest: Dict[str, List[str]] = {
            dst: [e.source_stage for e in edges] for dst, edges in incoming_by_dest.items()
        }

        # Execution order rules
        execution_order_deps = execution_order_deps or {}
        for stage_name, deps in execution_order_deps.items():
            if stage_name not in nodes:
                raise ValueError(f"Execution-order deps contain unknown target stage '{stage_name}'")
            for dep in deps:
                if dep.stage_name not in nodes:
                    raise ValueError(
                        f"Execution-order dependency for '{stage_name}' references unknown stage '{dep.stage_name}'"
                    )
            rules[stage_name] = StageTransitionRule(stage_name=stage_name, execution_order_dependencies=list(deps))

        # Input/topology consistency
        errors: List[str] = []
        for stage_name, stage in nodes.items():
            incoming_edges = incoming_by_dest.get(stage_name, [])
            mandatory_inputs = set(stage.__class__.mandatory_inputs())
            optional_inputs = set(stage.__class__.optional_inputs())
            all_declared = mandatory_inputs | optional_inputs

            seen_names: Set[str] = set()
            provided_names: Set[str] = set()
            for e in incoming_edges:
                if e.input_name in seen_names:
                    errors.append(
                        f"Stage '{stage_name}' has duplicate input_name '{e.input_name}' from multiple edges."
                    )
                seen_names.add(e.input_name)
                provided_names.add(e.input_name)

            missing_mandatory = mandatory_inputs - provided_names
            if missing_mandatory:
                errors.append(
                    f"Stage '{stage_name}' requires mandatory inputs {sorted(missing_mandatory)} "
                    f"but they are not provided by data flow edges. Provided={sorted(provided_names)}"
                )

            unexpected = provided_names - all_declared
            if unexpected:
                errors.append(
                    f"Stage '{stage_name}' will receive unexpected inputs {sorted(unexpected)} not declared "
                    f"in mandatory_inputs() or optional_inputs(). Declared={sorted(all_declared)}"
                )

        if errors:
            raise ValueError("Input/topology validation failed: " + "; ".join(errors))

        # Acyclic including exec-deps
        G = nx.DiGraph()
        G.add_nodes_from(nodes.keys())
        for e in data_flow_edges:
            G.add_edge(e.source_stage, e.destination_stage)
        for stage_name, rule in rules.items():
            for dep in rule.execution_order_dependencies:
                G.add_edge(dep.stage_name, stage_name)
        if not nx.is_directed_acyclic_graph(G):
            try:
                cycle_edges = nx.find_cycle(G, orientation="original")
                cycle_nodes = [cycle_edges[0][0]] + [v for (_, v, *_) in cycle_edges]
                cycle_desc = " -> ".join(cycle_nodes)
            except Exception:
                cycle_desc = "(could not extract cycle nodes)"
            raise ValueError(f"Cycle detected in DAG (including exec-order deps): {cycle_desc}")

        # Cacheability safety: no cacheable stage depends (data or exec) on non-cacheable
        def _assert_cache_safe(dst: str, src: str, kind: str) -> None:
            if nodes[dst].cacheable and not nodes[src].cacheable:
                raise ValueError(
                    f"Cacheability violation: cacheable '{dst}' depends on non-cacheable '{src}' via {kind}"
                )

        for dst, edges in incoming_by_dest.items():
            for e in edges:
                _assert_cache_safe(dst, e.source_stage, "data-flow")

        for stage_name, rule in rules.items():
            for dep in rule.execution_order_dependencies:
                _assert_cache_safe(stage_name, dep.stage_name, "execution-order")

        automata = cls(transition_rules=rules)
        automata.topology = _Topology(
            nodes=nodes,
            edges=data_flow_edges,
            incoming_by_dest=incoming_by_dest,
            preds_by_dest=preds_by_dest,
            exec_rules=rules,
        )
        return automata

    # --------------- core helpers ---------------

    def _pid(self, program: Program) -> str:
        try:
            return getattr(program, "id", "<?>")[:8]
        except Exception:
            return "<?>"

    def _compute_done_sets(self, program: Program, finished_this_run: Set[str]) -> Tuple[Set[str], Set[str]]:
        """Return (effective_done, effective_skipped) for logging & checks.

        - Cacheable: any FINAL historical result counts as done.
        - Non-cacheable: only stages finalized in THIS run count as done.
        """
        pid = self._pid(program)
        assert self.topology is not None

        cacheable_done: Set[str] = set()
        cacheable_skipped: Set[str] = set()

        for name, res in (program.stage_results or {}).items():
            if name not in self.topology.nodes:
                continue
            if self.topology.is_cacheable(name) and res.status in FINAL_STATES:
                cacheable_done.add(name)
                if res.status == StageState.SKIPPED:
                    cacheable_skipped.add(name)

        effective_done = cacheable_done | (finished_this_run & set(self.topology.nodes.keys()))
        effective_skipped = cacheable_skipped | {
            s for s in finished_this_run
            if (program.stage_results.get(s) and program.stage_results[s].status == StageState.SKIPPED)
        }

        logger.debug(
            "[Automata][{}] Done sets computed: done={} skipped={} finished_this_run={}",
            pid,
            sorted(list(effective_done)),
            sorted(list(effective_skipped)),
            sorted(list(finished_this_run)),
        )
        return effective_done, effective_skipped

    # --------------- readiness ---------------

    def get_ready_stages(
        self,
        program: Program,
        running: Set[str],
        launched_this_run: Set[str],
        finished_this_run: Set[str],
    ) -> Set[str]:
        """Return set of stage names ready to launch now."""
        assert self.topology is not None
        pid = self._pid(program)

        all_names = set(self.topology.nodes.keys())
        done, skipped = self._compute_done_sets(program, finished_this_run)

        ready: Set[str] = set()
        for stage_name in sorted(all_names - running - launched_this_run - skipped):
            stage = self.topology.nodes[stage_name]
            # Cacheables: NEVER re-run if FINAL result exists
            final_res = program.stage_results.get(stage_name)
            if stage.cacheable and final_res and final_res.status in FINAL_STATES:
                logger.debug(
                    "[Automata][{}] NOT READY '{}' (cache-hit: existing FINAL result {})",
                    pid, stage_name, final_res.status.name
                )
                continue

            # Exec-order gating (strict for non-cacheable deps)
            if not self._exec_order_ready(program, stage_name, finished_this_run):
                continue

            # Data-flow gating with clarified semantics
            if not self._passes_dataflow_gating(program, stage_name, finished_this_run):
                continue

            ready.add(stage_name)

        logger.debug("[Automata][{}] READY stages: {}", pid, sorted(list(ready)))
        return ready

    def _exec_order_ready(self, program: Program, stage_name: str, finished_this_run: Set[str]) -> bool:
        """Exec-order deps gate sequencing:

        Semantics for condition == "always":
          • CACHEABLE predecessor: satisfied if predecessor is in any FINAL state (historical OK).
          • NON-CACHEABLE predecessor: satisfied only if predecessor reached a FINAL state in THIS RUN.

        For "success"/"failure":
          • CACHEABLE predecessor: historical satisfaction must match the condition (success/failure).
          • NON-CACHEABLE predecessor: must satisfy the condition in THIS RUN.
        """
        assert self.topology is not None
        pid = self._pid(program)
        rule = self.transition_rules.get(stage_name)
        if not rule or not rule.execution_order_dependencies:
            return True

        for dep in rule.execution_order_dependencies:
            dep_stage = dep.stage_name
            dep_is_cacheable = self.topology.is_cacheable(dep_stage)
            res = program.stage_results.get(dep_stage)

            historical_ok = dep.is_satisfied_historically(res)

            if dep.condition == "always":
                if dep_is_cacheable:
                    satisfied = bool(res and res.status in FINAL_STATES)
                    detail = "cacheable-final-historical-ok" if satisfied else "cacheable-not-final-yet"
                else:
                    satisfied = bool(dep_stage in finished_this_run and res and res.status in FINAL_STATES)
                    detail = "noncacheable-final-thisrun-ok" if satisfied else "noncacheable-wait-final-thisrun"
            else:
                if dep_is_cacheable:
                    satisfied = historical_ok
                    detail = "cacheable-historical-ok" if satisfied else "cacheable-not-satisfied"
                else:
                    satisfied = bool(dep_stage in finished_this_run and res and dep._satisfied_by_status(res.status))
                    detail = "noncacheable-thisrun-ok" if satisfied else "noncacheable-thisrun-wait"

            logger.debug(
                "[Automata][{}] Exec-dep check for '{}' <- {}[{}]: dep_satisfied={} ({})",
                pid, stage_name, dep_stage, dep.condition, bool(satisfied), detail
            )
            if not satisfied:
                return False
        return True

    def _passes_dataflow_gating(self, program: Program, stage_name: str, finished_this_run: Set[str]) -> bool:
        """Data-flow gating:

        MANDATORY inputs -> must succeed:
          - CACHEABLE predecessor: historical COMPLETED ok, or COMPLETED finished_this_run.
          - NON-CACHEABLE predecessor: must be COMPLETED in THIS run.

        OPTIONAL inputs ->
          - If provider exists:
             * CACHEABLE predecessor: wait until producer is FINAL (historical FINAL ok, or FINAL finished_this_run).
             * NON-CACHEABLE predecessor: require FINAL in THIS run.
          - If no provider exists: ignore for gating.
        """
        assert self.topology is not None
        pid = self._pid(program)

        stage = self.topology.nodes[stage_name]
        incoming_edges = self.topology.incoming_by_dest.get(stage_name, [])
        mandatory_inputs, optional_inputs = self.topology.nodes[stage_name].__class__.mandatory_inputs(), self.topology.nodes[stage_name].__class__.optional_inputs()
        mandatory_inputs = set(mandatory_inputs)
        optional_inputs = set(optional_inputs)

        # Group incoming edges by input name
        edges_by_input: Dict[str, List[DataFlowEdge]] = {}
        for e in incoming_edges:
            edges_by_input.setdefault(e.input_name, []).append(e)

        # ---- MANDATORY ----
        for inp in sorted(mandatory_inputs):
            edges = edges_by_input.get(inp, [])
            if not edges:
                logger.error(
                    "[Automata][{}] MANDATORY input '{}' for '{}' has NO provider edges",
                    pid, inp, stage_name
                )
                return False

            for e in edges:
                pred = e.source_stage
                pred_stage = self.topology.nodes[pred]
                res = program.stage_results.get(pred)

                if pred_stage.cacheable:
                    ok = bool(
                        (res is not None and res.status == StageState.COMPLETED)
                        or (
                            pred in finished_this_run
                            and res is not None
                            and res.status == StageState.COMPLETED
                        )
                    )
                    logger.debug(
                        "[Automata][{}] Gate '{}'.MANDATORY '{}' <- {}: ok={} (cacheable, status={} finished_this_run={})",
                        pid, stage_name, inp, pred, bool(ok), (res.status.name if res else "NONE"), (pred in finished_this_run)
                    )
                else:
                    ok = bool(
                        pred in finished_this_run
                        and res is not None
                        and res.status == StageState.COMPLETED
                    )
                    logger.debug(
                        "[Automata][{}] Gate '{}'.MANDATORY '{}' <- {}: ok={} (non-cacheable, status={} finished_this_run={})",
                        pid, stage_name, inp, pred, bool(ok), (res.status.name if res else "NONE"), (pred in finished_this_run)
                    )

                if not ok:
                    return False

        # ---- OPTIONAL ----
        for inp in sorted(optional_inputs):
            edges = edges_by_input.get(inp, [])
            if not edges:
                logger.debug(
                    "[Automata][{}] '{}' OPTIONAL '{}' has NO provider edge -> ignored for gating",
                    pid, stage_name, inp
                )
                continue

            for e in edges:
                pred = e.source_stage
                pred_stage = self.topology.nodes[pred]
                res = program.stage_results.get(pred)

                if pred_stage.cacheable:
                    ok = bool(
                        (res is not None and res.status in FINAL_STATES)
                        or (pred in finished_this_run)
                    )
                    logger.debug(
                        "[Automata][{}] Gate '{}'.OPTIONAL '{}' <- {}: ok={} (cacheable, status={} finished_this_run={})",
                        pid, stage_name, inp, pred, bool(ok), (res.status.name if res else "NONE"), (pred in finished_this_run)
                    )
                else:
                    # For non-cacheable optional inputs, require THIS-RUN FINAL
                    ok = bool(
                        pred in finished_this_run
                        and res is not None
                        and res.status in FINAL_STATES
                    )
                    logger.debug(
                        "[Automata][{}] Gate '{}'.OPTIONAL '{}' <- {}: ok={} (non-cacheable, status={} finished_this_run={})",
                        pid, stage_name, inp, pred, bool(ok), (res.status.name if res else "NONE"), (pred in finished_this_run)
                    )

                if not ok:
                    return False

        return True

    # --------------- diagnostics ---------------

    def explain_blockers(
        self,
        program: Program,
        running: Set[str],
        launched_this_run: Set[str],
        finished_this_run: Set[str],
    ) -> List[str]:
        """Return human-readable reasons why progress cannot be made."""
        assert self.topology is not None
        pid = self._pid(program)
        all_names = set(self.topology.nodes.keys())
        done, skipped = self._compute_done_sets(program, finished_this_run)

        def _exec_dep_reason(stage_name: str) -> List[str]:
            reasons: List[str] = []
            rule = self.transition_rules.get(stage_name)
            if not rule:
                return reasons
            for dep in rule.execution_order_dependencies:
                dep_stage = dep.stage_name
                res = program.stage_results.get(dep_stage)
                dep_cacheable = self.topology.is_cacheable(dep_stage)
                finished_now = dep_stage in finished_this_run
                if dep.condition == "always":
                    if dep_cacheable:
                        ok = bool(res and res.status in FINAL_STATES)
                        if not ok:
                            reasons.append(
                                f"exec: wait FINAL of {dep_stage} (cacheable, historical allowed; status={res.status})"
                            )
                    else:
                        ok = bool(finished_now and res and res.status in FINAL_STATES)
                        if not ok:
                            reasons.append(
                                f"exec: wait FINAL of {dep_stage} in this run (non-cacheable; status={res.status}, finished_this_run={finished_now})"
                            )
                else:
                    if dep_cacheable:
                        ok = bool(dep.is_satisfied_historically(res))
                        if not ok:
                            reasons.append(
                                f"exec: {dep_stage}[{dep.condition}] not satisfied historically (status={res.status})"
                            )
                    else:
                        ok = bool(finished_now and res and dep._satisfied_by_status(res.status))
                        if not ok:
                            reasons.append(
                                f"exec: {dep_stage}[{dep.condition}] not satisfied in this run (status={res.status}, finished_this_run={finished_now})"
                            )
            return reasons

        def _dataflow_reason(stage_name: str) -> List[str]:
            reasons: List[str] = []
            incoming_edges = self.topology.incoming_by_dest.get(stage_name, [])
            edges_by_input: Dict[str, List[DataFlowEdge]] = {}
            for e in incoming_edges:
                edges_by_input.setdefault(e.input_name, []).append(e)

            mand = set(self.topology.nodes[stage_name].__class__.mandatory_inputs())
            opt = set(self.topology.nodes[stage_name].__class__.optional_inputs())

            for inp in sorted(mand):
                if inp not in edges_by_input:
                    reasons.append(f"data: mandatory '{inp}' has NO provider")
                    continue
                for e in edges_by_input[inp]:
                    pred = e.source_stage
                    res = program.stage_results.get(pred)
                    cacheable = self.topology.is_cacheable(pred)
                    if cacheable:
                        ok = bool(res and res.status == StageState.COMPLETED)
                        if not ok:
                            reasons.append(
                                f"data: '{inp}' <- {pred} needs COMPLETED (cacheable; status={res.status})"
                            )
                    else:
                        ok = bool(pred in finished_this_run and res and res.status == StageState.COMPLETED)
                        if not ok:
                            reasons.append(
                                f"data: '{inp}' <- {pred} needs COMPLETED in this run (non-cacheable; status={res.status}, finished_this_run={pred in finished_this_run})"
                            )

            for inp in sorted(opt):
                if inp not in edges_by_input:
                    continue
                for e in edges_by_input[inp]:
                    pred = e.source_stage
                    res = program.stage_results.get(pred)
                    cacheable = self.topology.is_cacheable(pred)
                    if cacheable:
                        ok = bool(res and res.status in FINAL_STATES)
                        if not ok:
                            reasons.append(
                                f"data: optional '{inp}' <- {pred} waiting FINAL (cacheable; status={res.status})"
                            )
                    else:
                        ok = bool(pred in finished_this_run and res and res.status in FINAL_STATES)
                        if not ok:
                            reasons.append(
                                f"data: optional '{inp}' <- {pred} waiting FINAL this run (non-cacheable; status={res.status}, finished_this_run={pred in finished_this_run})"
                            )
            return reasons

        blockers: List[str] = []
        for s in sorted(all_names - done - skipped - running - launched_this_run):
            exec_reasons = _exec_dep_reason(s)
            df_reasons = _dataflow_reason(s)
            if not exec_reasons and not df_reasons:
                continue
            joined = "; ".join(exec_reasons + df_reasons) if (exec_reasons or df_reasons) else "pending"
            blockers.append(f"[Blocker] '{s}': {joined}")

        if not blockers:
            blockers.append("[Blocker] No blockers detected; check worker pool, result persistence, or scheduler state.")
        logger.debug("[Automata][{}] Blocker diagnostics computed: {}", pid, blockers)
        return blockers

    def summarize_blockers_for_log(
        self,
        program: Program,
        running: Set[str],
        launched_this_run: Set[str],
        finished_this_run: Set[str],
    ) -> str:
        lines = self.explain_blockers(program, running, launched_this_run, finished_this_run)
        return "\n".join(lines)

    # --------------- auto-skip ---------------

    def get_stages_to_skip(
        self,
        program: Program,
        running: Set[str],
        launched_this_run: Set[str],
        finished_this_run: Set[str],
    ) -> Set[str]:
        """Stages to auto-skip when exec-order deps contradict (e.g., required SUCCESS but got FAILED)."""
        assert self.topology is not None
        pid = self._pid(program)

        all_names = set(self.topology.nodes.keys())
        _, skipped = self._compute_done_sets(program, finished_this_run)

        to_consider = all_names - running - launched_this_run - skipped
        to_skip: Set[str] = set()

        for stage_name in sorted(to_consider):
            rule = self.transition_rules.get(stage_name)
            if not rule or not rule.execution_order_dependencies:
                continue

            # If any exec-dep is definitively unmet (and won't change this run), skip.
            should_skip = False
            reasons: List[str] = []

            for dep in rule.execution_order_dependencies:
                res = program.stage_results.get(dep.stage_name)
                if res is None or res.status in (StageState.PENDING, StageState.RUNNING):
                    continue
                satisfied = dep.is_satisfied_historically(res)
                if not satisfied and dep.condition != "always":
                    should_skip = True
                    reasons.append(f"{dep.stage_name}[{dep.condition}] -> {res.status.name}")

            if should_skip:
                to_skip.add(stage_name)
                logger.debug(
                    "[Automata][{}] Auto-skip '{}' due to exec deps: {}",
                    pid, stage_name, reasons
                )

        logger.debug("[Automata][{}] Auto-skip stages: {}", pid, sorted(list(to_skip)))
        return to_skip

    def create_skip_result(self, stage_name: str, program: Program) -> ProgramStageResult:
        pid = self._pid(program)
        logger.debug("[Automata][{}] Creating SKIP result for '{}'", pid, stage_name)
        return ProgramStageResult(
            status=StageState.SKIPPED,
            error=format_error_for_llm(
                error="Stage skipped due to dependency issue",
                stage_name=stage_name,
                context="Exec-order dependency not satisfied",
            ),
            started_at=datetime.now(timezone.utc),
            finished_at=datetime.now(timezone.utc),
        )

    # --------------- runtime input wiring ---------------

    def build_named_inputs(self, program: Program, stage_name: str) -> Dict[str, Any]:
        """Build named inputs from COMPLETED producers only; reason-code why others aren't wired."""
        assert self.topology is not None
        pid = self._pid(program)
        named: Dict[str, Any] = {}

        for edge in self.topology.incoming_by_dest.get(stage_name, []):
            res = program.stage_results.get(edge.source_stage)
            if res and res.status == StageState.COMPLETED and res.output is not None:
                if edge.input_name in named:
                    logger.error(
                        "[Automata][{}] Duplicate input '{}' into '{}' from '{}'",
                        pid, edge.input_name, stage_name, edge.source_stage
                    )
                    continue
                named[edge.input_name] = res.output
                logger.debug(
                    "[Automata][{}] Input wired for '{}': {} <- {} (COMPLETED)",
                    pid, stage_name, edge.input_name, edge.source_stage
                )
            else:
                if res is None:
                    reason = "no result yet"
                elif res.status != StageState.COMPLETED:
                    reason = f"status={res.status.name}"
                else:
                    reason = "COMPLETED but output is None"
                logger.debug(
                    "[Automata][{}] Input NOT available for '{}': {} <- {} ({})",
                    pid, stage_name, edge.input_name, edge.source_stage, reason
                )

        return named

