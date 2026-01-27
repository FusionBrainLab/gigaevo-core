from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Type

import networkx as nx
from pydantic import BaseModel, ConfigDict, Field

from gigaevo.programs.core_types import FINAL_STATES, StageError, StageIO
from gigaevo.programs.dag.compatibility import (
    _covariant_type_compatible,
    _normalize_annotation,
    _type_origin_args,
)
from gigaevo.programs.program import Program, ProgramStageResult, StageState
from gigaevo.programs.stages.base import Stage


class DataFlowEdge(BaseModel):
    """Represents a data flow connection between stages with semantic input naming."""

    source_stage: str = Field(
        ..., description="Name of the source stage that produces data"
    )
    destination_stage: str = Field(
        ..., description="Name of the destination stage that consumes data"
    )
    input_name: str = Field(
        ..., description="Semantic name for this input in the destination stage"
    )

    @classmethod
    def create(cls, source: str, destination: str, input_name: str) -> "DataFlowEdge":
        return cls(
            source_stage=source, destination_stage=destination, input_name=input_name
        )


class ExecutionOrderDependency(BaseModel):
    stage_name: str = Field(
        ..., description="Name of the stage this dependency refers to"
    )
    condition: Literal["success", "failure", "always"] = Field(
        ..., description="When this dependency is considered satisfied"
    )

    def _satisfied_by_status(self, status: StageState) -> bool:
        if self.condition == "always":
            return status in FINAL_STATES
        if self.condition == "success":
            return status == StageState.COMPLETED
        if self.condition == "failure":
            return status in (
                StageState.FAILED,
                StageState.CANCELLED,
                StageState.SKIPPED,
            )
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
    execution_order_dependencies: List[ExecutionOrderDependency] = Field(
        default_factory=list
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)


@dataclass(frozen=True)
class DAGTopology:
    """Encapsulates the static structure of the DAG."""

    nodes: Dict[str, Stage]
    edges: List[DataFlowEdge]
    incoming_by_dest: Dict[str, List[DataFlowEdge]]
    preds_by_dest: Dict[str, List[str]]
    exec_rules: Dict[str, StageTransitionRule]

    def declared_inputs(self, stage_name: str) -> Tuple[Set[str], Set[str]]:
        st = self.nodes[stage_name].__class__
        return set(st._required_names), set(st._optional_names)

    def get_incoming_edges(self, stage_name: str) -> List[DataFlowEdge]:
        return self.incoming_by_dest.get(stage_name, [])

    def get_stage_class(self, stage_name: str) -> Type[Stage]:
        return self.nodes[stage_name].__class__


class DAGValidator:
    """Static validation logic for DAG structure."""

    @staticmethod
    def validate_structure(
        stage_classes: dict[str, Type[Stage]],
        data_flow_edges: list[DataFlowEdge],
        execution_order_deps: dict[str, list[ExecutionOrderDependency]] | None = None,
    ) -> list[str]:
        """
        Validate DAG structure using stage classes (no instances required).
        Returns a list of validation error messages. Empty list means valid.
        """
        errors: list[str] = []

        # Validate that all values are Stage classes
        bad_nodes = [
            k
            for k, v in stage_classes.items()
            if not (isinstance(v, type) and issubclass(v, Stage))
        ]
        if bad_nodes:
            errors.append(
                f"Non-Stage classes registered as nodes: {', '.join(sorted(bad_nodes))}"
            )
            return errors

        # Validate edge references
        incoming_by_dest: dict[str, list[DataFlowEdge]] = {}
        for e in data_flow_edges:
            if e.source_stage not in stage_classes:
                errors.append(
                    f"Data flow edge references unknown source '{e.source_stage}'"
                )
            if e.destination_stage not in stage_classes:
                errors.append(
                    f"Data flow edge references unknown destination '{e.destination_stage}'"
                )
            incoming_by_dest.setdefault(e.destination_stage, []).append(e)

        # Validate execution order dependencies
        execution_order_deps = execution_order_deps or {}
        for stage_name, deps in execution_order_deps.items():
            if stage_name not in stage_classes:
                errors.append(
                    f"Execution-order deps contain unknown target stage '{stage_name}'"
                )
            for dep in deps:
                if dep.stage_name not in stage_classes:
                    errors.append(
                        f"Execution-order dependency for '{stage_name}' references unknown stage '{dep.stage_name}'"
                    )

        # Return early if basic structure is invalid
        if errors:
            return errors

        # Validate input/output type compatibility
        errors.extend(DAGValidator._validate_types(stage_classes, incoming_by_dest))

        # Validate cycles
        errors.extend(
            DAGValidator._validate_cycles(
                stage_classes, data_flow_edges, execution_order_deps
            )
        )

        return errors

    @staticmethod
    def _validate_types(
        stage_classes: dict[str, Type[Stage]],
        incoming_by_dest: dict[str, list[DataFlowEdge]],
    ) -> list[str]:
        errors = []
        for stage_name, stage_cls in stage_classes.items():
            incoming_edges = incoming_by_dest.get(stage_name, [])
            seen: set[str] = set()
            dst_inputs_model: Type[StageIO] = stage_cls.InputsModel
            declared = set(dst_inputs_model.model_fields.keys())

            for e in incoming_edges:
                if e.input_name in seen:
                    errors.append(
                        f"Stage '{stage_name}' has duplicate input_name '{e.input_name}' from multiple edges."
                    )
                seen.add(e.input_name)

                # TYPE CHECK
                src_cls = stage_classes[e.source_stage]
                src_out_model = src_cls.OutputModel

                if e.input_name not in declared:
                    errors.append(
                        f"Stage '{stage_name}' will receive unexpected input '{e.input_name}'. Declared={sorted(declared)}"
                    )
                    continue

                ann = dst_inputs_model.model_fields[e.input_name].annotation
                accepts = _normalize_annotation(ann)
                if accepts is None:
                    # Any → allow
                    pass
                elif not accepts:
                    errors.append(
                        f"Input type for {e.destination_stage}.{e.input_name} must be a valid type "
                        f"(BaseModel/typing, Optional/Union allowed). Got {ann!r}"
                    )
                else:
                    # generic-covariant satisfiability
                    if not any(
                        _covariant_type_compatible(src_out_model, alt)
                        for alt in accepts
                    ):
                        errors.append(
                            f"Type mismatch: {e.source_stage} produces {DAGValidator._fmt_type(src_out_model)}, "
                            f"but {e.destination_stage}.{e.input_name} expects {DAGValidator._fmt_type(ann)}"
                        )

            # Check for missing mandatory inputs
            required = set(stage_cls._required_names)
            provided = seen
            missing = required - provided
            if missing:
                errors.append(
                    f"Stage '{stage_name}' missing required inputs: {sorted(missing)}"
                )
        return errors

    @staticmethod
    def _validate_cycles(
        stage_classes: dict[str, Type[Stage]],
        data_flow_edges: list[DataFlowEdge],
        execution_order_deps: dict[str, list[ExecutionOrderDependency]],
    ) -> list[str]:
        errors = []
        G = nx.DiGraph()
        G.add_nodes_from(stage_classes.keys())
        for e in data_flow_edges:
            G.add_edge(e.source_stage, e.destination_stage)
        for stage_name, deps in execution_order_deps.items():
            for dep in deps:
                G.add_edge(dep.stage_name, stage_name)

        if not nx.is_directed_acyclic_graph(G):
            try:
                cycle_edges = nx.find_cycle(G, orientation="original")
                cycle_nodes = [cycle_edges[0][0]] + [v for (_, v, *_) in cycle_edges]
                cycle_desc = " -> ".join(cycle_nodes)
            except Exception:
                cycle_desc = "(could not extract cycle nodes)"
            errors.append(
                f"Cycle detected in DAG (including exec-order deps): {cycle_desc}"
            )
        return errors

    @staticmethod
    def _fmt_type(t: Any) -> str:
        o, a = _type_origin_args(t)
        name = getattr(o, "__name__", str(o))
        if not a:
            return name
        inner = ", ".join(DAGValidator._fmt_type(x) for x in a)
        return f"{name}[{inner}]"


class DAGAutomata(BaseModel):
    transition_rules: dict[str, StageTransitionRule] = Field(default_factory=dict)
    topology: DAGTopology | None = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    class GateState(Enum):
        READY = "READY"
        WAIT = "WAIT"
        IMPOSSIBLE = "IMPOSSIBLE"

    @dataclass(frozen=True)
    class StageStatus:
        res: Optional[ProgramStageResult]
        finalized: bool
        completed: bool
        finalized_this_run: bool
        status_name: str

    @classmethod
    def build(
        cls,
        nodes: dict[str, Stage],
        data_flow_edges: list[DataFlowEdge],
        execution_order_deps: dict[str, list[ExecutionOrderDependency]] | None = None,
    ) -> "DAGAutomata":
        # Validate that all nodes are Stage instances
        bad_nodes = [k for k, v in nodes.items() if not isinstance(v, Stage)]
        if bad_nodes:
            raise ValueError(
                f"Non-Stage objects registered as nodes: {', '.join(sorted(bad_nodes))}"
            )

        # Extract stage classes from instances for validation
        stage_classes = {name: stage.__class__ for name, stage in nodes.items()}

        # Use DAGValidator to check the DAG structure
        validation_errors = DAGValidator.validate_structure(
            stage_classes, data_flow_edges, execution_order_deps
        )
        if validation_errors:
            raise ValueError(
                "DAG structure validation failed:\n  - "
                + "\n  - ".join(validation_errors)
            )

        # Build transition rules
        rules: dict[str, StageTransitionRule] = {}
        execution_order_deps = execution_order_deps or {}
        for stage_name, deps in execution_order_deps.items():
            rules[stage_name] = StageTransitionRule(
                stage_name=stage_name, execution_order_dependencies=list(deps)
            )

        # Build topology data structures
        incoming_by_dest: dict[str, list[DataFlowEdge]] = {}
        for e in data_flow_edges:
            incoming_by_dest.setdefault(e.destination_stage, []).append(e)

        preds_by_dest: dict[str, list[str]] = {
            dst: [e.source_stage for e in edges]
            for dst, edges in incoming_by_dest.items()
        }

        # Build the automata with validated topology
        automata = cls(transition_rules=rules)
        automata.topology = DAGTopology(
            nodes=nodes,
            edges=data_flow_edges,
            incoming_by_dest=incoming_by_dest,
            preds_by_dest=preds_by_dest,
            exec_rules=rules,
        )
        return automata

    def _get_stage_status(
        self, program: Program, stage_name: str, finished_this_run: set[str]
    ) -> "StageStatus":
        assert self.topology is not None
        res = program.stage_results.get(stage_name)
        finalized = bool(res and res.status in FINAL_STATES)
        completed = bool(res and res.status == StageState.COMPLETED)
        finished_now = stage_name in finished_this_run
        return self.StageStatus(
            res=res,
            finalized=finalized,
            completed=completed,
            finalized_this_run=finished_now and finalized,
            status_name=(res.status.name if res else "NONE"),
        )

    def _edges_by_input(self, stage_name: str) -> dict[str, list[DataFlowEdge]]:
        assert self.topology is not None
        edges = self.topology.get_incoming_edges(stage_name)
        by_input: dict[str, list[DataFlowEdge]] = {}
        for e in edges:
            by_input.setdefault(e.input_name, []).append(e)
        return by_input

    def _check_dependency_gate(
        self,
        program: Program,
        dep: ExecutionOrderDependency,
        finished_this_run: set[str],
    ) -> tuple["GateState", str]:
        """Check if an execution order dependency is satisfied."""
        status = self._get_stage_status(program, dep.stage_name, finished_this_run)

        if dep.condition == "always":
            if status.finalized_this_run:
                return (self.GateState.READY, "")
            return (
                self.GateState.WAIT,
                f"exec: wait FINAL of {dep.stage_name} in this run",
            )

        expected_ok = {
            "success": status.completed,
            "failure": bool(
                status.res
                and status.res.status
                in (StageState.FAILED, StageState.CANCELLED, StageState.SKIPPED)
            ),
        }[dep.condition]

        if not status.finalized_this_run:
            return (
                self.GateState.WAIT,
                f"exec: {dep.stage_name}[{dep.condition}] pending this run (status={status.status_name})",
            )
        if expected_ok:
            return (self.GateState.READY, "")
        return (
            self.GateState.IMPOSSIBLE,
            f"exec: {dep.stage_name}[{dep.condition}] failed this run (status={status.status_name})",
        )

    def _check_dataflow_gate(
        self, program: Program, stage_name: str, finished_this_run: set[str]
    ) -> tuple["GateState", list[str]]:
        """Check if all data flow requirements are satisfied."""
        assert self.topology is not None
        reasons: list[str] = []
        edges_by_input = self._edges_by_input(stage_name)
        st_cls = self.topology.get_stage_class(stage_name)
        mandatory = set(st_cls._required_names)
        optional = set(st_cls._optional_names)

        # Mandatory inputs
        for inp in sorted(mandatory):
            edges = edges_by_input.get(inp, [])
            if not edges:
                return (
                    self.GateState.IMPOSSIBLE,
                    [f"data: mandatory '{inp}' has NO provider"],
                )
            e = edges[0]
            status = self._get_stage_status(program, e.source_stage, finished_this_run)

            # If the source stage is COMPLETED, we are good.
            if status.finalized_this_run and status.completed:
                continue

            # If the source stage is FINALIZED but NOT COMPLETED (e.g. FAILED, SKIPPED, CANCELLED)
            # then the mandatory input can NEVER arrive. Impossible.
            if status.finalized_this_run and not status.completed:
                return (
                    self.GateState.IMPOSSIBLE,
                    [
                        f"data: mandatory '{inp}' <- {e.source_stage} finalized as {status.status_name} this run (non-cacheable)"
                    ],
                )

            # Otherwise, we wait.
            reasons.append(
                f"data: '{inp}' <- {e.source_stage} needs COMPLETED this run (non-cacheable; status={status.status_name})"
            )

        # Optional inputs
        for inp in sorted(optional):
            edges = edges_by_input.get(inp, [])
            if not edges:
                continue
            for e in edges:
                status = self._get_stage_status(
                    program, e.source_stage, finished_this_run
                )
                if status.finalized_this_run:
                    continue
                reasons.append(
                    f"data: optional '{inp}' <- {e.source_stage} wait FINAL this run (non-cacheable; status={status.status_name})"
                )

        if reasons:
            return (self.GateState.WAIT, reasons)
        return (self.GateState.READY, [])

    def _diagnose_stage(
        self, program: Program, stage_name: str, finished_this_run: set[str]
    ) -> tuple["GateState", list[str]]:
        """Combine exec-order and data-flow checks."""
        rule = self.transition_rules.get(stage_name)

        # Check execution order dependencies
        exec_state = self.GateState.READY
        exec_reasons: list[str] = []

        if rule and rule.execution_order_dependencies:
            for dep in rule.execution_order_dependencies:
                state, reason = self._check_dependency_gate(
                    program, dep, finished_this_run
                )
                if state is self.GateState.IMPOSSIBLE:
                    return (self.GateState.IMPOSSIBLE, [reason])
                if state is self.GateState.WAIT:
                    exec_state = self.GateState.WAIT
                    if reason:
                        exec_reasons.append(reason)

        # Check data flow dependencies
        df_state, df_reasons = self._check_dataflow_gate(
            program, stage_name, finished_this_run
        )

        if df_state is self.GateState.IMPOSSIBLE:
            return (df_state, df_reasons)
        if exec_state is self.GateState.WAIT or df_state is self.GateState.WAIT:
            return (self.GateState.WAIT, exec_reasons + df_reasons)
        return (self.GateState.READY, [])

    def _compute_done_sets(
        self, program: Program, finished_this_run: set[str]
    ) -> tuple[set[str], set[str]]:
        assert self.topology is not None
        effective_done = finished_this_run & set(self.topology.nodes.keys())
        effective_skipped = {
            s
            for s in finished_this_run
            if (
                program.stage_results.get(s)
                and program.stage_results[s].status == StageState.SKIPPED
            )
        }
        return effective_done, effective_skipped

    def get_ready_stages(
        self,
        program: Program,
        running: set[str],
        launched_this_run: set[str],
        finished_this_run: set[str],
    ) -> tuple[set[str], set[str]]:
        """Return (ready_stages, newly_cached_stages).

        ready_stages: Stages that are ready to launch now.
        newly_cached_stages: Stages that can use cached results and should be
                            added to finished_this_run by the caller.
        """
        assert self.topology is not None
        all_names = set(self.topology.nodes.keys())
        done, skipped = self._compute_done_sets(program, finished_this_run)

        ready: set[str] = set()
        newly_cached: set[str] = set()

        for stage_name in sorted(
            all_names - running - launched_this_run - skipped - done
        ):
            # 1. Check if the stage is ready (dependencies satisfied)
            state, _ = self._diagnose_stage(program, stage_name, finished_this_run)
            if state is not self.GateState.READY:
                continue

            # 2. If ready, check if we can skip execution using cache
            st = self.topology.nodes[stage_name]
            res = program.stage_results.get(stage_name)
            cache_handler = st.get_cache_handler()

            is_cached = False
            if res and res.status in FINAL_STATES:
                # Build inputs to compute hash for cache check
                inputs_hash = None
                try:
                    named_inputs = self.build_named_inputs(program, stage_name)
                    st_cls = self.topology.get_stage_class(stage_name)
                    inputs_hash = st_cls.compute_hash_from_inputs(named_inputs)
                except Exception:
                    inputs_hash = None

                if not cache_handler.should_rerun(res, inputs_hash, finished_this_run):
                    is_cached = True

            if is_cached:
                newly_cached.add(stage_name)
            else:
                ready.add(stage_name)

        return ready, newly_cached

    def explain_blockers(
        self,
        program: Program,
        running: set[str],
        launched_this_run: set[str],
        finished_this_run: set[str],
    ) -> list[str]:
        """Return human-readable reasons why progress cannot be made."""
        assert self.topology is not None
        all_names = set(self.topology.nodes.keys())
        done, skipped = self._compute_done_sets(program, finished_this_run)

        blockers: list[str] = []
        for s in sorted(all_names - done - skipped - running - launched_this_run):
            state, reasons = self._diagnose_stage(program, s, finished_this_run)
            if state is self.GateState.READY:
                continue
            joined = "; ".join(reasons) if reasons else "pending"
            blockers.append(f"[Blocker] '{s}': {joined}")

        if not blockers:
            blockers.append(
                "[Blocker] No blockers detected; check worker pool, result persistence, or scheduler state."
            )
        return blockers

    def summarize_blockers_for_log(
        self,
        program: Program,
        running: set[str],
        launched_this_run: set[str],
        finished_this_run: set[str],
    ) -> str:
        lines = self.explain_blockers(
            program, running, launched_this_run, finished_this_run
        )
        return "\n".join(lines)

    def get_stages_to_skip(
        self,
        program: Program,
        running: set[str],
        launched_this_run: set[str],
        finished_this_run: set[str],
    ) -> set[str]:
        """Stages to auto-skip when deps are IMPOSSIBLE this run."""
        assert self.topology is not None
        all_names = set(self.topology.nodes.keys())
        _, skipped = self._compute_done_sets(program, finished_this_run)

        to_skip: set[str] = set()
        for stage_name in sorted(all_names - running - launched_this_run - skipped):
            state, _ = self._diagnose_stage(program, stage_name, finished_this_run)
            if state is self.GateState.IMPOSSIBLE:
                to_skip.add(stage_name)
        return to_skip

    def create_skip_result(
        self, stage_name: str, program: Program
    ) -> ProgramStageResult:
        return ProgramStageResult(
            status=StageState.SKIPPED,
            error=StageError(
                type="Skip",
                message="Stage skipped due to dependency issue",
                stage=stage_name,
            ),
        )

    def build_named_inputs(self, program: Program, stage_name: str) -> dict[str, Any]:
        """Build named inputs from COMPLETED producers only."""
        assert self.topology is not None
        named: dict[str, Any] = {}

        for edge in self.topology.get_incoming_edges(stage_name):
            res = program.stage_results.get(edge.source_stage)
            if res and res.status == StageState.COMPLETED and res.output is not None:
                if edge.input_name in named:
                    # Should be prevented by build(); minimal guard
                    continue
                named[edge.input_name] = res.output

        return named
