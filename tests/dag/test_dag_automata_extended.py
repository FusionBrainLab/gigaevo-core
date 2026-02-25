"""Extended unit tests for DAGAutomata — targeting specifically uncovered paths.

Covers:
  I.   ExecutionOrderDependency.is_satisfied_historically — direct unit tests
  II.  DAGValidator.validate_structure — non-Stage node detection
  III. DAGValidator.validate_structure — duplicate input_name and missing required inputs
  IV.  DAGAutomata._check_dataflow_gate — mandatory input with no provider (IMPOSSIBLE)
  V.   DAGAutomata.explain_blockers — default diagnostic message (no blockers found)

Each test section is designed to exercise a real bug scenario, not just line coverage.
The core concern for each:

  I.   is_satisfied_historically diverges from _check_dependency_gate: the former
       ignores the `finalized_this_run` constraint while the latter enforces it.
       If a caller uses is_satisfied_historically for scheduling decisions (rather
       than going through _check_dependency_gate), stale cached results from a prior
       run would be treated as satisfied — the scheduler would incorrectly mark a
       stage READY before its dep has actually been re-executed in the current run.

  II.  Non-Stage classes passed as node values are caught early and a clear error is
       produced, before any edge or type validation is attempted.

  III. Duplicate input edges and un-provided required inputs are caught by structural
       validation so that they surface as clear errors rather than silent wrong
       behavior at execution time.

  IV.  At runtime the gate logic returns IMPOSSIBLE for a required input that has
       no incoming edge — no deadlock, no hang, just an IMPOSSIBLE verdict. This path
       is unreachable via the normal build() pathway (which validates structure up
       front), but can be hit if someone constructs automata topology manually or if
       the topology is mutated post-build.

  V.   When all stages are already accounted for (done / running / launched), the
       explain_blockers method returns a diagnostic message rather than an empty list,
       so callers always get a non-empty response they can log.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import pytest

from gigaevo.programs.core_types import (
    FINAL_STATES,
    ProgramStageResult,
    StageIO,
    StageState,
)
from gigaevo.programs.dag.automata import (
    DAGAutomata,
    DAGTopology,
    DAGValidator,
    DataFlowEdge,
    ExecutionOrderDependency,
)
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState
from gigaevo.programs.stages.base import Stage
from tests.conftest import (
    ChainedStage,
    FailingStage,
    FastStage,
    MockOutput,
    VoidStage,
)

# ---------------------------------------------------------------------------
# Helpers shared across all test sections
# ---------------------------------------------------------------------------


def _make_result(
    status: StageState,
    *,
    input_hash: Optional[str] = None,
    output: Optional[StageIO] = None,
) -> ProgramStageResult:
    """Construct a ProgramStageResult with a given status."""
    now = datetime.now(timezone.utc)
    return ProgramStageResult(
        status=status,
        started_at=now,
        finished_at=now if status in FINAL_STATES else None,
        input_hash=input_hash,
        output=output,
    )


def _make_program(
    stage_results: dict[str, ProgramStageResult] | None = None,
) -> Program:
    """Create a minimal RUNNING program with optional pre-populated stage results."""
    p = Program(
        code="def solve(): return 1",
        state=ProgramState.RUNNING,
        atomic_counter=999_999_999,
    )
    if stage_results:
        p.stage_results = stage_results
    return p


def _build_automata(
    nodes: dict,
    edges: list[DataFlowEdge] | None = None,
    exec_deps: dict | None = None,
) -> DAGAutomata:
    return DAGAutomata.build(
        nodes=nodes,
        data_flow_edges=edges or [],
        execution_order_deps=exec_deps,
    )


# ---------------------------------------------------------------------------
# Additional stage mocks needed by these extended tests
# ---------------------------------------------------------------------------


class RequiredInputStage(Stage):
    """A stage with a single mandatory (non-optional) input field."""

    class _Inputs(StageIO):
        data: MockOutput

    InputsModel = _Inputs
    OutputModel = MockOutput

    async def compute(self, program: Program) -> MockOutput:
        return MockOutput(value=self.params.data.value + 100)


# ===================================================================
# Section I: ExecutionOrderDependency.is_satisfied_historically
# ===================================================================


class TestIsSatisfiedHistorically:
    """Direct unit tests for ExecutionOrderDependency.is_satisfied_historically.

    This method is a simpler, run-agnostic variant of _check_dependency_gate:
    it answers "was this dep ever satisfied, ignoring whether it was in the
    current run?" — i.e. it DOES NOT require finalized_this_run.

    The key divergence from _check_dependency_gate:
      - _check_dependency_gate returns WAIT for any result not in
        finished_this_run (i.e. stale prior-run results count as WAIT).
      - is_satisfied_historically returns True for a cached final result
        that has the correct status, regardless of which run produced it.

    If this distinction is ignored and someone uses is_satisfied_historically
    to make scheduling decisions, stale results from prior runs will be
    treated as satisfied — creating incorrect READY decisions that bypass
    the "must re-run in the current run" invariant.
    """

    # -- None result ----------------------------------------------------------

    def test_returns_false_when_result_is_none(self):
        """No result at all means the dep is not satisfied."""
        dep = ExecutionOrderDependency.on_success("a")
        assert dep.is_satisfied_historically(None) is False

    def test_returns_false_for_none_regardless_of_condition(self):
        """All three conditions return False when result is None."""
        for condition in ("success", "failure", "always"):
            dep = ExecutionOrderDependency(stage_name="x", condition=condition)
            assert dep.is_satisfied_historically(None) is False, (
                f"Expected False for condition={condition!r} with None result"
            )

    # -- PENDING / RUNNING states (non-final) ---------------------------------

    def test_returns_false_for_pending_result(self):
        """PENDING status is not a final state — dep not satisfied."""
        dep = ExecutionOrderDependency.always_after("a")
        # PENDING has no finished_at, but is_satisfied_historically checks status directly
        # We construct manually to bypass the _make_result finished_at logic
        result_pending = ProgramStageResult(
            status=StageState.PENDING,
            started_at=datetime.now(timezone.utc),
            finished_at=None,
        )
        assert dep.is_satisfied_historically(result_pending) is False

    def test_returns_false_for_running_result(self):
        """RUNNING status is not a final state — dep not satisfied."""
        dep = ExecutionOrderDependency.always_after("a")
        result_running = ProgramStageResult(
            status=StageState.RUNNING,
            started_at=datetime.now(timezone.utc),
            finished_at=None,
        )
        assert dep.is_satisfied_historically(result_running) is False

    # -- success condition ----------------------------------------------------

    def test_success_condition_true_for_completed(self):
        """on_success: True for a COMPLETED result (the happy path)."""
        dep = ExecutionOrderDependency.on_success("a")
        assert dep.is_satisfied_historically(_make_result(StageState.COMPLETED)) is True

    def test_success_condition_false_for_failed(self):
        """on_success: False for a FAILED result."""
        dep = ExecutionOrderDependency.on_success("a")
        assert dep.is_satisfied_historically(_make_result(StageState.FAILED)) is False

    def test_success_condition_false_for_skipped(self):
        """on_success: False for a SKIPPED result."""
        dep = ExecutionOrderDependency.on_success("a")
        assert dep.is_satisfied_historically(_make_result(StageState.SKIPPED)) is False

    def test_success_condition_false_for_cancelled(self):
        """on_success: False for a CANCELLED result."""
        dep = ExecutionOrderDependency.on_success("a")
        assert (
            dep.is_satisfied_historically(_make_result(StageState.CANCELLED)) is False
        )

    # -- failure condition ----------------------------------------------------

    def test_failure_condition_true_for_failed(self):
        """on_failure: True for a FAILED result."""
        dep = ExecutionOrderDependency.on_failure("a")
        assert dep.is_satisfied_historically(_make_result(StageState.FAILED)) is True

    def test_failure_condition_true_for_skipped(self):
        """on_failure: True for a SKIPPED result (SKIPPED is in the failure set)."""
        dep = ExecutionOrderDependency.on_failure("a")
        assert dep.is_satisfied_historically(_make_result(StageState.SKIPPED)) is True

    def test_failure_condition_true_for_cancelled(self):
        """on_failure: True for a CANCELLED result (CANCELLED is in the failure set)."""
        dep = ExecutionOrderDependency.on_failure("a")
        assert dep.is_satisfied_historically(_make_result(StageState.CANCELLED)) is True

    def test_failure_condition_false_for_completed(self):
        """on_failure: False for a COMPLETED result."""
        dep = ExecutionOrderDependency.on_failure("a")
        assert (
            dep.is_satisfied_historically(_make_result(StageState.COMPLETED)) is False
        )

    # -- always condition -----------------------------------------------------

    def test_always_condition_true_for_completed(self):
        """always_after: True for COMPLETED — any final state satisfies 'always'."""
        dep = ExecutionOrderDependency.always_after("a")
        assert dep.is_satisfied_historically(_make_result(StageState.COMPLETED)) is True

    def test_always_condition_true_for_failed(self):
        """always_after: True for FAILED — any final state satisfies 'always'."""
        dep = ExecutionOrderDependency.always_after("a")
        assert dep.is_satisfied_historically(_make_result(StageState.FAILED)) is True

    def test_always_condition_true_for_skipped(self):
        """always_after: True for SKIPPED — any final state satisfies 'always'."""
        dep = ExecutionOrderDependency.always_after("a")
        assert dep.is_satisfied_historically(_make_result(StageState.SKIPPED)) is True

    def test_always_condition_true_for_cancelled(self):
        """always_after: True for CANCELLED — any final state satisfies 'always'."""
        dep = ExecutionOrderDependency.always_after("a")
        assert dep.is_satisfied_historically(_make_result(StageState.CANCELLED)) is True

    # -- The core divergence: is_satisfied_historically vs _check_dependency_gate --

    def test_divergence_from_check_dependency_gate_for_stale_failure_result(self):
        """Critical: is_satisfied_historically diverges from _check_dependency_gate
        for stale results (results from prior runs NOT in finished_this_run).

        Scenario: dep stage 'a' has a FAILED result from a *prior* run.
          - is_satisfied_historically: returns True for on_failure dep
            (sees the FAILED status, ignores which run produced it)
          - _check_dependency_gate: returns WAIT (not in finished_this_run)

        This divergence is intentional — _check_dependency_gate enforces the
        "must re-execute in the current run" invariant. But if a caller were
        to use is_satisfied_historically for scheduling, they would incorrectly
        treat the stale result as satisfying the dep and schedule the dependent
        stage prematurely.
        """
        automata = _build_automata(
            {"a": FailingStage(timeout=5.0), "b": FastStage(timeout=5.0)},
            exec_deps={"b": [ExecutionOrderDependency.on_failure("a")]},
        )
        dep = ExecutionOrderDependency.on_failure("a")
        stale_failed_result = _make_result(StageState.FAILED)

        # is_satisfied_historically sees FAILED -> True (ignores run context)
        assert dep.is_satisfied_historically(stale_failed_result) is True

        # _check_dependency_gate requires it to be in finished_this_run -> WAIT
        prog = _make_program(stage_results={"a": stale_failed_result})
        gate_state, reason = automata._check_dependency_gate(
            prog,
            dep,
            finished_this_run=set(),  # "a" was NOT finished this run
        )
        assert gate_state is DAGAutomata.GateState.WAIT, (
            "Stale FAILED result must not satisfy on_failure dep in _check_dependency_gate; "
            "got {} with reason {!r}".format(gate_state, reason)
        )

    def test_divergence_from_check_dependency_gate_for_stale_completed_result(self):
        """Analogous divergence for on_success dep with stale COMPLETED result.

        is_satisfied_historically returns True (COMPLETED satisfies on_success).
        _check_dependency_gate returns WAIT (not in finished_this_run).

        This is the scenario that caused the historical deadlock bug: a stale
        COMPLETED result from a prior run would be seen as satisfying on_success
        by naive callers, allowing the scheduler to mark downstream stages READY
        before the upstream re-ran in the current run.
        """
        dep = ExecutionOrderDependency.on_success("a")
        stale_completed_result = _make_result(
            StageState.COMPLETED, output=MockOutput(value=7)
        )

        # is_satisfied_historically: True
        assert dep.is_satisfied_historically(stale_completed_result) is True

        # _check_dependency_gate with empty finished_this_run: WAIT
        automata = _build_automata(
            {"a": FastStage(timeout=5.0), "b": FastStage(timeout=5.0)},
            exec_deps={"b": [ExecutionOrderDependency.on_success("a")]},
        )
        prog = _make_program(stage_results={"a": stale_completed_result})
        gate_state, _ = automata._check_dependency_gate(
            prog, dep, finished_this_run=set()
        )
        assert gate_state is DAGAutomata.GateState.WAIT

    def test_is_satisfied_historically_agrees_with_check_dependency_gate_when_in_this_run(
        self,
    ):
        """When the result IS in finished_this_run, both methods agree on COMPLETED.

        This confirms the two methods produce consistent answers for the normal
        (non-stale) case: dep stage completed this run -> both report satisfied.
        """
        dep = ExecutionOrderDependency.on_success("a")
        fresh_result = _make_result(StageState.COMPLETED, output=MockOutput(value=5))

        # is_satisfied_historically: True
        assert dep.is_satisfied_historically(fresh_result) is True

        # _check_dependency_gate with "a" in finished_this_run: READY
        automata = _build_automata(
            {"a": FastStage(timeout=5.0), "b": FastStage(timeout=5.0)},
            exec_deps={"b": [ExecutionOrderDependency.on_success("a")]},
        )
        prog = _make_program(stage_results={"a": fresh_result})
        gate_state, _ = automata._check_dependency_gate(
            prog, dep, finished_this_run={"a"}
        )
        assert gate_state is DAGAutomata.GateState.READY


# ===================================================================
# Section II: DAGValidator.validate_structure — non-Stage node detection
# ===================================================================


class TestDAGValidatorNonStageNodes:
    """Tests for DAGValidator.validate_structure when node values are not Stage subclasses.

    The validator must catch non-Stage values early (before any edge or type
    validation) and return a clear error message naming the bad keys.
    This path is distinct from DAGAutomata.build's instance-level check —
    validate_structure takes Type[Stage] values, so passing e.g. `dict` or
    `int` (which are types but not Stage subclasses) must be flagged.
    """

    def test_dict_class_as_node_is_rejected(self):
        """Passing `dict` (a type, but not a Stage subclass) produces a validation error."""
        errors = DAGValidator.validate_structure(
            stage_classes={"bad_node": dict},  # type: ignore[dict-item]
            data_flow_edges=[],
        )
        assert errors, "Expected at least one validation error for non-Stage node"
        assert any("bad_node" in e for e in errors), (
            "Error message must name the offending node key 'bad_node'"
        )
        assert any("Non-Stage" in e for e in errors), (
            "Error message must mention 'Non-Stage'"
        )

    def test_plain_class_not_subclassing_stage_is_rejected(self):
        """A custom class that does not inherit from Stage must be rejected."""

        class NotAStage:
            pass

        errors = DAGValidator.validate_structure(
            stage_classes={"some_node": NotAStage},  # type: ignore[dict-item]
            data_flow_edges=[],
        )
        assert errors
        assert any("some_node" in e for e in errors)

    def test_non_stage_node_error_returns_early_before_edge_validation(self):
        """When a non-Stage node is present, validation returns immediately.

        The early-return guard (line 136) means that edge-reference errors and
        type errors are NOT checked — only the non-Stage error is reported.
        This prevents a confusing cascade of errors when the root problem is
        a completely wrong node type.
        """
        errors = DAGValidator.validate_structure(
            stage_classes={"good": FastStage, "bad": dict},  # type: ignore[dict-item]
            data_flow_edges=[
                # Edge that references unknown stage — would produce edge errors
                # if we got past the non-Stage check
                DataFlowEdge.create("nonexistent_src", "good", "data"),
            ],
        )
        # Should only see the non-Stage error, not edge-reference errors
        assert len(errors) == 1, (
            f"Expected exactly 1 error (non-Stage guard), got {errors}"
        )
        assert "bad" in errors[0]

    def test_multiple_non_stage_nodes_all_named_in_error(self):
        """Multiple bad nodes are all listed in the same error message."""
        errors = DAGValidator.validate_structure(
            stage_classes={
                "alpha": dict,  # type: ignore[dict-item]
                "beta": int,  # type: ignore[dict-item]
                "gamma": str,  # type: ignore[dict-item]
            },
            data_flow_edges=[],
        )
        assert errors
        # All three bad node names must appear in the combined error
        combined = " ".join(errors)
        assert "alpha" in combined
        assert "beta" in combined
        assert "gamma" in combined

    def test_valid_stage_classes_produce_no_non_stage_error(self):
        """Valid Stage subclasses do NOT trigger the non-Stage validation error."""
        errors = DAGValidator.validate_structure(
            stage_classes={"a": FastStage, "b": VoidStage},
            data_flow_edges=[],
        )
        # No non-Stage error — any errors here are about other things (e.g. missing inputs)
        assert not any("Non-Stage" in e for e in errors), (
            f"Unexpected Non-Stage error for valid stages: {errors}"
        )

    def test_dag_automata_build_rejects_non_stage_instance(self):
        """DAGAutomata.build raises ValueError when a node value is not a Stage instance.

        This is the instance-level analog of the class-level validate_structure check.
        build() checks isinstance(v, Stage) for each node value.
        """
        with pytest.raises(ValueError, match="Non-Stage objects"):
            DAGAutomata.build(
                nodes={"bad": "not_a_stage_instance"},  # type: ignore[dict-item]
                data_flow_edges=[],
            )


# ===================================================================
# Section III: DAGValidator — duplicate input_name and missing required inputs
# ===================================================================


class TestDAGValidatorInputEdgeErrors:
    """Tests for DAGValidator.validate_structure input-edge validation.

    Two distinct error conditions are tested:
      A) Duplicate input_name: two edges feed the same input field of one stage.
         This is a wiring mistake that would silently drop one data source.
      B) Missing required input: a stage has a required (non-optional) field
         but no incoming edge provides it.
         This would cause a KeyError at execution time, not at build time.
    """

    # -- A: Duplicate input_name ----------------------------------------------

    def test_duplicate_input_name_from_two_sources_is_rejected(self):
        """Two edges feeding the same input_name to one destination stage is an error."""
        errors = DAGValidator.validate_structure(
            stage_classes={
                "producer_a": FastStage,
                "producer_b": FastStage,
                "consumer": ChainedStage,
            },
            data_flow_edges=[
                # Both edges try to feed "data" to consumer — a duplicate
                DataFlowEdge.create("producer_a", "consumer", "data"),
                DataFlowEdge.create("producer_b", "consumer", "data"),
            ],
        )
        assert errors, "Expected duplicate input_name error"
        assert any("duplicate" in e.lower() for e in errors), (
            "Error message must mention 'duplicate'"
        )
        assert any("data" in e for e in errors), (
            "Error message must name the duplicated input field 'data'"
        )
        assert any("consumer" in e for e in errors), (
            "Error message must name the destination stage"
        )

    def test_non_duplicate_inputs_on_different_fields_are_allowed(self):
        """Two edges feeding *different* input fields to the same stage is valid."""
        # FanInStage has two required fields: 'data' and 'score'
        from tests.dag.test_dag_automata import FanInStage, ProducerB

        errors = DAGValidator.validate_structure(
            stage_classes={
                "prod_data": FastStage,
                "prod_score": ProducerB,
                "fan_in": FanInStage,
            },
            data_flow_edges=[
                DataFlowEdge.create("prod_data", "fan_in", "data"),
                DataFlowEdge.create("prod_score", "fan_in", "score"),
            ],
        )
        # Should not have duplicate errors — different fields
        assert not any("duplicate" in e.lower() for e in errors), (
            f"Unexpected duplicate error for different fields: {errors}"
        )

    def test_dag_automata_build_raises_for_duplicate_input_name(self):
        """DAGAutomata.build raises ValueError when duplicate input_name is detected."""
        with pytest.raises(ValueError, match="duplicate input_name"):
            DAGAutomata.build(
                nodes={
                    "src1": FastStage(timeout=5.0),
                    "src2": FastStage(timeout=5.0),
                    "dst": ChainedStage(timeout=5.0),
                },
                data_flow_edges=[
                    DataFlowEdge.create("src1", "dst", "data"),
                    DataFlowEdge.create("src2", "dst", "data"),
                ],
            )

    # -- B: Missing required input --------------------------------------------

    def test_missing_required_input_with_no_edge_is_rejected(self):
        """A stage with a required input but no incoming edge is a validation error."""
        errors = DAGValidator.validate_structure(
            stage_classes={
                "lonely": ChainedStage,  # requires 'data': MockOutput
            },
            data_flow_edges=[],  # no edges at all — 'data' has no provider
        )
        assert errors, "Expected missing-input error for ChainedStage with no edges"
        assert any("missing required" in e.lower() for e in errors), (
            "Error message must mention 'missing required'"
        )
        assert any("data" in e for e in errors), (
            "Error message must name the missing field 'data'"
        )
        assert any("lonely" in e for e in errors), (
            "Error message must name the stage with the missing input"
        )

    def test_missing_one_of_two_required_inputs_is_rejected(self):
        """Providing only one of two required inputs is still a validation error."""
        from tests.dag.test_dag_automata import FanInStage

        errors = DAGValidator.validate_structure(
            stage_classes={
                "prod_data": FastStage,
                "fan_in": FanInStage,  # requires both 'data' and 'score'
            },
            data_flow_edges=[
                # Only 'data' is provided; 'score' is missing
                DataFlowEdge.create("prod_data", "fan_in", "data"),
            ],
        )
        assert errors, "Expected error for missing 'score' input"
        combined = " ".join(errors)
        assert "score" in combined, "Error must mention the missing 'score' field"

    def test_optional_input_with_no_edge_is_not_a_validation_error(self):
        """A stage with only optional inputs and no edges passes validation."""
        from tests.conftest import OptionalInputStage

        errors = DAGValidator.validate_structure(
            stage_classes={"opt": OptionalInputStage},
            data_flow_edges=[],
        )
        # OptionalInputStage has only an optional 'data' field — no required inputs
        assert not any("missing required" in e.lower() for e in errors), (
            f"Optional-only stage must not generate missing-required errors: {errors}"
        )

    def test_dag_automata_build_raises_for_missing_required_input(self):
        """DAGAutomata.build raises ValueError when a required input has no edge."""
        with pytest.raises(ValueError, match="missing required"):
            DAGAutomata.build(
                nodes={"lonely": ChainedStage(timeout=5.0)},
                data_flow_edges=[],  # ChainedStage requires 'data'
            )

    def test_unexpected_input_name_in_edge_is_rejected(self):
        """An edge whose input_name is not a declared field of the destination is rejected."""
        errors = DAGValidator.validate_structure(
            stage_classes={"src": FastStage, "dst": VoidStage},
            data_flow_edges=[
                # VoidStage (VoidInput) has no fields; 'nonexistent' is not declared
                DataFlowEdge.create("src", "dst", "nonexistent_field"),
            ],
        )
        assert errors, "Expected error for undeclared input_name"
        assert any("unexpected input" in e.lower() for e in errors), (
            "Error must mention 'unexpected input'"
        )
        assert any("nonexistent_field" in e for e in errors)


# ===================================================================
# Section IV: _check_dataflow_gate with no provider for mandatory input
# ===================================================================


class TestCheckDataflowGateNoProvider:
    """Tests for _check_dataflow_gate returning IMPOSSIBLE when a required input
    has no incoming edge at all in the topology.

    This path (automata.py line 441) is guarded at build-time by DAGValidator,
    so it is normally unreachable through DAGAutomata.build. However, it can
    be triggered by:
      1. Directly constructing DAGAutomata with a hand-assembled DAGTopology.
      2. Post-build mutation of topology (not supported but defensive).

    Testing this path ensures the runtime gate logic is correct independent
    of the build-time validator, and confirms IMPOSSIBLE is returned rather
    than a hang or incorrect WAIT.
    """

    def _build_automata_bypassing_validation(
        self,
        stage: Stage,
        stage_name: str,
    ) -> DAGAutomata:
        """Construct a DAGAutomata with a required-input stage but NO incoming edges.

        This bypasses DAGAutomata.build (which would reject the topology) by
        assembling the topology directly, mirroring what build() does internally
        minus the validation step.
        """
        nodes = {stage_name: stage}
        stage_cls = type(stage)

        automata = DAGAutomata(transition_rules={})
        automata.topology = DAGTopology(
            nodes=nodes,
            edges=[],  # No edges — required input has no provider
            incoming_by_dest={},  # Empty: no incoming edges for any stage
            preds_by_dest={},
            exec_rules={},
            incoming_by_input={},  # Empty: no input->edge mapping
            sorted_required_names={stage_name: sorted(stage_cls._required_names)},
            sorted_optional_names={stage_name: sorted(stage_cls._optional_names)},
        )
        return automata

    def test_mandatory_input_with_no_incoming_edge_returns_impossible(self):
        """_check_dataflow_gate returns IMPOSSIBLE when a required field has no edge.

        This is the precise scenario described in the code review: line 441
        returns (IMPOSSIBLE, ["data: mandatory 'X' has NO provider"]).
        """
        stage = RequiredInputStage(timeout=5.0)
        automata = self._build_automata_bypassing_validation(stage, "required_stage")
        prog = _make_program()

        gate_state, reasons = automata._check_dataflow_gate(
            prog, "required_stage", finished_this_run=set()
        )

        assert gate_state is DAGAutomata.GateState.IMPOSSIBLE, (
            "Expected IMPOSSIBLE when mandatory input has no provider, "
            f"got {gate_state} with reasons={reasons}"
        )
        assert reasons, "Expected a non-empty reasons list"
        assert any("NO provider" in r for r in reasons), (
            f"Reason must mention 'NO provider'; got {reasons}"
        )
        # The missing field name should appear in the reason
        assert any("data" in r for r in reasons), (
            f"Reason must mention the missing field name 'data'; got {reasons}"
        )

    def test_mandatory_input_no_edge_impossible_regardless_of_finished_this_run(self):
        """IMPOSSIBLE for missing provider holds whether finished_this_run is empty or not.

        This confirms the gate is not WAIT (which would require waiting for some
        upstream stage to finish) but truly IMPOSSIBLE — there is no upstream
        stage to wait for.
        """
        stage = RequiredInputStage(timeout=5.0)
        automata = self._build_automata_bypassing_validation(stage, "s")

        for finished in [set(), {"some_other_stage"}, {"s"}]:
            prog = _make_program()
            gate_state, _ = automata._check_dataflow_gate(
                prog, "s", finished_this_run=finished
            )
            assert gate_state is DAGAutomata.GateState.IMPOSSIBLE, (
                f"Expected IMPOSSIBLE for finished_this_run={finished}, got {gate_state}"
            )

    def test_mandatory_input_no_edge_leads_to_skip_via_diagnose_stage(self):
        """_diagnose_stage propagates IMPOSSIBLE from _check_dataflow_gate correctly.

        When the dataflow gate is IMPOSSIBLE (no provider), _diagnose_stage must
        also return IMPOSSIBLE — so the stage ends up in get_stages_to_skip.
        """
        stage = RequiredInputStage(timeout=5.0)
        automata = self._build_automata_bypassing_validation(stage, "req")
        prog = _make_program()

        diag_state, diag_reasons = automata._diagnose_stage(
            prog, "req", finished_this_run=set()
        )

        assert diag_state is DAGAutomata.GateState.IMPOSSIBLE
        assert diag_reasons

    def test_get_stages_to_skip_includes_stage_with_no_mandatory_provider(self):
        """get_stages_to_skip returns the stage when its mandatory input has no edge.

        End-to-end test of the IMPOSSIBLE path: a stage with no provider for its
        required input will appear in the to-skip set, ensuring the DAG main loop
        will issue a skip result rather than hanging.
        """
        stage = RequiredInputStage(timeout=5.0)
        automata = self._build_automata_bypassing_validation(stage, "unprovidable")
        prog = _make_program()

        to_skip = automata.get_stages_to_skip(
            prog,
            running=set(),
            launched_this_run=set(),
            finished_this_run=set(),
        )

        assert "unprovidable" in to_skip, (
            "Stage with no mandatory input provider must appear in get_stages_to_skip"
        )

    def test_optional_only_stage_with_no_edges_is_not_impossible(self):
        """Contrast: a stage with only optional inputs and no edges is READY, not IMPOSSIBLE.

        This ensures the IMPOSSIBLE path is specific to *required* inputs.
        """
        from tests.conftest import OptionalInputStage

        opt_stage = OptionalInputStage(timeout=5.0)
        # Build normally — OptionalInputStage has no required fields so validation passes
        automata = _build_automata({"opt": opt_stage})
        prog = _make_program()

        gate_state, reasons = automata._check_dataflow_gate(
            prog, "opt", finished_this_run=set()
        )

        assert gate_state is DAGAutomata.GateState.READY, (
            f"Optional-only stage with no edges must be READY, got {gate_state}"
        )
        assert not reasons


# ===================================================================
# Section V: explain_blockers — default diagnostic message
# ===================================================================


class TestExplainBlockersDefaultMessage:
    """Tests for DAGAutomata.explain_blockers when no blocking constraints are found.

    The key invariant: explain_blockers NEVER returns an empty list. When all
    stages are either done, running, launched, or skipped — and the method finds
    nothing to report — it appends a diagnostic fallback message (line 617-619).

    This prevents callers (e.g. the stall watchdog in dag.py) from silently
    receiving an empty log, which would be harder to debug than the explicit
    "no blockers detected" message.
    """

    def test_explain_blockers_returns_nonempty_when_all_stages_done(self):
        """When all stages are in finished_this_run, explain_blockers returns the
        fallback diagnostic message, not an empty list.
        """
        automata = _build_automata(
            {"a": FastStage(timeout=5.0), "b": VoidStage(timeout=5.0)}
        )
        prog = _make_program(
            stage_results={
                "a": _make_result(StageState.COMPLETED),
                "b": _make_result(StageState.COMPLETED),
            }
        )

        # Both stages done this run — no candidates remain for blocker analysis
        blockers = automata.explain_blockers(
            prog,
            running=set(),
            launched_this_run={"a", "b"},
            finished_this_run={"a", "b"},
        )

        assert blockers, (
            "explain_blockers must return a non-empty list even when all stages are done"
        )
        assert len(blockers) == 1, (
            f"Expected exactly one fallback message, got {blockers}"
        )
        assert "No blockers detected" in blockers[0], (
            f"Fallback message must contain 'No blockers detected'; got {blockers[0]!r}"
        )

    def test_explain_blockers_default_message_content(self):
        """The default message instructs the user to check worker pool / scheduler state."""
        automata = _build_automata({"only": VoidStage(timeout=5.0)})
        prog = _make_program(stage_results={"only": _make_result(StageState.COMPLETED)})

        blockers = automata.explain_blockers(
            prog,
            running=set(),
            launched_this_run={"only"},
            finished_this_run={"only"},
        )

        assert len(blockers) == 1
        msg = blockers[0]
        # The message must give actionable debugging guidance, not just say "done"
        assert "worker pool" in msg.lower() or "scheduler" in msg.lower(), (
            f"Fallback message must mention 'worker pool' or 'scheduler'; got {msg!r}"
        )

    def test_explain_blockers_returns_real_blockers_when_stage_is_waiting(self):
        """When a stage is genuinely blocked, explain_blockers reports it — not the fallback.

        This ensures the fallback message only appears when there are truly no
        blocker candidates, not as a replacement for real blocker analysis.
        """
        automata = _build_automata(
            {"a": FastStage(timeout=5.0), "b": ChainedStage(timeout=5.0)},
            edges=[DataFlowEdge.create("a", "b", "data")],
        )
        prog = _make_program()  # Neither stage has run yet

        blockers = automata.explain_blockers(
            prog,
            running=set(),
            launched_this_run=set(),
            finished_this_run=set(),
        )

        # 'b' is waiting on 'a' — should get a real blocker, not the fallback
        assert any("[Blocker]" in b for b in blockers), (
            f"Expected [Blocker] entries for blocked stage 'b'; got {blockers}"
        )
        assert not any("No blockers detected" in b for b in blockers), (
            "Fallback message must NOT appear when there are real blockers"
        )

    def test_explain_blockers_default_message_when_all_stages_are_running(self):
        """When all stages are running, there are no waiting candidates — returns fallback."""
        automata = _build_automata(
            {"a": FastStage(timeout=5.0), "b": FastStage(timeout=5.0)}
        )
        prog = _make_program()

        # Both stages are running — nothing is left to analyze as a blocker
        blockers = automata.explain_blockers(
            prog,
            running={"a", "b"},
            launched_this_run={"a", "b"},
            finished_this_run=set(),
        )

        assert blockers
        assert any("No blockers detected" in b for b in blockers), (
            f"Expected fallback message when all stages are running; got {blockers}"
        )

    def test_explain_blockers_default_message_when_all_stages_launched(self):
        """When all stages are in launched_this_run (but not yet finished), returns fallback."""
        automata = _build_automata({"x": VoidStage(timeout=5.0)})
        prog = _make_program()

        blockers = automata.explain_blockers(
            prog,
            running=set(),
            launched_this_run={"x"},
            finished_this_run=set(),
        )

        assert blockers
        assert any("No blockers detected" in b for b in blockers)

    def test_summarize_blockers_for_log_returns_nonempty_string_when_no_blockers(self):
        """summarize_blockers_for_log (used by stall watchdog) never returns empty string."""
        automata = _build_automata({"done": VoidStage(timeout=5.0)})
        prog = _make_program(stage_results={"done": _make_result(StageState.COMPLETED)})

        summary = automata.summarize_blockers_for_log(
            prog,
            running=set(),
            launched_this_run={"done"},
            finished_this_run={"done"},
        )

        assert summary, (
            "summarize_blockers_for_log must return a non-empty string even when "
            "no blockers are found"
        )
        assert "No blockers detected" in summary

    def test_explain_blockers_impossible_stage_appears_as_blocker(self):
        """A stage with IMPOSSIBLE deps appears as a [Blocker] entry (not the fallback).

        IMPOSSIBLE stages are not classified as 'ready', so they are candidates
        for the blocker analysis — their reasons should be reported.
        """
        automata = _build_automata(
            {"a": FailingStage(timeout=5.0), "b": ChainedStage(timeout=5.0)},
            edges=[DataFlowEdge.create("a", "b", "data")],
        )
        prog = _make_program(stage_results={"a": _make_result(StageState.FAILED)})

        # 'a' failed this run -> 'b' has IMPOSSIBLE mandatory dep
        blockers = automata.explain_blockers(
            prog,
            running=set(),
            launched_this_run=set(),
            finished_this_run={"a"},
        )

        assert any("'b'" in b for b in blockers), (
            f"Expected 'b' to appear as a blocker; got {blockers}"
        )
        assert not any("No blockers detected" in b for b in blockers)
