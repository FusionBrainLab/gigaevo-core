"""Edge-case and boundary tests for DAGAutomata.

Covers:
  I.    ExecutionOrderDependency.is_satisfied_historically — direct unit tests
        for the run-agnostic satisfaction check, including divergence from
        _check_dependency_gate for stale prior-run results.
  II.   DAGValidator.validate_structure — non-Stage node detection.
        Non-Stage classes passed as node values are caught early and a clear
        error is produced, before any edge or type validation is attempted.
  III.  DAGValidator.validate_structure — duplicate input_name and missing
        required inputs. Duplicate input edges and un-provided required inputs
        are caught by structural validation so that they surface as clear errors
        rather than silent wrong behavior at execution time.
  IV.   DAGAutomata._check_dataflow_gate — mandatory input with no provider
        (IMPOSSIBLE). At runtime the gate logic returns IMPOSSIBLE for a required
        input that has no incoming edge. This path is unreachable via the normal
        build() pathway (which validates structure up front), but can be hit if
        someone constructs automata topology manually or if the topology is
        mutated post-build.
  V.    DAGAutomata.explain_blockers — default diagnostic message (no blockers
        found). When all stages are already accounted for (done / running /
        launched), the explain_blockers method returns a diagnostic message
        rather than an empty list, so callers always get a non-empty response
        they can log.
  VI.   build_named_inputs output verification — correct dict contents for
        completed, failed, skipped, cancelled, and partial-completion scenarios.
  VII.  _check_dependency_gate — "always" gate with previous-run FINAL results
        not in finished_this_run (lines 398-404).
  VIII. _check_dependency_gate — "success"/"failure" gate with
        finalized_this_run=False (line 415).
  IX.   finalized_this_run compound flag with RUNNING/PENDING status in
        finished_this_run (line 381).
  X.    Multiple exec deps — IMPOSSIBLE short-circuits, WAIT accumulates
        (lines 496-506).
  XI.   Exec-order IMPOSSIBLE preempts / interacts with dataflow check
        (lines 501-502, 513-514).
  XII.  Optional input with previously-failed source — WAIT vs READY
        (lines 468-480).
  XIII. build_named_inputs — only COMPLETED outputs, first edge wins
        (lines 668-681).
  XIV.  get_ready_stages — cache hash=None forces rerun (lines 577-587).
  XV.   get_stages_to_skip vs get_ready_stages consistency (lines 558, 654).
  XVI.  Pydantic rejects invalid ExecutionOrderDependency condition.
  XVII. is_satisfied_historically — exhaustive status x condition combinations.
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
    VoidInput,
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
from gigaevo.programs.stages.cache_handler import InputHashCache
from tests.conftest import (
    ChainedStage,
    FailingStage,
    FastStage,
    MockOutput,
    OptionalInputStage,
    VoidStage,
)

# ---------------------------------------------------------------------------
# Helpers from extended tests
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
# Helpers from adversarial tests
# ---------------------------------------------------------------------------


def _prog(**stage_results) -> Program:
    p = Program(code="x=1", state=ProgramState.RUNNING)
    p.stage_results = {k: v for k, v in stage_results.items()}
    return p


def _result(status: StageState, output=None) -> ProgramStageResult:
    return ProgramStageResult(
        status=status,
        output=output,
        started_at=datetime.now(timezone.utc),
        finished_at=datetime.now(timezone.utc),
    )


def _automata(nodes, edges=None, exec_deps=None) -> DAGAutomata:
    return DAGAutomata.build(nodes, edges or [], exec_deps)


# ---------------------------------------------------------------------------
# Additional stage mocks needed by extended tests
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
        """is_satisfied_historically diverges from _check_dependency_gate
        for stale results (results from prior runs NOT in finished_this_run).

        Scenario: dep stage 'a' has a FAILED result from a *prior* run.
          - is_satisfied_historically: returns True for on_failure dep
            (sees the FAILED status, ignores which run produced it)
          - _check_dependency_gate: returns WAIT (not in finished_this_run)

        This divergence is intentional — _check_dependency_gate enforces the
        "must re-execute in the current run" invariant.
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


# ===================================================================
# Section VI: build_named_inputs output verification
# ===================================================================


class TestBuildNamedInputsOutputVerification:
    """Tests for DAGAutomata.build_named_inputs asserting actual dict contents.

    build_named_inputs collects output objects from COMPLETED producer stages
    and returns them keyed by the edge's input_name. These tests verify:
      - Correct key names in the returned dict
      - Correct output values (the actual StageIO objects)
      - Empty dict when no producers are completed
      - Optional inputs from failed producers are omitted
      - Multiple inputs from multiple producers are all present
    """

    def test_single_mandatory_input_returns_correct_value(self):
        """Single edge a->b with input_name='data': build_named_inputs returns {'data': <output>}."""
        automata = _build_automata(
            {"a": FastStage(timeout=5.0), "b": ChainedStage(timeout=5.0)},
            edges=[DataFlowEdge.create("a", "b", "data")],
        )
        output_a = MockOutput(value=42)
        prog = _make_program(
            stage_results={"a": _make_result(StageState.COMPLETED, output=output_a)}
        )

        named = automata.build_named_inputs(prog, "b")
        assert "data" in named, f"Expected 'data' key in named inputs; got {named}"
        assert named["data"] is output_a, (
            f"Expected the exact output object from 'a'; got {named['data']}"
        )
        assert named["data"].value == 42

    def test_multiple_inputs_from_different_producers(self):
        """Fan-in: two producers feeding different input fields to one stage."""
        from tests.dag.test_dag_automata import FanInStage, ProducerB, SecondInput

        automata = _build_automata(
            {
                "prod_data": FastStage(timeout=5.0),
                "prod_score": ProducerB(timeout=5.0),
                "fan_in": FanInStage(timeout=5.0),
            },
            edges=[
                DataFlowEdge.create("prod_data", "fan_in", "data"),
                DataFlowEdge.create("prod_score", "fan_in", "score"),
            ],
        )
        data_output = MockOutput(value=10)
        score_output = SecondInput(score=3.5)
        prog = _make_program(
            stage_results={
                "prod_data": _make_result(StageState.COMPLETED, output=data_output),
                "prod_score": _make_result(StageState.COMPLETED, output=score_output),
            }
        )

        named = automata.build_named_inputs(prog, "fan_in")
        assert set(named.keys()) == {"data", "score"}, (
            f"Expected keys {{'data', 'score'}}; got {set(named.keys())}"
        )
        assert named["data"] is data_output
        assert named["data"].value == 10
        assert named["score"] is score_output
        assert named["score"].score == 3.5

    def test_empty_dict_when_no_producers_completed(self):
        """Returns empty dict when no upstream stage has completed."""
        automata = _build_automata(
            {"a": FastStage(timeout=5.0), "b": ChainedStage(timeout=5.0)},
            edges=[DataFlowEdge.create("a", "b", "data")],
        )
        prog = _make_program()  # No stage results at all

        named = automata.build_named_inputs(prog, "b")
        assert named == {}, (
            f"Expected empty dict when no producers completed; got {named}"
        )

    def test_failed_producer_output_not_included(self):
        """FAILED producer's output is not included in named inputs.

        build_named_inputs only includes outputs from COMPLETED producers.
        A FAILED producer (even if it somehow has an output) is excluded.
        """
        automata = _build_automata(
            {"a": FailingStage(timeout=5.0), "b": ChainedStage(timeout=5.0)},
            edges=[DataFlowEdge.create("a", "b", "data")],
        )
        # FAILED with no output
        prog = _make_program(stage_results={"a": _make_result(StageState.FAILED)})

        named = automata.build_named_inputs(prog, "b")
        assert "data" not in named, (
            "FAILED producer's output must NOT appear in named inputs"
        )

    def test_skipped_producer_output_not_included(self):
        """SKIPPED producer's output is not included in named inputs."""
        automata = _build_automata(
            {"a": FastStage(timeout=5.0), "b": ChainedStage(timeout=5.0)},
            edges=[DataFlowEdge.create("a", "b", "data")],
        )
        prog = _make_program(stage_results={"a": _make_result(StageState.SKIPPED)})

        named = automata.build_named_inputs(prog, "b")
        assert "data" not in named, (
            "SKIPPED producer's output must NOT appear in named inputs"
        )

    def test_cancelled_producer_output_not_included(self):
        """CANCELLED producer's output is not included in named inputs."""
        automata = _build_automata(
            {"a": FastStage(timeout=5.0), "b": ChainedStage(timeout=5.0)},
            edges=[DataFlowEdge.create("a", "b", "data")],
        )
        prog = _make_program(stage_results={"a": _make_result(StageState.CANCELLED)})

        named = automata.build_named_inputs(prog, "b")
        assert "data" not in named, (
            "CANCELLED producer's output must NOT appear in named inputs"
        )

    def test_completed_with_none_output_not_included(self):
        """COMPLETED producer with output=None is not included.

        build_named_inputs checks `res.output is not None`, so a COMPLETED
        result with no actual output object is excluded.
        """
        automata = _build_automata(
            {"a": FastStage(timeout=5.0), "b": ChainedStage(timeout=5.0)},
            edges=[DataFlowEdge.create("a", "b", "data")],
        )
        prog = _make_program(
            stage_results={"a": _make_result(StageState.COMPLETED, output=None)}
        )

        named = automata.build_named_inputs(prog, "b")
        assert "data" not in named, (
            "COMPLETED with None output must NOT appear in named inputs"
        )

    def test_stage_with_no_incoming_edges_returns_empty_dict(self):
        """Root stage with no incoming edges returns empty dict."""
        automata = _build_automata({"a": FastStage(timeout=5.0)})
        prog = _make_program()

        named = automata.build_named_inputs(prog, "a")
        assert named == {}, (
            f"Root stage with no edges must return empty dict; got {named}"
        )

    def test_optional_input_from_completed_producer_is_included(self):
        """Optional input from a COMPLETED producer IS included in named inputs."""
        automata = _build_automata(
            {"a": FastStage(timeout=5.0), "opt": OptionalInputStage(timeout=5.0)},
            edges=[DataFlowEdge.create("a", "opt", "data")],
        )
        output_a = MockOutput(value=77)
        prog = _make_program(
            stage_results={"a": _make_result(StageState.COMPLETED, output=output_a)}
        )

        named = automata.build_named_inputs(prog, "opt")
        assert "data" in named
        assert named["data"].value == 77

    def test_optional_input_from_failed_producer_is_excluded(self):
        """Optional input from a FAILED producer is excluded (empty dict, not None value)."""
        automata = _build_automata(
            {"a": FailingStage(timeout=5.0), "opt": OptionalInputStage(timeout=5.0)},
            edges=[DataFlowEdge.create("a", "opt", "data")],
        )
        prog = _make_program(stage_results={"a": _make_result(StageState.FAILED)})

        named = automata.build_named_inputs(prog, "opt")
        assert "data" not in named, (
            "Optional input from FAILED producer must not be in named inputs"
        )

    def test_partial_completion_returns_only_completed_inputs(self):
        """Only COMPLETED producers' outputs appear; others are omitted."""
        from tests.dag.test_dag_automata import (
            MultiOptionalStage,
            ProducerB,
        )

        automata = _build_automata(
            {
                "a": FastStage(timeout=5.0),
                "b": ProducerB(timeout=5.0),
                "multi": MultiOptionalStage(timeout=5.0),
            },
            edges=[
                DataFlowEdge.create("a", "multi", "data"),
                DataFlowEdge.create("b", "multi", "score"),
            ],
        )
        output_a = MockOutput(value=55)
        prog = _make_program(
            stage_results={
                "a": _make_result(StageState.COMPLETED, output=output_a),
                "b": _make_result(StageState.FAILED),  # b failed
            }
        )

        named = automata.build_named_inputs(prog, "multi")
        assert "data" in named, "COMPLETED 'a' should provide 'data'"
        assert named["data"].value == 55
        assert "score" not in named, "FAILED 'b' should not provide 'score'"


# ===================================================================
# Section VII: "always" gate — previous-run FINAL result NOT in finished_this_run
# Target: automata.py lines 398-404
# ===================================================================


class TestAlwaysGatePreviousRun:
    """The 'always' condition only checks finalized_this_run (line 399).
    A stage that finished in a PREVIOUS run (FINAL status, but NOT in
    finished_this_run) returns WAIT — not READY.
    """

    def test_completed_previous_run_not_in_finished_returns_wait(self):
        """Stage COMPLETED previously but not this run -> WAIT."""
        aut = _automata(
            {"a": FastStage(timeout=5), "b": FastStage(timeout=5)},
            exec_deps={"b": [ExecutionOrderDependency.always_after("a")]},
        )
        prog = _prog(a=_result(StageState.COMPLETED))
        dep = ExecutionOrderDependency.always_after("a")
        state, _ = aut._check_dependency_gate(prog, dep, finished_this_run=set())
        assert state is DAGAutomata.GateState.WAIT

    def test_completed_this_run_returns_ready(self):
        """Same stage COMPLETED AND in finished_this_run -> READY."""
        aut = _automata(
            {"a": FastStage(timeout=5), "b": FastStage(timeout=5)},
            exec_deps={"b": [ExecutionOrderDependency.always_after("a")]},
        )
        prog = _prog(a=_result(StageState.COMPLETED))
        dep = ExecutionOrderDependency.always_after("a")
        state, _ = aut._check_dependency_gate(prog, dep, finished_this_run={"a"})
        assert state is DAGAutomata.GateState.READY

    @pytest.mark.parametrize(
        "final_status",
        [
            StageState.COMPLETED,
            StageState.FAILED,
            StageState.CANCELLED,
            StageState.SKIPPED,
        ],
    )
    def test_always_gate_ready_for_all_final_states_this_run(self, final_status):
        """always_after is satisfied by ANY final state, as long as it's this run."""
        aut = _automata(
            {"a": FastStage(timeout=5), "b": FastStage(timeout=5)},
            exec_deps={"b": [ExecutionOrderDependency.always_after("a")]},
        )
        prog = _prog(a=_result(final_status))
        dep = ExecutionOrderDependency.always_after("a")
        state, _ = aut._check_dependency_gate(prog, dep, finished_this_run={"a"})
        assert state is DAGAutomata.GateState.READY

    @pytest.mark.parametrize(
        "final_status",
        [
            StageState.COMPLETED,
            StageState.FAILED,
            StageState.CANCELLED,
            StageState.SKIPPED,
        ],
    )
    def test_always_gate_wait_for_all_final_states_previous_run(self, final_status):
        """always_after returns WAIT for any final state NOT in finished_this_run."""
        aut = _automata(
            {"a": FastStage(timeout=5), "b": FastStage(timeout=5)},
            exec_deps={"b": [ExecutionOrderDependency.always_after("a")]},
        )
        prog = _prog(a=_result(final_status))
        dep = ExecutionOrderDependency.always_after("a")
        state, _ = aut._check_dependency_gate(prog, dep, finished_this_run=set())
        assert state is DAGAutomata.GateState.WAIT


# ===================================================================
# Section VIII: "success"/"failure" gate — finalized_this_run=False means WAIT
# Target: automata.py line 415
# ===================================================================


class TestSuccessFailureGatePreviousRun:
    """For success/failure conditions, if finalized_this_run=False, the gate
    returns WAIT regardless of actual status (line 415).
    """

    def test_on_success_completed_previous_run_returns_wait(self):
        """on_success: stage COMPLETED previously but not this run -> WAIT."""
        aut = _automata(
            {"a": FastStage(timeout=5), "b": FastStage(timeout=5)},
            exec_deps={"b": [ExecutionOrderDependency.on_success("a")]},
        )
        prog = _prog(a=_result(StageState.COMPLETED))
        dep = ExecutionOrderDependency.on_success("a")
        state, _ = aut._check_dependency_gate(prog, dep, finished_this_run=set())
        assert state is DAGAutomata.GateState.WAIT

    def test_on_failure_failed_previous_run_returns_wait(self):
        """on_failure: stage FAILED previously but not this run -> WAIT."""
        aut = _automata(
            {"a": FastStage(timeout=5), "b": FastStage(timeout=5)},
            exec_deps={"b": [ExecutionOrderDependency.on_failure("a")]},
        )
        prog = _prog(a=_result(StageState.FAILED))
        dep = ExecutionOrderDependency.on_failure("a")
        state, _ = aut._check_dependency_gate(prog, dep, finished_this_run=set())
        assert state is DAGAutomata.GateState.WAIT

    def test_on_success_completed_this_run_returns_ready(self):
        """on_success: stage COMPLETED this run -> READY."""
        aut = _automata(
            {"a": FastStage(timeout=5), "b": FastStage(timeout=5)},
            exec_deps={"b": [ExecutionOrderDependency.on_success("a")]},
        )
        prog = _prog(a=_result(StageState.COMPLETED))
        dep = ExecutionOrderDependency.on_success("a")
        state, _ = aut._check_dependency_gate(prog, dep, finished_this_run={"a"})
        assert state is DAGAutomata.GateState.READY

    def test_on_success_failed_this_run_returns_impossible(self):
        """on_success: stage FAILED this run -> IMPOSSIBLE."""
        aut = _automata(
            {"a": FastStage(timeout=5), "b": FastStage(timeout=5)},
            exec_deps={"b": [ExecutionOrderDependency.on_success("a")]},
        )
        prog = _prog(a=_result(StageState.FAILED))
        dep = ExecutionOrderDependency.on_success("a")
        state, _ = aut._check_dependency_gate(prog, dep, finished_this_run={"a"})
        assert state is DAGAutomata.GateState.IMPOSSIBLE

    def test_on_success_cancelled_this_run_returns_impossible(self):
        """on_success with CANCELLED dep -> IMPOSSIBLE (line 407: completed=False)."""
        aut = _automata(
            {"a": FastStage(timeout=5), "b": FastStage(timeout=5)},
            exec_deps={"b": [ExecutionOrderDependency.on_success("a")]},
        )
        prog = _prog(a=_result(StageState.CANCELLED))
        dep = ExecutionOrderDependency.on_success("a")
        state, _ = aut._check_dependency_gate(prog, dep, finished_this_run={"a"})
        assert state is DAGAutomata.GateState.IMPOSSIBLE

    def test_on_failure_skipped_this_run_returns_ready(self):
        """on_failure: SKIPPED counts as failure (line 54-58)."""
        aut = _automata(
            {"a": FastStage(timeout=5), "b": FastStage(timeout=5)},
            exec_deps={"b": [ExecutionOrderDependency.on_failure("a")]},
        )
        prog = _prog(a=_result(StageState.SKIPPED))
        dep = ExecutionOrderDependency.on_failure("a")
        state, _ = aut._check_dependency_gate(prog, dep, finished_this_run={"a"})
        assert state is DAGAutomata.GateState.READY

    def test_on_failure_completed_this_run_returns_impossible(self):
        """on_failure: stage COMPLETED this run -> IMPOSSIBLE."""
        aut = _automata(
            {"a": FastStage(timeout=5), "b": FastStage(timeout=5)},
            exec_deps={"b": [ExecutionOrderDependency.on_failure("a")]},
        )
        prog = _prog(a=_result(StageState.COMPLETED))
        dep = ExecutionOrderDependency.on_failure("a")
        state, _ = aut._check_dependency_gate(prog, dep, finished_this_run={"a"})
        assert state is DAGAutomata.GateState.IMPOSSIBLE


# ===================================================================
# Section IX: finalized_this_run with RUNNING status in finished_this_run
# Target: automata.py line 381
# ===================================================================


class TestFinalizedThisRunWithNonFinalStatus:
    """finalized_this_run = finished_now AND finalized (line 381).
    If a stage is in finished_this_run but has RUNNING status,
    finalized=False -> finalized_this_run=False -> gate returns WAIT.
    """

    def test_running_in_finished_this_run_yields_wait(self):
        aut = _automata(
            {"a": FastStage(timeout=5), "b": FastStage(timeout=5)},
            exec_deps={"b": [ExecutionOrderDependency.always_after("a")]},
        )
        prog = _prog(
            a=ProgramStageResult(
                status=StageState.RUNNING,
                started_at=datetime.now(timezone.utc),
            )
        )
        dep = ExecutionOrderDependency.always_after("a")
        state, _ = aut._check_dependency_gate(prog, dep, finished_this_run={"a"})
        assert state is DAGAutomata.GateState.WAIT

    def test_pending_in_finished_this_run_yields_wait(self):
        aut = _automata(
            {"a": FastStage(timeout=5), "b": FastStage(timeout=5)},
            exec_deps={"b": [ExecutionOrderDependency.always_after("a")]},
        )
        prog = _prog(a=ProgramStageResult(status=StageState.PENDING))
        dep = ExecutionOrderDependency.always_after("a")
        state, _ = aut._check_dependency_gate(prog, dep, finished_this_run={"a"})
        assert state is DAGAutomata.GateState.WAIT

    def test_no_result_in_finished_this_run_yields_wait(self):
        """Stage in finished_this_run but no result at all -> WAIT."""
        aut = _automata(
            {"a": FastStage(timeout=5), "b": FastStage(timeout=5)},
            exec_deps={"b": [ExecutionOrderDependency.always_after("a")]},
        )
        prog = _prog()  # No result for 'a'
        dep = ExecutionOrderDependency.always_after("a")
        state, _ = aut._check_dependency_gate(prog, dep, finished_this_run={"a"})
        assert state is DAGAutomata.GateState.WAIT

    def test_get_stage_status_fields_with_running(self):
        """Verify all StageStatus fields for the RUNNING-in-finished edge case."""
        aut = _automata({"a": FastStage(timeout=5)})
        prog = _prog(
            a=ProgramStageResult(
                status=StageState.RUNNING,
                started_at=datetime.now(timezone.utc),
            )
        )
        status = aut._get_stage_status(prog, "a", finished_this_run={"a"})
        assert status.finalized is False
        assert status.completed is False
        assert status.finalized_this_run is False
        assert status.status_name == "RUNNING"


# ===================================================================
# Section X: Multiple exec deps — IMPOSSIBLE short-circuits, WAIT accumulates
# Target: automata.py lines 496-506
# ===================================================================


class TestMultipleExecDepsOrdering:
    """_diagnose_stage: IMPOSSIBLE > WAIT > READY priority."""

    def test_impossible_overrides_wait(self):
        """[READY, WAIT, IMPOSSIBLE] -> IMPOSSIBLE (short-circuit at line 502)."""
        aut = _automata(
            {
                "d1": FastStage(timeout=5),
                "d2": FastStage(timeout=5),
                "d3": FastStage(timeout=5),
                "t": FastStage(timeout=5),
            },
            exec_deps={
                "t": [
                    ExecutionOrderDependency.always_after("d1"),  # READY
                    ExecutionOrderDependency.always_after("d2"),  # WAIT
                    ExecutionOrderDependency.on_success("d3"),  # IMPOSSIBLE
                ]
            },
        )
        prog = _prog(
            d1=_result(StageState.COMPLETED),
            d3=_result(StageState.FAILED),
        )
        state, reasons = aut._diagnose_stage(prog, "t", finished_this_run={"d1", "d3"})
        assert state is DAGAutomata.GateState.IMPOSSIBLE
        # Only one reason (from IMPOSSIBLE dep), WAIT reason not included
        assert len(reasons) == 1

    def test_wait_when_no_impossible(self):
        """[READY, WAIT, READY] -> WAIT."""
        aut = _automata(
            {
                "d1": FastStage(timeout=5),
                "d2": FastStage(timeout=5),
                "d3": FastStage(timeout=5),
                "t": FastStage(timeout=5),
            },
            exec_deps={
                "t": [
                    ExecutionOrderDependency.always_after("d1"),
                    ExecutionOrderDependency.always_after("d2"),
                    ExecutionOrderDependency.always_after("d3"),
                ]
            },
        )
        prog = _prog(
            d1=_result(StageState.COMPLETED),
            d3=_result(StageState.COMPLETED),
        )
        state, _ = aut._diagnose_stage(prog, "t", finished_this_run={"d1", "d3"})
        assert state is DAGAutomata.GateState.WAIT

    def test_all_ready(self):
        """[READY, READY, READY] -> READY."""
        aut = _automata(
            {
                "d1": FastStage(timeout=5),
                "d2": FastStage(timeout=5),
                "d3": FastStage(timeout=5),
                "t": FastStage(timeout=5),
            },
            exec_deps={
                "t": [
                    ExecutionOrderDependency.always_after("d1"),
                    ExecutionOrderDependency.always_after("d2"),
                    ExecutionOrderDependency.always_after("d3"),
                ]
            },
        )
        prog = _prog(
            d1=_result(StageState.COMPLETED),
            d2=_result(StageState.FAILED),
            d3=_result(StageState.SKIPPED),
        )
        state, reasons = aut._diagnose_stage(
            prog,
            "t",
            finished_this_run={"d1", "d2", "d3"},
        )
        assert state is DAGAutomata.GateState.READY
        assert reasons == []


# ===================================================================
# Section XI: Exec IMPOSSIBLE preempts dataflow check
# Target: automata.py lines 501-502, 513-514
# ===================================================================


class TestExecImpossiblePreemptsDataflow:
    """If exec-order returns IMPOSSIBLE, dataflow is still checked
    (lines 508-510), but if dataflow is also IMPOSSIBLE it returns that.
    Key: exec IMPOSSIBLE returns at line 502 BEFORE dataflow check."""

    def test_exec_impossible_dataflow_ready_returns_impossible(self):
        aut = _automata(
            {
                "a": FastStage(timeout=5),
                "c": FastStage(timeout=5),
                "b": ChainedStage(timeout=5),
            },
            edges=[DataFlowEdge.create("a", "b", "data")],
            exec_deps={"b": [ExecutionOrderDependency.on_success("c")]},
        )
        prog = _prog(
            a=_result(StageState.COMPLETED, output=MockOutput(value=1)),
            c=_result(StageState.FAILED),
        )
        state, _ = aut._diagnose_stage(prog, "b", finished_this_run={"a", "c"})
        assert state is DAGAutomata.GateState.IMPOSSIBLE

    def test_exec_ready_dataflow_impossible_returns_impossible(self):
        aut = _automata(
            {
                "a": FailingStage(timeout=5),
                "c": FastStage(timeout=5),
                "b": ChainedStage(timeout=5),
            },
            edges=[DataFlowEdge.create("a", "b", "data")],
            exec_deps={"b": [ExecutionOrderDependency.always_after("c")]},
        )
        prog = _prog(
            a=_result(StageState.FAILED),
            c=_result(StageState.COMPLETED),
        )
        state, _ = aut._diagnose_stage(prog, "b", finished_this_run={"a", "c"})
        assert state is DAGAutomata.GateState.IMPOSSIBLE

    def test_exec_wait_dataflow_ready_returns_wait(self):
        aut = _automata(
            {
                "a": FastStage(timeout=5),
                "c": FastStage(timeout=5),
                "b": ChainedStage(timeout=5),
            },
            edges=[DataFlowEdge.create("a", "b", "data")],
            exec_deps={"b": [ExecutionOrderDependency.always_after("c")]},
        )
        prog = _prog(
            a=_result(StageState.COMPLETED, output=MockOutput(value=1)),
        )
        state, _ = aut._diagnose_stage(prog, "b", finished_this_run={"a"})
        assert state is DAGAutomata.GateState.WAIT


# ===================================================================
# Section XII: Optional input with previously-failed source -> WAIT
# Target: automata.py lines 468-480
# ===================================================================


class TestOptionalInputPreviouslyFailed:
    """For optional inputs, a source that FAILED in a previous run
    (finalized=True but finalized_this_run=False) causes WAIT.
    """

    def test_optional_source_failed_previous_run_yields_wait(self):
        aut = _automata(
            {"a": FastStage(timeout=5), "b": OptionalInputStage(timeout=5)},
            edges=[DataFlowEdge.create("a", "b", "data")],
        )
        prog = _prog(a=_result(StageState.FAILED))
        state, reasons = aut._check_dataflow_gate(
            prog,
            "b",
            finished_this_run=set(),
        )
        assert state is DAGAutomata.GateState.WAIT
        assert any("optional" in r for r in reasons)

    def test_optional_source_failed_this_run_yields_ready(self):
        """Failed this run -> finalized_this_run=True -> optional gate passes."""
        aut = _automata(
            {"a": FastStage(timeout=5), "b": OptionalInputStage(timeout=5)},
            edges=[DataFlowEdge.create("a", "b", "data")],
        )
        prog = _prog(a=_result(StageState.FAILED))
        state, reasons = aut._check_dataflow_gate(
            prog,
            "b",
            finished_this_run={"a"},
        )
        assert state is DAGAutomata.GateState.READY
        assert reasons == []


# ===================================================================
# Section XIII: build_named_inputs — only COMPLETED outputs, first edge wins
# Target: automata.py lines 668-681
# ===================================================================


class TestBuildNamedInputsOrdering:
    """build_named_inputs only includes COMPLETED producers.
    For duplicate input_name (prevented by build(), but guard at line 676),
    the first edge wins.
    """

    def test_failed_producer_excluded(self):
        aut = _automata(
            {"a": FailingStage(timeout=5), "b": OptionalInputStage(timeout=5)},
            edges=[DataFlowEdge.create("a", "b", "data")],
        )
        prog = _prog(a=_result(StageState.FAILED))
        named = aut.build_named_inputs(prog, "b")
        assert "data" not in named

    def test_completed_producer_included(self):
        aut = _automata(
            {"a": FastStage(timeout=5), "b": ChainedStage(timeout=5)},
            edges=[DataFlowEdge.create("a", "b", "data")],
        )
        prog = _prog(a=_result(StageState.COMPLETED, output=MockOutput(value=7)))
        named = aut.build_named_inputs(prog, "b")
        assert "data" in named
        assert named["data"].value == 7

    def test_completed_with_none_output_excluded(self):
        """COMPLETED but output=None -> excluded (line 675: res.output is not None)."""
        aut = _automata(
            {"a": FastStage(timeout=5), "b": OptionalInputStage(timeout=5)},
            edges=[DataFlowEdge.create("a", "b", "data")],
        )
        prog = _prog(
            a=ProgramStageResult(
                status=StageState.COMPLETED,
                output=None,
                started_at=datetime.now(timezone.utc),
            )
        )
        named = aut.build_named_inputs(prog, "b")
        assert "data" not in named


# ===================================================================
# Section XIV: get_ready_stages — cache hash=None forces rerun
# Target: automata.py lines 577-587
# ===================================================================


class TestGetReadyStageCacheHashNone:
    """When compute_hash_from_inputs raises (line 583), inputs_hash=None.
    For InputHashCache, stored_hash=non-None vs inputs_hash=None -> rerun.
    """

    def test_hash_exception_forces_rerun(self):
        """Stage with COMPLETED result + stored hash, but hash computation fails
        -> stage appears in ready_with_inputs (not newly_cached)."""

        class BadHashStage(Stage):
            InputsModel = VoidInput
            OutputModel = MockOutput
            cache_handler = InputHashCache()

            @classmethod
            def compute_hash_from_inputs(cls, inputs):  # noqa: ARG003
                raise RuntimeError("hash broken")

            async def compute(self, program):  # noqa: ARG002
                return MockOutput(value=1)

        aut = _automata({"s": BadHashStage(timeout=5)})
        prog = _prog(
            s=ProgramStageResult(
                status=StageState.COMPLETED,
                input_hash="stored_hash_abc",
                started_at=datetime.now(timezone.utc),
                finished_at=datetime.now(timezone.utc),
            )
        )
        ready, cached = aut.get_ready_stages(
            prog,
            running=set(),
            launched_this_run=set(),
            finished_this_run=set(),
        )
        # Hash failure -> inputs_hash=None -> should_rerun=True -> in ready, not cached
        assert "s" in ready
        assert "s" not in cached


# ===================================================================
# Section XV: get_stages_to_skip vs get_ready_stages consistency
# Target: automata.py lines 558, 654
# ===================================================================


class TestSkipAndReadyConsistency:
    """get_stages_to_skip and get_ready_stages should never both claim
    the same stage — a stage is either skippable or ready, never both.
    """

    def test_impossible_stage_in_skip_not_in_ready(self):
        aut = _automata(
            {"a": FastStage(timeout=5), "b": FastStage(timeout=5)},
            exec_deps={"b": [ExecutionOrderDependency.on_success("a")]},
        )
        prog = _prog(a=_result(StageState.FAILED))
        finished = {"a"}

        to_skip = aut.get_stages_to_skip(
            prog,
            running=set(),
            launched_this_run=set(),
            finished_this_run=finished,
        )
        ready, cached = aut.get_ready_stages(
            prog,
            running=set(),
            launched_this_run=set(),
            finished_this_run=finished,
        )
        # b should be in to_skip (IMPOSSIBLE), not in ready
        assert "b" in to_skip
        assert "b" not in ready
        assert "b" not in cached

    def test_ready_stage_not_in_skip(self):
        aut = _automata(
            {"a": FastStage(timeout=5), "b": FastStage(timeout=5)},
            exec_deps={"b": [ExecutionOrderDependency.always_after("a")]},
        )
        prog = _prog(a=_result(StageState.COMPLETED))
        finished = {"a"}

        to_skip = aut.get_stages_to_skip(
            prog,
            running=set(),
            launched_this_run=set(),
            finished_this_run=finished,
        )
        ready, _ = aut.get_ready_stages(
            prog,
            running=set(),
            launched_this_run=set(),
            finished_this_run=finished,
        )
        assert "b" in ready
        assert "b" not in to_skip


# ===================================================================
# Section XVI: Pydantic rejects invalid condition
# Target: ExecutionOrderDependency.condition Literal validation
# ===================================================================


class TestInvalidConditionRejected:
    def test_pydantic_rejects_unknown_condition(self):
        with pytest.raises(Exception):  # ValidationError
            ExecutionOrderDependency(stage_name="a", condition="unknown")


# ===================================================================
# Section XVII: is_satisfied_historically — exhaustive combinations
# ===================================================================


class TestIsSatisfiedHistoricallyCombinations:
    """Exhaustive status x condition combinations for is_satisfied_historically."""

    @pytest.mark.parametrize(
        "status,condition,expected",
        [
            (StageState.COMPLETED, "success", True),
            (StageState.COMPLETED, "failure", False),
            (StageState.COMPLETED, "always", True),
            (StageState.FAILED, "success", False),
            (StageState.FAILED, "failure", True),
            (StageState.FAILED, "always", True),
            (StageState.CANCELLED, "success", False),
            (StageState.CANCELLED, "failure", True),
            (StageState.CANCELLED, "always", True),
            (StageState.SKIPPED, "success", False),
            (StageState.SKIPPED, "failure", True),
            (StageState.SKIPPED, "always", True),
            (StageState.PENDING, "success", False),
            (StageState.PENDING, "failure", False),
            (StageState.PENDING, "always", False),
            (StageState.RUNNING, "success", False),
            (StageState.RUNNING, "failure", False),
            (StageState.RUNNING, "always", False),
        ],
    )
    def test_all_status_condition_combos(self, status, condition, expected):
        dep = ExecutionOrderDependency(stage_name="x", condition=condition)
        result = ProgramStageResult(
            status=status,
            started_at=datetime.now(timezone.utc),
        )
        assert dep.is_satisfied_historically(result) is expected

    def test_none_result_never_satisfied(self):
        for cond in ("success", "failure", "always"):
            dep = ExecutionOrderDependency(stage_name="x", condition=cond)
            assert dep.is_satisfied_historically(None) is False
