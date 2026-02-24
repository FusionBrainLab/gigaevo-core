"""DAGAutomata unit and integration tests.

Covers scheduling logic in isolation (DAGAutomata.build + method calls) and
DAG-level integration for multi-stage pipelines, failure propagation, stale
results, and concurrency/semaphore semantics.

Categories:
  A. _check_dependency_gate unit tests
  B. _check_dataflow_gate unit tests
  C. get_stages_to_skip unit tests
  D. get_ready_stages with caching unit tests
  E. DAG integration: multi-stage pipeline correctness
  F. DAG integration: failure propagation
  G. DAG integration: re-run with stale results (deadlock regression)
  H. Concurrency and semaphore
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import time
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
    DataFlowEdge,
    ExecutionOrderDependency,
)
from gigaevo.programs.dag.dag import DAG
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState
from gigaevo.programs.stages.base import Stage
from gigaevo.programs.stages.cache_handler import NO_CACHE
from tests.conftest import (
    ChainedStage,
    FailingStage,
    FastStage,
    MockOutput,
    NeverCachedStage,
    NullWriter,
    OptionalInputStage,
    SlowStage,
    VoidStage,
)

# ---------------------------------------------------------------------------
# Additional stage mocks used only in this module
# ---------------------------------------------------------------------------


class ControlledStage(Stage):
    """Stage whose execution is gated by an asyncio.Event for determinism."""

    InputsModel = VoidInput
    OutputModel = MockOutput

    # Each instance has its own event set by the test
    def __init__(self, *, timeout: float):
        super().__init__(timeout=timeout)
        self.gate: asyncio.Event = asyncio.Event()
        self.started: asyncio.Event = asyncio.Event()

    async def compute(self, program: Program) -> MockOutput:
        self.started.set()
        await self.gate.wait()
        return MockOutput(value=99)


class CountingStage(Stage):
    """Counts how many times compute() was called (for cache regression tests)."""

    InputsModel = VoidInput
    OutputModel = MockOutput

    call_count: int = 0  # class-level counter; reset between tests via fixture

    async def compute(self, program: Program) -> MockOutput:
        CountingStage.call_count += 1
        return MockOutput(value=CountingStage.call_count)


class NoCacheCountingStage(Stage):
    """NO_CACHE version of CountingStage — always re-executes."""

    InputsModel = VoidInput
    OutputModel = MockOutput
    cache_handler = NO_CACHE

    call_count: int = 0

    async def compute(self, program: Program) -> MockOutput:
        NoCacheCountingStage.call_count += 1
        return MockOutput(value=NoCacheCountingStage.call_count)


class SecondInput(StageIO):
    """A second distinct output type for multi-input testing."""

    score: float = 1.0


class ProducerB(Stage):
    """Produces SecondInput."""

    InputsModel = VoidInput
    OutputModel = SecondInput

    async def compute(self, program: Program) -> SecondInput:
        return SecondInput(score=2.0)


class DualMandatoryInput(StageIO):
    data: MockOutput
    score: SecondInput


class FanInStage(Stage):
    """Requires both 'data' (MockOutput) and 'score' (SecondInput) as mandatory inputs."""

    InputsModel = DualMandatoryInput
    OutputModel = MockOutput

    async def compute(self, program: Program) -> MockOutput:
        total = int(self.params.data.value + self.params.score.score)
        return MockOutput(value=total)


class DualOptionalInput(StageIO):
    data: Optional[MockOutput] = None
    score: Optional[SecondInput] = None


class MultiOptionalStage(Stage):
    """Has two optional inputs from different producers."""

    InputsModel = DualOptionalInput
    OutputModel = MockOutput

    async def compute(self, program: Program) -> MockOutput:
        val = 0
        if self.params.data is not None:
            val += self.params.data.value
        if self.params.score is not None:
            val += int(self.params.score.score)
        return MockOutput(value=val)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(
    status: StageState,
    *,
    input_hash: Optional[str] = None,
    output: Optional[StageIO] = None,
) -> ProgramStageResult:
    """Construct a ProgramStageResult with a given status and optional hash."""
    now = datetime.now(timezone.utc)
    return ProgramStageResult(
        status=status,
        started_at=now,
        finished_at=now if status in FINAL_STATES else None,
        input_hash=input_hash,
        output=output,
    )


def _make_dag(nodes, edges, state_manager, *, exec_deps=None, **kwargs) -> DAG:
    return DAG(
        nodes=nodes,
        data_flow_edges=edges,
        execution_order_deps=exec_deps,
        state_manager=state_manager,
        writer=NullWriter(),
        **kwargs,
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
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_counting_stages():
    """Reset class-level call counters before every test."""
    CountingStage.call_count = 0
    NoCacheCountingStage.call_count = 0
    yield
    CountingStage.call_count = 0
    NoCacheCountingStage.call_count = 0


# ===================================================================
# Category A: _check_dependency_gate unit tests
# ===================================================================


class TestCheckDependencyGate:
    """Unit tests for DAGAutomata._check_dependency_gate.

    The method evaluates whether a single execution-order dependency is
    READY, WAIT, or IMPOSSIBLE based on the dependency's condition
    ("always", "success", "failure") and the dep stage's current status.
    """

    # -- always condition -------------------------------------------------------

    def test_always_wait_when_dep_not_finalized_this_run(self):
        """always: WAIT when dep stage has not been finalized in the current run."""
        automata = _build_automata(
            {"a": FastStage(timeout=5.0), "b": FastStage(timeout=5.0)},
            exec_deps={"b": [ExecutionOrderDependency.always_after("a")]},
        )
        prog = _make_program(stage_results={"a": _make_result(StageState.COMPLETED)})
        dep = ExecutionOrderDependency.always_after("a")
        # "a" is COMPLETED in stage_results but NOT in finished_this_run
        state, _ = automata._check_dependency_gate(prog, dep, finished_this_run=set())
        assert state is DAGAutomata.GateState.WAIT

    def test_always_ready_when_dep_completed_this_run(self):
        """always: READY when dep finalized as COMPLETED in this run."""
        automata = _build_automata(
            {"a": FastStage(timeout=5.0), "b": FastStage(timeout=5.0)},
            exec_deps={"b": [ExecutionOrderDependency.always_after("a")]},
        )
        prog = _make_program(stage_results={"a": _make_result(StageState.COMPLETED)})
        dep = ExecutionOrderDependency.always_after("a")
        state, _ = automata._check_dependency_gate(prog, dep, finished_this_run={"a"})
        assert state is DAGAutomata.GateState.READY

    def test_always_ready_when_dep_failed_this_run(self):
        """always: READY when dep finalized as FAILED in this run (any final state)."""
        automata = _build_automata(
            {"a": FailingStage(timeout=5.0), "b": FastStage(timeout=5.0)},
            exec_deps={"b": [ExecutionOrderDependency.always_after("a")]},
        )
        prog = _make_program(stage_results={"a": _make_result(StageState.FAILED)})
        dep = ExecutionOrderDependency.always_after("a")
        state, _ = automata._check_dependency_gate(prog, dep, finished_this_run={"a"})
        assert state is DAGAutomata.GateState.READY

    def test_always_ready_when_dep_skipped_this_run(self):
        """always: READY when dep finalized as SKIPPED in this run."""
        automata = _build_automata(
            {"a": FastStage(timeout=5.0), "b": FastStage(timeout=5.0)},
            exec_deps={"b": [ExecutionOrderDependency.always_after("a")]},
        )
        prog = _make_program(stage_results={"a": _make_result(StageState.SKIPPED)})
        dep = ExecutionOrderDependency.always_after("a")
        state, _ = automata._check_dependency_gate(prog, dep, finished_this_run={"a"})
        assert state is DAGAutomata.GateState.READY

    def test_always_wait_when_dep_has_no_result_yet(self):
        """always: WAIT when dep stage has no result at all (PENDING implicitly)."""
        automata = _build_automata(
            {"a": FastStage(timeout=5.0), "b": FastStage(timeout=5.0)},
            exec_deps={"b": [ExecutionOrderDependency.always_after("a")]},
        )
        prog = _make_program()  # no stage_results
        dep = ExecutionOrderDependency.always_after("a")
        state, _ = automata._check_dependency_gate(prog, dep, finished_this_run=set())
        assert state is DAGAutomata.GateState.WAIT

    # -- success condition ------------------------------------------------------

    def test_success_ready_when_dep_completed_this_run(self):
        """success: READY when dep finalized as COMPLETED in this run."""
        automata = _build_automata(
            {"a": FastStage(timeout=5.0), "b": FastStage(timeout=5.0)},
            exec_deps={"b": [ExecutionOrderDependency.on_success("a")]},
        )
        prog = _make_program(stage_results={"a": _make_result(StageState.COMPLETED)})
        dep = ExecutionOrderDependency.on_success("a")
        state, _ = automata._check_dependency_gate(prog, dep, finished_this_run={"a"})
        assert state is DAGAutomata.GateState.READY

    def test_success_impossible_when_dep_failed_this_run(self):
        """success: IMPOSSIBLE when dep finalized as FAILED in this run."""
        automata = _build_automata(
            {"a": FailingStage(timeout=5.0), "b": FastStage(timeout=5.0)},
            exec_deps={"b": [ExecutionOrderDependency.on_success("a")]},
        )
        prog = _make_program(stage_results={"a": _make_result(StageState.FAILED)})
        dep = ExecutionOrderDependency.on_success("a")
        state, _ = automata._check_dependency_gate(prog, dep, finished_this_run={"a"})
        assert state is DAGAutomata.GateState.IMPOSSIBLE

    def test_success_impossible_when_dep_skipped_this_run(self):
        """success: IMPOSSIBLE when dep finalized as SKIPPED in this run."""
        automata = _build_automata(
            {"a": FastStage(timeout=5.0), "b": FastStage(timeout=5.0)},
            exec_deps={"b": [ExecutionOrderDependency.on_success("a")]},
        )
        prog = _make_program(stage_results={"a": _make_result(StageState.SKIPPED)})
        dep = ExecutionOrderDependency.on_success("a")
        state, _ = automata._check_dependency_gate(prog, dep, finished_this_run={"a"})
        assert state is DAGAutomata.GateState.IMPOSSIBLE

    def test_success_wait_when_dep_pending(self):
        """success: WAIT when dep has no result (effectively PENDING)."""
        automata = _build_automata(
            {"a": FastStage(timeout=5.0), "b": FastStage(timeout=5.0)},
            exec_deps={"b": [ExecutionOrderDependency.on_success("a")]},
        )
        prog = _make_program()
        dep = ExecutionOrderDependency.on_success("a")
        state, _ = automata._check_dependency_gate(prog, dep, finished_this_run=set())
        assert state is DAGAutomata.GateState.WAIT

    def test_success_wait_when_dep_finalized_but_not_in_this_run(self):
        """success: WAIT when dep is COMPLETED from a prior run (not in finished_this_run)."""
        automata = _build_automata(
            {"a": FastStage(timeout=5.0), "b": FastStage(timeout=5.0)},
            exec_deps={"b": [ExecutionOrderDependency.on_success("a")]},
        )
        prog = _make_program(stage_results={"a": _make_result(StageState.COMPLETED)})
        dep = ExecutionOrderDependency.on_success("a")
        # Completed but NOT in finished_this_run → stale result → WAIT
        state, _ = automata._check_dependency_gate(prog, dep, finished_this_run=set())
        assert state is DAGAutomata.GateState.WAIT

    # -- failure condition -------------------------------------------------------

    def test_failure_ready_when_dep_failed_this_run(self):
        """failure: READY when dep finalized as FAILED in this run."""
        automata = _build_automata(
            {"a": FailingStage(timeout=5.0), "b": FastStage(timeout=5.0)},
            exec_deps={"b": [ExecutionOrderDependency.on_failure("a")]},
        )
        prog = _make_program(stage_results={"a": _make_result(StageState.FAILED)})
        dep = ExecutionOrderDependency.on_failure("a")
        state, _ = automata._check_dependency_gate(prog, dep, finished_this_run={"a"})
        assert state is DAGAutomata.GateState.READY

    def test_failure_ready_when_dep_skipped_this_run(self):
        """failure: READY when dep finalized as SKIPPED in this run."""
        automata = _build_automata(
            {"a": FastStage(timeout=5.0), "b": FastStage(timeout=5.0)},
            exec_deps={"b": [ExecutionOrderDependency.on_failure("a")]},
        )
        prog = _make_program(stage_results={"a": _make_result(StageState.SKIPPED)})
        dep = ExecutionOrderDependency.on_failure("a")
        state, _ = automata._check_dependency_gate(prog, dep, finished_this_run={"a"})
        assert state is DAGAutomata.GateState.READY

    def test_failure_impossible_when_dep_completed_this_run(self):
        """failure: IMPOSSIBLE when dep finalized as COMPLETED in this run."""
        automata = _build_automata(
            {"a": FastStage(timeout=5.0), "b": FastStage(timeout=5.0)},
            exec_deps={"b": [ExecutionOrderDependency.on_failure("a")]},
        )
        prog = _make_program(stage_results={"a": _make_result(StageState.COMPLETED)})
        dep = ExecutionOrderDependency.on_failure("a")
        state, _ = automata._check_dependency_gate(prog, dep, finished_this_run={"a"})
        assert state is DAGAutomata.GateState.IMPOSSIBLE

    def test_failure_wait_when_dep_not_yet_finalized_this_run(self):
        """failure: WAIT when dep has no result (not yet finalized)."""
        automata = _build_automata(
            {"a": FastStage(timeout=5.0), "b": FastStage(timeout=5.0)},
            exec_deps={"b": [ExecutionOrderDependency.on_failure("a")]},
        )
        prog = _make_program()
        dep = ExecutionOrderDependency.on_failure("a")
        state, _ = automata._check_dependency_gate(prog, dep, finished_this_run=set())
        assert state is DAGAutomata.GateState.WAIT


# ===================================================================
# Category B: _check_dataflow_gate unit tests
# ===================================================================


class TestCheckDataflowGate:
    """Unit tests for DAGAutomata._check_dataflow_gate.

    Verifies how the automata decides READY / WAIT / IMPOSSIBLE for a stage's
    incoming data-flow edges, both for mandatory and optional inputs.
    """

    def test_mandatory_input_ready_when_producer_completed_this_run(self):
        """Mandatory input: READY when producer COMPLETED in finished_this_run."""
        automata = _build_automata(
            {"a": FastStage(timeout=5.0), "b": ChainedStage(timeout=5.0)},
            edges=[DataFlowEdge.create("a", "b", "data")],
        )
        prog = _make_program(
            stage_results={
                "a": _make_result(StageState.COMPLETED, output=MockOutput(value=1))
            }
        )
        state, _ = automata._check_dataflow_gate(prog, "b", finished_this_run={"a"})
        assert state is DAGAutomata.GateState.READY

    def test_mandatory_input_impossible_when_producer_failed_this_run(self):
        """Mandatory input: IMPOSSIBLE when producer FAILED in this run."""
        automata = _build_automata(
            {"a": FailingStage(timeout=5.0), "b": ChainedStage(timeout=5.0)},
            edges=[DataFlowEdge.create("a", "b", "data")],
        )
        prog = _make_program(stage_results={"a": _make_result(StageState.FAILED)})
        state, _ = automata._check_dataflow_gate(prog, "b", finished_this_run={"a"})
        assert state is DAGAutomata.GateState.IMPOSSIBLE

    def test_mandatory_input_impossible_when_producer_skipped_this_run(self):
        """Mandatory input: IMPOSSIBLE when producer SKIPPED in this run."""
        automata = _build_automata(
            {"a": FastStage(timeout=5.0), "b": ChainedStage(timeout=5.0)},
            edges=[DataFlowEdge.create("a", "b", "data")],
        )
        prog = _make_program(stage_results={"a": _make_result(StageState.SKIPPED)})
        state, _ = automata._check_dataflow_gate(prog, "b", finished_this_run={"a"})
        assert state is DAGAutomata.GateState.IMPOSSIBLE

    def test_mandatory_input_wait_when_producer_still_running(self):
        """Mandatory input: WAIT when producer not yet finalized in this run."""
        automata = _build_automata(
            {"a": FastStage(timeout=5.0), "b": ChainedStage(timeout=5.0)},
            edges=[DataFlowEdge.create("a", "b", "data")],
        )
        prog = _make_program()  # no results yet
        state, _ = automata._check_dataflow_gate(prog, "b", finished_this_run=set())
        assert state is DAGAutomata.GateState.WAIT

    def test_mandatory_input_wait_when_producer_completed_stale(self):
        """Mandatory input: WAIT when producer COMPLETED but NOT in this run (stale)."""
        automata = _build_automata(
            {"a": FastStage(timeout=5.0), "b": ChainedStage(timeout=5.0)},
            edges=[DataFlowEdge.create("a", "b", "data")],
        )
        prog = _make_program(
            stage_results={
                "a": _make_result(StageState.COMPLETED, output=MockOutput(value=1))
            }
        )
        # "a" has COMPLETED result but is NOT in finished_this_run
        state, _ = automata._check_dataflow_gate(prog, "b", finished_this_run=set())
        assert state is DAGAutomata.GateState.WAIT

    def test_optional_input_wait_when_producer_not_finalized(self):
        """Optional input: WAIT when producer has not been finalized this run."""
        automata = _build_automata(
            {"a": FastStage(timeout=5.0), "opt": OptionalInputStage(timeout=5.0)},
            edges=[DataFlowEdge.create("a", "opt", "data")],
        )
        prog = _make_program()
        state, _ = automata._check_dataflow_gate(prog, "opt", finished_this_run=set())
        assert state is DAGAutomata.GateState.WAIT

    def test_optional_input_ready_when_producer_completed_this_run(self):
        """Optional input: READY when producer COMPLETED in this run."""
        automata = _build_automata(
            {"a": FastStage(timeout=5.0), "opt": OptionalInputStage(timeout=5.0)},
            edges=[DataFlowEdge.create("a", "opt", "data")],
        )
        prog = _make_program(
            stage_results={
                "a": _make_result(StageState.COMPLETED, output=MockOutput(value=42))
            }
        )
        state, _ = automata._check_dataflow_gate(prog, "opt", finished_this_run={"a"})
        assert state is DAGAutomata.GateState.READY

    def test_optional_input_ready_when_producer_failed_this_run(self):
        """Optional input: READY when producer FAILED in this run (optional does not block)."""
        automata = _build_automata(
            {"a": FailingStage(timeout=5.0), "opt": OptionalInputStage(timeout=5.0)},
            edges=[DataFlowEdge.create("a", "opt", "data")],
        )
        prog = _make_program(stage_results={"a": _make_result(StageState.FAILED)})
        state, _ = automata._check_dataflow_gate(prog, "opt", finished_this_run={"a"})
        assert state is DAGAutomata.GateState.READY

    def test_optional_input_ready_when_no_edge_provided(self):
        """Optional input with no incoming edge: READY (no constraint)."""
        automata = _build_automata(
            {"opt": OptionalInputStage(timeout=5.0)},
        )
        prog = _make_program()
        state, _ = automata._check_dataflow_gate(prog, "opt", finished_this_run=set())
        assert state is DAGAutomata.GateState.READY

    def test_multiple_optional_inputs_wait_until_all_finalized(self):
        """Multiple optional inputs: WAIT while any producer is not finalized."""
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
        prog = _make_program(
            stage_results={
                "a": _make_result(StageState.COMPLETED, output=MockOutput(value=1))
            }
        )
        # "a" is in finished_this_run but "b" is not yet finalized
        state, _ = automata._check_dataflow_gate(prog, "multi", finished_this_run={"a"})
        assert state is DAGAutomata.GateState.WAIT

    def test_multiple_optional_inputs_ready_when_all_finalized(self):
        """Multiple optional inputs: READY when all producers are finalized this run."""
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
        prog = _make_program(
            stage_results={
                "a": _make_result(StageState.COMPLETED, output=MockOutput(value=1)),
                "b": _make_result(StageState.COMPLETED, output=SecondInput(score=2.0)),
            }
        )
        state, _ = automata._check_dataflow_gate(
            prog, "multi", finished_this_run={"a", "b"}
        )
        assert state is DAGAutomata.GateState.READY


# ===================================================================
# Category C: get_stages_to_skip unit tests
# ===================================================================


class TestGetStagesToSkip:
    """Unit tests for DAGAutomata.get_stages_to_skip.

    Verifies which stages the automata decides should be auto-skipped based
    on impossible dependency chains.
    """

    def test_no_stages_skipped_when_no_deps_failed(self):
        """When all deps are satisfied, no stages are in to_skip."""
        automata = _build_automata(
            {"a": FastStage(timeout=5.0), "b": ChainedStage(timeout=5.0)},
            edges=[DataFlowEdge.create("a", "b", "data")],
        )
        prog = _make_program(
            stage_results={
                "a": _make_result(StageState.COMPLETED, output=MockOutput(value=1))
            }
        )
        to_skip = automata.get_stages_to_skip(
            prog, running=set(), launched_this_run=set(), finished_this_run={"a"}
        )
        assert "b" not in to_skip

    def test_downstream_skipped_when_mandatory_dep_fails(self):
        """b is in to_skip when its mandatory producer 'a' has FAILED this run."""
        automata = _build_automata(
            {"a": FailingStage(timeout=5.0), "b": ChainedStage(timeout=5.0)},
            edges=[DataFlowEdge.create("a", "b", "data")],
        )
        prog = _make_program(stage_results={"a": _make_result(StageState.FAILED)})
        to_skip = automata.get_stages_to_skip(
            prog, running=set(), launched_this_run=set(), finished_this_run={"a"}
        )
        assert "b" in to_skip

    def test_stages_already_finished_this_run_excluded_from_skip(self):
        """Stages already in finished_this_run are not candidates for skipping."""
        automata = _build_automata(
            {"a": FailingStage(timeout=5.0), "b": ChainedStage(timeout=5.0)},
            edges=[DataFlowEdge.create("a", "b", "data")],
        )
        prog = _make_program(
            stage_results={
                "a": _make_result(StageState.FAILED),
                "b": _make_result(StageState.SKIPPED),
            }
        )
        # "b" is already in finished_this_run — should not appear in to_skip again
        to_skip = automata.get_stages_to_skip(
            prog,
            running=set(),
            launched_this_run=set(),
            finished_this_run={"a", "b"},
        )
        assert "b" not in to_skip

    def test_running_stages_excluded_from_skip(self):
        """Stages currently running are not candidates for skipping."""
        automata = _build_automata(
            {"a": FailingStage(timeout=5.0), "b": ChainedStage(timeout=5.0)},
            edges=[DataFlowEdge.create("a", "b", "data")],
        )
        prog = _make_program(stage_results={"a": _make_result(StageState.FAILED)})
        # "b" is currently running; it should not be in to_skip
        to_skip = automata.get_stages_to_skip(
            prog,
            running={"b"},
            launched_this_run=set(),
            finished_this_run={"a"},
        )
        assert "b" not in to_skip

    def test_chain_of_skips_propagates_transitively_across_rounds(self):
        """A->B->C: get_stages_to_skip is not transitive in one call but converges
        across iterations as the DAG main loop applies skips round by round.

        Round 1: A failed this run -> B is IMPOSSIBLE -> to_skip={B}
        Round 2: B now skipped (in finished_this_run) -> C is IMPOSSIBLE -> to_skip={C}

        This tests that get_stages_to_skip correctly identifies C as needing a skip
        once B has been finalized as SKIPPED in a subsequent scheduler iteration.
        """
        automata = _build_automata(
            {
                "a": FailingStage(timeout=5.0),
                "b": ChainedStage(timeout=5.0),
                "c": ChainedStage(timeout=5.0),
            },
            edges=[
                DataFlowEdge.create("a", "b", "data"),
                DataFlowEdge.create("b", "c", "data"),
            ],
        )
        prog = _make_program(stage_results={"a": _make_result(StageState.FAILED)})

        # Round 1: A finalized as FAILED — B becomes IMPOSSIBLE immediately
        to_skip_round1 = automata.get_stages_to_skip(
            prog, running=set(), launched_this_run=set(), finished_this_run={"a"}
        )
        assert "b" in to_skip_round1, (
            "B must be in to_skip when A (mandatory dep) fails"
        )
        assert "c" not in to_skip_round1, (
            "C is not yet IMPOSSIBLE in round 1 because B has not been finalized yet"
        )

        # Simulate the DAG applying the skip for B (as the main loop does)
        prog.stage_results["b"] = _make_result(StageState.SKIPPED)

        # Round 2: B is now finalized as SKIPPED in this run — C becomes IMPOSSIBLE
        to_skip_round2 = automata.get_stages_to_skip(
            prog,
            running=set(),
            launched_this_run={"b"},
            finished_this_run={"a", "b"},
        )
        assert "c" in to_skip_round2, (
            "C must be in to_skip after B is finalized as SKIPPED"
        )

    def test_optional_downstream_not_in_to_skip_when_producer_fails(self):
        """Optional input stage is NOT in to_skip when its producer fails."""
        automata = _build_automata(
            {"a": FailingStage(timeout=5.0), "opt": OptionalInputStage(timeout=5.0)},
            edges=[DataFlowEdge.create("a", "opt", "data")],
        )
        prog = _make_program(stage_results={"a": _make_result(StageState.FAILED)})
        to_skip = automata.get_stages_to_skip(
            prog, running=set(), launched_this_run=set(), finished_this_run={"a"}
        )
        assert "opt" not in to_skip

    def test_exec_dep_failure_causes_skip(self):
        """Stage with on_success dep on failed stage is in to_skip."""
        automata = _build_automata(
            {"a": FailingStage(timeout=5.0), "b": FastStage(timeout=5.0)},
            exec_deps={"b": [ExecutionOrderDependency.on_success("a")]},
        )
        prog = _make_program(stage_results={"a": _make_result(StageState.FAILED)})
        to_skip = automata.get_stages_to_skip(
            prog, running=set(), launched_this_run=set(), finished_this_run={"a"}
        )
        assert "b" in to_skip

    def test_exec_dep_always_condition_never_causes_skip(self):
        """Stage with always_after dep on failed stage is NOT in to_skip."""
        automata = _build_automata(
            {"a": FailingStage(timeout=5.0), "b": FastStage(timeout=5.0)},
            exec_deps={"b": [ExecutionOrderDependency.always_after("a")]},
        )
        prog = _make_program(stage_results={"a": _make_result(StageState.FAILED)})
        to_skip = automata.get_stages_to_skip(
            prog, running=set(), launched_this_run=set(), finished_this_run={"a"}
        )
        assert "b" not in to_skip


# ===================================================================
# Category D: get_ready_stages with caching unit tests
# ===================================================================


class TestGetReadyStagesWithCaching:
    """Unit tests for DAGAutomata.get_ready_stages caching behavior.

    Verifies how the automata determines which stages are ready to launch and
    which can use their cached results from a prior run.
    """

    def test_stage_with_no_prior_result_is_ready_to_run(self):
        """Fresh stage with no prior result appears in ready_with_inputs."""
        automata = _build_automata({"a": FastStage(timeout=5.0)})
        prog = _make_program()
        ready, cached = automata.get_ready_stages(
            prog, running=set(), launched_this_run=set(), finished_this_run=set()
        )
        assert "a" in ready
        assert "a" not in cached

    def test_completed_stage_with_matching_hash_is_cached(self):
        """A stage with a prior COMPLETED result and matching hash is newly_cached."""
        automata = _build_automata({"a": FastStage(timeout=5.0)})
        # Compute what the hash would be for a VoidInput (no inputs)
        expected_hash = FastStage.compute_hash_from_inputs({})
        prog = _make_program(
            stage_results={
                "a": _make_result(StageState.COMPLETED, input_hash=expected_hash)
            }
        )
        ready, cached = automata.get_ready_stages(
            prog, running=set(), launched_this_run=set(), finished_this_run=set()
        )
        assert "a" in cached
        assert "a" not in ready

    def test_no_cache_stage_never_in_newly_cached(self):
        """NO_CACHE stage is always in ready_with_inputs, never in newly_cached."""
        automata = _build_automata({"nc": NeverCachedStage(timeout=5.0)})
        expected_hash = NeverCachedStage.compute_hash_from_inputs({})
        prog = _make_program(
            stage_results={
                "nc": _make_result(StageState.COMPLETED, input_hash=expected_hash)
            }
        )
        ready, cached = automata.get_ready_stages(
            prog, running=set(), launched_this_run=set(), finished_this_run=set()
        )
        assert "nc" in ready
        assert "nc" not in cached

    def test_stage_with_wrong_input_hash_is_not_cached(self):
        """Stage with a different input_hash from its prior run is not cached."""
        automata = _build_automata({"a": FastStage(timeout=5.0)})
        prog = _make_program(
            stage_results={
                "a": _make_result(
                    StageState.COMPLETED, input_hash="deliberately_wrong_hash"
                )
            }
        )
        ready, cached = automata.get_ready_stages(
            prog, running=set(), launched_this_run=set(), finished_this_run=set()
        )
        assert "a" in ready
        assert "a" not in cached

    def test_completed_stage_with_no_hash_stored_is_not_cached(self):
        """Stage with COMPLETED result but no input_hash is not cached (hash=None)."""
        automata = _build_automata({"a": FastStage(timeout=5.0)})
        prog = _make_program(
            stage_results={"a": _make_result(StageState.COMPLETED, input_hash=None)}
        )
        ready, cached = automata.get_ready_stages(
            prog, running=set(), launched_this_run=set(), finished_this_run=set()
        )
        assert "a" in ready
        assert "a" not in cached

    def test_running_stage_not_in_ready_or_cached(self):
        """Stage currently running does not appear in ready_with_inputs or newly_cached."""
        automata = _build_automata({"a": FastStage(timeout=5.0)})
        prog = _make_program()
        ready, cached = automata.get_ready_stages(
            prog, running={"a"}, launched_this_run=set(), finished_this_run=set()
        )
        assert "a" not in ready
        assert "a" not in cached

    def test_launched_stage_not_in_ready_or_cached(self):
        """Stage already launched this run does not appear again in ready or cached."""
        automata = _build_automata({"a": FastStage(timeout=5.0)})
        prog = _make_program()
        ready, cached = automata.get_ready_stages(
            prog, running=set(), launched_this_run={"a"}, finished_this_run=set()
        )
        assert "a" not in ready
        assert "a" not in cached

    def test_downstream_not_ready_until_producer_finalized_this_run(self):
        """b is not in ready_with_inputs when a has not yet finalized in this run."""
        automata = _build_automata(
            {"a": FastStage(timeout=5.0), "b": ChainedStage(timeout=5.0)},
            edges=[DataFlowEdge.create("a", "b", "data")],
        )
        prog = _make_program()
        ready, cached = automata.get_ready_stages(
            prog, running=set(), launched_this_run=set(), finished_this_run=set()
        )
        assert "a" in ready
        assert "b" not in ready

    def test_cache_hit_is_idempotent(self):
        """Calling get_ready_stages twice with same state returns same cached set."""
        automata = _build_automata({"a": FastStage(timeout=5.0)})
        expected_hash = FastStage.compute_hash_from_inputs({})
        prog = _make_program(
            stage_results={
                "a": _make_result(StageState.COMPLETED, input_hash=expected_hash)
            }
        )
        ready1, cached1 = automata.get_ready_stages(
            prog, running=set(), launched_this_run=set(), finished_this_run=set()
        )
        ready2, cached2 = automata.get_ready_stages(
            prog, running=set(), launched_this_run=set(), finished_this_run=set()
        )
        assert cached1 == cached2
        assert ready1.keys() == ready2.keys()


# ===================================================================
# Category E: DAG integration: multi-stage pipeline correctness
# ===================================================================


class TestDAGIntegrationPipelineCorrectness:
    """Integration tests for multi-stage pipeline topologies.

    Each test runs a complete DAG and verifies final stage states and output values.
    """

    async def test_linear_chain_abc_all_complete(self, state_manager, make_program):
        """A->B->C linear chain: all stages complete in dependency order."""
        dag = _make_dag(
            {
                "a": FastStage(timeout=5.0),
                "b": ChainedStage(timeout=5.0),
                "c": ChainedStage(timeout=5.0),
            },
            [
                DataFlowEdge.create("a", "b", "data"),
                DataFlowEdge.create("b", "c", "data"),
            ],
            state_manager,
        )
        prog = make_program()
        await dag.run(prog)

        assert prog.stage_results["a"].status == StageState.COMPLETED
        assert prog.stage_results["b"].status == StageState.COMPLETED
        assert prog.stage_results["c"].status == StageState.COMPLETED
        assert prog.stage_results["a"].output.value == 42
        assert prog.stage_results["b"].output.value == 43
        assert prog.stage_results["c"].output.value == 44

    async def test_fanout_a_to_b_and_c_run_in_parallel(
        self, state_manager, make_program
    ):
        """Fan-out: A->B and A->C — both B and C complete after A."""
        dag = _make_dag(
            {
                "a": FastStage(timeout=5.0),
                "b": ChainedStage(timeout=5.0),
                "c": OptionalInputStage(timeout=5.0),
            },
            [
                DataFlowEdge.create("a", "b", "data"),
                DataFlowEdge.create("a", "c", "data"),
            ],
            state_manager,
        )
        prog = make_program()
        await dag.run(prog)

        assert prog.stage_results["a"].status == StageState.COMPLETED
        assert prog.stage_results["b"].status == StageState.COMPLETED
        assert prog.stage_results["c"].status == StageState.COMPLETED
        # b: 42+1=43; c: optional data=42 -> 42+10=52
        assert prog.stage_results["b"].output.value == 43
        assert prog.stage_results["c"].output.value == 52

    async def test_fanin_b_and_c_to_d_waits_for_both(self, state_manager, make_program):
        """Fan-in: D requires both B (data) and C (score) before it runs."""
        dag = _make_dag(
            {
                "b": FastStage(timeout=5.0),
                "c": ProducerB(timeout=5.0),
                "d": FanInStage(timeout=5.0),
            },
            [
                DataFlowEdge.create("b", "d", "data"),
                DataFlowEdge.create("c", "d", "score"),
            ],
            state_manager,
        )
        prog = make_program()
        await dag.run(prog)

        assert prog.stage_results["b"].status == StageState.COMPLETED
        assert prog.stage_results["c"].status == StageState.COMPLETED
        assert prog.stage_results["d"].status == StageState.COMPLETED
        # d = data.value + score.score = 42 + 2 = 44
        assert prog.stage_results["d"].output.value == 44

    async def test_diamond_dag_completes_correctly(self, state_manager, make_program):
        """Diamond: A->B->D and A->C->D. D runs after both B and C complete."""
        dag = _make_dag(
            {
                "a": FastStage(timeout=5.0),
                "b": ChainedStage(timeout=5.0),
                "c": OptionalInputStage(timeout=5.0),
                "d": VoidStage(timeout=5.0),
            },
            [
                DataFlowEdge.create("a", "b", "data"),
                DataFlowEdge.create("a", "c", "data"),
            ],
            state_manager,
            exec_deps={
                "d": [
                    ExecutionOrderDependency.on_success("b"),
                    ExecutionOrderDependency.on_success("c"),
                ]
            },
        )
        prog = make_program()
        await dag.run(prog)

        for stage in ("a", "b", "c", "d"):
            assert prog.stage_results[stage].status == StageState.COMPLETED

    async def test_exec_order_and_data_flow_combined(self, state_manager, make_program):
        """Mixed: B has exec-dep on_success of A, plus a data-flow edge from A."""
        dag = _make_dag(
            {
                "a": FastStage(timeout=5.0),
                "b": ChainedStage(timeout=5.0),
            },
            [DataFlowEdge.create("a", "b", "data")],
            state_manager,
            exec_deps={"b": [ExecutionOrderDependency.on_success("a")]},
        )
        prog = make_program()
        await dag.run(prog)

        assert prog.stage_results["a"].status == StageState.COMPLETED
        assert prog.stage_results["b"].status == StageState.COMPLETED
        assert prog.stage_results["b"].output.value == 43

    async def test_all_stages_in_final_state_after_run(
        self, state_manager, make_program
    ):
        """Every stage in the DAG ends in exactly one final state."""
        dag = _make_dag(
            {
                "a": FastStage(timeout=5.0),
                "b": ChainedStage(timeout=5.0),
                "c": OptionalInputStage(timeout=5.0),
            },
            [
                DataFlowEdge.create("a", "b", "data"),
                DataFlowEdge.create("a", "c", "data"),
            ],
            state_manager,
        )
        prog = make_program()
        await dag.run(prog)

        for name in ("a", "b", "c"):
            assert prog.stage_results[name].status in FINAL_STATES, (
                f"Stage '{name}' ended in non-final state: {prog.stage_results[name].status}"
            )

    async def test_single_node_dag_completes(self, state_manager, make_program):
        """Single-node DAG with no edges runs and completes."""
        dag = _make_dag({"only": VoidStage(timeout=5.0)}, [], state_manager)
        prog = make_program()
        await dag.run(prog)

        assert prog.stage_results["only"].status == StageState.COMPLETED


# ===================================================================
# Category F: DAG integration: failure propagation
# ===================================================================


class TestDAGIntegrationFailurePropagation:
    """Integration tests for failure propagation through the DAG.

    Verifies that failures skip mandatory downstream stages, that optional
    stages still run, and that 'always' conditions override failure.
    """

    async def test_root_failure_skips_all_mandatory_downstream(
        self, state_manager, make_program
    ):
        """When root 'a' fails, all stages with mandatory dep on a are SKIPPED."""
        dag = _make_dag(
            {
                "a": FailingStage(timeout=5.0),
                "b": ChainedStage(timeout=5.0),
                "c": ChainedStage(timeout=5.0),
            },
            [
                DataFlowEdge.create("a", "b", "data"),
                DataFlowEdge.create("b", "c", "data"),
            ],
            state_manager,
        )
        prog = make_program()
        await dag.run(prog)

        assert prog.stage_results["a"].status == StageState.FAILED
        assert prog.stage_results["b"].status == StageState.SKIPPED
        assert prog.stage_results["c"].status == StageState.SKIPPED

    async def test_optional_downstream_continues_when_producer_fails(
        self, state_manager, make_program
    ):
        """Optional-input stage runs with data=None when its producer fails."""
        dag = _make_dag(
            {
                "a": FailingStage(timeout=5.0),
                "opt": OptionalInputStage(timeout=5.0),
            },
            [DataFlowEdge.create("a", "opt", "data")],
            state_manager,
        )
        prog = make_program()
        await dag.run(prog)

        assert prog.stage_results["a"].status == StageState.FAILED
        assert prog.stage_results["opt"].status == StageState.COMPLETED
        # data was None so value = -1
        assert prog.stage_results["opt"].output.value == -1

    async def test_partial_fanout_failure_other_branch_completes(
        self, state_manager, make_program
    ):
        """Fan-out: B fails, C (independent) still completes."""
        dag = _make_dag(
            {
                "a": FastStage(timeout=5.0),
                "b": FailingStage(timeout=5.0),
                "c": ChainedStage(timeout=5.0),
            },
            [DataFlowEdge.create("a", "c", "data")],
            state_manager,
        )
        prog = make_program()
        await dag.run(prog)

        assert prog.stage_results["a"].status == StageState.COMPLETED
        assert prog.stage_results["b"].status == StageState.FAILED
        assert prog.stage_results["c"].status == StageState.COMPLETED

    async def test_always_after_runs_even_when_dep_fails(
        self, state_manager, make_program
    ):
        """Stage with always_after dependency runs regardless of dep failure."""
        dag = _make_dag(
            {
                "a": FailingStage(timeout=5.0),
                "cleanup": FastStage(timeout=5.0),
            },
            [],
            state_manager,
            exec_deps={"cleanup": [ExecutionOrderDependency.always_after("a")]},
        )
        prog = make_program()
        await dag.run(prog)

        assert prog.stage_results["a"].status == StageState.FAILED
        assert prog.stage_results["cleanup"].status == StageState.COMPLETED

    async def test_on_failure_stage_runs_after_failure_and_skips_after_success(
        self, state_manager, make_program
    ):
        """on_failure dep: stage runs when dep fails, is SKIPPED when dep succeeds."""
        # Case 1: dep fails -> stage runs
        dag_fail = _make_dag(
            {"a": FailingStage(timeout=5.0), "b": FastStage(timeout=5.0)},
            [],
            state_manager,
            exec_deps={"b": [ExecutionOrderDependency.on_failure("a")]},
        )
        prog1 = make_program()
        await dag_fail.run(prog1)
        assert prog1.stage_results["a"].status == StageState.FAILED
        assert prog1.stage_results["b"].status == StageState.COMPLETED

        # Case 2: dep succeeds -> stage is SKIPPED
        dag_ok = _make_dag(
            {"a": FastStage(timeout=5.0), "b": FastStage(timeout=5.0)},
            [],
            state_manager,
            exec_deps={"b": [ExecutionOrderDependency.on_failure("a")]},
        )
        prog2 = make_program()
        await dag_ok.run(prog2)
        assert prog2.stage_results["a"].status == StageState.COMPLETED
        assert prog2.stage_results["b"].status == StageState.SKIPPED

    async def test_all_stages_end_in_final_state_when_root_fails(
        self, state_manager, make_program
    ):
        """All stages end in a final state even when root fails."""
        dag = _make_dag(
            {
                "a": FailingStage(timeout=5.0),
                "b": ChainedStage(timeout=5.0),
                "c": OptionalInputStage(timeout=5.0),
            },
            [
                DataFlowEdge.create("a", "b", "data"),
                DataFlowEdge.create("a", "c", "data"),
            ],
            state_manager,
        )
        prog = make_program()
        await dag.run(prog)

        for name in ("a", "b", "c"):
            assert prog.stage_results[name].status in FINAL_STATES


# ===================================================================
# Category G: DAG integration: re-run with stale results (deadlock regression)
# ===================================================================


class TestDAGIntegrationStaleResultRegression:
    """Integration regression tests for the stale-result deadlock.

    Before the fix: a stage with a stale COMPLETED result from a previous run
    could not be overwritten by SKIPPED when upstream failed in the current run,
    causing a deadlock (skip_progress=False + running={} -> RuntimeError).

    After the fix: the skip guard is gated on `stage_name in finished_this_run`,
    so stale results from prior runs are correctly replaced with SKIPPED.
    """

    async def test_full_chain_stale_root_now_fails(self, state_manager, make_program):
        """Full chain stale: A and B were COMPLETED; now A fails -> B gets SKIPPED."""
        dag = _make_dag(
            {
                "a": FailingStage(timeout=5.0),
                "b": ChainedStage(timeout=5.0),
            },
            [DataFlowEdge.create("a", "b", "data")],
            state_manager,
        )
        prog = make_program(
            stage_results={
                "a": ProgramStageResult.success(output=MockOutput(value=10)),
                "b": ProgramStageResult.success(output=MockOutput(value=11)),
            }
        )
        # Both a and b have stale COMPLETED results from a prior run.
        # Now a fails in the current run -> b's mandatory input is gone.
        await dag.run(prog)

        assert prog.stage_results["a"].status == StageState.FAILED
        assert prog.stage_results["b"].status == StageState.SKIPPED

    async def test_mixed_stale_root_fails_downstream_skipped(
        self, state_manager, make_program
    ):
        """Mixed stale: A stale COMPLETED, B has no prior result. A fails -> B SKIPPED."""
        dag = _make_dag(
            {
                "a": FailingStage(timeout=5.0),
                "b": ChainedStage(timeout=5.0),
            },
            [DataFlowEdge.create("a", "b", "data")],
            state_manager,
        )
        prog = make_program(
            stage_results={
                "a": ProgramStageResult.success(output=MockOutput(value=10)),
            }
        )
        await dag.run(prog)

        assert prog.stage_results["a"].status == StageState.FAILED
        assert prog.stage_results["b"].status == StageState.SKIPPED

    async def test_stale_completed_chain_all_cached_on_rerun(
        self, state_manager, make_program
    ):
        """When all prior results are COMPLETED with valid hashes, all are newly_cached."""
        dag1 = _make_dag(
            {"a": FastStage(timeout=5.0), "b": ChainedStage(timeout=5.0)},
            [DataFlowEdge.create("a", "b", "data")],
            state_manager,
        )
        prog = make_program()
        await dag1.run(prog)

        first_a = prog.stage_results["a"].started_at
        first_b = prog.stage_results["b"].started_at

        # Second run: results are still valid -> should be cached
        dag2 = _make_dag(
            {"a": FastStage(timeout=5.0), "b": ChainedStage(timeout=5.0)},
            [DataFlowEdge.create("a", "b", "data")],
            state_manager,
        )
        await dag2.run(prog)

        # started_at unchanged == cached (not re-executed)
        assert prog.stage_results["a"].started_at == first_a
        assert prog.stage_results["b"].started_at == first_b

    async def test_three_node_stale_chain_deadlock_prevented(
        self, state_manager, make_program
    ):
        """Three-node chain: A, B, C all stale COMPLETED. A fails this run.
        B and C must both be SKIPPED without deadlock.
        """
        dag = _make_dag(
            {
                "a": FailingStage(timeout=5.0),
                "b": ChainedStage(timeout=5.0),
                "c": ChainedStage(timeout=5.0),
            },
            [
                DataFlowEdge.create("a", "b", "data"),
                DataFlowEdge.create("b", "c", "data"),
            ],
            state_manager,
        )
        prog = make_program(
            stage_results={
                "a": ProgramStageResult.success(output=MockOutput(value=1)),
                "b": ProgramStageResult.success(output=MockOutput(value=2)),
                "c": ProgramStageResult.success(output=MockOutput(value=3)),
            }
        )
        # Must not raise RuntimeError (deadlock)
        await asyncio.wait_for(dag.run(prog), timeout=10.0)

        assert prog.stage_results["a"].status == StageState.FAILED
        assert prog.stage_results["b"].status == StageState.SKIPPED
        assert prog.stage_results["c"].status == StageState.SKIPPED

    async def test_stale_optional_input_runs_when_fresh_producer_fails(
        self, state_manager, make_program
    ):
        """Stale COMPLETED opt result + fresh failure of optional producer -> opt re-runs."""
        dag = _make_dag(
            {
                "a": FailingStage(timeout=5.0),
                "opt": OptionalInputStage(timeout=5.0),
            },
            [DataFlowEdge.create("a", "opt", "data")],
            state_manager,
        )
        prog = make_program(
            stage_results={
                "opt": ProgramStageResult.success(output=MockOutput(value=52)),
            }
        )
        # opt has stale COMPLETED result; a fails this run; opt is optional -> opt reruns
        await dag.run(prog)

        assert prog.stage_results["a"].status == StageState.FAILED
        # opt should have re-run: with a FAILED, data is None -> value = -1
        assert prog.stage_results["opt"].status == StageState.COMPLETED
        assert prog.stage_results["opt"].output.value == -1

    async def test_no_deadlock_on_multiple_stale_failed_deps(
        self, state_manager, make_program
    ):
        """Stale COMPLETED D with two failing mandatory producers -> D is SKIPPED, no deadlock.

        Uses exec-order deps (no type constraints) so both producers (a, b) can be
        FailingStage and D can be VoidStage. D has on_success deps on both a and b;
        when both fail, D is IMPOSSIBLE -> SKIPPED. Pre-existing stale result on D
        must be overwritten without triggering the deadlock guard.
        """
        dag = _make_dag(
            {
                "a": FailingStage(timeout=5.0),
                "b": FailingStage(timeout=5.0),
                "d": VoidStage(timeout=5.0),
            },
            [],
            state_manager,
            exec_deps={
                "d": [
                    ExecutionOrderDependency.on_success("a"),
                    ExecutionOrderDependency.on_success("b"),
                ],
            },
        )
        prog = make_program(
            stage_results={
                # Stale COMPLETED result from a prior run — must be overwritten with SKIPPED
                "d": ProgramStageResult.success(output=None),
            }
        )
        await asyncio.wait_for(dag.run(prog), timeout=10.0)

        assert prog.stage_results["a"].status == StageState.FAILED
        assert prog.stage_results["b"].status == StageState.FAILED
        assert prog.stage_results["d"].status == StageState.SKIPPED


# ===================================================================
# Category H: Concurrency and semaphore
# ===================================================================


class TestConcurrencyAndSemaphore:
    """Integration tests for DAG semaphore and concurrent execution.

    Verifies that the semaphore correctly limits parallelism and that all
    stages still complete with various semaphore limits.
    """

    async def test_max_parallel_1_serializes_execution(
        self, state_manager, make_program
    ):
        """max_parallel_stages=1 forces sequential execution of all stages.

        With two SlowStages (0.5s each) and serialization, total >= 1.0s.
        """
        dag = _make_dag(
            {"a": SlowStage(timeout=5.0), "b": SlowStage(timeout=5.0)},
            [],
            state_manager,
            max_parallel_stages=1,
        )
        prog = make_program()
        t0 = time.monotonic()
        await dag.run(prog)
        elapsed = time.monotonic() - t0

        assert elapsed >= 0.9, f"Expected >= 0.9s with semaphore=1, got {elapsed:.3f}s"
        assert prog.stage_results["a"].status == StageState.COMPLETED
        assert prog.stage_results["b"].status == StageState.COMPLETED

    async def test_max_parallel_2_allows_simultaneous_stages(
        self, state_manager, make_program
    ):
        """max_parallel_stages=2 allows two SlowStages to run simultaneously.

        Two 0.5s stages in parallel: total should be < 1.5s (not fully serialized).
        """
        dag = _make_dag(
            {"a": SlowStage(timeout=5.0), "b": SlowStage(timeout=5.0)},
            [],
            state_manager,
            max_parallel_stages=2,
        )
        prog = make_program()
        t0 = time.monotonic()
        await dag.run(prog)
        elapsed = time.monotonic() - t0

        # Parallel: ~0.5s; serial would be ~1.0s; allow generous 1.4s upper bound
        assert elapsed < 1.4, f"Expected < 1.4s with semaphore=2, got {elapsed:.3f}s"
        assert prog.stage_results["a"].status == StageState.COMPLETED
        assert prog.stage_results["b"].status == StageState.COMPLETED

    async def test_semaphore_respected_with_many_ready_stages(
        self, state_manager, make_program
    ):
        """5 independent SlowStages with semaphore=2: all complete, semaphore respected.

        At most 2 stages run at a time, so total >= 1.0s (3 batches at 0.5s each).
        """
        nodes = {f"s{i}": SlowStage(timeout=5.0) for i in range(5)}
        dag = _make_dag(nodes, [], state_manager, max_parallel_stages=2)
        prog = make_program()
        t0 = time.monotonic()
        await dag.run(prog)
        elapsed = time.monotonic() - t0

        for name in nodes:
            assert prog.stage_results[name].status == StageState.COMPLETED

        # 5 stages with semaphore=2: ceil(5/2)=3 batches * 0.5s = 1.5s min
        assert elapsed >= 1.0, (
            f"Expected >= 1.0s with semaphore=2 and 5 slow stages, got {elapsed:.3f}s"
        )

    async def test_no_cache_stage_always_reruns_on_second_dag(
        self, state_manager, make_program
    ):
        """NeverCachedStage re-executes every DAG run regardless of prior results."""
        dag1 = _make_dag({"nc": NeverCachedStage(timeout=5.0)}, [], state_manager)
        prog = make_program()
        await dag1.run(prog)
        first_started = prog.stage_results["nc"].started_at

        await asyncio.sleep(0.01)

        dag2 = _make_dag({"nc": NeverCachedStage(timeout=5.0)}, [], state_manager)
        await dag2.run(prog)

        assert prog.stage_results["nc"].started_at != first_started

    async def test_concurrent_programs_do_not_interfere(
        self, state_manager, make_program
    ):
        """Two programs run concurrently through separate DAGs with no cross-contamination."""
        prog_a = make_program(code="def a(): return 1")
        prog_b = make_program(code="def b(): return 2")

        async def run_dag(prog):
            dag = _make_dag({"fast": FastStage(timeout=5.0)}, [], state_manager)
            await dag.run(prog)

        await asyncio.gather(run_dag(prog_a), run_dag(prog_b))

        assert prog_a.stage_results["fast"].status == StageState.COMPLETED
        assert prog_b.stage_results["fast"].status == StageState.COMPLETED
        # Verify programs are distinct
        assert prog_a.id != prog_b.id

    async def test_counting_stage_called_once_per_fresh_run(
        self, state_manager, make_program
    ):
        """CountingStage (with hash cache) executes only once for a fresh program."""
        dag = _make_dag({"cnt": CountingStage(timeout=5.0)}, [], state_manager)
        prog = make_program()
        await dag.run(prog)

        assert CountingStage.call_count == 1
        assert prog.stage_results["cnt"].status == StageState.COMPLETED

    async def test_counting_stage_skips_on_second_run_with_matching_hash(
        self, state_manager, make_program
    ):
        """CountingStage (hash cache) runs once; second DAG run uses cache (call_count=1)."""
        dag1 = _make_dag({"cnt": CountingStage(timeout=5.0)}, [], state_manager)
        prog = make_program()
        await dag1.run(prog)
        assert CountingStage.call_count == 1

        dag2 = _make_dag({"cnt": CountingStage(timeout=5.0)}, [], state_manager)
        await dag2.run(prog)
        # Second run: no re-execution (cached) -> call_count stays at 1
        assert CountingStage.call_count == 1

    async def test_no_cache_counting_stage_reruns_every_time(
        self, state_manager, make_program
    ):
        """NoCacheCountingStage (NO_CACHE) always re-executes on every DAG run."""
        dag1 = _make_dag({"nc": NoCacheCountingStage(timeout=5.0)}, [], state_manager)
        prog = make_program()
        await dag1.run(prog)
        assert NoCacheCountingStage.call_count == 1

        dag2 = _make_dag({"nc": NoCacheCountingStage(timeout=5.0)}, [], state_manager)
        await dag2.run(prog)
        assert NoCacheCountingStage.call_count == 2

    async def test_finished_this_run_tracks_all_finalized_stages(
        self, state_manager, make_program
    ):
        """Every stage that finishes is tracked in finished_this_run (observable via results)."""
        dag = _make_dag(
            {
                "a": FastStage(timeout=5.0),
                "b": FailingStage(timeout=5.0),
                "c": ChainedStage(timeout=5.0),
                "opt": OptionalInputStage(timeout=5.0),
            },
            [
                DataFlowEdge.create("a", "c", "data"),
                DataFlowEdge.create("b", "opt", "data"),
            ],
            state_manager,
        )
        prog = make_program()
        await dag.run(prog)

        # After run: all 4 stages must be in a final state
        for name in ("a", "b", "c", "opt"):
            assert prog.stage_results[name].status in FINAL_STATES, (
                f"Stage '{name}' ended in non-final state: {prog.stage_results[name].status}"
            )
        # opt has optional input from b (which fails) -> data=None -> value=-1
        assert prog.stage_results["opt"].output.value == -1
