"""Integration tests: DAG execution ordering and DataFlowEdge injection.

Covers:
  1. Three-stage linear chain A → B → C via ExecutionOrderDependency (on_success).
     Verifies C only runs after B completes, and that execution order is A, B, C.
  2. 'always' condition: cleanup stage runs even when an upstream stage fails.
  3. Upstream failure with 'on_success' dep: downstream stage is SKIPPED (not run).
     Simultaneously, a sibling with 'always' dep DOES run.
  4. DataFlowEdge: upstream output is correctly injected into downstream stage's
     input field and visible via self.params inside compute().

All tests use simple FakeStage helpers with NO_CACHE so results are never
reused from a prior run of the same program object.
"""

from __future__ import annotations

from typing import Optional

import pytest  # noqa: F401  (used implicitly for fixtures)

from gigaevo.programs.core_types import (
    StageIO,
    StageState,
    VoidInput,
    VoidOutput,
)
from gigaevo.programs.dag.automata import DataFlowEdge, ExecutionOrderDependency
from gigaevo.programs.dag.dag import DAG
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState
from gigaevo.programs.stages.base import Stage
from gigaevo.programs.stages.cache_handler import NO_CACHE
from tests.conftest import NullWriter

# ---------------------------------------------------------------------------
# Shared I/O types
# ---------------------------------------------------------------------------


class IntOutput(StageIO):
    value: int = 0


class IntInput(StageIO):
    upstream: IntOutput


class OptIntInput(StageIO):
    upstream: Optional[IntOutput] = None


# ---------------------------------------------------------------------------
# Instrumented fake stages
# ---------------------------------------------------------------------------


class RecordingStage(Stage):
    """Records the order in which instances were executed (class-level list)."""

    InputsModel = VoidInput
    OutputModel = IntOutput
    cache_handler = NO_CACHE

    # Class-level execution log shared across all instances in a test;
    # reset between tests via the fixture below.
    _execution_log: list[str] = []

    def __init__(self, *, name: str, timeout: float = 10.0):
        super().__init__(timeout=timeout)
        self._name = name

    async def compute(self, program: Program) -> IntOutput:
        RecordingStage._execution_log.append(self._name)
        return IntOutput(value=len(RecordingStage._execution_log))


class AlwaysFailStage(Stage):
    """Always raises RuntimeError (for failure-propagation tests)."""

    InputsModel = VoidInput
    OutputModel = IntOutput
    cache_handler = NO_CACHE

    async def compute(self, program: Program) -> IntOutput:
        raise RuntimeError("intentional failure")


class CleanupStage(Stage):
    """Should run regardless of upstream outcome ('always' condition).

    Uses an optional input so it does NOT require its upstream to have completed.
    Records itself in RecordingStage._execution_log so ordering is verifiable.
    """

    InputsModel = VoidInput
    OutputModel = VoidOutput
    cache_handler = NO_CACHE

    _ran: bool = False

    async def compute(self, program: Program) -> None:
        CleanupStage._ran = True
        RecordingStage._execution_log.append("cleanup")


class DataConsumerStage(Stage):
    """Receives an IntOutput from an upstream stage via DataFlowEdge."""

    InputsModel = IntInput
    OutputModel = IntOutput
    cache_handler = NO_CACHE

    # Stores received value for test assertions
    received_value: int = -1

    async def compute(self, program: Program) -> IntOutput:
        DataConsumerStage.received_value = self.params.upstream.value
        return IntOutput(value=self.params.upstream.value * 10)


class BlockedBySuccessStage(Stage):
    """Has an on_success dep — must be SKIPPED when upstream fails."""

    InputsModel = VoidInput
    OutputModel = IntOutput
    cache_handler = NO_CACHE

    _ran: bool = False

    async def compute(self, program: Program) -> IntOutput:
        BlockedBySuccessStage._ran = True
        return IntOutput(value=999)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_class_state():
    """Reset class-level state between tests."""
    RecordingStage._execution_log = []
    CleanupStage._ran = False
    BlockedBySuccessStage._ran = False
    DataConsumerStage.received_value = -1
    yield


# ---------------------------------------------------------------------------
# Helper: build a minimal DAG and run it
# ---------------------------------------------------------------------------


async def _run_dag(
    nodes: dict,
    data_flow_edges: list,
    execution_order_deps: dict | None,
    state_manager,
) -> Program:
    """Build a DAG, create a RUNNING program, run the DAG, return the program."""
    dag = DAG(
        nodes=nodes,
        data_flow_edges=data_flow_edges,
        execution_order_deps=execution_order_deps,
        state_manager=state_manager,
        writer=NullWriter(),
        dag_timeout=10.0,
    )
    program = Program(
        code="def solve(): return 1",
        state=ProgramState.RUNNING,
        atomic_counter=999_999_999,
    )
    # The DAG calls write_exclusive during init; storage must know about the program.
    await state_manager.storage.add(program)
    await dag.run(program)
    return program


# ---------------------------------------------------------------------------
# Area 1, Test 1: linear chain A → B → C via ExecutionOrderDependency
# ---------------------------------------------------------------------------


class TestLinearChainExecutionOrder:
    async def test_three_stage_linear_chain_runs_in_order(self, state_manager) -> None:
        """A → B → C: C only runs after B, B only runs after A.

        The execution log must be exactly ['A', 'B', 'C'], never out of order.
        """
        stage_a = RecordingStage(name="A")
        stage_b = RecordingStage(name="B")
        stage_c = RecordingStage(name="C")

        nodes = {"A": stage_a, "B": stage_b, "C": stage_c}
        exec_deps = {
            "B": [ExecutionOrderDependency.on_success("A")],
            "C": [ExecutionOrderDependency.on_success("B")],
        }

        program = await _run_dag(
            nodes=nodes,
            data_flow_edges=[],
            execution_order_deps=exec_deps,
            state_manager=state_manager,
        )

        # All three stages must have completed
        assert program.stage_results["A"].status == StageState.COMPLETED
        assert program.stage_results["B"].status == StageState.COMPLETED
        assert program.stage_results["C"].status == StageState.COMPLETED

        # Execution order must be strictly A, B, C
        assert RecordingStage._execution_log == ["A", "B", "C"], (
            f"Expected A→B→C execution order, got: {RecordingStage._execution_log}"
        )

    async def test_c_does_not_run_before_b_finishes(self, state_manager) -> None:
        """C must not be launched until B is in finished_this_run.

        We verify this indirectly: the log entry for C appears strictly after B.
        """
        stage_a = RecordingStage(name="A")
        stage_b = RecordingStage(name="B")
        stage_c = RecordingStage(name="C")

        nodes = {"A": stage_a, "B": stage_b, "C": stage_c}
        exec_deps = {
            "B": [ExecutionOrderDependency.on_success("A")],
            "C": [ExecutionOrderDependency.on_success("B")],
        }

        await _run_dag(
            nodes=nodes,
            data_flow_edges=[],
            execution_order_deps=exec_deps,
            state_manager=state_manager,
        )

        log = RecordingStage._execution_log
        assert log.index("B") < log.index("C"), f"C ran before B finished. Log: {log}"


# ---------------------------------------------------------------------------
# Area 1, Test 2: 'always' condition — cleanup runs even after upstream failure
# ---------------------------------------------------------------------------


class TestAlwaysConditionCleanup:
    async def test_cleanup_runs_after_failing_upstream(self, state_manager) -> None:
        """'always' dep is satisfied by FAILED as well as COMPLETED.

        Setup:
            fail_stage  (always fails)
            cleanup     (always_after fail_stage)

        Expected:
            fail_stage  → FAILED
            cleanup     → COMPLETED (ran despite upstream failure)
        """
        fail_stage = AlwaysFailStage(timeout=5.0)
        cleanup = CleanupStage(timeout=5.0)

        nodes = {"fail_stage": fail_stage, "cleanup": cleanup}
        exec_deps = {
            "cleanup": [ExecutionOrderDependency.always_after("fail_stage")],
        }

        program = await _run_dag(
            nodes=nodes,
            data_flow_edges=[],
            execution_order_deps=exec_deps,
            state_manager=state_manager,
        )

        assert program.stage_results["fail_stage"].status == StageState.FAILED, (
            "fail_stage should be FAILED"
        )
        assert program.stage_results["cleanup"].status == StageState.COMPLETED, (
            "cleanup should COMPLETE despite upstream failure (condition='always')"
        )
        assert CleanupStage._ran is True, "CleanupStage.compute() was never called"

    async def test_cleanup_also_runs_after_successful_upstream(
        self, state_manager
    ) -> None:
        """'always' dep fires on success too — not only on failure."""
        stage_a = RecordingStage(name="A")
        cleanup = CleanupStage(timeout=5.0)

        nodes = {"A": stage_a, "cleanup": cleanup}
        exec_deps = {
            "cleanup": [ExecutionOrderDependency.always_after("A")],
        }

        program = await _run_dag(
            nodes=nodes,
            data_flow_edges=[],
            execution_order_deps=exec_deps,
            state_manager=state_manager,
        )

        assert program.stage_results["A"].status == StageState.COMPLETED
        assert program.stage_results["cleanup"].status == StageState.COMPLETED
        assert CleanupStage._ran is True


# ---------------------------------------------------------------------------
# Area 1, Test 3: upstream failure blocks on_success dep, 'always' still fires
# ---------------------------------------------------------------------------


class TestFailurePropagation:
    async def test_on_success_dep_skips_when_upstream_fails(
        self, state_manager
    ) -> None:
        """on_success dep: downstream is SKIPPED when upstream fails.

        Setup:
            fail_stage  (always fails)
            blocked     (on_success dep on fail_stage) → must be SKIPPED
            cleanup     (always_after fail_stage)      → must be COMPLETED
        """
        fail_stage = AlwaysFailStage(timeout=5.0)
        blocked = BlockedBySuccessStage(timeout=5.0)
        cleanup = CleanupStage(timeout=5.0)

        nodes = {
            "fail_stage": fail_stage,
            "blocked": blocked,
            "cleanup": cleanup,
        }
        exec_deps = {
            "blocked": [ExecutionOrderDependency.on_success("fail_stage")],
            "cleanup": [ExecutionOrderDependency.always_after("fail_stage")],
        }

        program = await _run_dag(
            nodes=nodes,
            data_flow_edges=[],
            execution_order_deps=exec_deps,
            state_manager=state_manager,
        )

        assert program.stage_results["fail_stage"].status == StageState.FAILED
        assert program.stage_results["blocked"].status == StageState.SKIPPED, (
            "'blocked' should be SKIPPED because its on_success dep failed; "
            f"got {program.stage_results['blocked'].status}"
        )
        assert program.stage_results["cleanup"].status == StageState.COMPLETED, (
            "'cleanup' should still COMPLETE (always_after)"
        )
        assert BlockedBySuccessStage._ran is False, (
            "BlockedBySuccessStage.compute() must not have been called"
        )

    async def test_skipped_stage_does_not_count_as_success(self, state_manager) -> None:
        """A stage that depends on_success of a SKIPPED stage must also be SKIPPED.

        Chain: fail_stage → [on_success] → B → [on_success] → C
        fail_stage fails → B is SKIPPED → C is also SKIPPED.
        """

        class StageB(Stage):
            InputsModel = VoidInput
            OutputModel = IntOutput
            cache_handler = NO_CACHE
            _ran: bool = False

            async def compute(self, program: Program) -> IntOutput:
                StageB._ran = True
                return IntOutput(value=2)

        class StageC(Stage):
            InputsModel = VoidInput
            OutputModel = IntOutput
            cache_handler = NO_CACHE
            _ran: bool = False

            async def compute(self, program: Program) -> IntOutput:
                StageC._ran = True
                return IntOutput(value=3)

        StageB._ran = False
        StageC._ran = False

        fail_stage = AlwaysFailStage(timeout=5.0)
        stage_b = StageB(timeout=5.0)
        stage_c = StageC(timeout=5.0)

        nodes = {"fail_stage": fail_stage, "B": stage_b, "C": stage_c}
        exec_deps = {
            "B": [ExecutionOrderDependency.on_success("fail_stage")],
            "C": [ExecutionOrderDependency.on_success("B")],
        }

        result = await _run_dag(
            nodes=nodes,
            data_flow_edges=[],
            execution_order_deps=exec_deps,
            state_manager=state_manager,
        )

        assert result.stage_results["fail_stage"].status == StageState.FAILED
        assert result.stage_results["B"].status == StageState.SKIPPED
        assert result.stage_results["C"].status == StageState.SKIPPED
        assert StageB._ran is False
        assert StageC._ran is False


# ---------------------------------------------------------------------------
# Area 1, Test 4: DataFlowEdge — upstream output injected into downstream input
# ---------------------------------------------------------------------------


class TestDataFlowEdgeInjection:
    async def test_upstream_output_injected_into_downstream_input(
        self, state_manager
    ) -> None:
        """DataFlowEdge: the output of stage A is passed to stage B as 'upstream'.

        Stage A returns IntOutput(value=7).
        Stage B receives that IntOutput via input_name='upstream' and returns value*10.
        We assert:
          - B's output value is 70 (= 7 * 10)
          - DataConsumerStage.received_value == 7
        """
        # Use a fixed-value stage so the assertion is fully deterministic
        stage_a_fixed = _FixedValueStage(value=7, timeout=5.0)
        stage_b = DataConsumerStage(timeout=5.0)

        nodes = {"A": stage_a_fixed, "B": stage_b}
        edges = [
            DataFlowEdge.create(source="A", destination="B", input_name="upstream")
        ]

        program = await _run_dag(
            nodes=nodes,
            data_flow_edges=edges,
            execution_order_deps=None,
            state_manager=state_manager,
        )

        assert program.stage_results["A"].status == StageState.COMPLETED
        assert program.stage_results["B"].status == StageState.COMPLETED

        # Verify the injected value was received correctly
        assert DataConsumerStage.received_value == 7, (
            f"Expected received_value=7, got {DataConsumerStage.received_value}"
        )

        # Verify B's output reflects the injected input
        b_output = program.stage_results["B"].output
        assert isinstance(b_output, IntOutput)
        assert b_output.value == 70, (
            f"Expected B output value=70 (=7*10), got {b_output.value}"
        )

    async def test_dataflow_combined_with_exec_order_dep(self, state_manager) -> None:
        """DataFlowEdge implies execution ordering — B cannot run until A's output is ready.

        We use a DataFlowEdge (not an explicit ExecOrderDep) to establish the
        dependency. A produces IntOutput(value=3). B consumes it.
        """
        stage_a = _FixedValueStage(value=3, timeout=5.0)
        stage_b = DataConsumerStage(timeout=5.0)

        nodes = {"A": stage_a, "B": stage_b}
        edges = [
            DataFlowEdge.create(source="A", destination="B", input_name="upstream")
        ]

        program = await _run_dag(
            nodes=nodes,
            data_flow_edges=edges,
            execution_order_deps=None,
            state_manager=state_manager,
        )

        assert program.stage_results["A"].status == StageState.COMPLETED
        assert program.stage_results["B"].status == StageState.COMPLETED
        assert DataConsumerStage.received_value == 3


# ---------------------------------------------------------------------------
# Helper stage: produces a fixed IntOutput value
# ---------------------------------------------------------------------------


class _FixedValueStage(Stage):
    """Returns a fixed IntOutput(value=N) for deterministic DataFlowEdge tests."""

    InputsModel = VoidInput
    OutputModel = IntOutput
    cache_handler = NO_CACHE

    def __init__(self, *, value: int, timeout: float = 5.0):
        super().__init__(timeout=timeout)
        self._value = value

    async def compute(self, program: Program) -> IntOutput:
        return IntOutput(value=self._value)
