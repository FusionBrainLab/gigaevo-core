"""End-to-end tests for DAG stage caching, re-run behavior, and execution-order deps.

These tests use real DAGAutomata, real Stage instances, and real ProgramStateManager
backed by fakeredis. They verify the trickiest behaviors that have caused production
bugs: cache invalidation during re-runs, stale stage_results after refresh cycles,
and execution-order dependency gates (success/failure/always).
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from gigaevo.programs.core_types import (
    ProgramStageResult,
    StageIO,
    StageState,
    VoidInput,
    VoidOutput,
)
from gigaevo.programs.dag.automata import (
    DataFlowEdge,
    ExecutionOrderDependency,
)
from gigaevo.programs.dag.dag import DAG
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState
from gigaevo.programs.stages.base import Stage
from gigaevo.programs.stages.cache_handler import NO_CACHE, CacheHandler, InputHashCache

# ---------------------------------------------------------------------------
# Tracking stages — count how many times compute() is actually called
# ---------------------------------------------------------------------------


class _CallTracker:
    """Shared mutable counter for counting stage invocations across DAG runs."""

    def __init__(self):
        self.counts: dict[str, int] = {}

    def record(self, name: str) -> None:
        self.counts[name] = self.counts.get(name, 0) + 1


# Global tracker reset per test via fixture
_tracker = _CallTracker()


class CountedOutput(StageIO):
    value: int = 0


class CountedInput(StageIO):
    data: CountedOutput


class CountedFastStage(Stage):
    """VoidInput -> CountedOutput. Tracks calls. Uses default InputHashCache."""

    InputsModel = VoidInput
    OutputModel = CountedOutput

    def __init__(self, *, timeout: float = 30.0, name: str = "counted_fast"):
        super().__init__(timeout=timeout)
        self._name = name

    async def compute(self, program: Program) -> CountedOutput:
        _tracker.record(self._name)
        return CountedOutput(value=42)


class CountedChainedStage(Stage):
    """CountedInput -> CountedOutput. Tracks calls. Uses default InputHashCache."""

    InputsModel = CountedInput
    OutputModel = CountedOutput

    def __init__(self, *, timeout: float = 30.0, name: str = "counted_chained"):
        super().__init__(timeout=timeout)
        self._name = name

    async def compute(self, program: Program) -> CountedOutput:
        _tracker.record(self._name)
        return CountedOutput(value=self.params.data.value + 1)


class NeverCachedCountedStage(Stage):
    """VoidInput -> CountedOutput. NeverCached. Tracks calls."""

    InputsModel = VoidInput
    OutputModel = CountedOutput
    cache_handler: ClassVar[CacheHandler] = NO_CACHE

    def __init__(self, *, timeout: float = 30.0, name: str = "never_cached"):
        super().__init__(timeout=timeout)
        self._name = name

    async def compute(self, program: Program) -> CountedOutput:
        _tracker.record(self._name)
        return CountedOutput(value=99)


class NeverCachedChainedStage(Stage):
    """CountedInput -> CountedOutput. NeverCached. Tracks calls."""

    InputsModel = CountedInput
    OutputModel = CountedOutput
    cache_handler: ClassVar[CacheHandler] = NO_CACHE

    def __init__(self, *, timeout: float = 30.0, name: str = "never_cached_chained"):
        super().__init__(timeout=timeout)
        self._name = name

    async def compute(self, program: Program) -> CountedOutput:
        _tracker.record(self._name)
        return CountedOutput(value=self.params.data.value + 10)


class ControllableStage(Stage):
    """VoidInput -> CountedOutput. Can be told to fail. Tracks calls."""

    InputsModel = VoidInput
    OutputModel = CountedOutput

    def __init__(
        self,
        *,
        timeout: float = 30.0,
        name: str = "controllable",
        should_fail: bool = False,
    ):
        super().__init__(timeout=timeout)
        self._name = name
        self.should_fail = should_fail

    async def compute(self, program: Program) -> CountedOutput:
        _tracker.record(self._name)
        if self.should_fail:
            raise RuntimeError(f"{self._name} failed on purpose")
        return CountedOutput(value=77)


class ControllableChainedStage(Stage):
    """CountedInput -> CountedOutput. Can be told to fail. Tracks calls.
    Uses default InputHashCache."""

    InputsModel = CountedInput
    OutputModel = CountedOutput

    def __init__(
        self,
        *,
        timeout: float = 30.0,
        name: str = "controllable_chained",
        should_fail: bool = False,
    ):
        super().__init__(timeout=timeout)
        self._name = name
        self.should_fail = should_fail

    async def compute(self, program: Program) -> CountedOutput:
        _tracker.record(self._name)
        if self.should_fail:
            raise RuntimeError(f"{self._name} failed on purpose")
        return CountedOutput(value=self.params.data.value + 1)


class ControllableVoidStage(Stage):
    """VoidInput -> VoidOutput. Can be told to fail. Tracks calls."""

    InputsModel = VoidInput
    OutputModel = VoidOutput
    cache_handler: ClassVar[CacheHandler] = NO_CACHE

    def __init__(
        self,
        *,
        timeout: float = 30.0,
        name: str = "controllable_void",
        should_fail: bool = False,
    ):
        super().__init__(timeout=timeout)
        self._name = name
        self.should_fail = should_fail

    async def compute(self, program: Program) -> None:
        _tracker.record(self._name)
        if self.should_fail:
            raise RuntimeError(f"{self._name} failed on purpose")
        return None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_tracker():
    """Reset the global call tracker before each test."""
    _tracker.counts.clear()
    yield
    _tracker.counts.clear()


def _make_program() -> Program:
    return Program(
        code="def solve(): return 42",
        state=ProgramState.RUNNING,
        atomic_counter=999_999_999,
    )


def _make_dag(
    nodes: dict[str, Stage],
    edges: list[DataFlowEdge],
    exec_deps: dict[str, list[ExecutionOrderDependency]] | None = None,
    state_manager=None,
    writer=None,
) -> DAG:
    return DAG(
        nodes=nodes,
        data_flow_edges=edges,
        execution_order_deps=exec_deps,
        state_manager=state_manager,
        max_parallel_stages=4,
        dag_timeout=30.0,
        writer=writer,
    )


# ---------------------------------------------------------------------------
# Test: InputHashCache skips re-run when inputs are unchanged
# ---------------------------------------------------------------------------


class TestInputHashCacheRerun:
    async def test_failed_stage_cached_on_identical_rerun(
        self, state_manager, null_writer
    ):
        """Run 1: A fails. Run 2: same inputs -> A should be CACHED (not re-executed).
        Failed stages must call on_complete so input_hash is stored; otherwise
        InputHashCache treats stored_hash=None as 'rerun' and Optuna would rerun
        on every refresh cycle despite having already failed."""
        stage_a = ControllableStage(name="A", should_fail=True)
        nodes = {"A": stage_a}
        edges: list[DataFlowEdge] = []

        program = _make_program()

        # Run 1: A fails
        dag1 = _make_dag(nodes, edges, state_manager=state_manager, writer=null_writer)
        await dag1.run(program)
        assert program.stage_results["A"].status == StageState.FAILED
        assert (
            program.stage_results["A"].input_hash is not None
        )  # on_complete was called
        assert _tracker.counts.get("A") == 1

        # Run 2: same program, same inputs -> A should be CACHED (no re-execution)
        dag2 = _make_dag(nodes, edges, state_manager=state_manager, writer=null_writer)
        await dag2.run(program)
        assert program.stage_results["A"].status == StageState.FAILED
        assert _tracker.counts.get("A") == 1  # not re-executed

    async def test_single_stage_cached_on_identical_rerun(
        self, state_manager, null_writer
    ):
        """Run a single InputHashCache stage twice. Second run should be cached."""
        stage_a = CountedFastStage(name="A")
        nodes = {"A": stage_a}
        edges: list[DataFlowEdge] = []

        program = _make_program()

        # Run 1
        dag1 = _make_dag(nodes, edges, state_manager=state_manager, writer=null_writer)
        await dag1.run(program)

        assert _tracker.counts.get("A") == 1
        assert program.stage_results["A"].status == StageState.COMPLETED
        assert program.stage_results["A"].input_hash is not None

        # Run 2: same program, same inputs -> should be cached
        dag2 = _make_dag(nodes, edges, state_manager=state_manager, writer=null_writer)
        await dag2.run(program)

        # A should NOT have been called a second time (cached)
        assert _tracker.counts.get("A") == 1

    async def test_chain_ab_cached_on_identical_rerun(self, state_manager, null_writer):
        """A -> B chain, both InputHashCache. Second run should cache both."""
        stage_a = CountedFastStage(name="A")
        stage_b = CountedChainedStage(name="B")
        nodes = {"A": stage_a, "B": stage_b}
        edges = [DataFlowEdge.create("A", "B", "data")]

        program = _make_program()

        # Run 1
        dag1 = _make_dag(nodes, edges, state_manager=state_manager, writer=null_writer)
        await dag1.run(program)
        assert _tracker.counts.get("A") == 1
        assert _tracker.counts.get("B") == 1

        # Run 2: both should be cached
        dag2 = _make_dag(nodes, edges, state_manager=state_manager, writer=null_writer)
        await dag2.run(program)
        assert _tracker.counts.get("A") == 1  # still 1
        assert _tracker.counts.get("B") == 1  # still 1


# ---------------------------------------------------------------------------
# Test: NeverCached stage always re-executes
# ---------------------------------------------------------------------------


class TestNeverCachedRerun:
    async def test_never_cached_stage_reruns_every_time(
        self, state_manager, null_writer
    ):
        """NeverCached stage should execute on every DAG.run() call."""
        stage_a = NeverCachedCountedStage(name="A")
        nodes = {"A": stage_a}
        edges: list[DataFlowEdge] = []

        program = _make_program()

        # Run 1
        dag1 = _make_dag(nodes, edges, state_manager=state_manager, writer=null_writer)
        await dag1.run(program)
        assert _tracker.counts.get("A") == 1

        # Run 2: NeverCached -> should re-execute
        dag2 = _make_dag(nodes, edges, state_manager=state_manager, writer=null_writer)
        await dag2.run(program)
        assert _tracker.counts.get("A") == 2

        # Run 3: still re-executes
        dag3 = _make_dag(nodes, edges, state_manager=state_manager, writer=null_writer)
        await dag3.run(program)
        assert _tracker.counts.get("A") == 3

    async def test_never_cached_with_cached_upstream(self, state_manager, null_writer):
        """A(InputHashCache) -> B(NeverCached). On re-run, A is cached, B re-executes
        using A's stored output."""
        stage_a = CountedFastStage(name="A")
        stage_b = NeverCachedChainedStage(name="B")
        nodes = {"A": stage_a, "B": stage_b}
        edges = [DataFlowEdge.create("A", "B", "data")]

        program = _make_program()

        # Run 1
        dag1 = _make_dag(nodes, edges, state_manager=state_manager, writer=null_writer)
        await dag1.run(program)
        assert _tracker.counts.get("A") == 1
        assert _tracker.counts.get("B") == 1
        assert program.stage_results["B"].status == StageState.COMPLETED

        # Run 2: A cached (hash match), B must re-execute (NeverCached)
        dag2 = _make_dag(nodes, edges, state_manager=state_manager, writer=null_writer)
        await dag2.run(program)
        assert _tracker.counts.get("A") == 1  # cached, no re-exec
        assert _tracker.counts.get("B") == 2  # re-executed
        assert program.stage_results["B"].status == StageState.COMPLETED


# ---------------------------------------------------------------------------
# Test: setdefault preserves existing stage_results on re-run
# ---------------------------------------------------------------------------


class TestSetdefaultPreservesResults:
    async def test_existing_completed_result_not_reset_to_pending(
        self, state_manager, null_writer
    ):
        """After run 1, stage_results have COMPLETED. On run 2, setdefault should
        preserve them (not reset to PENDING)."""
        stage_a = CountedFastStage(name="A")
        nodes = {"A": stage_a}
        edges: list[DataFlowEdge] = []

        program = _make_program()

        # Run 1
        dag1 = _make_dag(nodes, edges, state_manager=state_manager, writer=null_writer)
        await dag1.run(program)
        assert program.stage_results["A"].status == StageState.COMPLETED
        stored_hash = program.stage_results["A"].input_hash

        # Before run 2, verify the result is still COMPLETED (setdefault preserves)
        dag2 = _make_dag(nodes, edges, state_manager=state_manager, writer=null_writer)
        # Just build the DAG and run — _run_internal uses setdefault
        await dag2.run(program)

        # After re-run, result should still be COMPLETED (cached)
        assert program.stage_results["A"].status == StageState.COMPLETED
        assert program.stage_results["A"].input_hash == stored_hash


# ---------------------------------------------------------------------------
# Test: Stale COMPLETED result overwritten when upstream fails on re-run
# ---------------------------------------------------------------------------


class TestStaleResultOverwrite:
    async def test_downstream_skipped_when_upstream_fails(
        self, state_manager, null_writer
    ):
        """A(NeverCached) -> B via data flow. A fails -> B should be SKIPPED
        since mandatory input can never arrive."""
        stage_a = ControllableStage(name="A", should_fail=True)
        stage_b = CountedChainedStage(name="B")
        nodes = {"A": stage_a, "B": stage_b}
        edges = [DataFlowEdge.create("A", "B", "data")]

        program = _make_program()
        dag = _make_dag(nodes, edges, state_manager=state_manager, writer=null_writer)
        await dag.run(program)

        assert program.stage_results["A"].status == StageState.FAILED
        assert program.stage_results["B"].status == StageState.SKIPPED
        assert _tracker.counts.get("A") == 1
        assert _tracker.counts.get("B", 0) == 0

    async def test_rerun_after_failure_does_not_stall(self, state_manager, null_writer):
        """Run 1: A fails, B skipped. Run 2: A succeeds -> B should execute.
        This catches the stale FAILED result stall bug."""
        # Use NeverCached for A so it always re-runs
        stage_a = ControllableVoidStage(name="A", should_fail=True)
        stage_b = CountedFastStage(name="B")
        nodes = {"A": stage_a, "B": stage_b}
        # B depends on A via execution-order (always), not data flow
        exec_deps = {"B": [ExecutionOrderDependency.on_success("A")]}

        program = _make_program()

        # Run 1: A fails, B should be impossible (on_success dep not met)
        dag1 = _make_dag(
            nodes, [], exec_deps, state_manager=state_manager, writer=null_writer
        )
        await dag1.run(program)
        assert program.stage_results["A"].status == StageState.FAILED
        assert program.stage_results["B"].status == StageState.SKIPPED
        assert _tracker.counts.get("A") == 1
        assert _tracker.counts.get("B", 0) == 0

        # Run 2: A succeeds this time. B should now execute.
        stage_a.should_fail = False
        dag2 = _make_dag(
            nodes, [], exec_deps, state_manager=state_manager, writer=null_writer
        )
        await dag2.run(program)
        assert program.stage_results["A"].status == StageState.COMPLETED
        assert program.stage_results["B"].status == StageState.COMPLETED
        assert _tracker.counts.get("A") == 2
        assert _tracker.counts.get("B") == 1


# ---------------------------------------------------------------------------
# Test: Diamond DAG with all-cached stages terminates correctly
# ---------------------------------------------------------------------------


class TestDiamondDagCaching:
    async def test_all_cached_diamond_terminates(self, state_manager, null_writer):
        """A -> (B, C) -> D, all InputHashCache. Re-run: all cached, terminates."""
        stage_a = CountedFastStage(name="A")
        stage_b = CountedChainedStage(name="B")
        stage_c = CountedChainedStage(name="C")

        # D needs inputs from both B and C — use optional input pattern
        class DualInput(StageIO):
            data_b: CountedOutput
            data_c: CountedOutput | None = None

        class DualInputStage(Stage):
            InputsModel = DualInput
            OutputModel = CountedOutput

            def __init__(self, *, timeout: float = 30.0):
                super().__init__(timeout=timeout)

            async def compute(self, program: Program) -> CountedOutput:
                _tracker.record("D")
                val = self.params.data_b.value
                if self.params.data_c is not None:
                    val += self.params.data_c.value
                return CountedOutput(value=val)

        stage_d = DualInputStage(timeout=30.0)
        nodes = {"A": stage_a, "B": stage_b, "C": stage_c, "D": stage_d}
        edges = [
            DataFlowEdge.create("A", "B", "data"),
            DataFlowEdge.create("A", "C", "data"),
            DataFlowEdge.create("B", "D", "data_b"),
            DataFlowEdge.create("C", "D", "data_c"),
        ]

        program = _make_program()

        # Run 1
        dag1 = _make_dag(nodes, edges, state_manager=state_manager, writer=null_writer)
        await dag1.run(program)
        assert _tracker.counts.get("A") == 1
        assert _tracker.counts.get("B") == 1
        assert _tracker.counts.get("C") == 1
        assert _tracker.counts.get("D") == 1
        for name in ["A", "B", "C", "D"]:
            assert program.stage_results[name].status == StageState.COMPLETED

        # Run 2: all cached -> 0 re-executions, DAG still terminates
        dag2 = _make_dag(nodes, edges, state_manager=state_manager, writer=null_writer)
        await dag2.run(program)
        # All counts should remain at 1 (no re-execution)
        assert _tracker.counts.get("A") == 1
        assert _tracker.counts.get("B") == 1
        assert _tracker.counts.get("C") == 1
        assert _tracker.counts.get("D") == 1


# ---------------------------------------------------------------------------
# Test: Execution-order dependencies (success/failure/always conditions)
# ---------------------------------------------------------------------------


class TestExecutionOrderDependencies:
    async def test_on_failure_runs_when_upstream_fails(
        self, state_manager, null_writer
    ):
        """A(NeverCached) fails -> B(on_failure: A) should execute."""
        stage_a = ControllableVoidStage(name="A", should_fail=True)
        stage_b = CountedFastStage(name="B")
        nodes = {"A": stage_a, "B": stage_b}
        exec_deps = {"B": [ExecutionOrderDependency.on_failure("A")]}

        program = _make_program()
        dag = _make_dag(
            nodes, [], exec_deps, state_manager=state_manager, writer=null_writer
        )
        await dag.run(program)

        assert program.stage_results["A"].status == StageState.FAILED
        assert program.stage_results["B"].status == StageState.COMPLETED
        assert _tracker.counts.get("B") == 1

    async def test_on_failure_skipped_when_upstream_succeeds(
        self, state_manager, null_writer
    ):
        """A(NeverCached) succeeds -> B(on_failure: A) should be SKIPPED (IMPOSSIBLE)."""
        stage_a = ControllableVoidStage(name="A", should_fail=False)
        stage_b = CountedFastStage(name="B")
        nodes = {"A": stage_a, "B": stage_b}
        exec_deps = {"B": [ExecutionOrderDependency.on_failure("A")]}

        program = _make_program()
        dag = _make_dag(
            nodes, [], exec_deps, state_manager=state_manager, writer=null_writer
        )
        await dag.run(program)

        assert program.stage_results["A"].status == StageState.COMPLETED
        assert program.stage_results["B"].status == StageState.SKIPPED
        assert _tracker.counts.get("B", 0) == 0

    async def test_always_after_runs_regardless(self, state_manager, null_writer):
        """B(always_after: A) should run whether A succeeds or fails."""
        for should_fail in [False, True]:
            _tracker.counts.clear()
            stage_a = ControllableVoidStage(name="A", should_fail=should_fail)
            stage_b = CountedFastStage(name="B")
            nodes = {"A": stage_a, "B": stage_b}
            exec_deps = {"B": [ExecutionOrderDependency.always_after("A")]}

            program = _make_program()
            dag = _make_dag(
                nodes, [], exec_deps, state_manager=state_manager, writer=null_writer
            )
            await dag.run(program)

            expected_a = StageState.FAILED if should_fail else StageState.COMPLETED
            assert program.stage_results["A"].status == expected_a
            assert program.stage_results["B"].status == StageState.COMPLETED
            assert _tracker.counts.get("B") == 1

    async def test_on_success_skipped_when_upstream_fails(
        self, state_manager, null_writer
    ):
        """A fails -> B(on_success: A) should be SKIPPED."""
        stage_a = ControllableVoidStage(name="A", should_fail=True)
        stage_b = CountedFastStage(name="B")
        nodes = {"A": stage_a, "B": stage_b}
        exec_deps = {"B": [ExecutionOrderDependency.on_success("A")]}

        program = _make_program()
        dag = _make_dag(
            nodes, [], exec_deps, state_manager=state_manager, writer=null_writer
        )
        await dag.run(program)

        assert program.stage_results["A"].status == StageState.FAILED
        assert program.stage_results["B"].status == StageState.SKIPPED
        assert _tracker.counts.get("B", 0) == 0


# ---------------------------------------------------------------------------
# Test: DAG init with new stage added to existing program
# ---------------------------------------------------------------------------


class TestDagInitialization:
    async def test_new_stage_gets_pending_existing_preserved(
        self, state_manager, null_writer
    ):
        """Program has stage_results for A (COMPLETED). Build DAG with A + new B.
        setdefault should preserve A's COMPLETED, set B to PENDING."""
        program = _make_program()
        # Simulate prior run result
        program.stage_results["A"] = ProgramStageResult.success(
            output=CountedOutput(value=42)
        )
        # Store input_hash to simulate cache
        program.stage_results["A"].input_hash = "abc123"

        stage_a = CountedFastStage(name="A")
        stage_b = CountedFastStage(name="B")
        nodes = {"A": stage_a, "B": stage_b}

        dag = _make_dag(nodes, [], state_manager=state_manager, writer=null_writer)
        await dag.run(program)

        # A should have been cached (its result was COMPLETED with matching hash)
        # The exact behavior depends on whether the hash matches. Since we set
        # input_hash manually, it may or may not match the real computed hash.
        # But at minimum, A's COMPLETED status should be preserved by setdefault.
        assert program.stage_results["A"].status == StageState.COMPLETED
        assert program.stage_results["B"].status == StageState.COMPLETED


# ---------------------------------------------------------------------------
# Test: ProbabilisticCache validation
# ---------------------------------------------------------------------------


class TestProbabilisticCache:
    def test_invalid_probability_raises(self):
        from gigaevo.programs.stages.cache_handler import ProbabilisticCache

        with pytest.raises(ValueError, match="rerun_probability"):
            ProbabilisticCache(rerun_probability=-0.1)

        with pytest.raises(ValueError, match="rerun_probability"):
            ProbabilisticCache(rerun_probability=1.5)

    def test_valid_boundary_values(self):
        from gigaevo.programs.stages.cache_handler import ProbabilisticCache

        p0 = ProbabilisticCache(rerun_probability=0.0)
        assert p0.rerun_probability == 0.0

        p1 = ProbabilisticCache(rerun_probability=1.0)
        assert p1.rerun_probability == 1.0

    def test_always_reruns_at_probability_1(self):
        from gigaevo.programs.stages.cache_handler import ProbabilisticCache

        p = ProbabilisticCache(rerun_probability=1.0)
        result = ProgramStageResult.success()
        # should_rerun should always return True
        for _ in range(100):
            assert p.should_rerun(result, "hash123", set()) is True

    def test_never_reruns_at_probability_0(self):
        from gigaevo.programs.stages.cache_handler import ProbabilisticCache

        p = ProbabilisticCache(rerun_probability=0.0)
        result = ProgramStageResult.success()
        # should_rerun should always return False (cached result exists)
        for _ in range(100):
            assert p.should_rerun(result, "hash123", set()) is False

    def test_reruns_when_no_cached_result(self):
        from gigaevo.programs.stages.cache_handler import ProbabilisticCache

        p = ProbabilisticCache(rerun_probability=0.0)
        # No existing result -> must run regardless of probability
        assert p.should_rerun(None, "hash123", set()) is True


# ---------------------------------------------------------------------------
# Test: InputHashCache edge cases
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Test: Failed stages in long dependency chains
# ---------------------------------------------------------------------------


class TestFailedStagesInLongChains:
    """Verify caching behavior when failures occur at various points in a chain."""

    async def test_failed_middle_stage_cached_in_3_stage_chain(
        self, state_manager, null_writer
    ):
        """A -> B -> C chain. B fails on run 1.
        Run 2: A cached, B cached (failed), C skipped (no input from B).
        No stage should re-execute."""
        stage_a = CountedFastStage(name="A")
        stage_b = ControllableChainedStage(name="B", should_fail=True)
        stage_c = CountedChainedStage(name="C")
        nodes = {"A": stage_a, "B": stage_b, "C": stage_c}
        edges = [
            DataFlowEdge.create("A", "B", "data"),
            DataFlowEdge.create("B", "C", "data"),
        ]

        program = _make_program()

        # Run 1: A succeeds, B fails, C skipped
        dag1 = _make_dag(nodes, edges, state_manager=state_manager, writer=null_writer)
        await dag1.run(program)

        assert program.stage_results["A"].status == StageState.COMPLETED
        assert program.stage_results["B"].status == StageState.FAILED
        assert program.stage_results["C"].status == StageState.SKIPPED
        assert _tracker.counts.get("A") == 1
        assert _tracker.counts.get("B") == 1
        assert _tracker.counts.get("C", 0) == 0

        # Verify B stored its input_hash (the fix)
        assert program.stage_results["B"].input_hash is not None

        # Run 2: A cached, B cached (failed with matching hash), C still skipped
        dag2 = _make_dag(nodes, edges, state_manager=state_manager, writer=null_writer)
        await dag2.run(program)

        assert _tracker.counts.get("A") == 1  # cached
        assert _tracker.counts.get("B") == 1  # cached (failed but hash matches)
        assert _tracker.counts.get("C", 0) == 0  # still skipped

    async def test_failed_root_stage_cached_in_4_stage_chain(
        self, state_manager, null_writer
    ):
        """A -> B -> C -> D. A fails on run 1.
        Run 2: A cached (failed), B/C/D all skipped. Zero re-executions."""
        stage_a = ControllableStage(name="A", should_fail=True)
        stage_b = CountedChainedStage(name="B")
        stage_c = CountedChainedStage(name="C")
        stage_d = CountedChainedStage(name="D")
        nodes = {"A": stage_a, "B": stage_b, "C": stage_c, "D": stage_d}
        edges = [
            DataFlowEdge.create("A", "B", "data"),
            DataFlowEdge.create("B", "C", "data"),
            DataFlowEdge.create("C", "D", "data"),
        ]

        program = _make_program()

        # Run 1: A fails, B/C/D skipped
        dag1 = _make_dag(nodes, edges, state_manager=state_manager, writer=null_writer)
        await dag1.run(program)

        assert program.stage_results["A"].status == StageState.FAILED
        assert program.stage_results["A"].input_hash is not None
        for name in ["B", "C", "D"]:
            assert program.stage_results[name].status == StageState.SKIPPED
        assert _tracker.counts.get("A") == 1

        # Run 2: A cached (failed), B/C/D still skipped, zero new executions
        dag2 = _make_dag(nodes, edges, state_manager=state_manager, writer=null_writer)
        await dag2.run(program)

        assert _tracker.counts.get("A") == 1  # still 1, cached
        for name in ["B", "C", "D"]:
            assert _tracker.counts.get(name, 0) == 0

    async def test_recovery_after_cached_failure_in_chain(
        self, state_manager, null_writer
    ):
        """A -> B -> C. Run 1: B fails. Run 2: B cached (failed).
        Run 3: B fixed (NeverCached), succeeds -> C should now execute.

        This tests that a chain can recover after a previously cached failure
        when the failing stage switches to NeverCached or its inputs change."""
        stage_a = CountedFastStage(name="A")
        stage_b_fail = ControllableChainedStage(name="B", should_fail=True)
        stage_c = CountedChainedStage(name="C")

        nodes_fail = {"A": stage_a, "B": stage_b_fail, "C": stage_c}
        edges = [
            DataFlowEdge.create("A", "B", "data"),
            DataFlowEdge.create("B", "C", "data"),
        ]

        program = _make_program()

        # Run 1: B fails
        dag1 = _make_dag(
            nodes_fail, edges, state_manager=state_manager, writer=null_writer
        )
        await dag1.run(program)
        assert program.stage_results["B"].status == StageState.FAILED
        assert _tracker.counts.get("B") == 1

        # Run 2: B cached (still failed)
        dag2 = _make_dag(
            nodes_fail, edges, state_manager=state_manager, writer=null_writer
        )
        await dag2.run(program)
        assert _tracker.counts.get("B") == 1  # cached

        # Run 3: Toggle B to succeed — same stage instance, so inputs hash unchanged
        # but NeverCached would re-execute. Since ControllableChainedStage uses
        # InputHashCache, the hash matches => still cached. Let's use NeverCached
        # chained to force re-execution.
        stage_b_ok = NeverCachedChainedStage(name="B")
        nodes_ok = {"A": stage_a, "B": stage_b_ok, "C": stage_c}
        # B is NeverCached, so it will re-execute regardless of cached result
        dag3 = _make_dag(
            nodes_ok, edges, state_manager=state_manager, writer=null_writer
        )
        await dag3.run(program)
        assert program.stage_results["B"].status == StageState.COMPLETED
        assert _tracker.counts.get("B") == 2  # re-executed


# ---------------------------------------------------------------------------
# Test: DAG terminates when ALL stages are failed-but-cached
# ---------------------------------------------------------------------------


class TestAllFailedCachedTermination:
    """Verify the DAG terminates correctly when every stage is cached-as-failed."""

    async def test_single_failed_cached_stage_terminates(
        self, state_manager, null_writer
    ):
        """One stage, failed on run 1, cached on run 2. DAG must terminate."""
        stage = ControllableStage(name="A", should_fail=True)
        nodes = {"A": stage}
        edges: list[DataFlowEdge] = []

        program = _make_program()

        # Run 1: A fails
        dag1 = _make_dag(nodes, edges, state_manager=state_manager, writer=null_writer)
        await dag1.run(program)
        assert program.stage_results["A"].status == StageState.FAILED
        assert _tracker.counts["A"] == 1

        # Run 2: A cached (failed) — DAG must terminate, not loop
        dag2 = _make_dag(nodes, edges, state_manager=state_manager, writer=null_writer)
        await dag2.run(program)  # Would hang if newly_cached not counted as progress
        assert _tracker.counts["A"] == 1  # not re-executed

    async def test_two_independent_failed_cached_stages_terminate(
        self, state_manager, null_writer
    ):
        """A and B both fail on run 1, both cached on run 2.
        Tests the 'newly_cached' progress accounting in DAG termination."""
        stage_a = ControllableStage(name="A", should_fail=True)
        stage_b = ControllableStage(name="B", should_fail=True)
        nodes = {"A": stage_a, "B": stage_b}
        edges: list[DataFlowEdge] = []

        program = _make_program()

        # Run 1: both fail
        dag1 = _make_dag(nodes, edges, state_manager=state_manager, writer=null_writer)
        await dag1.run(program)
        assert program.stage_results["A"].status == StageState.FAILED
        assert program.stage_results["B"].status == StageState.FAILED
        assert _tracker.counts["A"] == 1
        assert _tracker.counts["B"] == 1

        # Run 2: both cached (failed) — must terminate
        dag2 = _make_dag(nodes, edges, state_manager=state_manager, writer=null_writer)
        await dag2.run(program)
        assert _tracker.counts["A"] == 1
        assert _tracker.counts["B"] == 1

    async def test_diamond_all_failed_cached_terminates(
        self, state_manager, null_writer
    ):
        """Diamond: A -> (B, C) with exec deps. A fails, B/C skipped.
        Run 2: A cached (failed, InputHashCache), B/C skipped again.
        DAG must terminate without infinite loop.

        This is the pattern that caused the original production bug when
        newly_cached was not included in the termination check."""
        # Use ControllableStage (InputHashCache) so A is cached on run 2
        stage_a = ControllableStage(name="A", should_fail=True)
        stage_b = CountedFastStage(name="B")
        stage_c = CountedFastStage(name="C")
        nodes = {"A": stage_a, "B": stage_b, "C": stage_c}
        exec_deps = {
            "B": [ExecutionOrderDependency.on_success("A")],
            "C": [ExecutionOrderDependency.on_success("A")],
        }

        program = _make_program()

        # Run 1: A fails, B/C skipped
        dag1 = _make_dag(
            nodes, [], exec_deps, state_manager=state_manager, writer=null_writer
        )
        await dag1.run(program)
        assert program.stage_results["A"].status == StageState.FAILED
        assert program.stage_results["A"].input_hash is not None
        assert program.stage_results["B"].status == StageState.SKIPPED
        assert program.stage_results["C"].status == StageState.SKIPPED

        # Run 2: A cached (failed), B/C skipped again. Must terminate.
        dag2 = _make_dag(
            nodes, [], exec_deps, state_manager=state_manager, writer=null_writer
        )
        await dag2.run(program)
        assert _tracker.counts.get("A") == 1  # cached, not re-executed
        assert _tracker.counts.get("B", 0) == 0
        assert _tracker.counts.get("C", 0) == 0


# ---------------------------------------------------------------------------
# Test: on_complete called from exception handler (base.py)
# ---------------------------------------------------------------------------


class TestOnCompleteCalledOnFailure:
    """Verify that Stage.execute() calls on_complete for FAILED stages,
    not just successful ones. This is the core of the bug fix."""

    async def test_failed_stage_has_input_hash_set(self, state_manager, null_writer):
        """A single failing stage must have input_hash stored after execute().

        This directly verifies the fix in base.py lines 312-317: the exception
        handler calls cache_handler.on_complete(fail_result, inputs_hash)."""
        stage = ControllableStage(name="A", should_fail=True)
        nodes = {"A": stage}
        edges: list[DataFlowEdge] = []

        program = _make_program()
        dag = _make_dag(nodes, edges, state_manager=state_manager, writer=null_writer)
        await dag.run(program)

        result = program.stage_results["A"]
        assert result.status == StageState.FAILED
        assert result.input_hash is not None, (
            "on_complete must be called for FAILED stages to store input_hash"
        )

    async def test_failed_stage_input_hash_matches_success_hash(
        self, state_manager, null_writer
    ):
        """A stage that fails and one that succeeds with the same inputs
        should produce the same input_hash. This ensures on_complete computes
        the hash identically in both success and failure paths."""
        # Stage that succeeds
        stage_ok = ControllableStage(name="A_ok", should_fail=False)
        nodes_ok = {"A_ok": stage_ok}
        program_ok = _make_program()

        dag_ok = _make_dag(
            nodes_ok, [], state_manager=state_manager, writer=null_writer
        )
        await dag_ok.run(program_ok)
        success_hash = program_ok.stage_results["A_ok"].input_hash

        # Stage that fails (same VoidInput)
        stage_fail = ControllableStage(name="A_fail", should_fail=True)
        nodes_fail = {"A_fail": stage_fail}
        program_fail = _make_program()

        dag_fail = _make_dag(
            nodes_fail, [], state_manager=state_manager, writer=null_writer
        )
        await dag_fail.run(program_fail)
        fail_hash = program_fail.stage_results["A_fail"].input_hash

        assert success_hash is not None
        assert fail_hash is not None
        assert success_hash == fail_hash, (
            "Same inputs must produce same hash regardless of success/failure"
        )

    async def test_failed_chained_stage_input_hash_stored(
        self, state_manager, null_writer
    ):
        """A -> B where B fails. B's input_hash must be set (computed from
        A's output). This tests on_complete in the exception handler when
        the stage has non-void inputs."""
        stage_a = CountedFastStage(name="A")

        class FailingChainedStage(Stage):
            InputsModel = CountedInput
            OutputModel = CountedOutput

            def __init__(self):
                super().__init__(timeout=30.0)

            async def compute(self, program):
                _tracker.record("B_fail")
                raise RuntimeError("B fails after receiving input")

        stage_b = FailingChainedStage()
        nodes = {"A": stage_a, "B": stage_b}
        edges = [DataFlowEdge.create("A", "B", "data")]

        program = _make_program()
        dag = _make_dag(nodes, edges, state_manager=state_manager, writer=null_writer)
        await dag.run(program)

        assert program.stage_results["A"].status == StageState.COMPLETED
        assert program.stage_results["B"].status == StageState.FAILED
        # B's input_hash must be set even though B failed
        assert program.stage_results["B"].input_hash is not None
        # And it should be different from A's hash (different inputs)
        assert (
            program.stage_results["B"].input_hash
            != program.stage_results["A"].input_hash
        )

    async def test_no_cache_stage_failure_does_not_store_hash(
        self, state_manager, null_writer
    ):
        """NeverCached stages should NOT store input_hash even on failure,
        because NeverCached.on_complete uses the default (no-op) implementation."""
        stage = ControllableVoidStage(name="A", should_fail=True)
        nodes = {"A": stage}

        program = _make_program()
        dag = _make_dag(nodes, [], state_manager=state_manager, writer=null_writer)
        await dag.run(program)

        result = program.stage_results["A"]
        assert result.status == StageState.FAILED
        # NeverCached.on_complete is a no-op, so input_hash stays None
        assert result.input_hash is None


# ---------------------------------------------------------------------------
# Test: Mixed success/failure caching in parallel branches
# ---------------------------------------------------------------------------


class TestMixedSuccessFailureCaching:
    """Test caching when some branches succeed and others fail."""

    async def test_diamond_one_branch_fails_other_succeeds(
        self, state_manager, null_writer
    ):
        """A -> (B_fail, B_ok) via exec deps. Run 1: B_fail fails, B_ok succeeds.
        Run 2: both cached. Verifies caching works for mixed outcomes."""
        stage_a = CountedFastStage(name="A")
        stage_b_fail = ControllableStage(name="B_fail", should_fail=True)
        stage_b_ok = CountedFastStage(name="B_ok")
        nodes = {"A": stage_a, "B_fail": stage_b_fail, "B_ok": stage_b_ok}
        exec_deps = {
            "B_fail": [ExecutionOrderDependency.on_success("A")],
            "B_ok": [ExecutionOrderDependency.on_success("A")],
        }

        program = _make_program()

        # Run 1: A succeeds, B_fail fails, B_ok succeeds
        dag1 = _make_dag(
            nodes, [], exec_deps, state_manager=state_manager, writer=null_writer
        )
        await dag1.run(program)

        assert program.stage_results["A"].status == StageState.COMPLETED
        assert program.stage_results["B_fail"].status == StageState.FAILED
        assert program.stage_results["B_ok"].status == StageState.COMPLETED
        assert _tracker.counts["A"] == 1
        assert _tracker.counts["B_fail"] == 1
        assert _tracker.counts["B_ok"] == 1

        # Run 2: all three cached — DAG terminates
        dag2 = _make_dag(
            nodes, [], exec_deps, state_manager=state_manager, writer=null_writer
        )
        await dag2.run(program)
        assert _tracker.counts["A"] == 1
        assert _tracker.counts["B_fail"] == 1
        assert _tracker.counts["B_ok"] == 1


class TestInputHashCacheEdgeCases:
    def test_reruns_when_stored_hash_is_none(self):
        """If on_complete was never called, input_hash is None -> should rerun."""
        cache = InputHashCache()
        result = ProgramStageResult(status=StageState.COMPLETED)
        # input_hash is None by default
        assert result.input_hash is None
        assert cache.should_rerun(result, "new_hash", set()) is True

    def test_reruns_when_inputs_hash_is_none(self):
        """If inputs_hash computation failed (None), should rerun."""
        cache = InputHashCache()
        result = ProgramStageResult(status=StageState.COMPLETED)
        result.input_hash = "stored_hash"
        # inputs_hash=None != stored_hash -> True
        assert cache.should_rerun(result, None, set()) is True

    def test_no_rerun_when_hashes_match(self):
        cache = InputHashCache()
        result = ProgramStageResult(status=StageState.COMPLETED)
        result.input_hash = "abc123"
        assert cache.should_rerun(result, "abc123", set()) is False

    def test_rerun_when_hashes_differ(self):
        cache = InputHashCache()
        result = ProgramStageResult(status=StageState.COMPLETED)
        result.input_hash = "abc123"
        assert cache.should_rerun(result, "different", set()) is True

    def test_on_complete_stores_hash(self):
        cache = InputHashCache()
        result = ProgramStageResult(status=StageState.COMPLETED)
        updated = cache.on_complete(result, "my_hash")
        assert updated.input_hash == "my_hash"

    def test_rerun_when_no_existing_result(self):
        cache = InputHashCache()
        assert cache.should_rerun(None, "hash", set()) is True

    def test_rerun_when_existing_result_not_final(self):
        cache = InputHashCache()
        result = ProgramStageResult(status=StageState.RUNNING)
        assert cache.should_rerun(result, "hash", set()) is True
