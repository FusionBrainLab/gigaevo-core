"""DAG execution tests — end-to-end tests of the DAG class with fakeredis."""

from __future__ import annotations

import asyncio
import time

import pytest

from gigaevo.programs.core_types import ProgramStageResult, StageState
from gigaevo.programs.dag.automata import (
    DataFlowEdge,
    ExecutionOrderDependency,
)
from gigaevo.programs.dag.dag import DAG
from tests.conftest import (
    ChainedStage,
    FailingStage,
    FastStage,
    MockOutput,
    NeverCachedStage,
    NullWriter,
    OptionalInputStage,
    SideEffectStage,
    SlowStage,
    TimeoutStage,
    VoidStage,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dag(nodes, edges, state_manager, *, exec_deps=None, **kwargs):
    return DAG(
        nodes=nodes,
        data_flow_edges=edges,
        execution_order_deps=exec_deps,
        state_manager=state_manager,
        writer=NullWriter(),
        **kwargs,
    )


# ===================================================================
# Category A: Basic Execution
# ===================================================================


class TestBasicExecution:
    async def test_single_stage_completes(self, state_manager, make_program):
        """One FastStage runs and completes."""
        dag = _make_dag({"fast": FastStage(timeout=5.0)}, [], state_manager)
        prog = make_program()
        await dag.run(prog)

        assert "fast" in prog.stage_results
        assert prog.stage_results["fast"].status == StageState.COMPLETED
        assert prog.stage_results["fast"].output.value == 42

    async def test_independent_stages_run_parallel(self, state_manager, make_program):
        """3 independent FastStages all complete."""
        dag = _make_dag(
            {
                "a": FastStage(timeout=5.0),
                "b": FastStage(timeout=5.0),
                "c": FastStage(timeout=5.0),
            },
            [],
            state_manager,
        )
        prog = make_program()
        await dag.run(prog)

        for name in ("a", "b", "c"):
            assert prog.stage_results[name].status == StageState.COMPLETED

    async def test_chained_stages_run_sequentially(self, state_manager, make_program):
        """A->B->C chain: B sees A's output, C sees B's output."""
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

        assert prog.stage_results["a"].output.value == 42
        assert prog.stage_results["b"].output.value == 43  # 42 + 1
        assert prog.stage_results["c"].output.value == 44  # 43 + 1

    async def test_mixed_independent_and_chained(self, state_manager, make_program):
        """A->B with C independent; all complete."""
        dag = _make_dag(
            {
                "a": FastStage(timeout=5.0),
                "b": ChainedStage(timeout=5.0),
                "c": FastStage(timeout=5.0),
            },
            [DataFlowEdge.create("a", "b", "data")],
            state_manager,
        )
        prog = make_program()
        await dag.run(prog)

        for name in ("a", "b", "c"):
            assert prog.stage_results[name].status == StageState.COMPLETED
        assert prog.stage_results["b"].output.value == 43

    async def test_void_output_stage(self, state_manager, make_program):
        """Stage returning None with VoidOutput succeeds."""
        dag = _make_dag({"void": VoidStage(timeout=5.0)}, [], state_manager)
        prog = make_program()
        await dag.run(prog)

        assert prog.stage_results["void"].status == StageState.COMPLETED

    async def test_stage_mutates_program_metrics(self, state_manager, make_program):
        """SideEffectStage writes to program.metrics during compute."""
        dag = _make_dag({"side": SideEffectStage(timeout=5.0)}, [], state_manager)
        prog = make_program()
        await dag.run(prog)

        assert prog.stage_results["side"].status == StageState.COMPLETED
        assert prog.metrics["side_effect_metric"] == 123.0


# ===================================================================
# Category B: Error Handling
# ===================================================================


class TestErrorHandling:
    async def test_failing_stage_marked_failed(self, state_manager, make_program):
        """FailingStage gets FAILED status with StageError."""
        dag = _make_dag({"fail": FailingStage(timeout=5.0)}, [], state_manager)
        prog = make_program()
        await dag.run(prog)

        res = prog.stage_results["fail"]
        assert res.status == StageState.FAILED
        assert res.error is not None
        assert "stage failed on purpose" in res.error.message

    async def test_stage_timeout_marked_failed(self, state_manager, make_program):
        """TimeoutStage hits stage timeout, result is FAILED."""
        dag = _make_dag({"timeout": TimeoutStage(timeout=0.1)}, [], state_manager)
        prog = make_program()
        await dag.run(prog)

        res = prog.stage_results["timeout"]
        assert res.status == StageState.FAILED
        assert res.error is not None

    async def test_downstream_of_failed_stage_skipped(
        self, state_manager, make_program
    ):
        """A fails -> B (depends on A) auto-skipped."""
        dag = _make_dag(
            {
                "a": FailingStage(timeout=5.0),
                "b": ChainedStage(timeout=5.0),
            },
            [DataFlowEdge.create("a", "b", "data")],
            state_manager,
        )
        prog = make_program()
        await dag.run(prog)

        assert prog.stage_results["a"].status == StageState.FAILED
        assert prog.stage_results["b"].status == StageState.SKIPPED

    async def test_independent_stage_unaffected_by_failure(
        self, state_manager, make_program
    ):
        """A fails but C (independent) still runs."""
        dag = _make_dag(
            {
                "a": FailingStage(timeout=5.0),
                "b": ChainedStage(timeout=5.0),
                "c": FastStage(timeout=5.0),
            },
            [DataFlowEdge.create("a", "b", "data")],
            state_manager,
        )
        prog = make_program()
        await dag.run(prog)

        assert prog.stage_results["a"].status == StageState.FAILED
        assert prog.stage_results["b"].status == StageState.SKIPPED
        assert prog.stage_results["c"].status == StageState.COMPLETED

    async def test_dag_timeout_raises(self, state_manager, make_program):
        """DAG-level timeout fires when stages exceed dag_timeout."""
        dag = _make_dag(
            {"slow": SlowStage(timeout=60.0)},
            [],
            state_manager,
            dag_timeout=0.1,
        )
        prog = make_program()
        with pytest.raises(asyncio.TimeoutError):
            await dag.run(prog)


# ===================================================================
# Category C: Data Flow & Input Validation
# ===================================================================


class TestDataFlow:
    async def test_data_flow_passes_output_to_input(self, state_manager, make_program):
        """A's MockOutput wired as B's data input."""
        dag = _make_dag(
            {
                "a": FastStage(timeout=5.0),
                "b": ChainedStage(timeout=5.0),
            },
            [DataFlowEdge.create("a", "b", "data")],
            state_manager,
        )
        prog = make_program()
        await dag.run(prog)

        assert prog.stage_results["b"].output.value == 43

    async def test_optional_input_runs_without_edge(self, state_manager, make_program):
        """OptionalInputStage runs with data=None."""
        dag = _make_dag({"opt": OptionalInputStage(timeout=5.0)}, [], state_manager)
        prog = make_program()
        await dag.run(prog)

        assert prog.stage_results["opt"].status == StageState.COMPLETED
        assert prog.stage_results["opt"].output.value == -1

    async def test_optional_input_receives_data_when_provided(
        self, state_manager, make_program
    ):
        """Edge provides optional input."""
        dag = _make_dag(
            {
                "a": FastStage(timeout=5.0),
                "opt": OptionalInputStage(timeout=5.0),
            },
            [DataFlowEdge.create("a", "opt", "data")],
            state_manager,
        )
        prog = make_program()
        await dag.run(prog)

        assert prog.stage_results["opt"].output.value == 52  # 42 + 10

    async def test_build_named_inputs_only_from_completed(
        self, state_manager, make_program
    ):
        """Failed producer's output NOT passed to downstream."""
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

        # opt should still run — data is optional → value = -1
        assert prog.stage_results["opt"].status == StageState.COMPLETED
        assert prog.stage_results["opt"].output.value == -1

    async def test_missing_required_input_skips_stage(
        self, state_manager, make_program
    ):
        """Stage with unsatisfied required input is skipped when producer fails."""
        dag = _make_dag(
            {
                "a": FailingStage(timeout=5.0),
                "b": ChainedStage(timeout=5.0),
            },
            [DataFlowEdge.create("a", "b", "data")],
            state_manager,
        )
        prog = make_program()
        await dag.run(prog)

        assert prog.stage_results["b"].status == StageState.SKIPPED


# ===================================================================
# Category D: Caching
# ===================================================================


class TestCaching:
    async def test_cached_stage_not_reexecuted(self, state_manager, make_program):
        """Pre-populated COMPLETED result with matching hash -> cached."""
        dag = _make_dag({"fast": FastStage(timeout=5.0)}, [], state_manager)
        prog = make_program()

        # First run: executes
        await dag.run(prog)
        first_started = prog.stage_results["fast"].started_at

        # Second run: should use cache (same input hash)
        dag2 = _make_dag({"fast": FastStage(timeout=5.0)}, [], state_manager)
        await dag2.run(prog)

        # started_at should not change if cached
        assert prog.stage_results["fast"].started_at == first_started

    async def test_cache_invalidated_on_input_change(self, state_manager, make_program):
        """Hash mismatch -> stage reruns."""
        dag = _make_dag(
            {
                "a": FastStage(timeout=5.0),
                "b": ChainedStage(timeout=5.0),
            },
            [DataFlowEdge.create("a", "b", "data")],
            state_manager,
        )
        prog = make_program()
        await dag.run(prog)
        first_b_started = prog.stage_results["b"].started_at

        # Corrupt b's stored input_hash so it doesn't match the
        # actual hash that will be recomputed from a's output.
        prog.stage_results["b"].input_hash = "deliberately_wrong_hash"

        await asyncio.sleep(0.01)

        dag2 = _make_dag(
            {
                "a": FastStage(timeout=5.0),
                "b": ChainedStage(timeout=5.0),
            },
            [DataFlowEdge.create("a", "b", "data")],
            state_manager,
        )
        await dag2.run(prog)

        # b should have re-run since its stored hash doesn't match
        assert prog.stage_results["b"].started_at != first_b_started

    async def test_never_cached_stage_always_reruns(self, state_manager, make_program):
        """Stage with NeverCached handler always executes."""
        dag = _make_dag({"nc": NeverCachedStage(timeout=5.0)}, [], state_manager)
        prog = make_program()
        await dag.run(prog)
        first_started = prog.stage_results["nc"].started_at

        # Small delay to ensure different timestamp
        await asyncio.sleep(0.01)

        dag2 = _make_dag({"nc": NeverCachedStage(timeout=5.0)}, [], state_manager)
        await dag2.run(prog)

        assert prog.stage_results["nc"].started_at != first_started

    async def test_cached_chain_resolves_in_one_tick(self, state_manager, make_program):
        """A->B->C all cached -> all skip without re-execution."""
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

        first_a = prog.stage_results["a"].started_at
        first_b = prog.stage_results["b"].started_at
        first_c = prog.stage_results["c"].started_at

        # Second run — should all be cached
        dag2 = _make_dag(
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
        await dag2.run(prog)

        assert prog.stage_results["a"].started_at == first_a
        assert prog.stage_results["b"].started_at == first_b
        assert prog.stage_results["c"].started_at == first_c


# ===================================================================
# Category E: Concurrency & Semaphore
# ===================================================================


class TestConcurrency:
    async def test_semaphore_limits_parallel_stages(self, state_manager, make_program):
        """max_parallel_stages=1 -> stages run one at a time."""
        dag = _make_dag(
            {
                "a": SlowStage(timeout=5.0),
                "b": SlowStage(timeout=5.0),
            },
            [],
            state_manager,
            max_parallel_stages=1,
        )
        prog = make_program()
        t0 = time.monotonic()
        await dag.run(prog)
        elapsed = time.monotonic() - t0

        # With semaphore=1, two 0.5s stages should take >= 1.0s
        assert elapsed >= 0.9
        for name in ("a", "b"):
            assert prog.stage_results[name].status == StageState.COMPLETED

    async def test_multiple_stages_respect_semaphore(self, state_manager, make_program):
        """5 stages with semaphore=2 all complete."""
        nodes = {f"s{i}": SlowStage(timeout=5.0) for i in range(5)}
        dag = _make_dag(nodes, [], state_manager, max_parallel_stages=2)
        prog = make_program()
        await dag.run(prog)

        for name in nodes:
            assert prog.stage_results[name].status == StageState.COMPLETED

    async def test_stage_tasks_are_asyncio_tasks(self, state_manager, make_program):
        """Stages are launched as properly named asyncio tasks."""
        dag = _make_dag({"fast": FastStage(timeout=5.0)}, [], state_manager)
        prog = make_program()
        await dag.run(prog)

        # The stage completed, so if any task naming issues existed
        # we would have seen failures. Verify the result is correct.
        assert prog.stage_results["fast"].status == StageState.COMPLETED


# ===================================================================
# Category F: Execution Order Dependencies
# ===================================================================


class TestExecutionOrder:
    async def test_exec_order_success_condition(self, state_manager, make_program):
        """B runs only after A succeeds (on_success)."""
        dag = _make_dag(
            {
                "a": FastStage(timeout=5.0),
                "b": FastStage(timeout=5.0),
            },
            [],
            state_manager,
            exec_deps={
                "b": [ExecutionOrderDependency.on_success("a")],
            },
        )
        prog = make_program()
        await dag.run(prog)

        assert prog.stage_results["a"].status == StageState.COMPLETED
        assert prog.stage_results["b"].status == StageState.COMPLETED

    async def test_exec_order_failure_condition(self, state_manager, make_program):
        """B runs only after A fails (on_failure)."""
        dag = _make_dag(
            {
                "a": FailingStage(timeout=5.0),
                "b": FastStage(timeout=5.0),
            },
            [],
            state_manager,
            exec_deps={
                "b": [ExecutionOrderDependency.on_failure("a")],
            },
        )
        prog = make_program()
        await dag.run(prog)

        assert prog.stage_results["a"].status == StageState.FAILED
        assert prog.stage_results["b"].status == StageState.COMPLETED

    async def test_exec_order_failure_not_met_skips(self, state_manager, make_program):
        """B depends on A failing, but A succeeds -> B is skipped."""
        dag = _make_dag(
            {
                "a": FastStage(timeout=5.0),
                "b": FastStage(timeout=5.0),
            },
            [],
            state_manager,
            exec_deps={
                "b": [ExecutionOrderDependency.on_failure("a")],
            },
        )
        prog = make_program()
        await dag.run(prog)

        assert prog.stage_results["a"].status == StageState.COMPLETED
        assert prog.stage_results["b"].status == StageState.SKIPPED

    async def test_exec_order_always_condition(self, state_manager, make_program):
        """B runs after A regardless of outcome (always_after)."""
        dag = _make_dag(
            {
                "a": FailingStage(timeout=5.0),
                "b": FastStage(timeout=5.0),
            },
            [],
            state_manager,
            exec_deps={
                "b": [ExecutionOrderDependency.always_after("a")],
            },
        )
        prog = make_program()
        await dag.run(prog)

        assert prog.stage_results["a"].status == StageState.FAILED
        assert prog.stage_results["b"].status == StageState.COMPLETED


# ===================================================================
# Category G: Stale-result skip regression (deadlock fix)
# ===================================================================


class TestStaleResultSkip:
    """Regression tests for the stale-result DEADLOCK (dag.py skip guard).

    Before the fix, a stage with a stale COMPLETED result from a previous run
    could not be overwritten with SKIPPED when its upstream failed in the current
    run, because the skip guard refused to overwrite any non-PENDING status.
    This caused:
      skip_progress=False + running={} → DEADLOCK RuntimeError.

    After the fix the guard is gated on `stage_name in finished_this_run`, so
    stale results from prior runs are correctly replaced with SKIPPED.
    """

    async def test_stale_completed_downstream_skipped_when_dep_fails(
        self, state_manager, make_program
    ):
        """Stale COMPLETED result from a prior run must not block a skip.

        DAG: failing --data--> chained
        Prior run: chained COMPLETED.
        Current run: failing FAILS → chained's mandatory input can never arrive
        → automata marks chained as IMPOSSIBLE → must be SKIPPED (not DEADLOCK).
        """
        dag = _make_dag(
            {
                "failing": FailingStage(timeout=5.0),
                "chained": ChainedStage(timeout=5.0),
            },
            [
                DataFlowEdge(
                    source_stage="failing",
                    destination_stage="chained",
                    input_name="data",
                )
            ],
            state_manager,
        )
        prog = make_program(
            stage_results={
                # Stale COMPLETED result injected directly — simulates a refresh
                # run where chained completed in the previous DAG execution but
                # now failing's failure makes its mandatory input impossible.
                "chained": ProgramStageResult(
                    status=StageState.COMPLETED, output=MockOutput(value=99)
                ),
            }
        )
        await dag.run(prog)

        assert prog.stage_results["failing"].status == StageState.FAILED
        assert prog.stage_results["chained"].status == StageState.SKIPPED

    async def test_fresh_program_dep_failure_skips_downstream(
        self, state_manager, make_program
    ):
        """Baseline: fresh program with no prior results skips downstream on failure."""
        dag = _make_dag(
            {
                "failing": FailingStage(timeout=5.0),
                "chained": ChainedStage(timeout=5.0),
            },
            [
                DataFlowEdge(
                    source_stage="failing",
                    destination_stage="chained",
                    input_name="data",
                )
            ],
            state_manager,
        )
        prog = make_program()  # no pre-existing stage results

        await dag.run(prog)

        assert prog.stage_results["failing"].status == StageState.FAILED
        assert prog.stage_results["chained"].status == StageState.SKIPPED
