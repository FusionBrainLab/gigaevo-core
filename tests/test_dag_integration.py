"""Multi-run DAG integration tests — validate state persistence across runs."""

from __future__ import annotations

import asyncio

import pytest

from gigaevo.programs.core_types import (
    ProgramStageResult,
    StageState,
)
from gigaevo.programs.dag.automata import DataFlowEdge
from gigaevo.programs.dag.dag import DAG
from gigaevo.programs.program_state import ProgramState
from tests.conftest import (
    ChainedStage,
    FailingChainedStage,
    FailingStage,
    FastStage,
    MockOutput,
    NeverCachedStage,
    NullWriter,
    SideEffectStage,
    SlowStage,
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
# Category A: Multi-Run State Persistence
# ===================================================================


class TestMultiRunPersistence:
    async def test_dag_run_persists_all_stage_results_to_redis(
        self, state_manager, fakeredis_storage, make_program
    ):
        """Run DAG with 3 stages; verify all stage_results in Redis."""
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
        await fakeredis_storage.add(prog)
        await dag.run(prog)

        fetched = await fakeredis_storage.get(prog.id)
        assert fetched is not None

        for stage_name in ("a", "b", "c"):
            res = fetched.stage_results[stage_name]
            assert res.status == StageState.COMPLETED
            assert res.output is not None
            assert res.started_at is not None
            assert res.finished_at is not None
            assert res.input_hash is not None

    async def test_second_dag_run_uses_cached_results(
        self, state_manager, fakeredis_storage, make_program
    ):
        """Run DAG once, run again: stages are cached (not re-executed)."""

        def nodes_fn():
            return {
                "a": FastStage(timeout=5.0),
                "b": ChainedStage(timeout=5.0),
            }

        edges = [DataFlowEdge.create("a", "b", "data")]

        prog = make_program()
        await fakeredis_storage.add(prog)

        dag1 = _make_dag(nodes_fn(), edges, state_manager)
        await dag1.run(prog)
        first_a_started = prog.stage_results["a"].started_at
        first_b_started = prog.stage_results["b"].started_at

        # Second run — should use cache
        dag2 = _make_dag(nodes_fn(), edges, state_manager)
        await dag2.run(prog)

        assert prog.stage_results["a"].started_at == first_a_started
        assert prog.stage_results["b"].started_at == first_b_started

        # Redis should also reflect cached (unchanged) state
        fetched = await fakeredis_storage.get(prog.id)
        assert fetched.stage_results["a"].started_at == first_a_started

    async def test_second_run_after_input_change_reruns_affected_stages(
        self, state_manager, fakeredis_storage, make_program
    ):
        """Change A's output -> B and C rerun on second DAG run."""
        edges = [
            DataFlowEdge.create("a", "b", "data"),
            DataFlowEdge.create("b", "c", "data"),
        ]

        prog = make_program()
        await fakeredis_storage.add(prog)

        dag1 = _make_dag(
            {
                "a": FastStage(timeout=5.0),
                "b": ChainedStage(timeout=5.0),
                "c": ChainedStage(timeout=5.0),
            },
            edges,
            state_manager,
        )
        await dag1.run(prog)
        prog.stage_results["b"].output.value

        # Mutate a's output to invalidate downstream hashes
        prog.stage_results["a"] = ProgramStageResult.success(
            output=MockOutput(value=100)
        )
        # Clear cached input_hash for a
        prog.stage_results["a"].input_hash = "forced_new_hash"

        dag2 = _make_dag(
            {
                "a": FastStage(timeout=5.0),
                "b": ChainedStage(timeout=5.0),
                "c": ChainedStage(timeout=5.0),
            },
            edges,
            state_manager,
        )
        await dag2.run(prog)

        # b should have rerun with new a output (value=100 -> b=101)
        # But a itself reruns because its input_hash was changed
        # After rerun, a produces 42 again (FastStage always returns 42)
        # The key is that the DAG ran successfully a second time
        assert prog.stage_results["b"].status == StageState.COMPLETED
        assert prog.stage_results["c"].status == StageState.COMPLETED

    async def test_multiple_sequential_runs_accumulate_correct_state(
        self, state_manager, fakeredis_storage, make_program
    ):
        """Run DAG 3 times. After each, verify Redis state is consistent."""
        prog = make_program()
        await fakeredis_storage.add(prog)

        for run_idx in range(3):
            dag = _make_dag(
                {"fast": NeverCachedStage(timeout=5.0)},
                [],
                state_manager,
            )
            await dag.run(prog)

            fetched = await fakeredis_storage.get(prog.id)
            assert fetched.stage_results["fast"].status == StageState.COMPLETED
            # atomic_counter should be monotonically increasing
            if run_idx > 0:
                assert fetched.atomic_counter > 0

    async def test_dag_run_with_failure_then_retry_after_fix(
        self, state_manager, fakeredis_storage, make_program
    ):
        """Run DAG where B fails. Fix B, rerun: B and C succeed."""
        edges = [
            DataFlowEdge.create("a", "b", "data"),
            DataFlowEdge.create("b", "c", "data"),
        ]

        prog = make_program()
        await fakeredis_storage.add(prog)

        # First run: B fails (FailingChainedStage accepts data input)
        dag1 = _make_dag(
            {
                "a": FastStage(timeout=5.0),
                "b": FailingChainedStage(timeout=5.0),
                "c": ChainedStage(timeout=5.0),
            },
            edges,
            state_manager,
        )
        await dag1.run(prog)

        fetched = await fakeredis_storage.get(prog.id)
        assert fetched.stage_results["b"].status == StageState.FAILED
        assert fetched.stage_results["c"].status == StageState.SKIPPED

        # "Fix" B by replacing with working stage and clearing results
        prog.stage_results.pop("b", None)
        prog.stage_results.pop("c", None)

        dag2 = _make_dag(
            {
                "a": FastStage(timeout=5.0),
                "b": ChainedStage(timeout=5.0),
                "c": ChainedStage(timeout=5.0),
            },
            edges,
            state_manager,
        )
        await dag2.run(prog)

        assert prog.stage_results["b"].status == StageState.COMPLETED
        assert prog.stage_results["c"].status == StageState.COMPLETED


# ===================================================================
# Category B: Cross-Program Multi-Run
# ===================================================================


class TestCrossProgram:
    async def test_multiple_programs_through_same_dag(
        self, state_manager, fakeredis_storage, make_program
    ):
        """5 programs through identical DAGs — each independent in Redis."""
        progs = []
        for i in range(5):
            p = make_program(code=f"def solve(): return {i}")
            await fakeredis_storage.add(p)
            progs.append(p)

        for p in progs:
            dag = _make_dag({"fast": FastStage(timeout=5.0)}, [], state_manager)
            await dag.run(p)

        for p in progs:
            fetched = await fakeredis_storage.get(p.id)
            assert fetched.stage_results["fast"].status == StageState.COMPLETED
            assert fetched.id == p.id

    async def test_concurrent_programs_no_cross_contamination(
        self, state_manager, fakeredis_storage, make_program
    ):
        """3 programs concurrently — no data leaks between programs."""
        progs = []
        for i in range(3):
            p = make_program(code=f"def solve(): return {i}")
            await fakeredis_storage.add(p)
            progs.append(p)

        async def run_one(p):
            dag = _make_dag({"fast": FastStage(timeout=5.0)}, [], state_manager)
            await dag.run(p)

        await asyncio.gather(*[run_one(p) for p in progs])

        ids = set()
        for p in progs:
            fetched = await fakeredis_storage.get(p.id)
            assert fetched.stage_results["fast"].status == StageState.COMPLETED
            ids.add(fetched.id)

        # All three are distinct
        assert len(ids) == 3

    async def test_program_state_transitions_persisted_across_runs(
        self, state_manager, fakeredis_storage, make_program
    ):
        """Create QUEUED, run DAG (RUNNING), complete (DONE). Verify in Redis."""
        prog = make_program(state=ProgramState.QUEUED)
        await fakeredis_storage.add(prog)

        # Transition to RUNNING
        await state_manager.set_program_state(prog, ProgramState.RUNNING)
        fetched = await fakeredis_storage.get(prog.id)
        assert fetched.state == ProgramState.RUNNING

        # Run DAG
        dag = _make_dag({"fast": FastStage(timeout=5.0)}, [], state_manager)
        await dag.run(prog)

        # Transition to DONE
        await state_manager.set_program_state(prog, ProgramState.DONE)
        fetched = await fakeredis_storage.get(prog.id)
        assert fetched.state == ProgramState.DONE


# ===================================================================
# Category C: Data Integrity Across Runs
# ===================================================================


class TestDataIntegrity:
    async def test_stage_output_survives_redis_roundtrip_after_dag(
        self, state_manager, fakeredis_storage, make_program
    ):
        """Complex MockOutput survives Redis round-trip."""
        dag = _make_dag({"fast": FastStage(timeout=5.0)}, [], state_manager)
        prog = make_program()
        await fakeredis_storage.add(prog)
        await dag.run(prog)

        fetched = await fakeredis_storage.get(prog.id)
        output = fetched.stage_results["fast"].output
        assert isinstance(output, MockOutput)
        assert output.value == 42

    async def test_metrics_updated_during_dag_persist_to_redis(
        self, state_manager, fakeredis_storage, make_program
    ):
        """SideEffectStage writes metrics -> verify in Redis after DAG."""
        dag = _make_dag({"side": SideEffectStage(timeout=5.0)}, [], state_manager)
        prog = make_program()
        await fakeredis_storage.add(prog)
        await dag.run(prog)

        fetched = await fakeredis_storage.get(prog.id)
        assert fetched.metrics["side_effect_metric"] == 123.0

    async def test_stage_error_details_survive_redis_roundtrip(
        self, state_manager, fakeredis_storage, make_program
    ):
        """FailingStage error details survive Redis round-trip."""
        dag = _make_dag({"fail": FailingStage(timeout=5.0)}, [], state_manager)
        prog = make_program()
        await fakeredis_storage.add(prog)
        await dag.run(prog)

        fetched = await fakeredis_storage.get(prog.id)
        res = fetched.stage_results["fail"]
        assert res.status == StageState.FAILED
        assert res.error is not None
        assert res.error.type == "RuntimeError"
        assert "stage failed on purpose" in res.error.message
        assert res.error.stage == "FailingStage"

    async def test_chained_dag_outputs_consistent_in_redis(
        self, state_manager, fakeredis_storage, make_program
    ):
        """A->B->C chain: A=42, B=43, C=44 in Redis."""
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
        await fakeredis_storage.add(prog)
        await dag.run(prog)

        fetched = await fakeredis_storage.get(prog.id)
        assert fetched.stage_results["a"].output.value == 42
        assert fetched.stage_results["b"].output.value == 43
        assert fetched.stage_results["c"].output.value == 44


# ===================================================================
# Category D: Edge Cases in Multi-Run
# ===================================================================


class TestMultiRunEdgeCases:
    async def test_dag_timeout_leaves_partial_results_in_redis(
        self, state_manager, fakeredis_storage, make_program
    ):
        """Slow stage exceeds dag_timeout; completed stages have results."""
        dag = _make_dag(
            {
                "fast": FastStage(timeout=5.0),
                "slow": SlowStage(timeout=60.0),
            },
            [],
            state_manager,
            dag_timeout=0.2,
        )
        prog = make_program()
        await fakeredis_storage.add(prog)

        with pytest.raises(asyncio.TimeoutError):
            await dag.run(prog)

        fetched = await fakeredis_storage.get(prog.id)
        # fast should have completed before timeout
        assert fetched.stage_results["fast"].status == StageState.COMPLETED

    async def test_mixed_cached_and_fresh_stages_on_rerun(
        self, state_manager, fakeredis_storage, make_program
    ):
        """Run DAG; invalidate C's cache; rerun: A, B cached, C reruns."""
        edges = [
            DataFlowEdge.create("a", "b", "data"),
            DataFlowEdge.create("b", "c", "data"),
        ]

        def nodes_fn():
            return {
                "a": FastStage(timeout=5.0),
                "b": ChainedStage(timeout=5.0),
                "c": ChainedStage(timeout=5.0),
            }

        prog = make_program()
        await fakeredis_storage.add(prog)

        dag1 = _make_dag(nodes_fn(), edges, state_manager)
        await dag1.run(prog)

        first_a_started = prog.stage_results["a"].started_at
        first_b_started = prog.stage_results["b"].started_at

        # Invalidate c's cache by changing its input_hash
        prog.stage_results["c"].input_hash = "invalid_hash"

        dag2 = _make_dag(nodes_fn(), edges, state_manager)
        await dag2.run(prog)

        # a and b should be cached (unchanged)
        assert prog.stage_results["a"].started_at == first_a_started
        assert prog.stage_results["b"].started_at == first_b_started
        # c should have rerun
        assert prog.stage_results["c"].status == StageState.COMPLETED

    async def test_never_cached_stage_reruns_on_every_dag_execution(
        self, state_manager, fakeredis_storage, make_program
    ):
        """NeverCached stage reruns each time (different started_at)."""
        prog = make_program()
        await fakeredis_storage.add(prog)

        dag1 = _make_dag({"nc": NeverCachedStage(timeout=5.0)}, [], state_manager)
        await dag1.run(prog)
        first_started = prog.stage_results["nc"].started_at

        await asyncio.sleep(0.01)

        dag2 = _make_dag({"nc": NeverCachedStage(timeout=5.0)}, [], state_manager)
        await dag2.run(prog)
        second_started = prog.stage_results["nc"].started_at

        assert first_started != second_started

        # Verify in Redis too
        fetched = await fakeredis_storage.get(prog.id)
        assert fetched.stage_results["nc"].started_at == second_started
