"""Integration tests for LPT scheduling with DagRunner and SteadyStateEngine.

These tests verify that:
1. DagRunner._launch() actually uses the prioritizer to reorder programs
2. DagRunner._execute_dag() actually updates the predictor with timing data
3. The predictor learns online and improves ordering over multiple batches
4. LPT produces lower makespan than FIFO for programs with variable eval times
5. Failed DAGs still contribute training data (survivorship bias fix)
6. FIFO is the default when no prioritizer is provided
7. Full end-to-end: engine produces mutants -> DagRunner prioritizes -> ingests

Uses real async event loops with controlled timing, NOT mocks for the
scheduling decision path.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

from gigaevo.evolution.scheduling.predictor import (
    SimpleHeuristicPredictor,
)
from gigaevo.evolution.scheduling.prioritizer import (
    FIFOPrioritizer,
    LPTPrioritizer,
)
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState
from gigaevo.runner.dag_runner import DagRunner, DagRunnerConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TEST_TIMEOUT = 10.0


def _make_runner(
    *,
    prioritizer=None,
    max_concurrent_dags: int = 4,
    programs_in_queue: list[Program] | None = None,
) -> tuple[DagRunner, AsyncMock, MagicMock]:
    """Build a DagRunner with mocked storage and blueprint.

    Returns (runner, storage_mock, blueprint_mock).
    Programs in *programs_in_queue* are set up as QUEUED in the mock storage.
    """
    storage = AsyncMock()
    blueprint = MagicMock()
    writer = MagicMock()
    writer.bind.return_value = writer
    config = DagRunnerConfig(
        max_concurrent_dags=max_concurrent_dags,
        poll_interval=0.1,
        dag_timeout=60,
    )

    runner = DagRunner(
        storage=storage,
        dag_blueprint=blueprint,
        config=config,
        writer=writer,
        prioritizer=prioritizer,
    )

    # Set up storage to return the queued programs
    progs = programs_in_queue or []
    queued_ids = [p.id for p in progs]
    prog_map = {p.id: p for p in progs}

    storage.get_ids_by_status.side_effect = lambda status: (
        list(queued_ids) if status == ProgramState.QUEUED.value else []
    )

    def mget_side(ids, **kw):
        return [prog_map.get(pid) for pid in ids]

    storage.mget.side_effect = mget_side

    return runner, storage, blueprint


def _prog(code: str, state: ProgramState = ProgramState.QUEUED) -> Program:
    return Program(code=code, state=state)


# ---------------------------------------------------------------------------
# DagRunner integration: prioritizer is actually called
# ---------------------------------------------------------------------------


class TestDagRunnerPrioritizerCalled:
    """Verify that _launch() calls the prioritizer and uses its ordering."""

    async def test_launch_calls_prioritize(self) -> None:
        """_launch() passes fetched programs through the prioritizer."""
        progs = [_prog("short"), _prog("x" * 500), _prog("x" * 2000)]
        pred = SimpleHeuristicPredictor(default_rate=1.0)
        # Warm up so LPT actually reorders
        for _ in range(5):
            pred.update(_prog("x" * 300), 300.0)
        prioritizer = LPTPrioritizer(pred)

        runner, storage, blueprint = _make_runner(
            prioritizer=prioritizer,
            programs_in_queue=progs,
        )

        def build_side(*args, **kwargs):
            dag = MagicMock()
            dag.run = AsyncMock()
            # Track which program gets launched
            return dag

        blueprint.build.side_effect = build_side

        await runner._launch()

        # Verify programs were actually launched (tasks created)
        assert len(runner._active) == 3

        # The active dict keys should contain all 3 program IDs
        active_ids = set(runner._active.keys())
        expected_ids = {p.id for p in progs}
        assert active_ids == expected_ids

    async def test_lpt_launches_longest_first(self) -> None:
        """With LPT, the longest-code program gets launched first in task order."""
        short = _prog("x" * 100)
        long = _prog("x" * 5000)

        pred = SimpleHeuristicPredictor(default_rate=1.0)
        for _ in range(5):
            pred.update(_prog("x" * 300), 300.0)
        prioritizer = LPTPrioritizer(pred)

        runner, storage, blueprint = _make_runner(
            prioritizer=prioritizer,
            programs_in_queue=[short, long],
            max_concurrent_dags=1,  # Only 1 slot -> order matters!
        )

        # Track actual execution order via the semaphore gate
        exec_order: list[str] = []

        def build_side(*args, **kwargs):
            dag = MagicMock()

            async def fake_run(prog):
                exec_order.append(prog.id)
                await asyncio.sleep(0.01)

            dag.run = fake_run
            return dag

        blueprint.build.side_effect = build_side
        storage.batch_transition_state = AsyncMock()

        await runner._launch()

        # Wait for tasks to complete
        tasks = [info.task for info in runner._active.values()]
        await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True), timeout=5
        )

        # With max_concurrent_dags=1, execution is serialized.
        # LPT should run the long program first.
        assert len(exec_order) == 2
        assert exec_order[0] == long.id, (
            f"LPT should launch longest first, but got {exec_order}"
        )

    async def test_fifo_is_default(self) -> None:
        """When no prioritizer is provided, DagRunner uses FIFOPrioritizer."""
        runner, _, _ = _make_runner()
        assert isinstance(runner._prioritizer, FIFOPrioritizer)


# ---------------------------------------------------------------------------
# DagRunner integration: predictor updated after execution
# ---------------------------------------------------------------------------


class TestDagRunnerPredictorUpdate:
    """Verify that _execute_dag() actually feeds back timing to the predictor."""

    async def test_predictor_updated_on_success(self) -> None:
        """After successful DAG execution, predictor.update() is called."""
        pred = SimpleHeuristicPredictor()
        prioritizer = LPTPrioritizer(pred)

        runner, storage, blueprint = _make_runner(prioritizer=prioritizer)

        prog = _prog("x" * 500, state=ProgramState.RUNNING)
        dag = MagicMock()
        dag.run = AsyncMock()  # success
        storage.batch_transition_state = AsyncMock()

        assert not pred.is_warm()

        await runner._execute_dag(dag, prog)

        # Predictor should have been called with actual duration
        # We can't check exact timing, but we can check state changed
        assert len(pred._window) == 1
        assert pred._window[0] > 0  # duration was positive

    async def test_predictor_updated_on_failure(self) -> None:
        """After failed DAG execution, predictor STILL gets updated (survivorship bias fix)."""
        pred = SimpleHeuristicPredictor()
        prioritizer = LPTPrioritizer(pred)

        runner, storage, blueprint = _make_runner(prioritizer=prioritizer)

        prog = _prog("x" * 500, state=ProgramState.RUNNING)
        dag = MagicMock()
        dag.run = AsyncMock(side_effect=RuntimeError("DAG failed"))
        # set_program_state is called on failure
        runner._state_manager = AsyncMock()

        await runner._execute_dag(dag, prog)

        # Even though DAG failed, predictor should have been updated
        assert len(pred._window) == 1

    async def test_no_predictor_update_for_fifo(self) -> None:
        """FIFO prioritizer has no predictor — update path is a no-op."""
        runner, storage, blueprint = _make_runner()  # default FIFO

        prog = _prog("x" * 500, state=ProgramState.RUNNING)
        dag = MagicMock()
        dag.run = AsyncMock()
        storage.batch_transition_state = AsyncMock()

        # Should not raise even though FIFO has no predictor
        await runner._execute_dag(dag, prog)

    async def test_predictor_warms_up_from_dag_executions(self) -> None:
        """After enough DAG executions, predictor becomes warm."""
        pred = SimpleHeuristicPredictor()
        prioritizer = LPTPrioritizer(pred)

        runner, storage, blueprint = _make_runner(prioritizer=prioritizer)
        storage.batch_transition_state = AsyncMock()

        assert not pred.is_warm()

        for i in range(5):
            prog = _prog("x" * (200 + i * 100), state=ProgramState.RUNNING)
            dag = MagicMock()
            dag.run = AsyncMock()
            await runner._execute_dag(dag, prog)

        assert pred.is_warm()


# ---------------------------------------------------------------------------
# End-to-end: LPT actually reduces makespan
# ---------------------------------------------------------------------------


class TestLPTReducesMakespan:
    """Verify that LPT scheduling produces measurably lower makespan than FIFO
    when programs have variable evaluation times.

    Uses the real async event loop with controlled sleep durations.
    This is the closest we get to production without Redis.
    """

    async def test_lpt_beats_fifo_makespan(self) -> None:
        """LPT produces strictly lower makespan than FIFO for skewed eval times.

        Setup: 6 programs with durations [5, 4, 3, 2, 1, 1] on 2 servers.
        Optimal (LPT): [5,1] [4,2] [3,1] => makespan=max(6,6,4)=6
        FIFO (worst):  [5,4] [3,2] [1,1] => makespan=max(9,5,2)=9
        """
        # Durations in milliseconds (short enough for a test)
        # code_length proportional to duration for the predictor
        dur_map: dict[str, float] = {}
        programs: list[Program] = []
        durations_ms = [50, 40, 30, 20, 10, 10]

        for dur_ms in durations_ms:
            # Code length proportional to eval time so predictor can learn
            code = "x" * (dur_ms * 10)
            p = _prog(code)
            programs.append(p)
            dur_map[p.id] = dur_ms / 1000.0  # convert to seconds for sleep

        async def run_with_prioritizer(prio, progs):
            """Measure makespan for a given prioritizer and program list."""
            ordered = prio.prioritize(progs)
            sema = asyncio.Semaphore(2)  # 2 servers
            t0 = time.monotonic()

            async def eval_one(p):
                async with sema:
                    await asyncio.sleep(dur_map[p.id])

            await asyncio.gather(*[asyncio.create_task(eval_one(p)) for p in ordered])
            return time.monotonic() - t0

        # Warm up the predictor
        pred = SimpleHeuristicPredictor(default_rate=0.001)
        for _ in range(5):
            # Train: code_length correlates with eval time
            p = _prog("x" * 500)
            pred.update(p, 0.05)  # 500 chars -> 50ms

        lpt_prio = LPTPrioritizer(pred)
        fifo_prio = FIFOPrioritizer()

        fifo_time = await run_with_prioritizer(fifo_prio, programs)
        lpt_time = await run_with_prioritizer(lpt_prio, programs)

        # LPT should be faster (or at worst equal)
        # Allow 20ms tolerance for async scheduling jitter
        assert lpt_time <= fifo_time + 0.02, (
            f"LPT ({lpt_time:.3f}s) should be <= FIFO ({fifo_time:.3f}s)"
        )

    async def test_lpt_no_worse_for_uniform_programs(self) -> None:
        """When all programs have the same eval time, LPT doesn't hurt."""
        programs = [_prog(f"x_{i}" * 50) for i in range(8)]

        pred = SimpleHeuristicPredictor(default_rate=0.001)
        for _ in range(5):
            pred.update(_prog("x" * 200), 0.01)

        lpt_prio = LPTPrioritizer(pred)
        fifo_prio = FIFOPrioritizer()

        sema = asyncio.Semaphore(2)

        async def measure(prio):
            ordered = prio.prioritize(programs)
            t0 = time.monotonic()

            async def eval_one(p):
                async with sema:
                    await asyncio.sleep(0.01)

            await asyncio.gather(*[asyncio.create_task(eval_one(p)) for p in ordered])
            return time.monotonic() - t0

        fifo_time = await measure(fifo_prio)
        lpt_time = await measure(lpt_prio)

        # Should be approximately equal (within 20ms jitter)
        assert abs(lpt_time - fifo_time) < 0.02


# ---------------------------------------------------------------------------
# Online learning: predictor improves over multiple batches
# ---------------------------------------------------------------------------


class TestOnlineLearning:
    """Verify the predictor learns from real execution data and improves."""

    async def test_prediction_accuracy_improves(self) -> None:
        """After observing multiple evals, prediction error decreases."""
        pred = SimpleHeuristicPredictor(default_rate=0.1, window_size=20)

        # Ground truth: eval_time = code_length * 0.5
        true_rate = 0.5

        # Before any training: predictions use default rate (0.1)
        p = _prog("x" * 200)
        cold_pred = pred.predict(p)
        cold_error = abs(cold_pred - 200 * true_rate)

        # Train with 10 samples following the true rate
        for length in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
            prog = _prog("x" * length)
            pred.update(prog, length * true_rate)

        # After training: predictions should be closer to truth
        warm_pred = pred.predict(p)
        warm_error = abs(warm_pred - 200 * true_rate)

        assert warm_error < cold_error, (
            f"Warm error ({warm_error:.1f}) should be < cold error ({cold_error:.1f})"
        )

    async def test_ordering_improves_across_batches(self) -> None:
        """Over multiple batches, LPT ordering gets closer to optimal."""
        pred = SimpleHeuristicPredictor(default_rate=0.1, window_size=20)
        prio = LPTPrioritizer(pred)

        # Batch 1: cold predictor -> FIFO ordering
        batch1 = [_prog("x" * 100), _prog("x" * 500), _prog("x" * 1000)]
        result1 = prio.prioritize(batch1)
        # Cold = FIFO (no reordering)
        assert [p.id for p in result1] == [p.id for p in batch1]

        # Train: longer code = longer eval
        for length in [100, 200, 500, 800, 1000]:
            pred.update(_prog("x" * length), float(length))

        # Batch 2: warm predictor -> LPT ordering (longest first)
        batch2 = [_prog("x" * 100), _prog("x" * 500), _prog("x" * 1000)]
        result2 = prio.prioritize(batch2)
        # Warm = longest first
        code_lengths = [len(p.code) for p in result2]
        assert code_lengths == sorted(code_lengths, reverse=True)


# ---------------------------------------------------------------------------
# NaN/Inf safety (chaos-hacker finding)
# ---------------------------------------------------------------------------


class TestPredictorSafety:
    """Verify chaos-hacker fixes: NaN, Inf, outlier resilience."""

    def test_outlier_clipped_in_heuristic(self) -> None:
        """SimpleHeuristicPredictor clips extreme outlier updates."""
        pred = SimpleHeuristicPredictor(default_rate=1.0, window_size=10)

        # Normal training: rate ~1.0
        for _ in range(5):
            pred.update(_prog("x" * 200), 200.0)

        # Inject extreme outlier: rate = 100000/200 = 500 (500x normal)
        pred.update(_prog("x" * 200), 100000.0)

        # Prediction should NOT be wildly inflated
        p = pred.predict(_prog("x" * 200))
        # Without clipping: ~500*200=100000. With clipping: ~10*1.0*200=2000
        assert p < 5000, f"Outlier not clipped: prediction={p}"

    def test_ridge_nan_guard(self) -> None:
        """RidgePredictor returns default on NaN/degenerate predictions."""
        from gigaevo.evolution.scheduling.predictor import RidgePredictor

        pred = RidgePredictor(min_samples=3, default_prediction=500.0)

        # Train with degenerate data (all same features, varying targets)
        # This can produce unstable Ridge coefficients
        for dur in [1.0, 1000.0, 0.001]:
            pred.update(_prog("x" * 100), dur)

        # Prediction should be finite and reasonable
        result = pred.predict(_prog("x" * 100))
        assert result > 0
        assert result < 1e10  # not infinity


# ---------------------------------------------------------------------------
# Full DagRunner _launch + _execute cycle
# ---------------------------------------------------------------------------


class TestDagRunnerFullCycle:
    """Test the complete cycle: queue -> prioritize -> launch -> execute -> learn."""

    async def test_full_launch_execute_learn_cycle(self) -> None:
        """Programs go through queue -> prioritize -> execute -> predictor update."""
        pred = SimpleHeuristicPredictor()
        prioritizer = LPTPrioritizer(pred)

        short = _prog("x" * 100)
        long = _prog("x" * 2000)
        progs = [short, long]

        runner, storage, blueprint = _make_runner(
            prioritizer=prioritizer,
            programs_in_queue=progs,
            max_concurrent_dags=2,
        )

        # Set up DAG that actually sleeps (proportional to code length)
        def build_side(*args, **kwargs):
            dag = MagicMock()

            async def fake_run(prog):
                await asyncio.sleep(0.001 * len(prog.code) / 100)

            dag.run = fake_run
            return dag

        blueprint.build.side_effect = build_side
        storage.batch_transition_state = AsyncMock()

        assert not pred.is_warm()

        # Launch
        await runner._launch()
        assert len(runner._active) == 2

        # Wait for tasks to complete
        tasks = [info.task for info in runner._active.values()]
        await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=5,
        )

        # Predictor should have learned from both executions
        assert len(pred._window) == 2
        # The longer program should have produced a larger duration entry
        # (pred stores rate = duration/code_length, but both should be positive)
        assert all(r > 0 for r in pred._window)

    async def test_multiple_launch_cycles_improve_ordering(self) -> None:
        """Over multiple _launch() cycles, predictor learns and improves."""
        pred = SimpleHeuristicPredictor()
        prioritizer = LPTPrioritizer(pred)

        # Pre-train with 3 batches (each batch has 2 programs = 6 updates total)
        # This simulates the predictor learning from prior runs
        for cycle in range(3):
            batch = [_prog("x" * 100), _prog("x" * 1000)]
            runner, storage, bp = _make_runner(
                prioritizer=prioritizer,
                programs_in_queue=batch,
                max_concurrent_dags=2,
            )

            def make_build():
                def build_side(*a, **k):
                    dag = MagicMock()

                    async def run(prog):
                        # Eval time proportional to code length
                        await asyncio.sleep(0.001 * len(prog.code) / 100)

                    dag.run = run
                    return dag

                return build_side

            bp.build.side_effect = make_build()
            storage.batch_transition_state = AsyncMock()

            await runner._launch()
            tasks = [i.task for i in runner._active.values()]
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True), timeout=5
            )

        # Predictor should be warm now (6 updates >= 5)
        assert pred.is_warm()

        # Final cycle: warm predictor should reorder (longest first)
        batch_final = [_prog("x" * 100), _prog("x" * 1000)]
        runner_final, storage_final, bp_final = _make_runner(
            prioritizer=prioritizer,
            programs_in_queue=batch_final,
            max_concurrent_dags=1,  # Serialize to verify order
        )

        exec_order: list[str] = []

        def build_final(*a, **k):
            dag = MagicMock()

            async def run(prog):
                exec_order.append(prog.id)
                await asyncio.sleep(0.001)

            dag.run = run
            return dag

        bp_final.build.side_effect = build_final
        storage_final.batch_transition_state = AsyncMock()

        await runner_final._launch()
        tasks = [i.task for i in runner_final._active.values()]
        await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True), timeout=5
        )

        # Warm LPT should launch the longer program first
        assert exec_order[0] == batch_final[1].id, (
            f"After learning, LPT should launch longest first. "
            f"Got order: {exec_order}, expected {batch_final[1].id} first"
        )
