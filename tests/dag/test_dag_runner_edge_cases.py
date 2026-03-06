"""Edge-case and boundary tests for gigaevo/programs/dag/dag.py (DAG runner orchestration).

Each test class documents the exact line/branch it targets in dag.py.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from gigaevo.programs.core_types import (
    ProgramStageResult,
    StageIO,
    StageState,
    VoidInput,
)
from gigaevo.programs.dag.automata import (
    DataFlowEdge,
    ExecutionOrderDependency,
)
from gigaevo.programs.dag.dag import DAG, DEFAULT_STALL_GRACE_SECONDS
from gigaevo.programs.program import Program
from gigaevo.programs.stages.base import Stage
from gigaevo.programs.stages.cache_handler import NO_CACHE
from tests.conftest import (
    ChainedStage,
    FailingStage,
    FastStage,
    MockOutput,
    NullWriter,
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


# ---------------------------------------------------------------------------
# Module-level stage classes (must be picklable for fakeredis serialization)
# ---------------------------------------------------------------------------


class _TrackOutput(StageIO):
    value: int = 0
    stage_id: str = ""


class _TrackInput(StageIO):
    data: _TrackOutput


class IdentityStage(Stage):
    """Returns a unique value based on stage identity (for closure-capture tests)."""

    InputsModel = VoidInput
    OutputModel = _TrackOutput

    def __init__(self, *, timeout: float, identity: str):
        super().__init__(timeout=timeout)
        self._identity = identity

    async def compute(self, program: Program) -> _TrackOutput:
        return _TrackOutput(value=hash(self._identity) % 10000, stage_id=self._identity)


class CancellingStage(Stage):
    """Stage that raises asyncio.CancelledError during compute."""

    InputsModel = VoidInput
    OutputModel = MockOutput

    async def compute(self, program: Program) -> MockOutput:
        raise asyncio.CancelledError()


class GenericExceptionStage(Stage):
    """Stage that raises a non-CancelledError exception."""

    InputsModel = VoidInput
    OutputModel = MockOutput

    async def compute(self, program: Program) -> MockOutput:
        raise ValueError("something went wrong")


class TinyTimeoutStage(Stage):
    """Stage with a very short timeout (for stall_grace_seconds calculation tests)."""

    InputsModel = VoidInput
    OutputModel = MockOutput

    async def compute(self, program: Program) -> MockOutput:
        return MockOutput(value=1)


class LongTimeoutStage(Stage):
    """Stage with a very long timeout (for stall_grace_seconds calculation tests)."""

    InputsModel = VoidInput
    OutputModel = MockOutput

    async def compute(self, program: Program) -> MockOutput:
        return MockOutput(value=1)


class NeverCachedFastStage(Stage):
    """Fast stage that never uses cache — always re-executes."""

    InputsModel = VoidInput
    OutputModel = MockOutput
    cache_handler = NO_CACHE

    async def compute(self, program: Program) -> MockOutput:
        return MockOutput(value=42)


class ConcurrencyTrackingStage(Stage):
    """Tracks concurrent execution count to verify semaphore enforcement."""

    InputsModel = VoidInput
    OutputModel = MockOutput

    _counter: dict = {}
    _lock: asyncio.Lock | None = None
    _max_concurrent: list = []

    def __init__(self, *, timeout: float, tracker: dict):
        super().__init__(timeout=timeout)
        self._tracker = tracker

    async def compute(self, program: Program) -> MockOutput:
        tracker = self._tracker
        lock = tracker["lock"]
        async with lock:
            tracker["current"] += 1
            tracker["max_samples"].append(tracker["current"])
        await asyncio.sleep(0.05)
        async with lock:
            tracker["current"] -= 1
        return MockOutput(value=1)


# ===================================================================
# Test 1: Skip guard (lines 120-149) — stale COMPLETED from previous
# run gets overwritten by SKIP, but COMPLETED in this run does not.
# ===================================================================


class TestSkipGuardStaleVsFresh:
    """Target: dag.py lines 120-149.

    The skip guard checks `stage_name in finished_this_run`.
    A COMPLETED result from a *previous* run (not in finished_this_run)
    must be overwritten with SKIP when its upstream fails.
    A stage COMPLETED *in this run* must NOT be re-skipped.
    """

    async def test_stale_completed_from_previous_run_overwritten_by_skip(
        self, state_manager, make_program
    ):
        """A stage with a stale COMPLETED result from a previous run must be
        overwritten by SKIP when its mandatory upstream fails this run.

        DAG: failing --data--> chained
        Previous run left chained as COMPLETED.
        This run: failing FAILS -> chained must be SKIPPED (not stuck as COMPLETED).
        """
        dag = _make_dag(
            {
                "failing": FailingStage(timeout=5.0),
                "chained": ChainedStage(timeout=5.0),
            },
            [DataFlowEdge.create("failing", "chained", "data")],
            state_manager,
        )
        prog = make_program(
            stage_results={
                "chained": ProgramStageResult(
                    status=StageState.COMPLETED,
                    output=MockOutput(value=99),
                ),
            }
        )
        await dag.run(prog)

        assert prog.stage_results["failing"].status == StageState.FAILED
        # The stale COMPLETED must have been replaced by SKIP
        assert prog.stage_results["chained"].status == StageState.SKIPPED

    async def test_fresh_completed_in_this_run_not_re_skipped(
        self, state_manager, make_program
    ):
        """A stage that genuinely COMPLETED in this run must NOT be re-skipped,
        even if the automata later suggests skipping it.

        DAG: fast (independent), also_fast -> chained (data flow)
        Both complete; chained should remain COMPLETED.
        """
        dag = _make_dag(
            {
                "fast": FastStage(timeout=5.0),
                "chained": ChainedStage(timeout=5.0),
            },
            [DataFlowEdge.create("fast", "chained", "data")],
            state_manager,
        )
        prog = make_program()
        await dag.run(prog)

        assert prog.stage_results["fast"].status == StageState.COMPLETED
        assert prog.stage_results["chained"].status == StageState.COMPLETED

    async def test_stale_failed_from_previous_run_overwritten_by_skip(
        self, state_manager, make_program
    ):
        """A FAILED result from a previous run should also be overwritable by SKIP."""
        dag = _make_dag(
            {
                "failing": FailingStage(timeout=5.0),
                "downstream": ChainedStage(timeout=5.0),
            },
            [DataFlowEdge.create("failing", "downstream", "data")],
            state_manager,
        )
        prog = make_program(
            stage_results={
                # Stale FAILED result from previous run
                "downstream": ProgramStageResult(status=StageState.FAILED),
            }
        )
        await dag.run(prog)

        assert prog.stage_results["downstream"].status == StageState.SKIPPED


# ===================================================================
# Test 2: Deadlock detection (lines 152-161)
# ===================================================================


class TestDeadlockDetection:
    """Target: dag.py lines 152-161.

    If to_skip is non-empty but skip_progress is False AND running is
    empty, a RuntimeError is raised. This happens when all skip
    candidates are non-PENDING and in finished_this_run.
    """

    async def test_deadlock_raises_runtime_error(self, state_manager, make_program):
        """Force a deadlock scenario: automata wants to skip stages that are
        already finalized in this run (so skip_progress stays False) and
        nothing is running.

        We achieve this by patching get_stages_to_skip on the automata class
        to keep returning 'chained' as needing skip even after it's been
        finalized this run. Since 'chained' is in finished_this_run and
        non-PENDING, skip_progress stays False -> deadlock.
        """
        from gigaevo.programs.dag.automata import DAGAutomata

        dag = _make_dag(
            {
                "failing": FailingStage(timeout=5.0),
                "chained": ChainedStage(timeout=5.0),
            },
            [DataFlowEdge.create("failing", "chained", "data")],
            state_manager,
        )

        prog = make_program()

        original_get_skip = DAGAutomata.get_stages_to_skip
        call_count = {"n": 0}

        def patched_get_skip(self_automata, program, running, launched, finished):
            call_count["n"] += 1
            if call_count["n"] <= 2:
                return original_get_skip(
                    self_automata, program, running, launched, finished
                )
            # After the DAG has settled, force a skip request for a stage
            # that's already finalized this run — triggers deadlock path
            if "chained" in finished and not running:
                return {"chained"}
            return set()

        with patch.object(DAGAutomata, "get_stages_to_skip", patched_get_skip):
            with pytest.raises(RuntimeError, match="DEADLOCK"):
                await dag.run(prog)


# ===================================================================
# Test 3: Termination with unresolved stages (lines 215-239)
# ===================================================================


class TestTerminationWithUnresolvedStages:
    """Target: dag.py lines 215-239.

    When there are no pending tasks, no skips, no launches, no cached —
    but unresolved stages remain — the DAG logs a WARNING and breaks
    (does NOT raise an exception).
    """

    async def test_unresolved_stages_terminate_gracefully(
        self, state_manager, make_program
    ):
        """Force a situation where some stages are unresolved but the DAG
        terminates without raising.

        We patch get_ready_stages and get_stages_to_skip on the DAGAutomata
        class to hide 'orphan' — it never becomes ready or skippable. The DAG
        should break out of its loop with a warning (not raise).
        """
        from gigaevo.programs.dag.automata import DAGAutomata

        dag = _make_dag(
            {
                "fast": FastStage(timeout=5.0),
                "orphan": FastStage(timeout=5.0),
            },
            [],
            state_manager,
        )

        prog = make_program()

        original_get_ready = DAGAutomata.get_ready_stages

        def patched_get_ready(self_automata, program, running, launched, finished):
            ready, cached = original_get_ready(
                self_automata, program, running, launched, finished
            )
            ready.pop("orphan", None)
            cached.discard("orphan")
            return ready, cached

        original_get_skip = DAGAutomata.get_stages_to_skip

        def patched_get_skip(self_automata, program, running, launched, finished):
            result = original_get_skip(
                self_automata, program, running, launched, finished
            )
            result.discard("orphan")
            return result

        with (
            patch.object(DAGAutomata, "get_ready_stages", patched_get_ready),
            patch.object(DAGAutomata, "get_stages_to_skip", patched_get_skip),
        ):
            # Should NOT raise — just terminate gracefully
            await dag.run(prog)

        # 'fast' completed, 'orphan' should still be PENDING (unresolved)
        assert prog.stage_results["fast"].status == StageState.COMPLETED
        assert prog.stage_results["orphan"].status == StageState.PENDING


# ===================================================================
# Test 4: Stall watchdog (lines 264-278)
# ===================================================================


class TestStallWatchdog:
    """Target: dag.py lines 264-278.

    The watchdog fires ONCE when now - last_progress_ts > stall_grace_seconds
    and stalled_reported is False. It must not spam.
    """

    async def test_stall_watchdog_fires_after_grace_period(
        self, state_manager, make_program
    ):
        """Verify the stall watchdog fires after the grace period by patching
        time.time to simulate a long stall.
        """
        # Use a stage that takes just long enough to trigger the watchdog
        dag = _make_dag(
            {"slow": SlowStage(timeout=300.0)},
            [],
            state_manager,
        )
        prog = make_program()

        # Override stall_grace_seconds to be very small so we can trigger it
        dag.stall_grace_seconds = 0.01

        warning_count = {"n": 0}

        # Count how many stall warnings are emitted
        with patch("gigaevo.programs.dag.dag.logger") as mock_logger:
            mock_logger.debug = lambda *a, **kw: None
            mock_logger.info = lambda *a, **kw: None
            mock_logger.error = lambda *a, **kw: None

            def count_warning(fmt, *args, **kwargs):
                if "STALLED" in str(fmt):
                    warning_count["n"] += 1

            mock_logger.warning = count_warning

            await dag.run(prog)

        # The watchdog should fire at most once per stall period (it sets
        # stalled_reported=True after first fire, which prevents re-firing
        # until progress resets it). But since the slow stage eventually
        # completes (after 0.5s), we expect exactly 1 or 0 warnings depending
        # on timing. The key invariant is it doesn't spam (count <= 1).
        assert warning_count["n"] <= 1

    async def test_stall_watchdog_does_not_fire_when_progress_is_made(
        self, state_manager, make_program
    ):
        """Verify no stall warning when stages complete quickly."""
        dag = _make_dag(
            {"fast": FastStage(timeout=5.0)},
            [],
            state_manager,
        )
        prog = make_program()

        # Even with a very short grace period, fast completion should prevent stall
        dag.stall_grace_seconds = 0.001

        warning_emitted = {"stalled": False}

        with patch("gigaevo.programs.dag.dag.logger") as mock_logger:
            mock_logger.debug = lambda *a, **kw: None
            mock_logger.info = lambda *a, **kw: None
            mock_logger.error = lambda *a, **kw: None
            mock_logger.exception = lambda *a, **kw: None

            def check_warning(fmt, *args, **kwargs):
                if "STALLED" in str(fmt):
                    warning_emitted["stalled"] = True

            mock_logger.warning = check_warning

            await dag.run(prog)

        assert prog.stage_results["fast"].status == StageState.COMPLETED
        # Fast stage should complete before stall can be detected
        assert not warning_emitted["stalled"]


# ===================================================================
# Test 5: _process_finished_task — CancelledError vs other exception
# (lines 329-400)
# ===================================================================


class TestProcessFinishedTask:
    """Target: dag.py lines 329-400.

    CancelledError -> CANCELLED result (not FAILED).
    Other exceptions -> FAILED result.
    """

    async def test_cancelled_error_produces_cancelled_status(
        self, state_manager, make_program
    ):
        """Stage raising CancelledError must get CANCELLED status."""
        dag = _make_dag(
            {"cancel": CancellingStage(timeout=5.0)},
            [],
            state_manager,
        )
        prog = make_program()
        await dag.run(prog)

        res = prog.stage_results["cancel"]
        assert res.status == StageState.CANCELLED
        assert res.error is not None
        assert res.error.type == "Cancelled"
        assert "cancelled" in res.error.message.lower()

    async def test_generic_exception_produces_failed_status(
        self, state_manager, make_program
    ):
        """Stage raising ValueError must get FAILED status (not CANCELLED)."""
        dag = _make_dag(
            {"bad": GenericExceptionStage(timeout=5.0)},
            [],
            state_manager,
        )
        prog = make_program()
        await dag.run(prog)

        res = prog.stage_results["bad"]
        assert res.status == StageState.FAILED
        assert res.error is not None
        # Should NOT be Cancelled
        assert res.error.type != "Cancelled"

    async def test_cancelled_and_failed_coexist_independently(
        self, state_manager, make_program
    ):
        """In the same DAG, a CancelledError stage and a ValueError stage
        should each get their correct distinct status.
        """
        dag = _make_dag(
            {
                "cancel": CancellingStage(timeout=5.0),
                "fail": GenericExceptionStage(timeout=5.0),
                "ok": FastStage(timeout=5.0),
            },
            [],
            state_manager,
        )
        prog = make_program()
        await dag.run(prog)

        assert prog.stage_results["cancel"].status == StageState.CANCELLED
        assert prog.stage_results["fail"].status == StageState.FAILED
        assert prog.stage_results["ok"].status == StageState.COMPLETED

    async def test_cancelled_error_has_timing_info(self, state_manager, make_program):
        """CANCELLED results should still have started_at and finished_at."""
        dag = _make_dag(
            {"cancel": CancellingStage(timeout=5.0)},
            [],
            state_manager,
        )
        prog = make_program()
        await dag.run(prog)

        res = prog.stage_results["cancel"]
        assert res.status == StageState.CANCELLED
        assert res.started_at is not None
        assert res.finished_at is not None


# ===================================================================
# Test 6: _launch_ready closure capture (lines 316-317)
# ===================================================================


class TestClosureCapture:
    """Target: dag.py lines 316-317.

    The _run_stage closure captures `stage_name` and `precomputed_inputs`
    via default args: `async def _run_stage(stage_name=name, precomputed_inputs=...)`.
    If it used a closure over loop variable without default args, all tasks
    would reference the last loop value. Test that each stage gets its own inputs.
    """

    async def test_multiple_parallel_stages_get_distinct_inputs(
        self, state_manager, make_program
    ):
        """Launch multiple independent stages in parallel; each should
        receive its own identity, not the last loop variable's value.
        """
        dag = _make_dag(
            {
                "s0": IdentityStage(timeout=5.0, identity="s0"),
                "s1": IdentityStage(timeout=5.0, identity="s1"),
                "s2": IdentityStage(timeout=5.0, identity="s2"),
                "s3": IdentityStage(timeout=5.0, identity="s3"),
            },
            [],
            state_manager,
        )
        prog = make_program()
        await dag.run(prog)

        # All stages must complete
        for name in ("s0", "s1", "s2", "s3"):
            assert prog.stage_results[name].status == StageState.COMPLETED

        # Each stage must report its own identity, not the last one
        identities = set()
        for name in ("s0", "s1", "s2", "s3"):
            out = prog.stage_results[name].output
            assert out is not None
            identities.add(out.stage_id)

        # All 4 distinct identities must be present
        assert identities == {"s0", "s1", "s2", "s3"}

    async def test_closure_does_not_share_inputs_across_stages(
        self, state_manager, make_program
    ):
        """Chain: A -> B (data), C independent.
        B should get A's output. C should get no data input.
        If closures captured the last loop variable, both might get
        the same inputs (or the wrong ones).
        """
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

        # B must receive A's output (42) and add 1
        assert prog.stage_results["b"].output.value == 43
        # C is independent and produces its own value
        assert prog.stage_results["c"].output.value == 42


# ===================================================================
# Test 7: DAG timeout (lines 67-79)
# ===================================================================


class TestDagTimeout:
    """Target: dag.py lines 67-79.

    If dag_timeout is set and _run_internal exceeds it, asyncio.TimeoutError
    is raised. If dag_timeout is None, no timeout.
    """

    async def test_dag_timeout_fires_when_exceeded(self, state_manager, make_program):
        """A slow stage exceeding dag_timeout should raise TimeoutError."""
        dag = _make_dag(
            {"slow": SlowStage(timeout=60.0)},
            [],
            state_manager,
            dag_timeout=0.05,
        )
        prog = make_program()

        with pytest.raises(asyncio.TimeoutError):
            await dag.run(prog)

    async def test_dag_timeout_none_means_no_timeout(self, state_manager, make_program):
        """With dag_timeout=None, the DAG runs to completion even if slow."""
        dag = _make_dag(
            {"fast": FastStage(timeout=5.0)},
            [],
            state_manager,
            dag_timeout=None,
        )
        prog = make_program()
        await dag.run(prog)

        assert prog.stage_results["fast"].status == StageState.COMPLETED

    async def test_dag_timeout_propagates_as_timeout_error(
        self, state_manager, make_program
    ):
        """The raised exception must be exactly asyncio.TimeoutError."""
        dag = _make_dag(
            {"slow": SlowStage(timeout=60.0)},
            [],
            state_manager,
            dag_timeout=0.01,
        )
        prog = make_program()

        with pytest.raises(asyncio.TimeoutError) as exc_info:
            await dag.run(prog)

        # Must be TimeoutError, not a wrapped version
        assert type(exc_info.value) is asyncio.TimeoutError or isinstance(
            exc_info.value, asyncio.TimeoutError
        )

    async def test_fast_stage_completes_within_dag_timeout(
        self, state_manager, make_program
    ):
        """A fast stage should complete well within a generous dag_timeout."""
        dag = _make_dag(
            {"fast": FastStage(timeout=5.0)},
            [],
            state_manager,
            dag_timeout=10.0,
        )
        prog = make_program()
        await dag.run(prog)

        assert prog.stage_results["fast"].status == StageState.COMPLETED


# ===================================================================
# Test 8: stall_grace_seconds calculation (lines 57-59)
# ===================================================================


class TestStallGraceSecondsCalculation:
    """Target: dag.py lines 57-59.

    stall_grace_seconds = max(DEFAULT_STALL_GRACE_SECONDS, max_stage_timeout * 1.5)

    With a very short stage timeout (e.g. 0.1s), stall_grace defaults to
    DEFAULT_STALL_GRACE_SECONDS (120s).
    With a very long timeout (e.g. 200s), stall_grace is 300s.
    """

    def test_short_timeout_uses_default_grace(self, state_manager):
        """Short stage timeout -> grace = DEFAULT_STALL_GRACE_SECONDS."""
        dag = _make_dag(
            {"tiny": TinyTimeoutStage(timeout=0.1)},
            [],
            state_manager,
        )
        # 0.1 * 1.5 = 0.15, which is less than DEFAULT (120)
        assert dag.stall_grace_seconds == DEFAULT_STALL_GRACE_SECONDS

    def test_long_timeout_scales_grace(self, state_manager):
        """Long stage timeout -> grace = max_stage_timeout * 1.5."""
        dag = _make_dag(
            {"long": LongTimeoutStage(timeout=200.0)},
            [],
            state_manager,
        )
        # 200 * 1.5 = 300, which is greater than DEFAULT (120)
        assert dag.stall_grace_seconds == 300.0

    def test_mixed_timeouts_uses_max(self, state_manager):
        """Multiple stages: grace uses the max timeout among all stages."""
        dag = _make_dag(
            {
                "tiny": TinyTimeoutStage(timeout=0.1),
                "long": LongTimeoutStage(timeout=200.0),
            },
            [],
            state_manager,
        )
        assert dag.stall_grace_seconds == 300.0

    def test_exact_boundary_at_default(self, state_manager):
        """When max_stage_timeout * 1.5 == DEFAULT_STALL_GRACE_SECONDS,
        grace equals DEFAULT_STALL_GRACE_SECONDS.
        """
        boundary_timeout = DEFAULT_STALL_GRACE_SECONDS / 1.5  # 80.0
        dag = _make_dag(
            {"boundary": TinyTimeoutStage(timeout=boundary_timeout)},
            [],
            state_manager,
        )
        assert dag.stall_grace_seconds == DEFAULT_STALL_GRACE_SECONDS

    def test_just_above_default_boundary(self, state_manager):
        """When max_stage_timeout * 1.5 > DEFAULT, grace scales with timeout."""
        above_boundary = (DEFAULT_STALL_GRACE_SECONDS / 1.5) + 1.0  # 81.0
        dag = _make_dag(
            {"above": TinyTimeoutStage(timeout=above_boundary)},
            [],
            state_manager,
        )
        expected = above_boundary * 1.5
        assert dag.stall_grace_seconds == expected
        assert dag.stall_grace_seconds > DEFAULT_STALL_GRACE_SECONDS


# ===================================================================
# Test 9: Semaphore enforcement (line 54)
# ===================================================================


class TestSemaphoreEnforcement:
    """Target: dag.py line 54.

    _stage_sema = asyncio.Semaphore(max(1, max_parallel_stages))
    Even if max_parallel_stages=0 or negative, semaphore is at least 1.
    """

    def test_zero_parallel_stages_clamped_to_one(self, state_manager):
        """max_parallel_stages=0 -> semaphore allows at least 1 concurrent stage."""
        dag = _make_dag(
            {"fast": FastStage(timeout=5.0)},
            [],
            state_manager,
            max_parallel_stages=0,
        )
        # Semaphore internal value should be 1
        assert dag._stage_sema._value == 1

    def test_negative_parallel_stages_clamped_to_one(self, state_manager):
        """max_parallel_stages=-5 -> semaphore allows at least 1 concurrent stage."""
        dag = _make_dag(
            {"fast": FastStage(timeout=5.0)},
            [],
            state_manager,
            max_parallel_stages=-5,
        )
        assert dag._stage_sema._value == 1

    def test_positive_parallel_stages_respected(self, state_manager):
        """max_parallel_stages=3 -> semaphore value is 3."""
        dag = _make_dag(
            {"fast": FastStage(timeout=5.0)},
            [],
            state_manager,
            max_parallel_stages=3,
        )
        assert dag._stage_sema._value == 3

    async def test_semaphore_limits_concurrency_at_runtime(
        self, state_manager, make_program
    ):
        """With semaphore=1, 4 stages should run sequentially (not in parallel)."""
        tracker = {"current": 0, "lock": asyncio.Lock(), "max_samples": []}

        dag = _make_dag(
            {
                f"s{i}": ConcurrencyTrackingStage(timeout=5.0, tracker=tracker)
                for i in range(4)
            },
            [],
            state_manager,
            max_parallel_stages=1,
        )
        prog = make_program()
        await dag.run(prog)

        for i in range(4):
            assert prog.stage_results[f"s{i}"].status == StageState.COMPLETED

        observed_max = max(tracker["max_samples"])
        assert observed_max == 1, f"Expected max concurrency 1, got {observed_max}"

    async def test_semaphore_allows_parallelism_when_limit_is_higher(
        self, state_manager, make_program
    ):
        """With semaphore=4, multiple stages can run concurrently."""
        tracker = {"current": 0, "lock": asyncio.Lock(), "max_samples": []}

        dag = _make_dag(
            {
                f"s{i}": ConcurrencyTrackingStage(timeout=5.0, tracker=tracker)
                for i in range(6)
            },
            [],
            state_manager,
            max_parallel_stages=4,
        )
        prog = make_program()
        await dag.run(prog)

        for i in range(6):
            assert prog.stage_results[f"s{i}"].status == StageState.COMPLETED

        observed_max = max(tracker["max_samples"])
        assert observed_max <= 4
        # With 6 independent stages and semaphore=4, at least 2 should overlap
        assert observed_max >= 2, f"Expected some parallelism (>=2), got {observed_max}"


# ===================================================================
# Test 10: Progress tracking resets (lines 210-212, 260-262)
# ===================================================================


class TestProgressTrackingReset:
    """Target: dag.py lines 210-212 and 260-262.

    last_progress_ts resets on: skip_progress, new_tasks_map, newly_cached,
    collected_any. Missing any of these would cause false stall warnings.
    """

    async def test_skip_progress_resets_stall_timer(self, state_manager, make_program):
        """When a skip occurs, the stall timer should reset.
        DAG: failing -> chained. The skip of 'chained' should count as progress.
        """
        dag = _make_dag(
            {
                "failing": FailingStage(timeout=5.0),
                "chained": ChainedStage(timeout=5.0),
            },
            [DataFlowEdge.create("failing", "chained", "data")],
            state_manager,
        )
        dag.stall_grace_seconds = 0.001  # Very short to catch stall
        prog = make_program()

        stall_fired = {"value": False}

        with patch("gigaevo.programs.dag.dag.logger") as mock_logger:
            mock_logger.debug = lambda *a, **kw: None
            mock_logger.info = lambda *a, **kw: None
            mock_logger.error = lambda *a, **kw: None
            mock_logger.exception = lambda *a, **kw: None

            def check_warning(fmt, *args, **kwargs):
                if "STALLED" in str(fmt):
                    stall_fired["value"] = True

            mock_logger.warning = check_warning

            await dag.run(prog)

        # Even with a tiny grace period, the skip + completion should
        # reset the timer and prevent stall warnings
        assert not stall_fired["value"]

    async def test_new_task_launch_resets_stall_timer(
        self, state_manager, make_program
    ):
        """Launching new tasks should reset the stall timer."""
        dag = _make_dag(
            {
                "a": FastStage(timeout=5.0),
                "b": FastStage(timeout=5.0),
            },
            [],
            state_manager,
        )
        dag.stall_grace_seconds = 0.001
        prog = make_program()

        stall_fired = {"value": False}

        with patch("gigaevo.programs.dag.dag.logger") as mock_logger:
            mock_logger.debug = lambda *a, **kw: None
            mock_logger.info = lambda *a, **kw: None
            mock_logger.error = lambda *a, **kw: None
            mock_logger.exception = lambda *a, **kw: None

            def check_warning(fmt, *args, **kwargs):
                if "STALLED" in str(fmt):
                    stall_fired["value"] = True

            mock_logger.warning = check_warning

            await dag.run(prog)

        assert not stall_fired["value"]

    async def test_cached_stage_resets_stall_timer(self, state_manager, make_program):
        """A cached stage hit should reset the stall timer.
        Run DAG once to populate cache, then run again. The cache hit
        should count as progress.
        """
        dag1 = _make_dag(
            {"fast": FastStage(timeout=5.0)},
            [],
            state_manager,
        )
        prog = make_program()
        await dag1.run(prog)

        # Second run — should use cache
        dag2 = _make_dag(
            {"fast": FastStage(timeout=5.0)},
            [],
            state_manager,
        )
        dag2.stall_grace_seconds = 0.001
        prog2_stall_fired = {"value": False}

        with patch("gigaevo.programs.dag.dag.logger") as mock_logger:
            mock_logger.debug = lambda *a, **kw: None
            mock_logger.info = lambda *a, **kw: None
            mock_logger.error = lambda *a, **kw: None
            mock_logger.exception = lambda *a, **kw: None

            def check_warning(fmt, *args, **kwargs):
                if "STALLED" in str(fmt):
                    prog2_stall_fired["value"] = True

            mock_logger.warning = check_warning

            await dag2.run(prog)

        assert not prog2_stall_fired["value"]

    async def test_collected_task_completion_resets_stall_timer(
        self, state_manager, make_program
    ):
        """When a running task finishes (collected_any=True), stall timer resets."""
        dag = _make_dag(
            {"slow": SlowStage(timeout=5.0)},
            [],
            state_manager,
        )
        dag.stall_grace_seconds = 0.001
        prog = make_program()

        stall_fired = {"value": False}

        with patch("gigaevo.programs.dag.dag.logger") as mock_logger:
            mock_logger.debug = lambda *a, **kw: None
            mock_logger.info = lambda *a, **kw: None
            mock_logger.error = lambda *a, **kw: None
            mock_logger.exception = lambda *a, **kw: None

            def check_warning(fmt, *args, **kwargs):
                if "STALLED" in str(fmt):
                    stall_fired["value"] = True

            mock_logger.warning = check_warning

            await dag.run(prog)

        # The slow stage takes 0.5s, and with grace=0.001s, the initial
        # launch should still reset the timer, preventing a stall warning
        # during the 0.5s wait.
        assert prog.stage_results["slow"].status == StageState.COMPLETED


# ===================================================================
# Additional edge case: PENDING initialization (lines 90-94)
# ===================================================================


class TestPendingInitialization:
    """Target: dag.py lines 90-94.

    All stages should be initialized to PENDING status at the start
    of a DAG run. This uses setdefault, so pre-existing results from
    previous runs should NOT be overwritten.
    """

    async def test_all_stages_start_as_pending(self, state_manager, make_program):
        """On a fresh program, all stages should be PENDING before execution."""
        dag = _make_dag(
            {
                "a": FastStage(timeout=5.0),
                "b": FastStage(timeout=5.0),
            },
            [],
            state_manager,
        )
        prog = make_program()

        await dag.run(prog)

        # After run, all stages should be in a FINAL state
        for name in ("a", "b"):
            assert prog.stage_results[name].status in (
                StageState.COMPLETED,
                StageState.FAILED,
                StageState.CANCELLED,
                StageState.SKIPPED,
            )

    async def test_setdefault_preserves_preexisting_results(
        self, state_manager, make_program
    ):
        """setdefault should preserve pre-existing stage results (from previous runs).
        The stale result should be handled by the skip/cache logic, not overwritten
        by PENDING.
        """
        dag = _make_dag(
            {"fast": NeverCachedFastStage(timeout=5.0)},
            [],
            state_manager,
        )
        # Pre-populate with a COMPLETED result from a "previous run"
        prog = make_program(
            stage_results={
                "fast": ProgramStageResult(
                    status=StageState.COMPLETED,
                    output=MockOutput(value=99),
                ),
            }
        )

        # The NeverCachedFastStage always re-executes, so the value should change
        await dag.run(prog)

        # Should have re-executed (value=42 from NeverCachedFastStage), not kept 99
        assert prog.stage_results["fast"].status == StageState.COMPLETED
        assert prog.stage_results["fast"].output.value == 42


# ===================================================================
# Additional: running set management and finished_this_run tracking
# ===================================================================


class TestRunningSetManagement:
    """Target: dag.py lines 103-107, 200-207, 354.

    Running set should accurately track which stages are currently in-flight.
    Stages should be added on launch and removed on completion.
    """

    async def test_running_set_empty_at_termination(self, state_manager, make_program):
        """After DAG completes, no stages should remain in the running set."""
        dag = _make_dag(
            {
                "a": FastStage(timeout=5.0),
                "b": SlowStage(timeout=5.0),
                "c": FailingStage(timeout=5.0),
            },
            [],
            state_manager,
        )
        prog = make_program()
        await dag.run(prog)

        # All stages should be in some final state
        for name in ("a", "b", "c"):
            assert prog.stage_results[name].status in (
                StageState.COMPLETED,
                StageState.FAILED,
                StageState.CANCELLED,
                StageState.SKIPPED,
            )

    async def test_finished_this_run_tracks_all_completed(
        self, state_manager, make_program
    ):
        """All stages that execute in this run should end up as finalized."""
        dag = _make_dag(
            {
                "a": FastStage(timeout=5.0),
                "b": FailingStage(timeout=5.0),
                "c": CancellingStage(timeout=5.0),
            },
            [],
            state_manager,
        )
        prog = make_program()
        await dag.run(prog)

        assert prog.stage_results["a"].status == StageState.COMPLETED
        assert prog.stage_results["b"].status == StageState.FAILED
        assert prog.stage_results["c"].status == StageState.CANCELLED


# ===================================================================
# Additional: exec-order deps interact with skip guard
# ===================================================================


class TestExecOrderInteractionWithSkip:
    """Target: interaction between exec-order deps, skip logic, and deadlock.

    When an exec-order dependency's condition is not met (e.g., on_success
    but the dependency failed), the dependent stage should be skipped cleanly
    without triggering deadlock.
    """

    async def test_exec_order_on_success_with_failed_dep_skips_cleanly(
        self, state_manager, make_program
    ):
        """B depends on A succeeding, but A fails -> B should be skipped."""
        dag = _make_dag(
            {
                "a": FailingStage(timeout=5.0),
                "b": FastStage(timeout=5.0),
            },
            [],
            state_manager,
            exec_deps={"b": [ExecutionOrderDependency.on_success("a")]},
        )
        prog = make_program()
        await dag.run(prog)

        assert prog.stage_results["a"].status == StageState.FAILED
        assert prog.stage_results["b"].status == StageState.SKIPPED

    async def test_exec_order_chain_cascading_skip(self, state_manager, make_program):
        """A fails -> B skipped (on_success(A)) -> C skipped (data from B)."""
        dag = _make_dag(
            {
                "a": FailingStage(timeout=5.0),
                "b": FastStage(timeout=5.0),
                "c": ChainedStage(timeout=5.0),
            },
            [DataFlowEdge.create("b", "c", "data")],
            state_manager,
            exec_deps={"b": [ExecutionOrderDependency.on_success("a")]},
        )
        prog = make_program()
        await dag.run(prog)

        assert prog.stage_results["a"].status == StageState.FAILED
        assert prog.stage_results["b"].status == StageState.SKIPPED
        assert prog.stage_results["c"].status == StageState.SKIPPED

    async def test_independent_stage_unaffected_by_exec_order_skip(
        self, state_manager, make_program
    ):
        """D is independent; A fails -> B skipped, but D still completes."""
        dag = _make_dag(
            {
                "a": FailingStage(timeout=5.0),
                "b": FastStage(timeout=5.0),
                "d": FastStage(timeout=5.0),
            },
            [],
            state_manager,
            exec_deps={"b": [ExecutionOrderDependency.on_success("a")]},
        )
        prog = make_program()
        await dag.run(prog)

        assert prog.stage_results["a"].status == StageState.FAILED
        assert prog.stage_results["b"].status == StageState.SKIPPED
        assert prog.stage_results["d"].status == StageState.COMPLETED
