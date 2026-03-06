"""Tests for gigaevo/utils/metrics_tracker.py."""

from __future__ import annotations

import asyncio
import math
from typing import Any
from unittest.mock import AsyncMock

import pytest

from gigaevo.programs.metrics.context import VALIDITY_KEY, MetricsContext, MetricSpec
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState
from gigaevo.utils.metrics_tracker import MetricsTracker, _RunningStats
from gigaevo.utils.trackers.base import LogWriter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class RecordingWriter(LogWriter):
    """Captures all scalar() calls for assertion."""

    def __init__(self) -> None:
        self.scalars: list[tuple[str, float, dict[str, Any]]] = []

    def bind(self, path: list[str]) -> RecordingWriter:
        return self

    def scalar(self, metric: str, value: float, **kwargs: Any) -> None:
        self.scalars[len(self.scalars) :] = [(metric, value, kwargs)]

    def hist(self, metric: str, values: list[float], **kwargs: Any) -> None:
        pass

    def text(self, tag: str, text: str, **kwargs: Any) -> None:
        pass

    def close(self) -> None:
        pass


def _make_metrics_context(
    *,
    higher_is_better: bool = True,
    extra_specs: dict[str, MetricSpec] | None = None,
) -> MetricsContext:
    """Build a minimal MetricsContext with a primary 'score' metric."""
    specs: dict[str, MetricSpec] = {
        "score": MetricSpec(
            description="Primary score",
            is_primary=True,
            higher_is_better=higher_is_better,
        ),
    }
    if extra_specs:
        specs.update(extra_specs)
    return MetricsContext(specs=specs)


def _make_program(
    *,
    state: ProgramState = ProgramState.DONE,
    metrics: dict[str, float] | None = None,
    iteration: int = 1,
    generation: int = 1,
    prog_id: str | None = None,
) -> Program:
    """Create a minimal Program suitable for MetricsTracker tests."""
    p = Program(
        code="def solve(): return 42",
        state=state,
        atomic_counter=999_999_999,
    )
    if prog_id is not None:
        # overwrite the random UUID
        object.__setattr__(p, "id", prog_id)
    if metrics:
        p.add_metrics(metrics)
    p.metadata["iteration"] = iteration
    p.lineage.generation = generation
    return p


def _mock_storage(programs: list[Program]) -> AsyncMock:
    """Return an AsyncMock ProgramStorage that serves the given programs."""
    storage = AsyncMock()
    storage.get_all_program_ids = AsyncMock(return_value=[p.id for p in programs])
    storage.mget = AsyncMock(return_value=programs)
    return storage


# ---------------------------------------------------------------------------
# TestRunningStats
# ---------------------------------------------------------------------------


class TestRunningStats:
    """Unit tests for the Welford running-stats accumulator."""

    def test_initial_state(self) -> None:
        rs = _RunningStats()
        assert rs.n == 0
        assert rs.mean_value() == 0.0
        assert rs.std_value() == 0.0

    def test_single_value(self) -> None:
        rs = _RunningStats()
        rs.update(5.0)
        assert rs.n == 1
        assert rs.mean_value() == 5.0
        # std is 0 with a single sample (n <= 1)
        assert rs.std_value() == 0.0

    def test_multi_values_mean(self) -> None:
        rs = _RunningStats()
        values = [2.0, 4.0, 6.0, 8.0]
        for v in values:
            rs.update(v)
        assert rs.n == 4
        assert rs.mean_value() == pytest.approx(5.0)

    def test_welford_vs_numpy(self) -> None:
        """Running stats should match numpy-style calculations."""
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        rs = _RunningStats()
        for v in values:
            rs.update(v)

        expected_mean = sum(values) / len(values)
        variance = sum((x - expected_mean) ** 2 for x in values) / (len(values) - 1)
        expected_std = math.sqrt(variance)

        assert rs.mean_value() == pytest.approx(expected_mean)
        assert rs.std_value() == pytest.approx(expected_std)

    def test_negative_values(self) -> None:
        rs = _RunningStats()
        values = [-3.0, -1.0, 1.0, 3.0]
        for v in values:
            rs.update(v)
        assert rs.mean_value() == pytest.approx(0.0)
        variance = sum((x - 0.0) ** 2 for x in values) / (len(values) - 1)
        assert rs.std_value() == pytest.approx(math.sqrt(variance))

    def test_constant_values(self) -> None:
        rs = _RunningStats()
        for _ in range(10):
            rs.update(7.0)
        assert rs.mean_value() == pytest.approx(7.0)
        assert rs.std_value() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# TestProcessProgram
# ---------------------------------------------------------------------------


class TestProcessProgram:
    """Tests for MetricsTracker._process_program."""

    @pytest.mark.asyncio
    async def test_skip_queued_program(self) -> None:
        prog = _make_program(state=ProgramState.QUEUED)
        writer = RecordingWriter()
        ctx = _make_metrics_context()
        tracker = MetricsTracker(
            storage=_mock_storage([]),
            metrics_context=ctx,
            writer=writer,
        )
        result = await tracker._process_program(prog)
        assert result is False
        assert len(writer.scalars) == 0

    @pytest.mark.asyncio
    async def test_skip_running_program(self) -> None:
        prog = _make_program(state=ProgramState.RUNNING)
        writer = RecordingWriter()
        ctx = _make_metrics_context()
        tracker = MetricsTracker(
            storage=_mock_storage([]),
            metrics_context=ctx,
            writer=writer,
        )
        result = await tracker._process_program(prog)
        assert result is False
        assert len(writer.scalars) == 0

    @pytest.mark.asyncio
    async def test_skip_no_validity(self) -> None:
        """A DONE program without is_valid in metrics should be skipped."""
        prog = _make_program(metrics={"score": 42.0})
        writer = RecordingWriter()
        ctx = _make_metrics_context()
        tracker = MetricsTracker(
            storage=_mock_storage([]),
            metrics_context=ctx,
            writer=writer,
        )
        result = await tracker._process_program(prog)
        assert result is False

    @pytest.mark.asyncio
    async def test_invalid_program_counts(self) -> None:
        """An invalid program increments invalid_count, writes validity + counts."""
        prog = _make_program(metrics={VALIDITY_KEY: 0.0, "score": 10.0})
        writer = RecordingWriter()
        ctx = _make_metrics_context()
        tracker = MetricsTracker(
            storage=_mock_storage([]),
            metrics_context=ctx,
            writer=writer,
        )
        result = await tracker._process_program(prog)
        assert result is True
        assert tracker._invalid_count == 1
        assert tracker._valid_count == 0

        tag_to_val = {t: v for t, v, _ in writer.scalars}
        assert tag_to_val[VALIDITY_KEY] == 0.0
        assert tag_to_val["programs/invalid_count"] == pytest.approx(1.0)
        assert tag_to_val["programs/valid_count"] == pytest.approx(0.0)
        assert tag_to_val["programs/total_count"] == pytest.approx(1.0)
        # Per-metric tags must NOT appear for invalid programs
        assert "valid/program/score" not in tag_to_val
        assert "valid/frontier/score" not in tag_to_val

    @pytest.mark.asyncio
    async def test_valid_program_metrics(self) -> None:
        """A valid program writes per-program metrics and frontier with correct values."""
        prog = _make_program(
            metrics={VALIDITY_KEY: 1.0, "score": 99.0},
            iteration=3,
            generation=2,
        )
        writer = RecordingWriter()
        ctx = _make_metrics_context()
        tracker = MetricsTracker(
            storage=_mock_storage([]),
            metrics_context=ctx,
            writer=writer,
        )
        result = await tracker._process_program(prog)
        assert result is True
        assert tracker._valid_count == 1

        tag_to_val = {t: v for t, v, _ in writer.scalars}
        # Per-program value
        assert tag_to_val["valid/program/score"] == pytest.approx(99.0)
        # Frontier (first value, so always set)
        assert tag_to_val["valid/frontier/score"] == pytest.approx(99.0)
        # Iteration aggregates (single sample: mean=value, std=0)
        assert tag_to_val["valid/iter/score/mean"] == pytest.approx(99.0)
        assert tag_to_val["valid/iter/score/std"] == pytest.approx(0.0)
        # Generation aggregates
        assert tag_to_val["valid/gen/score/mean"] == pytest.approx(99.0)
        # Validity flag for valid program should be 1.0
        assert tag_to_val[VALIDITY_KEY] == pytest.approx(1.0)

        # Verify step arguments
        iter_mean = [
            (t, v, kw) for t, v, kw in writer.scalars if t == "valid/iter/score/mean"
        ]
        assert iter_mean[-1][2].get("step") == 3
        gen_mean = [
            (t, v, kw) for t, v, kw in writer.scalars if t == "valid/gen/score/mean"
        ]
        assert gen_mean[-1][2].get("step") == 2
        frontier = [
            (t, v, kw) for t, v, kw in writer.scalars if t == "valid/frontier/score"
        ]
        assert frontier[-1][2].get("step") == 3

    @pytest.mark.asyncio
    async def test_frontier_improves_on_higher_is_better(self) -> None:
        """Frontier updates when a higher score comes in (higher_is_better=True)."""
        writer = RecordingWriter()
        ctx = _make_metrics_context(higher_is_better=True)
        tracker = MetricsTracker(
            storage=_mock_storage([]),
            metrics_context=ctx,
            writer=writer,
        )

        prog1 = _make_program(metrics={VALIDITY_KEY: 1.0, "score": 10.0}, iteration=1)
        await tracker._process_program(prog1)
        assert tracker._best_valid["score"] == (10.0, 1)

        prog2 = _make_program(metrics={VALIDITY_KEY: 1.0, "score": 20.0}, iteration=2)
        await tracker._process_program(prog2)
        assert tracker._best_valid["score"] == (20.0, 2)

    @pytest.mark.asyncio
    async def test_frontier_no_regress(self) -> None:
        """Frontier does NOT update when new value is worse."""
        writer = RecordingWriter()
        ctx = _make_metrics_context(higher_is_better=True)
        tracker = MetricsTracker(
            storage=_mock_storage([]),
            metrics_context=ctx,
            writer=writer,
        )

        prog1 = _make_program(metrics={VALIDITY_KEY: 1.0, "score": 50.0}, iteration=1)
        await tracker._process_program(prog1)

        prog2 = _make_program(metrics={VALIDITY_KEY: 1.0, "score": 30.0}, iteration=2)
        await tracker._process_program(prog2)
        # Frontier should still be at the first value
        assert tracker._best_valid["score"] == (50.0, 1)

    @pytest.mark.asyncio
    async def test_valid_program_with_non_numeric_metric_ignored(self) -> None:
        """Non-numeric metric values should be silently skipped."""
        prog = _make_program(
            metrics={VALIDITY_KEY: 1.0, "score": 42.0},
            iteration=1,
        )
        # Manually inject a non-numeric metric
        prog.metrics["label"] = float("nan")  # float but we test the int/float guard
        # Actually, to test the isinstance guard we need a non-float. But
        # Program.add_metrics coerces everything to float. So we test the
        # VALIDITY_KEY skip path instead by verifying no "valid/program/is_valid" tag.
        writer = RecordingWriter()
        ctx = _make_metrics_context()
        tracker = MetricsTracker(
            storage=_mock_storage([]),
            metrics_context=ctx,
            writer=writer,
        )
        await tracker._process_program(prog)
        tags = [s[0] for s in writer.scalars]
        # is_valid should NOT appear as "valid/program/is_valid" because
        # the loop skips key == VALIDITY_KEY
        assert "valid/program/is_valid" not in tags


# ---------------------------------------------------------------------------
# TestDrainOnce
# ---------------------------------------------------------------------------


class TestDrainOnce:
    """Tests for MetricsTracker._drain_once."""

    @pytest.mark.asyncio
    async def test_new_programs_processed(self) -> None:
        prog = _make_program(metrics={VALIDITY_KEY: 1.0, "score": 5.0}, iteration=1)
        storage = _mock_storage([prog])
        writer = RecordingWriter()
        ctx = _make_metrics_context()
        tracker = MetricsTracker(
            storage=storage,
            metrics_context=ctx,
            writer=writer,
        )
        await tracker._drain_once()
        assert prog.id in tracker._seen_ids
        tag_to_val = {t: v for t, v, _ in writer.scalars}
        assert tag_to_val["valid/program/score"] == pytest.approx(5.0)
        assert tag_to_val["programs/valid_count"] == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_skip_already_seen(self) -> None:
        prog = _make_program(metrics={VALIDITY_KEY: 1.0, "score": 5.0}, iteration=1)
        storage = _mock_storage([prog])
        writer = RecordingWriter()
        ctx = _make_metrics_context()
        tracker = MetricsTracker(
            storage=storage,
            metrics_context=ctx,
            writer=writer,
        )
        # First drain
        await tracker._drain_once()
        count_after_first = len(writer.scalars)

        # Second drain — same ids returned but already seen
        await tracker._drain_once()
        assert len(writer.scalars) == count_after_first
        # mget must not be called for already-seen ids
        assert storage.mget.call_count == 1

    @pytest.mark.asyncio
    async def test_empty_storage(self) -> None:
        storage = _mock_storage([])
        writer = RecordingWriter()
        ctx = _make_metrics_context()
        tracker = MetricsTracker(
            storage=storage,
            metrics_context=ctx,
            writer=writer,
        )
        await tracker._drain_once()
        assert len(writer.scalars) == 0
        assert len(tracker._seen_ids) == 0

    @pytest.mark.asyncio
    async def test_none_program_in_mget(self) -> None:
        """If mget returns None entries, they should be skipped."""
        prog = _make_program(metrics={VALIDITY_KEY: 1.0, "score": 5.0}, iteration=1)
        storage = AsyncMock()
        storage.get_all_program_ids = AsyncMock(return_value=["id1", prog.id])
        # mget returns None for the first id, valid program for the second
        storage.mget = AsyncMock(return_value=[None, prog])
        writer = RecordingWriter()
        ctx = _make_metrics_context()
        tracker = MetricsTracker(
            storage=storage,
            metrics_context=ctx,
            writer=writer,
        )
        await tracker._drain_once()
        # Only the valid program should be marked as seen
        assert prog.id in tracker._seen_ids
        assert "id1" not in tracker._seen_ids


# ---------------------------------------------------------------------------
# TestMaybeUpdateFrontier
# ---------------------------------------------------------------------------


class TestMaybeUpdateFrontier:
    """Tests for MetricsTracker._maybe_update_frontier."""

    def test_first_value_always_updates(self) -> None:
        ctx = _make_metrics_context()
        writer = RecordingWriter()
        tracker = MetricsTracker(
            storage=_mock_storage([]),
            metrics_context=ctx,
            writer=writer,
        )
        result = tracker._maybe_update_frontier("score", 10.0, iteration=1)
        assert result is True
        assert tracker._best_valid["score"] == (10.0, 1)

    def test_lower_is_better(self) -> None:
        ctx = _make_metrics_context(higher_is_better=False)
        writer = RecordingWriter()
        tracker = MetricsTracker(
            storage=_mock_storage([]),
            metrics_context=ctx,
            writer=writer,
        )
        # First value
        tracker._maybe_update_frontier("score", 10.0, iteration=1)

        # Lower value should improve
        result = tracker._maybe_update_frontier("score", 5.0, iteration=2)
        assert result is True
        assert tracker._best_valid["score"] == (5.0, 2)

        # Higher value should NOT improve
        result = tracker._maybe_update_frontier("score", 15.0, iteration=3)
        assert result is False
        assert tracker._best_valid["score"] == (5.0, 2)

    def test_unknown_metric_defaults_higher_is_better(self) -> None:
        """A metric not in specs should default to higher_is_better=True."""
        ctx = _make_metrics_context()
        writer = RecordingWriter()
        tracker = MetricsTracker(
            storage=_mock_storage([]),
            metrics_context=ctx,
            writer=writer,
        )
        # "unknown_metric" is not in the specs
        tracker._maybe_update_frontier("unknown_metric", 10.0, iteration=1)
        assert tracker._best_valid["unknown_metric"] == (10.0, 1)

        # Higher value should improve (default higher_is_better)
        result = tracker._maybe_update_frontier("unknown_metric", 20.0, iteration=2)
        assert result is True
        assert tracker._best_valid["unknown_metric"] == (20.0, 2)

        # Lower value should NOT improve
        result = tracker._maybe_update_frontier("unknown_metric", 5.0, iteration=3)
        assert result is False
        assert tracker._best_valid["unknown_metric"] == (20.0, 2)


# ---------------------------------------------------------------------------
# TestLifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    """Tests for start/stop task management."""

    @pytest.mark.asyncio
    async def test_start_stop(self) -> None:
        storage = _mock_storage([])
        # Make get_all_program_ids return empty each time
        storage.get_all_program_ids = AsyncMock(return_value=[])
        writer = RecordingWriter()
        ctx = _make_metrics_context()
        tracker = MetricsTracker(
            storage=storage,
            metrics_context=ctx,
            writer=writer,
            interval=0.05,
        )

        loop = asyncio.get_running_loop()
        tracker.start(loop)

        assert tracker._task is not None
        assert not tracker._task.done()
        assert tracker._running is True

        # Let it tick at least once
        await asyncio.sleep(0.1)

        await tracker.stop()
        assert tracker._running is False
        assert tracker._task is None


# ---------------------------------------------------------------------------
# Additional tests from audit: value assertions, edge cases, missing paths
# ---------------------------------------------------------------------------


class TestValidityThresholdBoundary:
    """Verify the 0.5 threshold: v >= 0.5 is valid, v < 0.5 is invalid."""

    @pytest.mark.parametrize(
        "validity_val,expected_valid",
        [
            (0.0, False),
            (0.4999, False),
            (0.5, True),
            (0.5001, True),
            (1.0, True),
        ],
    )
    @pytest.mark.asyncio
    async def test_validity_threshold(
        self, validity_val: float, expected_valid: bool
    ) -> None:
        prog = _make_program(metrics={VALIDITY_KEY: validity_val, "score": 1.0})
        writer = RecordingWriter()
        ctx = _make_metrics_context()
        tracker = MetricsTracker(
            storage=_mock_storage([]), metrics_context=ctx, writer=writer
        )
        await tracker._process_program(prog)
        if expected_valid:
            assert tracker._valid_count == 1
            assert tracker._invalid_count == 0
        else:
            assert tracker._valid_count == 0
            assert tracker._invalid_count == 1


class TestMultiProgramDrain:
    """Multiple programs in a single _drain_once call — accumulation."""

    @pytest.mark.asyncio
    async def test_drain_once_accumulates_multiple_programs(self) -> None:
        programs = [
            _make_program(
                metrics={VALIDITY_KEY: 1.0, "score": s},
                iteration=1,
                prog_id=f"prog-{i}",
            )
            for i, s in enumerate([10.0, 20.0, 30.0])
        ]
        storage = _mock_storage(programs)
        writer = RecordingWriter()
        ctx = _make_metrics_context()
        tracker = MetricsTracker(storage=storage, metrics_context=ctx, writer=writer)
        await tracker._drain_once()

        assert tracker._valid_count == 3
        # Last total_count write should be 3.0
        total_writes = [v for t, v, _ in writer.scalars if t == "programs/total_count"]
        assert total_writes[-1] == pytest.approx(3.0)

        # Iter mean after 3 programs: (10+20+30)/3 = 20.0
        iter_mean = [v for t, v, _ in writer.scalars if t == "valid/iter/score/mean"]
        assert iter_mean[-1] == pytest.approx(20.0)

        # Frontier should be the best (30.0 for higher_is_better=True)
        assert tracker._best_valid["score"] == (30.0, 1)


class TestCrossIterationIsolation:
    """Programs from different iterations get separate _iter_stats buckets."""

    @pytest.mark.asyncio
    async def test_iter_stats_separated_by_iteration(self) -> None:
        prog1 = _make_program(metrics={VALIDITY_KEY: 1.0, "score": 10.0}, iteration=1)
        prog2 = _make_program(metrics={VALIDITY_KEY: 1.0, "score": 30.0}, iteration=2)
        writer = RecordingWriter()
        ctx = _make_metrics_context()
        tracker = MetricsTracker(
            storage=_mock_storage([]), metrics_context=ctx, writer=writer
        )
        await tracker._process_program(prog1)
        await tracker._process_program(prog2)

        assert tracker._iter_stats[1]["score"].n == 1
        assert tracker._iter_stats[2]["score"].n == 1
        assert tracker._iter_stats[1]["score"].mean_value() == pytest.approx(10.0)
        assert tracker._iter_stats[2]["score"].mean_value() == pytest.approx(30.0)

    @pytest.mark.asyncio
    async def test_gen_stats_separated_by_generation(self) -> None:
        prog1 = _make_program(
            metrics={VALIDITY_KEY: 1.0, "score": 5.0}, iteration=1, generation=1
        )
        prog2 = _make_program(
            metrics={VALIDITY_KEY: 1.0, "score": 15.0}, iteration=2, generation=2
        )
        writer = RecordingWriter()
        ctx = _make_metrics_context()
        tracker = MetricsTracker(
            storage=_mock_storage([]), metrics_context=ctx, writer=writer
        )
        await tracker._process_program(prog1)
        await tracker._process_program(prog2)

        assert tracker._gen_stats[1]["score"].mean_value() == pytest.approx(5.0)
        assert tracker._gen_stats[2]["score"].mean_value() == pytest.approx(15.0)


class TestFrontierNotWrittenWhenNotImproved:
    """Frontier scalars should NOT be re-written for non-improving programs."""

    @pytest.mark.asyncio
    async def test_no_frontier_write_for_worse_score(self) -> None:
        writer = RecordingWriter()
        ctx = _make_metrics_context(higher_is_better=True)
        tracker = MetricsTracker(
            storage=_mock_storage([]), metrics_context=ctx, writer=writer
        )

        prog1 = _make_program(metrics={VALIDITY_KEY: 1.0, "score": 50.0}, iteration=1)
        await tracker._process_program(prog1)
        frontier_count_after_first = len(
            [t for t, _, _ in writer.scalars if t == "valid/frontier/score"]
        )

        prog2 = _make_program(metrics={VALIDITY_KEY: 1.0, "score": 30.0}, iteration=2)
        await tracker._process_program(prog2)
        frontier_count_after_second = len(
            [t for t, _, _ in writer.scalars if t == "valid/frontier/score"]
        )
        assert frontier_count_after_second == frontier_count_after_first


class TestStartAlreadyRunning:
    """Calling start() twice should not create a second task."""

    @pytest.mark.asyncio
    async def test_double_start_is_noop(self) -> None:
        storage = _mock_storage([])
        storage.get_all_program_ids = AsyncMock(return_value=[])
        writer = RecordingWriter()
        ctx = _make_metrics_context()
        tracker = MetricsTracker(
            storage=storage, metrics_context=ctx, writer=writer, interval=0.05
        )
        loop = asyncio.get_running_loop()
        tracker.start(loop)
        first_task = tracker._task
        tracker.start(loop)  # should be a no-op
        assert tracker._task is first_task
        await tracker.stop()


class TestRunningStatsN2Boundary:
    """The n=2 transition from std=0 to a real standard deviation."""

    def test_two_element_std(self) -> None:
        rs = _RunningStats()
        rs.update(10.0)
        rs.update(20.0)
        assert rs.n == 2
        assert rs.mean_value() == pytest.approx(15.0)
        # Sample std of [10, 20] = sqrt((10-15)^2 + (20-15)^2) / (2-1)) = sqrt(50) ≈ 7.071
        assert rs.std_value() == pytest.approx(math.sqrt(50.0))
