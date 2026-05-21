"""Smoke tests for the live frontier-comparison daemon.

Mirrors the test style of ``tests/monitoring/test_live_profiler.py``:
we test the pure compute helper (``compute_snapshot``) and the
thread-start contract (``start_live_frontier_compare``) — not the actual
loop timing, which would make tests flaky on shared CI infrastructure.
"""

from __future__ import annotations

import threading

import pytest

from gigaevo.monitoring.live_frontier_compare import (
    FrontierCompareSnapshot,
    MetricComparison,
    _render_frontier_plot,
    compute_snapshot,
    format_snapshot,
    start_live_frontier_compare,
)


class TestComputeSnapshot:
    def test_returns_empty_snapshot_when_no_data(self) -> None:
        snap = compute_snapshot(
            metrics=["fitness"],
            frontier_history={"fitness": []},
            iter_mean_history={"fitness": []},
            program_history={"fitness": []},
            higher_is_better={"fitness": True},
        )
        assert snap.metrics == {}

    def test_higher_is_better_positive_delta_means_improvement(self) -> None:
        # frontier-best = 0.5 (steady), current iteration best = 0.7.
        # delta_best = current_best - frontier_best = +0.2 → improvement.
        snap = compute_snapshot(
            metrics=["fitness"],
            frontier_history={"fitness": [(0, 0.5), (1, 0.5)]},
            iter_mean_history={"fitness": [(0, 0.3), (1, 0.5)]},
            program_history={
                "fitness": [(0, 0.3), (0, 0.4), (1, 0.5), (1, 0.7)],
            },
            higher_is_better={"fitness": True},
        )
        comp = snap.metrics["fitness"]
        assert comp.frontier_best == pytest.approx(0.5)
        assert comp.current_best == pytest.approx(0.7)
        assert comp.delta_best == pytest.approx(0.2)
        # delta_best_sign tracks "improvement direction" relative to
        # higher_is_better. A positive delta_best with higher_is_better=True
        # means improvement.
        assert comp.delta_best_sign == "+"
        # current_mean = latest per-iter mean = 0.5; frontier_mean = mean over
        # frontier values = (0.5 + 0.5) / 2 = 0.5; delta_mean = 0.0.
        assert comp.current_mean == pytest.approx(0.5)
        assert comp.frontier_mean == pytest.approx(0.5)

    def test_lower_is_better_inverts_delta_sign(self) -> None:
        # frontier-best = 10.0 (lower is better), current best = 8.0.
        # delta_best = current - frontier = -2.0; for lower_is_better this is
        # an *improvement*, so delta_best_sign is "+".
        snap = compute_snapshot(
            metrics=["loss"],
            frontier_history={"loss": [(0, 10.0)]},
            iter_mean_history={"loss": [(0, 12.0)]},
            program_history={"loss": [(0, 8.0), (0, 9.0)]},
            higher_is_better={"loss": False},
        )
        comp = snap.metrics["loss"]
        assert comp.frontier_best == pytest.approx(10.0)
        # current best with lower_is_better is the *min* over the latest iter.
        assert comp.current_best == pytest.approx(8.0)
        assert comp.delta_best == pytest.approx(-2.0)
        assert comp.delta_best_sign == "+"

    def test_skips_metric_with_no_frontier_or_no_current(self) -> None:
        snap = compute_snapshot(
            metrics=["fitness", "ghost"],
            frontier_history={"fitness": [(0, 0.5)], "ghost": []},
            iter_mean_history={"fitness": [(0, 0.4)], "ghost": []},
            program_history={"fitness": [(0, 0.5)], "ghost": []},
            higher_is_better={"fitness": True, "ghost": True},
        )
        assert "fitness" in snap.metrics
        assert "ghost" not in snap.metrics


class TestFormatSnapshot:
    def test_format_emits_one_line_per_metric(self) -> None:
        snap = FrontierCompareSnapshot(
            metrics={
                "fitness": MetricComparison(
                    name="fitness",
                    current_best=0.7,
                    current_mean=0.5,
                    frontier_best=0.5,
                    frontier_mean=0.5,
                    delta_best=0.2,
                    delta_mean=0.0,
                    delta_best_sign="+",
                ),
            }
        )
        line = format_snapshot(snap)
        assert "fitness" in line
        assert "current_best=" in line
        assert "frontier_best=" in line
        assert "delta_best=" in line

    def test_format_returns_idle_marker_when_empty(self) -> None:
        snap = FrontierCompareSnapshot(metrics={})
        line = format_snapshot(snap)
        assert "(no frontier data yet)" in line


class TestRenderFrontierPlot:
    def test_writes_png_when_frontier_data_present(self, tmp_path) -> None:
        _render_frontier_plot(
            output_dir=tmp_path,
            metric="fitness",
            frontier_history=[(0, 0.3), (1, 0.5), (2, 0.7)],
            iter_mean_history=[(0, 0.2), (1, 0.4), (2, 0.6)],
            higher_is_better=True,
        )
        out = tmp_path / "frontier_fitness.png"
        assert out.exists()
        assert out.stat().st_size > 0

    def test_no_file_when_frontier_empty(self, tmp_path) -> None:
        _render_frontier_plot(
            output_dir=tmp_path,
            metric="fitness",
            frontier_history=[],
            iter_mean_history=[(0, 0.1)],
            higher_is_better=True,
        )
        assert not (tmp_path / "frontier_fitness.png").exists()

    def test_metric_with_slash_is_safe_in_filename(self, tmp_path) -> None:
        # MetricsTracker tags can contain '/'; renderer must sanitise it.
        _render_frontier_plot(
            output_dir=tmp_path,
            metric="loss/train",
            frontier_history=[(0, 1.0), (1, 0.5)],
            iter_mean_history=[],
            higher_is_better=False,
        )
        # No `/` in the produced filename — sanitised to '_'.
        out = tmp_path / "frontier_loss_train.png"
        assert out.exists()


class TestStartLiveFrontierCompare:
    def test_returns_event_and_starts_daemon_thread(self) -> None:
        # Bogus Redis URL — the thread will fail to connect, log a warning,
        # and retry on the next tick. We only verify the bootstrap contract.
        stop = start_live_frontier_compare(
            redis_url="redis://127.0.0.1:1/0",
            key_prefix="test:metrics",
            metrics=["fitness"],
            higher_is_better={"fitness": True},
            interval_s=3600.0,
        )
        try:
            assert isinstance(stop, threading.Event)
            threads = {t.name: t for t in threading.enumerate()}
            assert "live-frontier-compare" in threads
            assert threads["live-frontier-compare"].daemon is True
        finally:
            stop.set()

    def test_file_target_accepts_output_dir(self, tmp_path) -> None:
        stop = start_live_frontier_compare(
            redis_url="redis://127.0.0.1:1/0",
            key_prefix="test:metrics",
            metrics=["fitness"],
            higher_is_better={"fitness": True},
            interval_s=3600.0,
            emit_targets=("file",),
            output_dir=tmp_path,
        )
        try:
            assert isinstance(stop, threading.Event)
            threads = {t.name: t for t in threading.enumerate()}
            assert "live-frontier-compare" in threads
        finally:
            stop.set()

    def test_file_target_without_output_dir_still_starts(self) -> None:
        # Resilient: file emit silently drops if no output_dir was wired.
        # The daemon must still start so log/telegram targets work.
        stop = start_live_frontier_compare(
            redis_url="redis://127.0.0.1:1/0",
            key_prefix="test:metrics",
            metrics=["fitness"],
            higher_is_better={"fitness": True},
            interval_s=3600.0,
            emit_targets=("file",),
            output_dir=None,
        )
        try:
            assert isinstance(stop, threading.Event)
        finally:
            stop.set()

    def test_disabled_returns_event_without_starting_a_new_thread(self) -> None:
        # Snapshot the thread set before/after to avoid coupling to other
        # tests that may have started a still-shutting-down thread.
        before = {id(t) for t in threading.enumerate()}
        stop = start_live_frontier_compare(
            redis_url="redis://127.0.0.1:1/0",
            key_prefix="test:metrics",
            metrics=["fitness"],
            higher_is_better={"fitness": True},
            interval_s=3600.0,
            enabled=False,
        )
        try:
            assert isinstance(stop, threading.Event)
            after = {id(t) for t in threading.enumerate()}
            new_threads = after - before
            assert new_threads == set()
        finally:
            stop.set()
