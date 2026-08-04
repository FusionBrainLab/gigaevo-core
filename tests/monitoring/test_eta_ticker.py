"""Tests for eta_ticker module."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

from gigaevo.evolution.engine.stopper import (
    CompositeStopper,
    EvolutionStopper,
    FitnessPlateauStopper,
    MaxMutantsStopper,
    StopContext,
    WallClockStopper,
)
from gigaevo.monitoring.eta_ticker import ThroughputWindow, _humanize_seconds, _tick


def _engine(stopper: EvolutionStopper, samples: list[tuple[float, int]]) -> MagicMock:
    """Engine whose successive stop-contexts replay ``(elapsed, mutants)``."""
    engine = MagicMock()
    engine.stopper = stopper
    engine.build_stop_context.side_effect = [
        StopContext(
            total_mutants=mutants,
            elapsed_seconds=elapsed,
            best_fitness=0.5,
            programs_processed=mutants,
        )
        for elapsed, mutants in samples
    ]
    return engine


def _last_line(
    stopper: EvolutionStopper,
    samples: list[tuple[float, int]],
    *,
    window_seconds: float = 600.0,
    warmup_mutants: int = 3,
) -> str | None:
    """Drive one tick per sample; return the final tick's line."""
    engine = _engine(stopper, samples)
    window = ThroughputWindow(window_seconds)
    line = None
    for _ in samples:
        line = _tick(engine, window, warmup_mutants=warmup_mutants)
    return line


def _ramp(
    *,
    start_s: float,
    end_s: float,
    step_s: float,
    per_step: float,
    start_mutants: int,
) -> list[tuple[float, int]]:
    """Samples at a fixed cadence accruing ``per_step`` mutants each step."""
    samples = []
    mutants = float(start_mutants)
    t = start_s
    while t <= end_s:
        samples.append((t, int(mutants)))
        mutants += per_step
        t += step_s
    return samples


class TestHumanizeSeconds:
    def test_under_one_minute(self):
        assert _humanize_seconds(45) == "0m45s"

    def test_one_minute(self):
        assert _humanize_seconds(60) == "1m00s"

    def test_over_one_minute(self):
        assert _humanize_seconds(125) == "2m05s"

    def test_over_one_hour(self):
        # 3661 seconds = 1 hour, 1 minute, 1 second
        assert _humanize_seconds(3661) == "1:01:01"

    def test_zero(self):
        assert _humanize_seconds(0) == "0m00s"


class TestThroughputWindow:
    def test_first_observation_has_no_span(self):
        window = ThroughputWindow(600.0)
        span = window.observe(60.0, 10)
        assert span.seconds == 0.0
        assert span.mutants == 0

    def test_span_measures_delta_between_samples(self):
        window = ThroughputWindow(600.0)
        window.observe(60.0, 10)
        span = window.observe(120.0, 25)
        assert span.seconds == 60.0
        assert span.mutants == 15

    def test_samples_older_than_window_are_dropped(self):
        window = ThroughputWindow(600.0)
        for elapsed, mutants in _ramp(
            start_s=60.0, end_s=1200.0, step_s=60.0, per_step=1, start_mutants=0
        ):
            span = window.observe(elapsed, mutants)
        # Anchor sits at the window boundary, not at the run start.
        assert span.seconds == 600.0
        assert span.mutants == 10

    def test_anchor_is_retained_at_or_before_the_cutoff(self):
        # Sparse samples: the anchor must not be pruned past the cutoff,
        # otherwise the span collapses and the rate spikes.
        window = ThroughputWindow(600.0)
        window.observe(0.0, 0)
        span = window.observe(1000.0, 100)
        assert span.seconds == 1000.0
        assert span.mutants == 100


class TestTickWarmup:
    def test_first_tick_returns_none(self):
        line = _last_line(MaxMutantsStopper(max_mutants=100), [(60.0, 5)])
        assert line is None

    def test_partial_window_below_warmup_returns_none(self):
        line = _last_line(
            MaxMutantsStopper(max_mutants=100),
            [(60.0, 1), (120.0, 2)],
            warmup_mutants=3,
        )
        assert line is None

    def test_partial_window_at_warmup_emits_line(self):
        line = _last_line(
            MaxMutantsStopper(max_mutants=100),
            [(60.0, 1), (120.0, 5)],
            warmup_mutants=3,
        )
        assert line is not None
        assert "[eta]" in line


class TestRateExcludesUncountedTime:
    def test_restored_mutants_do_not_inflate_the_rate(self):
        # Resumed run: 400 mutants restored, new work proceeds at 2/min.
        # A lifetime average would divide 406 mutants by 180s of new-run
        # wall time and report an ETA of well under a minute.
        line = _last_line(
            MaxMutantsStopper(max_mutants=500),
            [(60.0, 402), (120.0, 404), (180.0, 406)],
        )
        assert line is not None
        assert "(2.0/min)" in line
        assert "remaining=94" in line
        assert "ETA=47m00s" in line

    def test_seed_drain_leaves_the_window(self):
        # 900s of initial-population evaluation creates no mutants, then
        # steady 5/min. Once the drain ages out, the rate is exact.
        samples = _ramp(
            start_s=60.0, end_s=900.0, step_s=60.0, per_step=0, start_mutants=0
        )
        samples += _ramp(
            start_s=960.0, end_s=1560.0, step_s=60.0, per_step=5, start_mutants=5
        )
        line = _last_line(MaxMutantsStopper(max_mutants=100), samples)
        assert line is not None
        assert "(5.0/min)" in line
        assert "remaining=45" in line
        assert "ETA=9m00s" in line

    def test_rate_tracks_a_slowdown(self):
        # 10/min for the first 600s, then 1/min. The ETA must reflect the
        # current rate, not the flattering lifetime average.
        samples = _ramp(
            start_s=60.0, end_s=600.0, step_s=60.0, per_step=10, start_mutants=10
        )
        samples += _ramp(
            start_s=660.0, end_s=1200.0, step_s=60.0, per_step=1, start_mutants=101
        )
        line = _last_line(MaxMutantsStopper(max_mutants=200), samples)
        assert line is not None
        assert "(1.0/min)" in line
        assert "ETA=1:30:00" in line


class TestTickStalled:
    def test_no_mutants_across_a_full_window_reports_unknown(self):
        samples = _ramp(
            start_s=60.0, end_s=600.0, step_s=60.0, per_step=5, start_mutants=5
        )
        samples += _ramp(
            start_s=660.0, end_s=1260.0, step_s=60.0, per_step=0, start_mutants=50
        )
        line = _last_line(MaxMutantsStopper(max_mutants=100), samples)
        assert line is not None
        assert "unknown" in line.lower()
        assert "no mutants" in line.lower()

    def test_wall_clock_bound_survives_a_stall(self):
        # A wall-clock budget does not depend on throughput, so a stalled
        # run still has a real ETA — reporting "unknown" here would throw
        # away an answer the stopper can give.
        samples = _ramp(
            start_s=60.0, end_s=600.0, step_s=60.0, per_step=5, start_mutants=5
        )
        samples += _ramp(
            start_s=660.0, end_s=1260.0, step_s=60.0, per_step=0, start_mutants=50
        )
        line = _last_line(WallClockStopper(budget_seconds=1800.0), samples)
        assert line is not None
        assert "unknown" not in line.lower()
        assert "(0.0/min)" in line
        assert "ETA=9m00s" in line


class TestTickBounded:
    def test_max_mutants_bounded(self):
        line = _last_line(
            MaxMutantsStopper(max_mutants=100),
            [(60.0, 10), (120.0, 20), (180.0, 30)],
        )
        assert line is not None
        assert "[eta]" in line
        assert "MaxMutantsStopper" in line
        assert "ETA=" in line
        assert "unknown" not in line.lower()

    def test_wall_clock_bounded(self):
        line = _last_line(
            WallClockStopper(budget_seconds=300.0),
            [(60.0, 10), (120.0, 20), (180.0, 30)],
        )
        assert line is not None
        assert "[eta]" in line
        assert "WallClockStopper" in line
        assert "ETA=2m00s" in line


class TestTickUnbounded:
    def test_fitness_plateau_unbounded(self):
        line = _last_line(
            FitnessPlateauStopper(window=5),
            [(60.0, 10), (120.0, 20), (180.0, 30)],
        )
        assert line is not None
        assert "[eta]" in line
        assert "unknown" in line.lower()
        assert "FitnessPlateauStopper" in line

    def test_composite_all_unbounded(self):
        stopper = CompositeStopper(
            mode="any",
            children=[
                FitnessPlateauStopper(window=5),
                FitnessPlateauStopper(window=10),
            ],
        )
        line = _last_line(stopper, [(60.0, 10), (120.0, 20), (180.0, 30)])
        assert line is not None
        assert "[eta]" in line
        assert "unknown" in line.lower()


class TestTickComposite:
    def test_composite_any_with_bounded_child(self):
        stopper = CompositeStopper(
            mode="any",
            children=[
                MaxMutantsStopper(max_mutants=100),
                FitnessPlateauStopper(window=10),
            ],
        )
        line = _last_line(stopper, [(60.0, 10), (120.0, 20), (180.0, 30)])
        assert line is not None
        assert "[eta]" in line
        assert "MaxMutantsStopper" in line
        assert "unknown" not in line.lower()

    def test_composite_all_with_unbounded_child_names_the_unbounded_one(self):
        # mode="all" only stops once every child fires, so an unbounded
        # child makes the whole estimate unknown — and the label must point
        # at that child, not at the bounded one.
        stopper = CompositeStopper(
            mode="all",
            children=[
                MaxMutantsStopper(max_mutants=100),
                FitnessPlateauStopper(window=10),
            ],
        )
        line = _last_line(stopper, [(60.0, 10), (120.0, 20), (180.0, 30)])
        assert line is not None
        assert "unknown" in line.lower()
        assert "FitnessPlateauStopper" in line


class TestStartEtaTicker:
    def test_start_eta_ticker_returns_stop_event(self):
        from gigaevo.monitoring.eta_ticker import start_eta_ticker

        engine = MagicMock()
        engine.build_stop_context.return_value = StopContext(
            total_mutants=5,
            elapsed_seconds=10.0,
            best_fitness=0.5,
            programs_processed=5,
        )
        engine.stopper = MaxMutantsStopper(max_mutants=100)

        stop = start_eta_ticker(engine, interval_s=0.1)
        assert isinstance(stop, threading.Event)

        # Cleanup.
        stop.set()
