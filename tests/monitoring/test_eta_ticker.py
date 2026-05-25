"""Tests for eta_ticker module."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

from gigaevo.evolution.engine.stopper import (
    CompositeStopper,
    FitnessPlateauStopper,
    MaxMutantsStopper,
    StopContext,
    WallClockStopper,
)
from gigaevo.monitoring.eta_ticker import _humanize_seconds, _tick


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


class TestTickWarmup:
    def test_warmup_phase_returns_none(self):
        engine = MagicMock()
        engine.build_stop_context.return_value = StopContext(
            total_mutants=2,
            elapsed_seconds=10.0,
            best_fitness=0.5,
            programs_processed=2,
        )
        engine.stopper = MaxMutantsStopper(max_mutants=100)

        line = _tick(engine, warmup_mutants=3)
        assert line is None

    def test_warmup_pass_emits_line(self):
        engine = MagicMock()
        engine.build_stop_context.return_value = StopContext(
            total_mutants=5,
            elapsed_seconds=10.0,
            best_fitness=0.5,
            programs_processed=5,
        )
        engine.stopper = MaxMutantsStopper(max_mutants=100)

        line = _tick(engine, warmup_mutants=3)
        assert line is not None
        assert "[eta]" in line


class TestTickBounded:
    def test_max_mutants_bounded(self):
        engine = MagicMock()
        engine.build_stop_context.return_value = StopContext(
            total_mutants=50,
            elapsed_seconds=100.0,
            best_fitness=0.5,
            programs_processed=50,
        )
        engine.stopper = MaxMutantsStopper(max_mutants=100)

        line = _tick(engine, warmup_mutants=3)
        assert line is not None
        assert "[eta]" in line
        assert "MaxMutantsStopper" in line
        assert "ETA=" in line
        assert "unknown" not in line.lower()

    def test_wall_clock_bounded(self):
        engine = MagicMock()
        engine.build_stop_context.return_value = StopContext(
            total_mutants=30,
            elapsed_seconds=50.0,
            best_fitness=0.5,
            programs_processed=30,
        )
        engine.stopper = WallClockStopper(budget_seconds=300.0)

        line = _tick(engine, warmup_mutants=3)
        assert line is not None
        assert "[eta]" in line
        assert "WallClockStopper" in line


class TestTickUnbounded:
    def test_fitness_plateau_unbounded(self):
        engine = MagicMock()
        engine.build_stop_context.return_value = StopContext(
            total_mutants=10,
            elapsed_seconds=20.0,
            best_fitness=0.5,
            programs_processed=10,
        )
        engine.stopper = FitnessPlateauStopper(window=5)

        line = _tick(engine, warmup_mutants=3)
        assert line is not None
        assert "[eta]" in line
        assert "unknown" in line.lower()
        assert "FitnessPlateauStopper" in line

    def test_composite_all_unbounded(self):
        engine = MagicMock()
        engine.build_stop_context.return_value = StopContext(
            total_mutants=10,
            elapsed_seconds=20.0,
            best_fitness=0.5,
            programs_processed=10,
        )
        engine.stopper = CompositeStopper(
            mode="any",
            children=[
                FitnessPlateauStopper(window=5),
                FitnessPlateauStopper(window=10),
            ],
        )

        line = _tick(engine, warmup_mutants=3)
        assert line is not None
        assert "[eta]" in line
        assert "unknown" in line.lower()


class TestTickComposite:
    def test_composite_any_with_bounded_child(self):
        engine = MagicMock()
        engine.build_stop_context.return_value = StopContext(
            total_mutants=30,
            elapsed_seconds=60.0,
            best_fitness=0.5,
            programs_processed=30,
        )
        engine.stopper = CompositeStopper(
            mode="any",
            children=[
                MaxMutantsStopper(max_mutants=100),
                FitnessPlateauStopper(window=10),
            ],
        )

        line = _tick(engine, warmup_mutants=3)
        assert line is not None
        assert "[eta]" in line
        assert "MaxMutantsStopper" in line
        assert "unknown" not in line.lower()


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
