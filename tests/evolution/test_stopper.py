"""Tests for EvolutionStopper — pluggable engine stopping criteria."""

from __future__ import annotations

import pytest

from gigaevo.evolution.engine.stopper import (
    CompositeStopper,
    EngineThroughput,
    EvolutionStopper,
    FitnessPlateauStopper,
    MaxMutantsStopper,
    StopContext,
    StopDecision,
    WallClockStopper,
)


def _ctx(
    *,
    total_mutants: int = 0,
    elapsed_seconds: float = 0.0,
    best_fitness: float | None = None,
    programs_processed: int = 0,
) -> StopContext:
    return StopContext(
        total_mutants=total_mutants,
        elapsed_seconds=elapsed_seconds,
        best_fitness=best_fitness,
        programs_processed=programs_processed,
    )


class TestStopDecision:
    def test_stop_decision_is_truthy_when_stop_true(self) -> None:
        d = StopDecision(stop=True, reason="done")
        assert d.stop is True
        assert d.reason == "done"

    def test_stop_decision_is_falsy_when_stop_false(self) -> None:
        d = StopDecision(stop=False, reason="")
        assert d.stop is False


class TestMaxMutantsStopper:
    def test_does_not_stop_before_cap(self) -> None:
        stopper = MaxMutantsStopper(max_mutants=10)
        assert stopper.should_stop(_ctx(total_mutants=9)).stop is False

    def test_stops_at_cap(self) -> None:
        stopper = MaxMutantsStopper(max_mutants=10)
        result = stopper.should_stop(_ctx(total_mutants=10))
        assert result.stop is True
        assert "10" in result.reason

    def test_stops_past_cap(self) -> None:
        stopper = MaxMutantsStopper(max_mutants=10)
        assert stopper.should_stop(_ctx(total_mutants=15)).stop is True

    def test_zero_generations_does_not_stop(self) -> None:
        stopper = MaxMutantsStopper(max_mutants=10)
        assert stopper.should_stop(_ctx(total_mutants=0)).stop is False


class TestWallClockStopper:
    def test_does_not_stop_before_budget(self) -> None:
        stopper = WallClockStopper(budget_seconds=3600)
        assert stopper.should_stop(_ctx(elapsed_seconds=3599)).stop is False

    def test_stops_at_budget(self) -> None:
        stopper = WallClockStopper(budget_seconds=3600)
        result = stopper.should_stop(_ctx(elapsed_seconds=3600))
        assert result.stop is True

    def test_stops_past_budget(self) -> None:
        stopper = WallClockStopper(budget_seconds=3600)
        assert stopper.should_stop(_ctx(elapsed_seconds=7200)).stop is True

    def test_reason_includes_budget(self) -> None:
        stopper = WallClockStopper(budget_seconds=3600)
        result = stopper.should_stop(_ctx(elapsed_seconds=3600))
        assert "3600" in result.reason


class TestFitnessPlateauStopper:
    def test_does_not_stop_when_fitness_improving(self) -> None:
        stopper = FitnessPlateauStopper(window=3, min_delta=0.001)
        for gen, fitness in [(0, 0.5), (1, 0.6), (2, 0.7)]:
            result = stopper.should_stop(_ctx(total_mutants=gen, best_fitness=fitness))
        assert result.stop is False

    def test_stops_after_plateau_window(self) -> None:
        stopper = FitnessPlateauStopper(window=3, min_delta=0.01)
        stopper.should_stop(_ctx(total_mutants=0, best_fitness=0.5))
        stopper.should_stop(_ctx(total_mutants=1, best_fitness=0.5))
        stopper.should_stop(_ctx(total_mutants=2, best_fitness=0.5))
        result = stopper.should_stop(_ctx(total_mutants=3, best_fitness=0.5))
        assert result.stop is True

    def test_resets_window_on_improvement(self) -> None:
        stopper = FitnessPlateauStopper(window=3, min_delta=0.01)
        stopper.should_stop(_ctx(total_mutants=0, best_fitness=0.5))
        stopper.should_stop(_ctx(total_mutants=1, best_fitness=0.5))
        stopper.should_stop(_ctx(total_mutants=2, best_fitness=0.6))
        result = stopper.should_stop(_ctx(total_mutants=3, best_fitness=0.6))
        assert result.stop is False

    def test_none_fitness_does_not_count(self) -> None:
        stopper = FitnessPlateauStopper(window=2, min_delta=0.001)
        stopper.should_stop(_ctx(total_mutants=0, best_fitness=None))
        stopper.should_stop(_ctx(total_mutants=1, best_fitness=None))
        result = stopper.should_stop(_ctx(total_mutants=2, best_fitness=None))
        assert result.stop is False

    def test_reason_includes_window(self) -> None:
        stopper = FitnessPlateauStopper(window=2, min_delta=0.01)
        stopper.should_stop(_ctx(total_mutants=0, best_fitness=0.5))
        stopper.should_stop(_ctx(total_mutants=1, best_fitness=0.5))
        result = stopper.should_stop(_ctx(total_mutants=2, best_fitness=0.5))
        assert result.stop is True
        assert "2" in result.reason


class TestCompositeStopper:
    def test_any_mode_stops_when_one_child_stops(self) -> None:
        stopper = CompositeStopper(
            mode="any",
            children=[
                MaxMutantsStopper(max_mutants=10),
                WallClockStopper(budget_seconds=3600),
            ],
        )
        result = stopper.should_stop(_ctx(total_mutants=10, elapsed_seconds=100))
        assert result.stop is True

    def test_any_mode_continues_when_no_child_stops(self) -> None:
        stopper = CompositeStopper(
            mode="any",
            children=[
                MaxMutantsStopper(max_mutants=10),
                WallClockStopper(budget_seconds=3600),
            ],
        )
        result = stopper.should_stop(_ctx(total_mutants=5, elapsed_seconds=100))
        assert result.stop is False

    def test_all_mode_stops_only_when_all_children_stop(self) -> None:
        stopper = CompositeStopper(
            mode="all",
            children=[
                MaxMutantsStopper(max_mutants=10),
                WallClockStopper(budget_seconds=3600),
            ],
        )
        result = stopper.should_stop(_ctx(total_mutants=10, elapsed_seconds=100))
        assert result.stop is False

    def test_all_mode_stops_when_all_satisfied(self) -> None:
        stopper = CompositeStopper(
            mode="all",
            children=[
                MaxMutantsStopper(max_mutants=10),
                WallClockStopper(budget_seconds=3600),
            ],
        )
        result = stopper.should_stop(_ctx(total_mutants=10, elapsed_seconds=3600))
        assert result.stop is True

    def test_reason_aggregates_child_reasons(self) -> None:
        stopper = CompositeStopper(
            mode="any",
            children=[
                MaxMutantsStopper(max_mutants=5),
                WallClockStopper(budget_seconds=100),
            ],
        )
        result = stopper.should_stop(_ctx(total_mutants=5, elapsed_seconds=200))
        assert result.stop is True
        assert result.reason  # non-empty

    def test_empty_children_any_mode_does_not_stop(self) -> None:
        stopper = CompositeStopper(mode="any", children=[])
        assert stopper.should_stop(_ctx()).stop is False

    def test_empty_children_all_mode_does_not_stop(self) -> None:
        stopper = CompositeStopper(mode="all", children=[])
        assert stopper.should_stop(_ctx()).stop is False


class TestBaseEvolutionStopper:
    def test_base_never_stops(self) -> None:
        stopper = EvolutionStopper()
        assert stopper.should_stop(_ctx(total_mutants=9999)).stop is False

    def test_base_estimate_remaining_is_none(self) -> None:
        stopper = EvolutionStopper()
        tp = EngineThroughput(mutants_per_second=1.0, elapsed_seconds=10.0)
        assert stopper.estimate_remaining(_ctx(total_mutants=5), tp) is None


class TestMaxMutantsEstimateRemaining:
    def test_positive_rate_returns_seconds_and_label(self) -> None:
        stopper = MaxMutantsStopper(max_mutants=100)
        tp = EngineThroughput(mutants_per_second=2.0, elapsed_seconds=20.0)
        result = stopper.estimate_remaining(_ctx(total_mutants=40), tp)
        assert result == (30.0, "MaxMutantsStopper")

    def test_zero_rate_returns_none(self) -> None:
        stopper = MaxMutantsStopper(max_mutants=100)
        tp = EngineThroughput(mutants_per_second=0.0, elapsed_seconds=20.0)
        assert stopper.estimate_remaining(_ctx(total_mutants=40), tp) is None

    def test_negative_rate_returns_none(self) -> None:
        stopper = MaxMutantsStopper(max_mutants=100)
        tp = EngineThroughput(mutants_per_second=-1.0, elapsed_seconds=20.0)
        assert stopper.estimate_remaining(_ctx(total_mutants=40), tp) is None

    def test_past_cap_clamps_to_zero(self) -> None:
        stopper = MaxMutantsStopper(max_mutants=100)
        tp = EngineThroughput(mutants_per_second=2.0, elapsed_seconds=200.0)
        result = stopper.estimate_remaining(_ctx(total_mutants=150), tp)
        assert result == (0.0, "MaxMutantsStopper")


class TestWallClockEstimateRemaining:
    def test_returns_budget_minus_elapsed(self) -> None:
        stopper = WallClockStopper(budget_seconds=3600)
        tp = EngineThroughput(mutants_per_second=1.0, elapsed_seconds=600.0)
        result = stopper.estimate_remaining(_ctx(elapsed_seconds=600.0), tp)
        assert result == (3000.0, "WallClockStopper")

    def test_past_budget_clamps_to_zero(self) -> None:
        stopper = WallClockStopper(budget_seconds=3600)
        tp = EngineThroughput(mutants_per_second=1.0, elapsed_seconds=7200.0)
        result = stopper.estimate_remaining(_ctx(elapsed_seconds=7200.0), tp)
        assert result == (0.0, "WallClockStopper")

    def test_throughput_is_ignored(self) -> None:
        stopper = WallClockStopper(budget_seconds=3600)
        tp_a = EngineThroughput(mutants_per_second=0.0, elapsed_seconds=100.0)
        tp_b = EngineThroughput(mutants_per_second=99.0, elapsed_seconds=100.0)
        assert stopper.estimate_remaining(
            _ctx(elapsed_seconds=100.0), tp_a
        ) == stopper.estimate_remaining(_ctx(elapsed_seconds=100.0), tp_b)


class TestFitnessPlateauEstimateRemaining:
    def test_always_returns_none(self) -> None:
        stopper = FitnessPlateauStopper(window=10, min_delta=0.01)
        tp = EngineThroughput(mutants_per_second=2.0, elapsed_seconds=100.0)
        assert (
            stopper.estimate_remaining(_ctx(total_mutants=50, best_fitness=0.5), tp)
            is None
        )


class TestCompositeEstimateRemaining:
    def test_any_mode_picks_min_bounded_with_winner_label(self) -> None:
        stopper = CompositeStopper(
            mode="any",
            children=[
                MaxMutantsStopper(max_mutants=100),
                WallClockStopper(budget_seconds=3600),
            ],
        )
        tp = EngineThroughput(mutants_per_second=2.0, elapsed_seconds=20.0)
        result = stopper.estimate_remaining(
            _ctx(total_mutants=40, elapsed_seconds=20.0), tp
        )
        assert result == (30.0, "MaxMutantsStopper")

    def test_all_mode_picks_max_bounded_with_winner_label(self) -> None:
        stopper = CompositeStopper(
            mode="all",
            children=[
                MaxMutantsStopper(max_mutants=100),
                WallClockStopper(budget_seconds=3600),
            ],
        )
        tp = EngineThroughput(mutants_per_second=2.0, elapsed_seconds=20.0)
        result = stopper.estimate_remaining(
            _ctx(total_mutants=40, elapsed_seconds=20.0), tp
        )
        assert result == (3580.0, "WallClockStopper")

    def test_mixed_bounded_and_unbounded_returns_bounded_min(self) -> None:
        stopper = CompositeStopper(
            mode="any",
            children=[
                MaxMutantsStopper(max_mutants=100),
                FitnessPlateauStopper(window=10),
            ],
        )
        tp = EngineThroughput(mutants_per_second=2.0, elapsed_seconds=20.0)
        result = stopper.estimate_remaining(
            _ctx(total_mutants=40, elapsed_seconds=20.0), tp
        )
        assert result == (30.0, "MaxMutantsStopper")

    def test_all_unbounded_returns_none(self) -> None:
        stopper = CompositeStopper(
            mode="any",
            children=[
                FitnessPlateauStopper(window=5),
                FitnessPlateauStopper(window=10),
            ],
        )
        tp = EngineThroughput(mutants_per_second=2.0, elapsed_seconds=20.0)
        assert stopper.estimate_remaining(_ctx(total_mutants=40), tp) is None

    def test_empty_children_returns_none(self) -> None:
        stopper = CompositeStopper(mode="any", children=[])
        tp = EngineThroughput(mutants_per_second=2.0, elapsed_seconds=20.0)
        assert stopper.estimate_remaining(_ctx(), tp) is None


class TestHydraInstantiation:
    """Verify each config/stopper/*.yaml round-trips through hydra.utils.instantiate."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_hydra(self) -> None:
        pytest.importorskip("hydra")

    def test_max_mutants_yaml(self) -> None:
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        # The yaml references ${max_mutants} from the parent Hydra config.
        # Wrap it in a parent scope that provides the value so resolution works
        # standalone in tests.
        stopper_cfg = OmegaConf.load("config/stopper/max_mutants.yaml")
        cfg = OmegaConf.create({"max_mutants": 25, "stopper": stopper_cfg})
        stopper = instantiate(cfg.stopper)
        assert isinstance(stopper, MaxMutantsStopper)
        assert stopper.max_mutants == 25

    def test_wall_clock_yaml(self) -> None:
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        cfg = OmegaConf.load("config/stopper/wall_clock.yaml")
        stopper = instantiate(cfg)
        assert isinstance(stopper, WallClockStopper)
        assert stopper.budget_seconds == 21600

    def test_fitness_plateau_yaml(self) -> None:
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        cfg = OmegaConf.load("config/stopper/fitness_plateau.yaml")
        stopper = instantiate(cfg)
        assert isinstance(stopper, FitnessPlateauStopper)
        assert stopper.window == 10

    def test_composite_yaml(self) -> None:
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        # Composite child refers to ${max_mutants} — wrap in a parent
        # config that provides the value.
        stopper_cfg = OmegaConf.load(
            "config/stopper/max_mutants_or_fitness_plateau.yaml"
        )
        cfg = OmegaConf.create({"max_mutants": 50, "stopper": stopper_cfg})
        stopper = instantiate(cfg.stopper)
        assert isinstance(stopper, CompositeStopper)
        assert stopper.mode == "any"
        assert len(stopper.children) == 2
        assert isinstance(stopper.children[0], MaxMutantsStopper)
        assert isinstance(stopper.children[1], FitnessPlateauStopper)
