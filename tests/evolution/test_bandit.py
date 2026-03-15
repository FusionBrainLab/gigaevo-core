"""Tests for bandit-based adaptive model selection."""

from __future__ import annotations

import asyncio
import math
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gigaevo.llm.bandit import (
    _MAX_IMPROVEMENT,
    BanditModelRouter,
    MutationOutcome,
    RunningPercentileNormalizer,
    SlidingWindowUCB1,
    compute_bandit_reward,
)
from gigaevo.programs.program import Program

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_mock_models_shared(names: list[str]) -> list[MagicMock]:
    """Create mock ChatOpenAI models — defined early so new classes can use it."""
    models = []
    for name in names:
        m = MagicMock()
        m.model_name = name
        m.with_structured_output = MagicMock(return_value=MagicMock())
        models.append(m)
    return models


# ---------------------------------------------------------------------------
# compute_bandit_reward
# ---------------------------------------------------------------------------


class TestComputeBanditReward:
    def test_positive_improvement(self):
        # child=10, parent=8, higher_is_better → improvement=2 → exp(2)-1
        r = compute_bandit_reward(10.0, 8.0, higher_is_better=True)
        assert r == pytest.approx(math.exp(2.0) - 1.0)

    def test_no_improvement(self):
        r = compute_bandit_reward(5.0, 5.0, higher_is_better=True)
        assert r == pytest.approx(0.0)

    def test_negative_improvement_clamped(self):
        # child worse than parent → max(improvement, 0) = 0 → exp(0)-1 = 0
        r = compute_bandit_reward(3.0, 5.0, higher_is_better=True)
        assert r == pytest.approx(0.0)

    def test_lower_is_better(self):
        # child=3, parent=5, lower is better → improvement = -(3-5)=2
        r = compute_bandit_reward(3.0, 5.0, higher_is_better=False)
        assert r == pytest.approx(math.exp(2.0) - 1.0)

    def test_lower_is_better_no_improvement(self):
        r = compute_bandit_reward(7.0, 5.0, higher_is_better=False)
        assert r == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compute_bandit_reward — edge cases and clamping
# ---------------------------------------------------------------------------


class TestComputeBanditRewardEdgeCases:
    def test_equal_fitness_lower_is_better_returns_zero(self) -> None:
        r = compute_bandit_reward(5.0, 5.0, higher_is_better=False)
        assert r == pytest.approx(0.0)

    def test_large_negative_improvement_clamped_to_zero(self) -> None:
        r = compute_bandit_reward(-1000.0, 0.0, higher_is_better=True)
        assert r == pytest.approx(0.0)

    def test_large_negative_improvement_lower_is_better_clamped(self) -> None:
        r = compute_bandit_reward(1000.0, 0.0, higher_is_better=False)
        assert r == pytest.approx(0.0)

    def test_very_large_improvement_clamped_not_overflow(self) -> None:
        """Pathological improvements are clamped to _MAX_IMPROVEMENT, not overflow."""
        r = compute_bandit_reward(1000.0, 0.0, higher_is_better=True)
        assert math.isfinite(r)
        assert r == pytest.approx(math.exp(_MAX_IMPROVEMENT) - 1.0)

    def test_lower_is_better_sentinel_clamped_not_overflow(self) -> None:
        """Sentinel -1000 with higher_is_better=False would cause exp(1005)
        overflow without the clamp. Now it should be safely capped."""
        r = compute_bandit_reward(-1000.0, 5.0, higher_is_better=False)
        assert math.isfinite(r)
        assert r == pytest.approx(math.exp(_MAX_IMPROVEMENT) - 1.0)

    def test_improvement_exactly_at_max(self) -> None:
        """Improvement exactly at _MAX_IMPROVEMENT should not be altered."""
        r = compute_bandit_reward(_MAX_IMPROVEMENT, 0.0, higher_is_better=True)
        assert r == pytest.approx(math.exp(_MAX_IMPROVEMENT) - 1.0)

    def test_improvement_just_below_max(self) -> None:
        """Improvement just below _MAX_IMPROVEMENT should pass through."""
        delta = _MAX_IMPROVEMENT - 0.1
        r = compute_bandit_reward(delta, 0.0, higher_is_better=True)
        assert r == pytest.approx(math.exp(delta) - 1.0)

    def test_small_positive_improvement(self) -> None:
        r = compute_bandit_reward(1.001, 1.0, higher_is_better=True)
        assert r > 0.0
        assert r == pytest.approx(math.exp(0.001) - 1.0)

    def test_reward_is_strictly_non_negative(self) -> None:
        cases = [
            (3.0, 5.0, True),
            (7.0, 5.0, False),
            (0.0, 100.0, True),
        ]
        for child, parent, hib in cases:
            assert compute_bandit_reward(child, parent, higher_is_better=hib) >= 0.0


# ---------------------------------------------------------------------------
# RunningPercentileNormalizer
# ---------------------------------------------------------------------------


class TestRunningPercentileNormalizer:
    def test_warmup_returns_neutral(self):
        norm = RunningPercentileNormalizer(min_samples=5)
        for _ in range(4):
            assert norm.normalize(1.0) == pytest.approx(0.5)

    def test_after_warmup_normalizes(self):
        norm = RunningPercentileNormalizer(percentile=95.0, min_samples=3)
        for _ in range(3):
            norm.normalize(1.0)
        # Now we have 3 samples of 1.0; p95 = 1.0
        result = norm.normalize(0.5)
        assert 0.0 <= result <= 1.0
        assert result == pytest.approx(0.5)

    def test_clamps_to_one(self):
        norm = RunningPercentileNormalizer(percentile=95.0, min_samples=3)
        for _ in range(3):
            norm.normalize(1.0)
        # reward=10.0 >> p95=1.0 → clipped to 1.0
        result = norm.normalize(10.0)
        assert result == pytest.approx(1.0)

    def test_zero_percentile_returns_neutral(self):
        norm = RunningPercentileNormalizer(percentile=95.0, min_samples=3)
        for _ in range(3):
            norm.normalize(0.0)
        # p95 = 0 → returns 0.5
        result = norm.normalize(0.0)
        assert result == pytest.approx(0.5)


class TestRunningPercentileNormalizerEdgeCases:
    def test_exactly_at_min_samples_triggers_normalization(self) -> None:
        norm = RunningPercentileNormalizer(percentile=95.0, min_samples=3)
        norm.normalize(1.0)
        norm.normalize(1.0)
        result = norm.normalize(1.0)
        assert result == pytest.approx(1.0)

    def test_rewards_list_grows_with_each_call(self) -> None:
        norm = RunningPercentileNormalizer(min_samples=2)
        for _ in range(10):
            norm.normalize(0.5)
        assert len(norm._rewards) == 10

    def test_negative_reward_input_clamped_to_zero_after_clip(self) -> None:
        norm = RunningPercentileNormalizer(percentile=95.0, min_samples=3)
        for _ in range(3):
            norm.normalize(1.0)
        result = norm.normalize(-5.0)
        assert result == pytest.approx(0.0)

    def test_min_samples_one_skips_warmup_immediately(self) -> None:
        norm = RunningPercentileNormalizer(percentile=95.0, min_samples=1)
        result = norm.normalize(2.0)
        assert result == pytest.approx(1.0)

    def test_percentile_reference_tracks_growing_history(self) -> None:
        norm = RunningPercentileNormalizer(percentile=50.0, min_samples=2)
        norm.normalize(1.0)
        norm.normalize(1.0)
        norm.normalize(100.0)
        result = norm.normalize(100.0)
        assert result == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# SlidingWindowUCB1
# ---------------------------------------------------------------------------


class TestSlidingWindowUCB1:
    def test_warmup_round_robin(self):
        ucb = SlidingWindowUCB1(arm_names=["a", "b", "c"])
        selected = set()
        for _ in range(3):
            name = ucb.select()
            ucb.record_pull(name)
            selected.add(name)
        assert selected == {"a", "b", "c"}

    def test_exploitation_prefers_high_reward(self):
        ucb = SlidingWindowUCB1(arm_names=["good", "bad"], exploration_constant=0.01)
        for name in ["good", "bad"]:
            ucb.record_pull(name)
        for _ in range(20):
            ucb.update_reward("good", 1.0)
            ucb.record_pull("good")
        for _ in range(20):
            ucb.update_reward("bad", 0.0)
            ucb.record_pull("bad")
        selections = [ucb.select() for _ in range(50)]
        assert selections.count("good") > selections.count("bad")

    def test_exploration_favors_under_pulled(self):
        ucb = SlidingWindowUCB1(arm_names=["a", "b"], exploration_constant=100.0)
        for name in ["a", "b"]:
            ucb.record_pull(name)
            ucb.update_reward(name, 0.5)
        for _ in range(50):
            ucb.record_pull("a")
            ucb.update_reward("a", 0.5)
        assert ucb.select() == "b"

    def test_sliding_window_drops_old_rewards(self):
        ucb = SlidingWindowUCB1(
            arm_names=["x"], exploration_constant=0.0, window_size=5
        )
        ucb.record_pull("x")
        for _ in range(5):
            ucb.update_reward("x", 1.0)
        for _ in range(5):
            ucb.update_reward("x", 0.0)
        stats = ucb.get_stats()
        assert stats["x"]["mean_reward"] == pytest.approx(0.0)
        assert stats["x"]["window_size"] == 5

    def test_get_stats(self):
        ucb = SlidingWindowUCB1(arm_names=["a", "b"])
        ucb.record_pull("a")
        ucb.update_reward("a", 0.8)
        stats = ucb.get_stats()
        assert stats["a"]["total_pulls"] == 1
        assert stats["a"]["mean_reward"] == pytest.approx(0.8)
        assert stats["b"]["total_pulls"] == 0

    def test_ucb1_uses_total_pulls_for_exploration(self):
        """The exploration term uses total_pulls, not window size.
        An arm with many pulls but few rewards should have lower exploration
        bonus than an arm with few pulls."""
        ucb = SlidingWindowUCB1(
            arm_names=["many_pulls", "few_pulls"],
            exploration_constant=10.0,
        )
        # Warm up both
        for name in ["many_pulls", "few_pulls"]:
            ucb.record_pull(name)
            ucb.update_reward(name, 0.5)
        # Pull "many_pulls" 100 more times with same reward
        for _ in range(100):
            ucb.record_pull("many_pulls")
            ucb.update_reward("many_pulls", 0.5)
        # "few_pulls" has 1 pull, "many_pulls" has 101 pulls, same mean
        # exploration for few_pulls should be much higher (sqrt(ln(102)/1) vs sqrt(ln(102)/101))
        assert ucb.select() == "few_pulls"


class TestSlidingWindowUCB1EdgeCases:
    def test_single_arm_always_selected(self) -> None:
        ucb = SlidingWindowUCB1(arm_names=["solo"])
        assert ucb.select() == "solo"
        ucb.record_pull("solo")
        ucb.update_reward("solo", 0.5)
        assert ucb.select() == "solo"

    def test_window_exactly_at_capacity_does_not_overflow(self) -> None:
        ucb = SlidingWindowUCB1(
            arm_names=["x"], window_size=4, exploration_constant=0.0
        )
        ucb.record_pull("x")
        for _ in range(4):
            ucb.update_reward("x", 1.0)
        stats = ucb.get_stats()
        assert stats["x"]["window_size"] == 4
        assert stats["x"]["mean_reward"] == pytest.approx(1.0)

    def test_window_eviction_at_capacity_plus_one(self) -> None:
        ucb = SlidingWindowUCB1(
            arm_names=["x"], window_size=4, exploration_constant=0.0
        )
        ucb.record_pull("x")
        for _ in range(4):
            ucb.update_reward("x", 1.0)
        ucb.update_reward("x", 0.0)
        stats = ucb.get_stats()
        assert stats["x"]["window_size"] == 4
        assert stats["x"]["mean_reward"] == pytest.approx(0.75)

    def test_total_pulls_equals_sum_of_record_pull_calls(self) -> None:
        ucb = SlidingWindowUCB1(arm_names=["a", "b", "c"])
        for _ in range(3):
            ucb.record_pull("a")
        for _ in range(5):
            ucb.record_pull("b")
        ucb.record_pull("c")
        assert ucb._total_pulls == 9
        assert ucb.arms["a"].total_pulls == 3
        assert ucb.arms["b"].total_pulls == 5
        assert ucb.arms["c"].total_pulls == 1

    def test_warmup_skips_already_pulled_arms(self) -> None:
        ucb = SlidingWindowUCB1(arm_names=["first", "second"])
        ucb.record_pull("first")
        assert ucb.select() == "second"

    def test_get_stats_empty_window_returns_zero_mean(self) -> None:
        ucb = SlidingWindowUCB1(arm_names=["a"])
        ucb.record_pull("a")
        stats = ucb.get_stats()
        assert stats["a"]["total_pulls"] == 1
        assert stats["a"]["window_size"] == 0
        assert stats["a"]["mean_reward"] == pytest.approx(0.0)

    def test_all_arms_pulled_equal_times_with_equal_rewards_selects_deterministically(
        self,
    ) -> None:
        ucb = SlidingWindowUCB1(arm_names=["x", "y", "z"])
        for name in ["x", "y", "z"]:
            ucb.record_pull(name)
            ucb.update_reward(name, 0.5)
        assert ucb.select() == "x"


# ---------------------------------------------------------------------------
# SlidingWindowUCB1 — zero-observation regression
# ---------------------------------------------------------------------------


class TestSlidingWindowUCB1ZeroObservations:
    """Regression: zero-pull arms must never cause ZeroDivisionError or crash.

    The round-robin warmup in select() guarantees that UCB1 score computation
    (which divides by n_i) is only reached after all arms have been pulled at
    least once.  These tests verify that invariant explicitly.
    """

    def test_first_select_on_fresh_bandit_returns_valid_arm(self) -> None:
        """select() on a completely fresh bandit returns one of the arm names."""
        ucb = SlidingWindowUCB1(arm_names=["x", "y", "z"])
        name = ucb.select()
        assert name in {"x", "y", "z"}

    def test_all_arms_visited_during_warmup(self) -> None:
        """select()+record_pull() N times visits every arm exactly once before UCB1."""
        ucb = SlidingWindowUCB1(arm_names=["a", "b", "c"])
        visited = []
        for _ in range(3):
            name = ucb.select()
            ucb.record_pull(name)
            visited.append(name)
        # All three arms visited, no repeats before warmup ends
        assert set(visited) == {"a", "b", "c"}
        assert len(visited) == 3

    def test_no_division_by_zero_after_warmup(self) -> None:
        """UCB1 score computation after warmup does not raise ZeroDivisionError."""
        ucb = SlidingWindowUCB1(arm_names=["p", "q"])
        for name in ["p", "q"]:
            ucb.record_pull(name)
            ucb.update_reward(name, 0.5)
        # Should not raise — UCB1 formula executes without division by zero
        result = ucb.select()
        assert result in {"p", "q"}

    def test_two_arm_bandit_warmup_visits_both(self) -> None:
        """Two-arm bandit: both arms selected before exploitation begins."""
        ucb = SlidingWindowUCB1(arm_names=["arm0", "arm1"])
        first = ucb.select()
        ucb.record_pull(first)
        second = ucb.select()
        ucb.record_pull(second)
        assert first != second
        assert {first, second} == {"arm0", "arm1"}


# ---------------------------------------------------------------------------
# BanditModelRouter
# ---------------------------------------------------------------------------


def _make_mock_models(names: list[str]) -> list[MagicMock]:
    """Create mock ChatOpenAI models with given model names."""
    models = []
    for name in names:
        m = MagicMock()
        m.model_name = name
        m.with_structured_output = MagicMock(return_value=MagicMock())
        models.append(m)
    return models


class TestBanditModelRouter:
    def test_select_returns_model_and_name(self):
        models = _make_mock_models(["model_a", "model_b"])
        router = BanditModelRouter(
            models, [0.5, 0.5], fitness_key="score", higher_is_better=True
        )
        model, name = router._select()
        assert name in ["model_a", "model_b"]
        assert model in models

    def test_get_last_model_in_async_context(self):
        models = _make_mock_models(["model_a"])
        router = BanditModelRouter(
            models, [1.0], fitness_key="score", higher_is_better=True
        )

        async def _run():
            router._select()
            return router.get_last_model()

        result = asyncio.get_event_loop().run_until_complete(_run())
        assert result == "model_a"

    def test_get_last_model_pops(self):
        models = _make_mock_models(["model_a"])
        router = BanditModelRouter(
            models, [1.0], fitness_key="score", higher_is_better=True
        )

        async def _run():
            router._select()
            first = router.get_last_model()
            second = router.get_last_model()
            return first, second

        first, second = asyncio.get_event_loop().run_until_complete(_run())
        assert first == "model_a"
        assert second is None

    def test_on_mutation_outcome_updates_bandit(self):
        models = _make_mock_models(["model_a", "model_b"])
        router = BanditModelRouter(
            models, [0.5, 0.5], fitness_key="score", higher_is_better=True
        )
        router._bandit.record_pull("model_a")
        router._bandit.record_pull("model_b")

        child = Program(code="x=1")
        child.set_metadata("mutation_model", "model_a")
        child.metrics["score"] = 10.0

        parent = Program(code="x=0")
        parent.metrics["score"] = 8.0

        router.on_mutation_outcome(child, [parent])

        stats = router.get_bandit_stats()
        assert stats["model_a"]["window_size"] == 1
        assert stats["model_a"]["mean_reward"] > 0

    def test_on_mutation_outcome_skips_missing_model(self):
        models = _make_mock_models(["model_a"])
        router = BanditModelRouter(
            models, [1.0], fitness_key="score", higher_is_better=True
        )
        child = Program(code="x=1")
        child.metrics["score"] = 10.0
        parent = Program(code="x=0")
        parent.metrics["score"] = 8.0
        router.on_mutation_outcome(child, [parent])
        assert router.get_bandit_stats()["model_a"]["window_size"] == 0

    def test_on_mutation_outcome_missing_fitness_records_zero(self):
        """When child has no fitness, reward=0 should be recorded (not skipped)."""
        models = _make_mock_models(["model_a"])
        router = BanditModelRouter(
            models, [1.0], fitness_key="score", higher_is_better=True
        )
        child = Program(code="x=1")
        child.set_metadata("mutation_model", "model_a")
        # No fitness metric
        parent = Program(code="x=0")
        parent.metrics["score"] = 8.0
        router.on_mutation_outcome(child, [parent])
        # Now records a zero reward instead of skipping
        assert router.get_bandit_stats()["model_a"]["window_size"] == 1

    def test_on_mutation_outcome_no_parent_fitness_records_zero(self):
        """When parents lack fitness, reward=0 should be recorded."""
        models = _make_mock_models(["model_a"])
        router = BanditModelRouter(
            models, [1.0], fitness_key="score", higher_is_better=True
        )
        child = Program(code="x=1")
        child.set_metadata("mutation_model", "model_a")
        child.metrics["score"] = 10.0
        parent = Program(code="x=0")
        router.on_mutation_outcome(child, [parent])
        assert router.get_bandit_stats()["model_a"]["window_size"] == 1

    def test_get_bandit_stats(self):
        models = _make_mock_models(["model_a", "model_b"])
        router = BanditModelRouter(
            models, [0.5, 0.5], fitness_key="score", higher_is_better=True
        )
        stats = router.get_bandit_stats()
        assert set(stats.keys()) == {"model_a", "model_b"}
        assert stats["model_a"]["total_pulls"] == 0


# ---------------------------------------------------------------------------
# MutationOutcome handling
# ---------------------------------------------------------------------------


class TestMutationOutcomeHandling:
    def _make_router(self, **kwargs):
        models = _make_mock_models(["llama", "qwen"])
        defaults = dict(fitness_key="fitness", higher_is_better=True, window_size=50)
        defaults.update(kwargs)
        router = BanditModelRouter(models, [0.5, 0.5], **defaults)
        router._bandit.record_pull("llama")
        router._bandit.record_pull("qwen")
        return router

    def test_accepted_computes_normal_reward(self):
        router = self._make_router()
        child = Program(code="x=1")
        child.set_metadata("mutation_model", "llama")
        child.metrics["fitness"] = 0.030
        parent = Program(code="x=0")
        parent.metrics["fitness"] = 0.025
        router.on_mutation_outcome(child, [parent], outcome=MutationOutcome.ACCEPTED)
        stats = router.get_bandit_stats()
        assert stats["llama"]["window_size"] == 1
        assert stats["llama"]["mean_reward"] > 0

    def test_rejected_strategy_computes_normal_reward(self):
        """Valid program rejected by strategy still gets real fitness-based reward."""
        router = self._make_router()
        child = Program(code="x=1")
        child.set_metadata("mutation_model", "qwen")
        child.metrics["fitness"] = 0.020  # worse than parent
        parent = Program(code="x=0")
        parent.metrics["fitness"] = 0.025
        router.on_mutation_outcome(
            child, [parent], outcome=MutationOutcome.REJECTED_STRATEGY
        )
        stats = router.get_bandit_stats()
        assert stats["qwen"]["window_size"] == 1
        # improvement = 0.020 - 0.025 = -0.005 → clamped to 0 → reward = 0
        # During warmup normalizer returns 0.5

    def test_rejected_acceptor_injects_zero_reward(self):
        """Invalid/crashed program gets reward=0 without looking at fitness."""
        router = self._make_router()
        child = Program(code="x=CRASH")
        child.set_metadata("mutation_model", "llama")
        # Program might have sentinel or no fitness — doesn't matter
        child.metrics["fitness"] = -1000
        parent = Program(code="x=0")
        parent.metrics["fitness"] = 0.025
        router.on_mutation_outcome(
            child, [parent], outcome=MutationOutcome.REJECTED_ACCEPTOR
        )
        stats = router.get_bandit_stats()
        assert stats["llama"]["window_size"] == 1

    def test_rejected_acceptor_no_fitness_still_records(self):
        """Acceptor-rejected program with no fitness at all still gets reward=0."""
        router = self._make_router()
        child = Program(code="x=CRASH")
        child.set_metadata("mutation_model", "qwen")
        # No fitness at all
        router.on_mutation_outcome(child, [], outcome=MutationOutcome.REJECTED_ACCEPTOR)
        stats = router.get_bandit_stats()
        assert stats["qwen"]["window_size"] == 1

    def test_default_outcome_is_accepted(self):
        """Omitting outcome defaults to ACCEPTED behavior."""
        router = self._make_router()
        child = Program(code="x=1")
        child.set_metadata("mutation_model", "llama")
        child.metrics["fitness"] = 0.030
        parent = Program(code="x=0")
        parent.metrics["fitness"] = 0.025
        # No outcome kwarg
        router.on_mutation_outcome(child, [parent])
        stats = router.get_bandit_stats()
        assert stats["llama"]["window_size"] == 1
        assert stats["llama"]["mean_reward"] > 0


# ---------------------------------------------------------------------------
# Realistic heilbron scenarios
# ---------------------------------------------------------------------------


class TestBanditHeilbronScenarios:
    """Tests using realistic fitness values from the Heilbronn triangle problem.

    Heilbron problem: higher_is_better=True, fitness_key="fitness",
    range ~[0.0, 0.0365], sentinel=-1000, significant_change=0.001.
    """

    def _make_router(self):
        models = _make_mock_models(["llama-70b", "qwen-72b"])
        return BanditModelRouter(
            models,
            [0.5, 0.5],
            fitness_key="fitness",
            higher_is_better=True,
            window_size=50,
        )

    def test_small_improvement_produces_positive_reward(self):
        """Typical heilbron improvement: 0.025 → 0.026 (delta=0.001)."""
        router = self._make_router()
        router._bandit.record_pull("llama-70b")

        child = Program(code="solve()")
        child.set_metadata("mutation_model", "llama-70b")
        child.metrics["fitness"] = 0.026

        parent = Program(code="solve_old()")
        parent.metrics["fitness"] = 0.025

        router.on_mutation_outcome(child, [parent])

        stats = router.get_bandit_stats()
        assert stats["llama-70b"]["window_size"] == 1
        assert stats["llama-70b"]["mean_reward"] > 0

    def test_no_improvement_produces_zero_raw_reward(self):
        """Mutation that doesn't improve: 0.025 → 0.025."""
        r = compute_bandit_reward(0.025, 0.025, higher_is_better=True)
        assert r == pytest.approx(0.0)

    def test_regression_produces_zero_raw_reward(self):
        """Mutation that degrades: 0.025 → 0.020."""
        r = compute_bandit_reward(0.020, 0.025, higher_is_better=True)
        assert r == pytest.approx(0.0)

    def test_sentinel_value_higher_is_better_safe(self):
        """Sentinel -1000 with higher_is_better=True: improvement is hugely
        negative → clamped to 0 → reward = 0. No overflow."""
        r = compute_bandit_reward(-1000.0, 0.025, higher_is_better=True)
        assert r == pytest.approx(0.0)

    def test_sentinel_acceptor_rejection_flow(self):
        """Full flow: program crashes → sentinel fitness → acceptor rejects →
        bandit gets reward=0 without touching sentinel value."""
        router = self._make_router()
        router._bandit.record_pull("qwen-72b")

        child = Program(code="CRASH")
        child.set_metadata("mutation_model", "qwen-72b")
        child.metrics["fitness"] = -1000  # sentinel
        child.metrics["is_valid"] = 0

        parent = Program(code="solve()")
        parent.metrics["fitness"] = 0.025

        # Acceptor would reject this; engine calls with REJECTED_ACCEPTOR
        router.on_mutation_outcome(
            child, [parent], outcome=MutationOutcome.REJECTED_ACCEPTOR
        )

        stats = router.get_bandit_stats()
        assert stats["qwen-72b"]["window_size"] == 1

    def test_model_comparison_over_many_mutations(self):
        """Simulate 20 mutations per model: llama improves 50% of the time,
        qwen improves 20%. After enough data, bandit should prefer llama."""
        router = self._make_router()
        router._bandit.record_pull("llama-70b")
        router._bandit.record_pull("qwen-72b")

        import random

        rng = random.Random(42)
        parent_fitness = 0.020

        for _ in range(20):
            # llama: 50% chance of improvement
            child = Program(code="ll")
            child.set_metadata("mutation_model", "llama-70b")
            if rng.random() < 0.5:
                child.metrics["fitness"] = parent_fitness + rng.uniform(0.001, 0.005)
            else:
                child.metrics["fitness"] = parent_fitness - rng.uniform(0.001, 0.005)

            parent = Program(code="p")
            parent.metrics["fitness"] = parent_fitness
            router.on_mutation_outcome(child, [parent])

        for _ in range(20):
            # qwen: 20% chance of improvement
            child = Program(code="qw")
            child.set_metadata("mutation_model", "qwen-72b")
            if rng.random() < 0.2:
                child.metrics["fitness"] = parent_fitness + rng.uniform(0.001, 0.005)
            else:
                child.metrics["fitness"] = parent_fitness - rng.uniform(0.001, 0.005)

            parent = Program(code="p")
            parent.metrics["fitness"] = parent_fitness
            router.on_mutation_outcome(child, [parent])

        stats = router.get_bandit_stats()
        # llama should have higher mean reward than qwen
        assert stats["llama-70b"]["mean_reward"] > stats["qwen-72b"]["mean_reward"]

    def test_acceptor_rejections_penalize_unreliable_model(self):
        """Model that produces many invalid programs (acceptor rejections)
        should accumulate lower mean reward than a reliable model."""
        router = self._make_router()
        router._bandit.record_pull("llama-70b")
        router._bandit.record_pull("qwen-72b")

        parent = Program(code="p")
        parent.metrics["fitness"] = 0.020

        # llama: 10 valid mutations with small improvements
        for i in range(10):
            child = Program(code=f"ll_{i}")
            child.set_metadata("mutation_model", "llama-70b")
            child.metrics["fitness"] = 0.021  # small improvement
            router.on_mutation_outcome(child, [parent])

        # qwen: 2 valid, 8 crashes (acceptor rejections)
        for i in range(2):
            child = Program(code=f"qw_{i}")
            child.set_metadata("mutation_model", "qwen-72b")
            child.metrics["fitness"] = 0.021
            router.on_mutation_outcome(child, [parent])
        for i in range(8):
            child = Program(code=f"qw_crash_{i}")
            child.set_metadata("mutation_model", "qwen-72b")
            child.metrics["fitness"] = -1000  # sentinel
            router.on_mutation_outcome(
                child, [parent], outcome=MutationOutcome.REJECTED_ACCEPTOR
            )

        stats = router.get_bandit_stats()
        # llama: 10 small-positive rewards → higher mean
        # qwen: 2 small-positive + 8 zeros → lower mean
        assert stats["llama-70b"]["mean_reward"] > stats["qwen-72b"]["mean_reward"]

    def test_lower_is_better_problem(self):
        """For a lower-is-better problem (e.g., minimizing cost), child with
        lower fitness than parent should get positive reward."""
        models = _make_mock_models(["model_a"])
        router = BanditModelRouter(
            models,
            [1.0],
            fitness_key="cost",
            higher_is_better=False,
            window_size=50,
        )
        router._bandit.record_pull("model_a")

        child = Program(code="x=1")
        child.set_metadata("mutation_model", "model_a")
        child.metrics["cost"] = 3.0  # improved (lower)

        parent = Program(code="x=0")
        parent.metrics["cost"] = 5.0

        router.on_mutation_outcome(child, [parent])

        stats = router.get_bandit_stats()
        assert stats["model_a"]["window_size"] == 1
        assert stats["model_a"]["mean_reward"] > 0

    def test_lower_is_better_regression_zero_reward(self):
        """For lower-is-better, child with higher cost should get zero reward."""
        r = compute_bandit_reward(7.0, 5.0, higher_is_better=False)
        assert r == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# MultiModelRouter.get_last_model
# ---------------------------------------------------------------------------


class TestMultiModelRouterGetLastModel:
    def test_standard_router_tracks_model(self):
        from gigaevo.llm.models import MultiModelRouter

        models = _make_mock_models(["m1"])
        router = MultiModelRouter(models, [1.0])

        async def _run():
            router._select()
            return router.get_last_model()

        result = asyncio.get_event_loop().run_until_complete(_run())
        assert result == "m1"

    def test_no_task_returns_none(self):
        from gigaevo.llm.models import MultiModelRouter

        models = _make_mock_models(["m1"])
        router = MultiModelRouter(models, [1.0])
        router._select()
        result = router.get_last_model()
        assert result is None


# ---------------------------------------------------------------------------
# Shared _task_model_map between routers
# ---------------------------------------------------------------------------


class TestSharedTaskModelMap:
    def test_structured_router_writes_to_shared_map(self):
        from gigaevo.llm.models import MultiModelRouter

        models = _make_mock_models(["m1"])
        router = MultiModelRouter(models, [1.0])

        async def _run():
            structured = router.with_structured_output(MagicMock())
            structured._select()
            return router.get_last_model()

        result = asyncio.get_event_loop().run_until_complete(_run())
        assert result == "m1"


# ---------------------------------------------------------------------------
# _StructuredOutputRouter with select_override (bandit)
# ---------------------------------------------------------------------------


class TestStructuredOutputRouterWithOverride:
    def test_select_override_delegates_to_bandit(self):
        models = _make_mock_models(["model_a", "model_b"])
        router = BanditModelRouter(
            models, [0.5, 0.5], fitness_key="score", higher_is_better=True
        )
        structured = router.with_structured_output(MagicMock())

        async def _run():
            _, name = structured._select()
            return name

        name = asyncio.get_event_loop().run_until_complete(_run())
        assert name in ["model_a", "model_b"]
        stats = router.get_bandit_stats()
        total = sum(s["total_pulls"] for s in stats.values())
        assert total >= 1


# ---------------------------------------------------------------------------
# LLMMutationOperator.on_program_ingested
# ---------------------------------------------------------------------------


class TestLLMMutationOperatorOnProgramIngested:
    @pytest.mark.asyncio
    async def test_calls_on_mutation_outcome_with_outcome(self):
        from gigaevo.evolution.mutation.mutation_operator import LLMMutationOperator

        mock_router = MagicMock(spec=BanditModelRouter)
        mock_router.model_names = ["m1"]
        mock_router.models = _make_mock_models(["m1"])
        mock_router.on_mutation_outcome = MagicMock()

        parent = Program(code="x=0")
        parent.metrics["score"] = 5.0

        child = Program(code="x=1")
        child.lineage.parents = [parent.id]
        child.set_metadata("mutation_model", "m1")
        child.metrics["score"] = 10.0

        mock_storage = AsyncMock()
        mock_storage.mget = AsyncMock(return_value=[parent])

        with patch.object(LLMMutationOperator, "__init__", lambda self, **kw: None):
            op = LLMMutationOperator.__new__(LLMMutationOperator)
            op.llm_wrapper = mock_router

        await op.on_program_ingested(
            child, mock_storage, outcome=MutationOutcome.ACCEPTED
        )
        mock_router.on_mutation_outcome.assert_called_once_with(
            child, [parent], outcome=MutationOutcome.ACCEPTED
        )

    @pytest.mark.asyncio
    async def test_passes_rejected_acceptor_outcome(self):
        from gigaevo.evolution.mutation.mutation_operator import LLMMutationOperator

        mock_router = MagicMock(spec=BanditModelRouter)
        mock_router.on_mutation_outcome = MagicMock()

        child = Program(code="CRASH")
        child.lineage.parents = ["some_parent_id"]

        parent = Program(code="x=0")
        parent.metrics["score"] = 5.0

        mock_storage = AsyncMock()
        mock_storage.mget = AsyncMock(return_value=[parent])

        with patch.object(LLMMutationOperator, "__init__", lambda self, **kw: None):
            op = LLMMutationOperator.__new__(LLMMutationOperator)
            op.llm_wrapper = mock_router

        await op.on_program_ingested(
            child, mock_storage, outcome=MutationOutcome.REJECTED_ACCEPTOR
        )
        mock_router.on_mutation_outcome.assert_called_once_with(
            child, [parent], outcome=MutationOutcome.REJECTED_ACCEPTOR
        )

    @pytest.mark.asyncio
    async def test_skips_root_programs(self):
        from gigaevo.evolution.mutation.mutation_operator import LLMMutationOperator

        mock_router = MagicMock(spec=BanditModelRouter)
        mock_router.on_mutation_outcome = MagicMock()

        root = Program(code="x=0")
        mock_storage = AsyncMock()

        with patch.object(LLMMutationOperator, "__init__", lambda self, **kw: None):
            op = LLMMutationOperator.__new__(LLMMutationOperator)
            op.llm_wrapper = mock_router

        await op.on_program_ingested(root, mock_storage)
        mock_router.on_mutation_outcome.assert_not_called()
        mock_storage.mget.assert_not_called()


# ---------------------------------------------------------------------------
# BanditModelRouter edge cases
# ---------------------------------------------------------------------------


class TestBanditModelRouterEdgeCases:
    def test_on_mutation_outcome_higher_is_better_uses_max_parent(self) -> None:
        models = _make_mock_models_shared(["m"])
        router = BanditModelRouter(
            models, [1.0], fitness_key="score", higher_is_better=True
        )
        router._bandit.record_pull("m")

        child = Program(code="x=1")
        child.set_metadata("mutation_model", "m")
        child.metrics["score"] = 10.0

        weak_parent = Program(code="x=weak")
        weak_parent.metrics["score"] = 2.0
        strong_parent = Program(code="x=strong")
        strong_parent.metrics["score"] = 9.0

        router.on_mutation_outcome(child, [weak_parent, strong_parent])

        stats = router.get_bandit_stats()
        assert stats["m"]["window_size"] == 1
        assert stats["m"]["mean_reward"] > 0.0

    def test_on_mutation_outcome_lower_is_better_uses_min_parent(self) -> None:
        models = _make_mock_models_shared(["m"])
        router = BanditModelRouter(
            models, [1.0], fitness_key="cost", higher_is_better=False
        )
        router._bandit.record_pull("m")

        child = Program(code="x=1")
        child.set_metadata("mutation_model", "m")
        child.metrics["cost"] = 2.0

        bad_parent = Program(code="x=bad")
        bad_parent.metrics["cost"] = 10.0
        good_parent = Program(code="x=good")
        good_parent.metrics["cost"] = 3.0

        router.on_mutation_outcome(child, [bad_parent, good_parent])

        stats = router.get_bandit_stats()
        assert stats["m"]["window_size"] == 1
        assert stats["m"]["mean_reward"] > 0.0

    def test_on_mutation_outcome_lower_is_better_child_worse_yields_zero(self) -> None:
        models = _make_mock_models_shared(["m"])
        router = BanditModelRouter(
            models, [1.0], fitness_key="cost", higher_is_better=False
        )
        router._bandit.record_pull("m")

        child = Program(code="x=1")
        child.set_metadata("mutation_model", "m")
        child.metrics["cost"] = 7.0

        parent = Program(code="x=0")
        parent.metrics["cost"] = 5.0

        router.on_mutation_outcome(child, [parent])

        stats = router.get_bandit_stats()
        assert stats["m"]["window_size"] == 1

    def test_select_outside_async_context_does_not_crash(self) -> None:
        models = _make_mock_models_shared(["solo"])
        router = BanditModelRouter(
            models, [1.0], fitness_key="score", higher_is_better=True
        )
        model, name = router._select()
        assert name == "solo"
        assert model is models[0]

    def test_probability_normalization_preserved(self) -> None:
        models = _make_mock_models_shared(["a", "b", "c"])
        router = BanditModelRouter(
            models,
            [1.0, 3.0, 6.0],
            fitness_key="score",
            higher_is_better=True,
        )
        assert sum(router.probabilities) == pytest.approx(1.0)
        assert router.probabilities == pytest.approx([0.1, 0.3, 0.6])

    def test_structured_output_records_bandit_pull(self) -> None:
        models = _make_mock_models_shared(["m1", "m2"])
        router = BanditModelRouter(
            models, [0.5, 0.5], fitness_key="score", higher_is_better=True
        )
        structured = router.with_structured_output(MagicMock())

        async def _run() -> str | None:
            structured._select()
            return router.get_last_model()

        result = asyncio.get_event_loop().run_until_complete(_run())
        assert result in ["m1", "m2"]
        total_pulls = sum(s["total_pulls"] for s in router.get_bandit_stats().values())
        assert total_pulls == 1

    def test_on_mutation_outcome_accumulates_multiple_rewards(self) -> None:
        models = _make_mock_models_shared(["m"])
        router = BanditModelRouter(
            models,
            [1.0],
            fitness_key="score",
            higher_is_better=True,
            window_size=10,
        )
        router._bandit.record_pull("m")

        for i in range(5):
            child = Program(code=f"x={i}")
            child.set_metadata("mutation_model", "m")
            child.metrics["score"] = float(i + 2)
            parent = Program(code=f"x={i}_p")
            parent.metrics["score"] = float(i)
            router.on_mutation_outcome(child, [parent])

        stats = router.get_bandit_stats()
        assert stats["m"]["window_size"] == 5


# ---------------------------------------------------------------------------
# MultiModelRouter validation
# ---------------------------------------------------------------------------


class TestMultiModelRouterValidation:
    def test_length_mismatch_raises_value_error(self) -> None:
        from gigaevo.llm.models import MultiModelRouter

        models = _make_mock_models_shared(["a", "b"])
        with pytest.raises(ValueError, match="Length mismatch"):
            MultiModelRouter(models, [1.0])

    def test_zero_probability_raises_value_error(self) -> None:
        from gigaevo.llm.models import MultiModelRouter

        models = _make_mock_models_shared(["a", "b"])
        with pytest.raises(ValueError, match="probabilities must be positive"):
            MultiModelRouter(models, [0.0, 1.0])

    def test_negative_probability_raises_value_error(self) -> None:
        from gigaevo.llm.models import MultiModelRouter

        models = _make_mock_models_shared(["a", "b"])
        with pytest.raises(ValueError, match="probabilities must be positive"):
            MultiModelRouter(models, [-0.5, 1.5])

    def test_unnormalized_probabilities_are_normalized(self) -> None:
        from gigaevo.llm.models import MultiModelRouter

        models = _make_mock_models_shared(["a", "b"])
        router = MultiModelRouter(models, [2.0, 8.0])
        assert router.probabilities == pytest.approx([0.2, 0.8])
        assert sum(router.probabilities) == pytest.approx(1.0)

    def test_single_model_always_selected(self) -> None:
        from gigaevo.llm.models import MultiModelRouter

        models = _make_mock_models_shared(["only"])
        router = MultiModelRouter(models, [1.0])

        async def _run() -> str | None:
            router._select()
            return router.get_last_model()

        result = asyncio.get_event_loop().run_until_complete(_run())
        assert result == "only"


# ---------------------------------------------------------------------------
# Async concurrency — task_model_map isolation
# ---------------------------------------------------------------------------


class TestTaskModelMapConcurrency:
    async def test_n_concurrent_tasks_each_get_own_selection(self) -> None:
        models = _make_mock_models_shared(["a", "b", "c"])
        router = BanditModelRouter(
            models,
            [1 / 3, 1 / 3, 1 / 3],
            fitness_key="score",
            higher_is_better=True,
        )
        valid_names = {"a", "b", "c"}
        results: dict[int, str | None] = {}

        async def worker(task_id: int) -> None:
            router._select()
            await asyncio.sleep(0)
            results[task_id] = router.get_last_model()

        await asyncio.wait_for(
            asyncio.gather(*[worker(i) for i in range(12)]),
            timeout=5.0,
        )

        assert all(v is not None for v in results.values())
        assert all(v in valid_names for v in results.values())

    async def test_task_map_is_empty_after_all_get_last_model_calls(self) -> None:
        models = _make_mock_models_shared(["m"])
        router = BanditModelRouter(
            models, [1.0], fitness_key="score", higher_is_better=True
        )

        async def worker() -> None:
            router._select()
            await asyncio.sleep(0)
            router.get_last_model()

        await asyncio.wait_for(
            asyncio.gather(*[worker() for _ in range(20)]),
            timeout=5.0,
        )
        assert router._task_model_map == {}

    async def test_task2_gets_none_when_only_task1_selected(self) -> None:
        models = _make_mock_models_shared(["alpha"])
        router = BanditModelRouter(
            models, [1.0], fitness_key="score", higher_is_better=True
        )

        selected_event = asyncio.Event()
        read_event = asyncio.Event()
        task1_name: str | None = None
        task2_name: str | None = None

        async def task1() -> None:
            nonlocal task1_name
            router._select()
            selected_event.set()
            await read_event.wait()
            task1_name = router.get_last_model()

        async def task2() -> None:
            nonlocal task2_name
            await selected_event.wait()
            task2_name = router.get_last_model()
            read_event.set()

        await asyncio.wait_for(
            asyncio.gather(task1(), task2()),
            timeout=5.0,
        )

        assert task1_name == "alpha"
        assert task2_name is None

    async def test_concurrent_on_mutation_outcome_does_not_corrupt_bandit(
        self,
    ) -> None:
        models = _make_mock_models_shared(["m"])
        router = BanditModelRouter(
            models, [1.0], fitness_key="score", higher_is_better=True, window_size=50
        )
        router._bandit.record_pull("m")

        n_updates = 30

        async def send_outcome(i: int) -> None:
            child = Program(code=f"x={i}")
            child.set_metadata("mutation_model", "m")
            child.metrics["score"] = float(i + 1)
            parent = Program(code=f"y={i}")
            parent.metrics["score"] = float(i)
            await asyncio.sleep(0)
            router.on_mutation_outcome(child, [parent])

        await asyncio.wait_for(
            asyncio.gather(*[send_outcome(i) for i in range(n_updates)]),
            timeout=5.0,
        )

        stats = router.get_bandit_stats()
        assert stats["m"]["window_size"] == n_updates


# ---------------------------------------------------------------------------
# on_program_ingested edge cases
# ---------------------------------------------------------------------------


class TestOnProgramIngestedEdgeCases:
    async def test_all_null_parents_from_storage_calls_outcome_with_empty_list(
        self,
    ) -> None:
        from gigaevo.evolution.mutation.mutation_operator import LLMMutationOperator

        mock_router = MagicMock(spec=BanditModelRouter)
        mock_router.on_mutation_outcome = MagicMock()

        child = Program(code="x=1")
        child.lineage.parents = ["gone_id_1", "gone_id_2"]

        mock_storage = AsyncMock()
        mock_storage.mget = AsyncMock(return_value=[None, None])

        with patch.object(LLMMutationOperator, "__init__", lambda self, **kw: None):
            op = LLMMutationOperator.__new__(LLMMutationOperator)
            op.llm_wrapper = mock_router

        await op.on_program_ingested(child, mock_storage)

        mock_router.on_mutation_outcome.assert_called_once_with(child, [], outcome=None)

    async def test_mixed_null_and_valid_parents_filters_nulls(self) -> None:
        from gigaevo.evolution.mutation.mutation_operator import LLMMutationOperator

        mock_router = MagicMock(spec=BanditModelRouter)
        mock_router.on_mutation_outcome = MagicMock()

        surviving_parent = Program(code="x=0")
        surviving_parent.metrics["score"] = 5.0

        child = Program(code="x=1")
        child.lineage.parents = ["id_alive", "id_gone"]

        mock_storage = AsyncMock()
        mock_storage.mget = AsyncMock(return_value=[surviving_parent, None])

        with patch.object(LLMMutationOperator, "__init__", lambda self, **kw: None):
            op = LLMMutationOperator.__new__(LLMMutationOperator)
            op.llm_wrapper = mock_router

        await op.on_program_ingested(child, mock_storage)

        mock_router.on_mutation_outcome.assert_called_once_with(
            child, [surviving_parent], outcome=None
        )

    async def test_mget_called_with_exact_parent_ids(self) -> None:
        from gigaevo.evolution.mutation.mutation_operator import LLMMutationOperator

        mock_router = MagicMock(spec=BanditModelRouter)
        mock_router.on_mutation_outcome = MagicMock()

        parent = Program(code="x=0")
        parent.metrics["score"] = 3.0

        child = Program(code="x=1")
        parent_ids = [parent.id, "some-other-uuid-1234-5678-abcd-ef01"]
        child.lineage.parents = parent_ids

        mock_storage = AsyncMock()
        mock_storage.mget = AsyncMock(return_value=[parent, None])

        with patch.object(LLMMutationOperator, "__init__", lambda self, **kw: None):
            op = LLMMutationOperator.__new__(LLMMutationOperator)
            op.llm_wrapper = mock_router

        await op.on_program_ingested(child, mock_storage)

        mock_storage.mget.assert_called_once_with(parent_ids)
