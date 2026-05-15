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


# ---------------------------------------------------------------------------
# Classifier-driven failure dispatch through invoke / ainvoke
# ---------------------------------------------------------------------------


class TestBanditFailureDispatchViaClassifier:
    """``_select`` records the pull before the LLM call. A failure between
    those two points used to inflate ``total_pulls`` with no matching reward
    entry, shrinking the UCB1 confidence term for the failing arm and
    underexploring flaky models. The new dispatch wraps the LLM call,
    classifies the exception via ``classify_call_result``, and injects a
    zero reward via ``_inject_failure_reward`` so pulls and the reward
    window stay in step on every failure path. The exception still
    propagates."""

    def _router_with_flaky_arm(
        self, exc: BaseException
    ) -> tuple[BanditModelRouter, MagicMock]:
        flaky = MagicMock()
        flaky.model_name = "flaky"
        flaky.invoke = MagicMock(side_effect=exc)
        flaky.ainvoke = AsyncMock(side_effect=exc)
        flaky.with_structured_output = MagicMock(return_value=MagicMock())

        healthy = MagicMock()
        healthy.model_name = "healthy"
        healthy.with_structured_output = MagicMock(return_value=MagicMock())

        router = BanditModelRouter(
            [flaky, healthy],
            [0.5, 0.5],
            fitness_key="score",
            higher_is_better=True,
        )
        router._langfuse = None
        router._bandit.select = lambda: "flaky"  # type: ignore[assignment]
        return router, flaky

    def test_sync_invoke_failure_records_zero_reward_and_propagates(
        self,
    ) -> None:
        router, _flaky = self._router_with_flaky_arm(RuntimeError("rate limited"))

        with pytest.raises(RuntimeError, match="rate limited"):
            router.invoke("hello")

        stats = router.get_bandit_stats()
        assert stats["flaky"]["total_pulls"] == 1
        assert stats["flaky"]["window_size"] == 1

    async def test_async_ainvoke_failure_records_zero_reward_and_propagates(
        self,
    ) -> None:
        router, _flaky = self._router_with_flaky_arm(RuntimeError("rate limited"))

        with pytest.raises(RuntimeError, match="rate limited"):
            await router.ainvoke("hello")

        stats = router.get_bandit_stats()
        assert stats["flaky"]["total_pulls"] == 1
        assert stats["flaky"]["window_size"] == 1

    async def test_repeated_ainvoke_failures_keep_ledgers_in_step(self) -> None:
        router, _flaky = self._router_with_flaky_arm(RuntimeError("boom"))

        for _ in range(7):
            with pytest.raises(RuntimeError):
                await router.ainvoke("hello")

        stats = router.get_bandit_stats()
        assert stats["flaky"]["total_pulls"] == 7
        assert stats["flaky"]["window_size"] == 7

    async def test_successful_call_does_not_inject_immediate_reward(self) -> None:
        # The success path defers the reward to on_mutation_outcome, which
        # runs later with the fitness result.
        model = MagicMock()
        model.model_name = "ok"
        model.ainvoke = AsyncMock(return_value=MagicMock())
        model.with_structured_output = MagicMock(return_value=MagicMock())

        router = BanditModelRouter(
            [model], [1.0], fitness_key="score", higher_is_better=True
        )
        router._langfuse = None

        await router.ainvoke("hello")

        stats = router.get_bandit_stats()
        assert stats["ok"]["total_pulls"] == 1
        # No reward entry yet — on_mutation_outcome drives the real reward.
        assert stats["ok"]["window_size"] == 0

    def test_structured_output_failure_also_injects_zero_reward(self) -> None:
        # The bandit's with_structured_output wires the failure_hook through
        # to _StructuredOutputRouter so the structured-output dispatch path
        # gets the same ledger-symmetry guarantee.
        flaky = MagicMock()
        flaky.model_name = "flaky"
        flaky.with_structured_output = MagicMock(return_value=flaky)
        flaky.invoke = MagicMock(side_effect=RuntimeError("structured failure"))

        router = BanditModelRouter(
            [flaky], [1.0], fitness_key="score", higher_is_better=True
        )
        router._langfuse = None

        structured = router.with_structured_output(dict)
        with pytest.raises(RuntimeError, match="structured failure"):
            structured.invoke("hello")

        stats = router.get_bandit_stats()
        assert stats["flaky"]["total_pulls"] == 1
        assert stats["flaky"]["window_size"] == 1

    def test_inject_failure_reward_skips_unknown_arm_silently(self) -> None:
        # Defense-in-depth symmetry with on_mutation_outcome's unknown-arm
        # guard. _select cannot normally return a name outside
        # self._bandit.arms, but if a future caller invokes
        # _inject_failure_reward directly with a stale name (loaded from a
        # snapshot, hand-built in a test, etc.) the helper must not raise
        # KeyError on top of the original exception.
        router, _ = self._router_with_flaky_arm(RuntimeError("boom"))

        # Should not raise.
        router._inject_failure_reward(RuntimeError("orig"), "not_an_arm")

        stats = router.get_bandit_stats()
        assert stats["flaky"]["window_size"] == 0
        assert stats["healthy"]["window_size"] == 0


# ---------------------------------------------------------------------------
# Re-raise integrity: cause chain, context, traceback survive failure hook
# ---------------------------------------------------------------------------


class TestBanditFailureReRaiseIntegrity:
    """The classifier dispatch wraps the call in ``except BaseException`` and
    bare-``raise``s after injecting the zero reward. The original exception
    object, its traceback, ``__cause__`` (explicit ``raise X from Y``), and
    ``__context__`` (implicit chain) must all survive untouched — otherwise
    higher-level retry / logging layers cannot tell the real failure apart
    from a bandit-side error."""

    def _make_router_raising(self, exc: BaseException) -> BanditModelRouter:
        flaky = MagicMock()
        flaky.model_name = "flaky"
        flaky.invoke = MagicMock(side_effect=exc)
        flaky.ainvoke = AsyncMock(side_effect=exc)
        flaky.with_structured_output = MagicMock(return_value=MagicMock())
        router = BanditModelRouter(
            [flaky], [1.0], fitness_key="score", higher_is_better=True
        )
        router._langfuse = None
        router._bandit.select = lambda: "flaky"  # type: ignore[assignment]
        return router

    def test_sync_invoke_preserves_explicit_cause_chain(self) -> None:
        root = ValueError("root cause")
        try:
            raise RuntimeError("surface") from root
        except RuntimeError as e:
            surface = e
        router = self._make_router_raising(surface)

        with pytest.raises(RuntimeError, match="surface") as exc_info:
            router.invoke("hello")

        # Same exception object identity is the strictest possible check.
        assert exc_info.value is surface
        assert exc_info.value.__cause__ is root
        assert exc_info.value.__traceback__ is not None

    async def test_async_ainvoke_preserves_explicit_cause_chain(self) -> None:
        root = ValueError("root cause async")
        try:
            raise RuntimeError("surface async") from root
        except RuntimeError as e:
            surface = e
        router = self._make_router_raising(surface)

        with pytest.raises(RuntimeError, match="surface async") as exc_info:
            await router.ainvoke("hello")
        assert exc_info.value is surface
        assert exc_info.value.__cause__ is root


# ---------------------------------------------------------------------------
# Hardened failure-hook: classifier-internal errors must not mask LLM exc
# ---------------------------------------------------------------------------


class TestBanditFailureHookErrorContainment:
    """If the classifier itself raises (e.g. a future schema change that
    refuses some attribute), ``_inject_failure_reward`` must not let the new
    exception replace the original LLM failure. Same for downstream
    normalizer / bandit calls. The original exception is what the caller
    asked to handle; the bandit hook is observability only."""

    def _make_router(self) -> BanditModelRouter:
        flaky = MagicMock()
        flaky.model_name = "flaky"
        flaky.invoke = MagicMock(side_effect=RuntimeError("real failure"))
        flaky.with_structured_output = MagicMock(return_value=MagicMock())
        router = BanditModelRouter(
            [flaky], [1.0], fitness_key="score", higher_is_better=True
        )
        router._langfuse = None
        router._bandit.select = lambda: "flaky"  # type: ignore[assignment]
        return router

    def test_classifier_raising_does_not_mask_real_exception(self) -> None:
        router = self._make_router()
        with patch(
            "gigaevo.llm.bandit.classify_call_result",
            side_effect=ValueError("classifier exploded"),
        ):
            with pytest.raises(RuntimeError, match="real failure"):
                router.invoke("hello")


# ---------------------------------------------------------------------------
# Structured-output: _process raising after a successful invoke
# ---------------------------------------------------------------------------


class TestStructuredOutputProcessFailureFiresHook:
    """``_StructuredOutputRouter.invoke`` wraps ``model.invoke`` in
    try/except so the failure_hook fires on transport errors. But the
    response goes through ``_process`` afterward to extract the parsed
    Pydantic object, and the response dict may contain a parser exception
    that the langchain wrapper surfaces as a ``response['parsing_error']``
    or by raising directly. If ``_process`` raises, the bandit ledger
    must still be told (otherwise we have an inflated pull count with no
    matching reward — exactly the desync the wiring was supposed to
    prevent)."""

    def test_structured_process_failure_fires_failure_hook(self) -> None:
        flaky = MagicMock()
        flaky.model_name = "flaky"

        # Return a malformed response that crashes _process. The simplest
        # repro: response.get("raw") evaluates fine, but then we patch the
        # tracker.track to raise — same effect, no need to mock pydantic.
        flaky.with_structured_output = MagicMock(return_value=flaky)
        flaky.invoke = MagicMock(return_value={"raw": MagicMock(), "parsed": None})

        router = BanditModelRouter(
            [flaky], [1.0], fitness_key="score", higher_is_better=True
        )
        router._langfuse = None
        broken_tracker = MagicMock()
        broken_tracker.track = MagicMock(side_effect=RuntimeError("track exploded"))
        router._tracker = broken_tracker  # type: ignore[assignment]

        structured = router.with_structured_output(dict)
        with pytest.raises(RuntimeError, match="track exploded"):
            structured.invoke("hello")

        # The bandit ledger must be in step: one pull, one reward injection.
        stats = router.get_bandit_stats()
        assert stats["flaky"]["total_pulls"] == 1
        assert stats["flaky"]["window_size"] == 1


# ---------------------------------------------------------------------------
# Success-path tracker exception must not leave ledger out of step
# ---------------------------------------------------------------------------


class TestBanditSuccessPathTrackerFailure:
    """The success path defers reward to ``on_mutation_outcome``. But if
    ``self._tracker.track`` raises (malformed token_usage from a hostile
    provider, telemetry-side bug, etc.) the exception leaks back to the
    caller without the bandit having recorded a reward — same desync as
    the original failure case. The fix: tracker errors should not be
    treated as bandit failures (the LLM call succeeded), but the caller
    deserves a usable response. We swallow tracker errors and continue."""

    def test_sync_success_with_tracker_exception_returns_response(self) -> None:
        response = MagicMock()
        model = MagicMock()
        model.model_name = "ok"
        model.invoke = MagicMock(return_value=response)
        model.with_structured_output = MagicMock(return_value=MagicMock())

        router = BanditModelRouter(
            [model], [1.0], fitness_key="score", higher_is_better=True
        )
        router._langfuse = None
        broken_tracker = MagicMock()
        broken_tracker.track = MagicMock(side_effect=RuntimeError("telemetry exploded"))
        router._tracker = broken_tracker  # type: ignore[assignment]

        # Tracker errors are telemetry; the caller must still get the LLM
        # response and the bandit must record the pull. The deferred reward
        # arrives via on_mutation_outcome.
        result = router.invoke("hello")
        assert result is response

        stats = router.get_bandit_stats()
        assert stats["ok"]["total_pulls"] == 1
        # No reward yet — deferred to on_mutation_outcome.
        assert stats["ok"]["window_size"] == 0

    async def test_async_success_with_tracker_exception_returns_response(
        self,
    ) -> None:
        response = MagicMock()
        model = MagicMock()
        model.model_name = "ok"
        model.ainvoke = AsyncMock(return_value=response)
        model.with_structured_output = MagicMock(return_value=MagicMock())

        router = BanditModelRouter(
            [model], [1.0], fitness_key="score", higher_is_better=True
        )
        router._langfuse = None
        broken_tracker = MagicMock()
        broken_tracker.track = MagicMock(side_effect=RuntimeError("telemetry exploded"))
        router._tracker = broken_tracker  # type: ignore[assignment]

        result = await router.ainvoke("hello")
        assert result is response

        stats = router.get_bandit_stats()
        assert stats["ok"]["total_pulls"] == 1
        assert stats["ok"]["window_size"] == 0


# ---------------------------------------------------------------------------
# ContextVar regression: get_selected_model after bandit routing
# ---------------------------------------------------------------------------


class TestBanditContextVarPropagation:
    """``MultiModelRouter._select`` calls ``_remember_selected_model`` so that
    downstream consumers (``MutationAgent``, ``BaseAgent``) can read the
    selected model via ``get_selected_model()``. ``BanditModelRouter._select``
    was wired without that call, so any agent stack that consumes
    ``get_selected_model()`` would see a stale value (whatever the last
    non-bandit selection left in the ContextVar, or ``None``).
    """

    async def test_get_selected_model_returns_bandit_arm(self) -> None:
        from gigaevo.llm.models import get_selected_model

        models = _make_mock_models(["arm_a", "arm_b"])
        router = BanditModelRouter(
            models, [0.5, 0.5], fitness_key="score", higher_is_better=True
        )

        async def _run() -> str | None:
            router._bandit.select = lambda: "arm_b"  # type: ignore[assignment]
            router._select()
            return get_selected_model()

        result = await _run()
        assert result == "arm_b"

    async def test_structured_select_override_sets_context_var(self) -> None:
        from gigaevo.llm.models import get_selected_model

        models = _make_mock_models(["arm_a", "arm_b"])
        router = BanditModelRouter(
            models, [0.5, 0.5], fitness_key="score", higher_is_better=True
        )

        async def _run() -> str | None:
            router._bandit.select = lambda: "arm_a"  # type: ignore[assignment]
            structured = router.with_structured_output(dict)
            structured._select()
            return get_selected_model()

        result = await _run()
        assert result == "arm_a"


class TestBanditStreamingFailureDispatch:
    """``stream`` and ``astream`` are inherited from ``MultiModelRouter``.
    Without an override on ``BanditModelRouter`` they call ``_select`` (which
    records the pull through the bandit's overridden ``_select``) but then
    iterate ``model.{,a}stream`` with no try/except — a mid-stream failure
    would inflate ``total_pulls`` for the failing arm without a matching
    window entry, exactly the asymmetry the classifier wiring exists to
    eliminate. Streaming must follow the same ledger-symmetry contract as
    ``invoke``/``ainvoke``."""

    def _flaky_streaming_router(
        self, exc: BaseException
    ) -> tuple[BanditModelRouter, MagicMock]:
        flaky = MagicMock()
        flaky.model_name = "flaky"

        def _sync_stream(*_args, **_kwargs):
            raise exc

        async def _async_stream(*_args, **_kwargs):
            raise exc
            yield  # pragma: no cover — unreachable, makes this an async generator

        flaky.stream = MagicMock(side_effect=_sync_stream)
        flaky.astream = _async_stream
        flaky.with_structured_output = MagicMock(return_value=MagicMock())

        router = BanditModelRouter(
            [flaky], [1.0], fitness_key="score", higher_is_better=True
        )
        router._langfuse = None
        return router, flaky

    def test_sync_stream_failure_records_zero_reward_and_propagates(self) -> None:
        router, _flaky = self._flaky_streaming_router(RuntimeError("stream boom"))

        with pytest.raises(RuntimeError, match="stream boom"):
            # ``stream`` is a generator — exhaust it to trigger the call.
            for _ in router.stream("hello"):
                pass

        stats = router.get_bandit_stats()
        assert stats["flaky"]["total_pulls"] == 1
        # Without the override the window would still be empty here, leaving
        # ``total_pulls`` and ``window_size`` permanently out of step.
        assert stats["flaky"]["window_size"] == 1

    async def test_async_astream_failure_records_zero_reward_and_propagates(
        self,
    ) -> None:
        router, _flaky = self._flaky_streaming_router(RuntimeError("astream boom"))

        with pytest.raises(RuntimeError, match="astream boom"):
            async for _ in router.astream("hello"):
                pass

        stats = router.get_bandit_stats()
        assert stats["flaky"]["total_pulls"] == 1
        assert stats["flaky"]["window_size"] == 1


class TestInjectFailureRewardCannotMaskOriginalException:
    """``BanditModelRouter.invoke``/``ainvoke`` call ``_inject_failure_reward``
    from inside an ``except`` block.  If the hook itself raises (a logger
    blowing up, a corrupted normalizer, a classifier regression, etc.) the
    naive ``raise`` at the end would surface the *hook's* exception instead
    of the original LLM failure — and the original traceback would be lost.
    The ``_StructuredOutputRouter`` path already protects against this via
    ``_maybe_fire_failure_hook``; the direct path must follow suit."""

    def _router_with_broken_hook(self) -> BanditModelRouter:
        model = MagicMock()
        model.model_name = "m"
        model.invoke = MagicMock(side_effect=RuntimeError("real failure"))
        model.ainvoke = AsyncMock(side_effect=RuntimeError("real failure"))
        model.with_structured_output = MagicMock(return_value=MagicMock())

        router = BanditModelRouter(
            [model], [1.0], fitness_key="score", higher_is_better=True
        )
        router._langfuse = None

        def _explode(_exc, _name):
            raise RuntimeError("hook bug — must not mask original")

        router._inject_failure_reward = _explode  # type: ignore[assignment]
        return router

    def test_sync_invoke_surfaces_original_exception_even_if_hook_explodes(
        self,
    ) -> None:
        router = self._router_with_broken_hook()
        with pytest.raises(RuntimeError, match="real failure"):
            router.invoke("hi")

    async def test_ainvoke_surfaces_original_exception_even_if_hook_explodes(
        self,
    ) -> None:
        router = self._router_with_broken_hook()
        with pytest.raises(RuntimeError, match="real failure"):
            await router.ainvoke("hi")


# ---------------------------------------------------------------------------
# Structured-output: failure_hook errors are warned (not silently swallowed)
# ---------------------------------------------------------------------------


class TestStructuredOutputFailureHookErrorsAreLogged:
    """``_StructuredOutputRouter._maybe_fire_failure_hook`` suppresses any
    exception raised by the hook so the original LLM failure still
    propagates. Previously the suppression was silent (`except Exception:
    pass`), losing telemetry whenever the hook itself had a bug. The
    suppression must remain (the hook is observability-only and must not
    mask the real failure), but the suppressed error has to be visible at
    warning level so a hook regression does not vanish into the void."""

    def test_warning_emitted_when_failure_hook_raises(self) -> None:
        from gigaevo.llm.models import _StructuredOutputRouter

        flaky = MagicMock()
        flaky.invoke = MagicMock(side_effect=RuntimeError("real failure"))

        def _explode_hook(_exc: BaseException, _name: str) -> None:
            raise RuntimeError("hook bug")

        router = _StructuredOutputRouter(
            [flaky],
            ["m"],
            [1.0],
            None,
            MagicMock(),
            failure_hook=_explode_hook,
        )

        with patch("gigaevo.llm.models.logger.warning") as mock_warning:
            with pytest.raises(RuntimeError, match="real failure"):
                router.invoke("hi")

        # The hook exception must have been logged at warning level so a
        # hook regression is visible in operator logs.
        assert mock_warning.called
        # The warning payload must reference the hook exception so the
        # operator can identify the broken hook from logs alone.
        call_args = mock_warning.call_args
        payload = repr(call_args)
        assert "hook bug" in payload or "RuntimeError" in payload


# ---------------------------------------------------------------------------
# Pre-_select exceptions: ledger invariant must hold
# ---------------------------------------------------------------------------


class TestPreSelectFailureLedgerInvariant:
    """If ``_select`` itself raises (e.g. a corrupted bandit state where
    ``self._bandit.select()`` blows up), the LLM call never happens and the
    try/except inside ``invoke`` never engages. The invariant the wiring
    promises is "pulls and rewards stay in step": if no pull was recorded
    (``record_pull`` is called *inside* ``_select``), no reward must be
    injected either. Verifying explicitly so a future refactor that moves
    ``record_pull`` outside ``_select`` doesn't quietly break the
    invariant."""

    def test_select_failure_before_record_pull_leaves_ledger_clean(self) -> None:
        models = _make_mock_models(["arm_a"])
        router = BanditModelRouter(
            models, [1.0], fitness_key="score", higher_is_better=True
        )
        router._langfuse = None

        # Force bandit.select to raise *before* record_pull has a chance.
        def _explode() -> str:
            raise RuntimeError("bandit corrupted")

        router._bandit.select = _explode  # type: ignore[assignment]

        with pytest.raises(RuntimeError, match="bandit corrupted"):
            router.invoke("hello")

        # No pull recorded, no reward injected — ledgers in step.
        stats = router.get_bandit_stats()
        assert stats["arm_a"]["total_pulls"] == 0
        assert stats["arm_a"]["window_size"] == 0

    async def test_aselect_failure_before_record_pull_leaves_ledger_clean(
        self,
    ) -> None:
        models = _make_mock_models(["arm_a"])
        router = BanditModelRouter(
            models, [1.0], fitness_key="score", higher_is_better=True
        )
        router._langfuse = None

        def _explode() -> str:
            raise RuntimeError("bandit corrupted")

        router._bandit.select = _explode  # type: ignore[assignment]

        with pytest.raises(RuntimeError, match="bandit corrupted"):
            await router.ainvoke("hello")

        stats = router.get_bandit_stats()
        assert stats["arm_a"]["total_pulls"] == 0
        assert stats["arm_a"]["window_size"] == 0
