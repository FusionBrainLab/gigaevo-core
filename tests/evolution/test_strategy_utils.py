"""Tests for gigaevo/evolution/strategies/utils.py

Covers:
  - weighted_sample_without_replacement: basic sampling, k > len, zero-weight fallback,
    single-item selection, empty population
  - extract_fitness_values: normal extraction, missing key, higher-is-better negation
  - dominates: strict domination, equal vectors, partial domination, single-objective
"""

from __future__ import annotations

import random

import pytest

from gigaevo.evolution.strategies.utils import (
    dominates,
    extract_fitness_values,
    weighted_sample_without_replacement,
)
from gigaevo.programs.program import Program

# ═══════════════════════════════════════════════════════════════════════════
# weighted_sample_without_replacement
# ═══════════════════════════════════════════════════════════════════════════


class TestWeightedSampleWithoutReplacement:
    def test_basic_sampling_returns_k_items(self) -> None:
        items = ["a", "b", "c", "d", "e"]
        weights = [1.0, 2.0, 3.0, 4.0, 5.0]
        random.seed(42)
        result = weighted_sample_without_replacement(items, weights, k=3)
        assert len(result) == 3
        assert len(set(result)) == 3  # no duplicates
        assert all(item in items for item in result)

    def test_k_larger_than_population_clamps(self) -> None:
        items = ["x", "y"]
        weights = [1.0, 1.0]
        result = weighted_sample_without_replacement(items, weights, k=10)
        assert len(result) == 2
        assert set(result) == {"x", "y"}

    def test_k_zero_returns_empty(self) -> None:
        items = ["a", "b", "c"]
        weights = [1.0, 2.0, 3.0]
        result = weighted_sample_without_replacement(items, weights, k=0)
        assert result == []

    def test_k_equals_population_returns_all(self) -> None:
        items = [1, 2, 3]
        weights = [1.0, 1.0, 1.0]
        result = weighted_sample_without_replacement(items, weights, k=3)
        assert len(result) == 3
        assert set(result) == {1, 2, 3}

    def test_single_item_selection(self) -> None:
        items = ["only"]
        weights = [5.0]
        result = weighted_sample_without_replacement(items, weights, k=1)
        assert result == ["only"]

    def test_empty_population_returns_empty(self) -> None:
        result = weighted_sample_without_replacement([], [], k=3)
        assert result == []

    def test_zero_weight_fallback_to_uniform(self) -> None:
        """When all weights are zero, sampling falls back to uniform random."""
        items = ["a", "b", "c", "d"]
        weights = [0.0, 0.0, 0.0, 0.0]
        random.seed(0)
        result = weighted_sample_without_replacement(items, weights, k=2)
        assert len(result) == 2
        assert all(item in items for item in result)
        # No duplicates even with uniform fallback
        assert len(set(result)) == 2

    def test_some_zero_weights_partial_fallback(self) -> None:
        """First pick uses non-zero weight; remaining become zero → uniform fallback."""
        items = ["a", "b", "c"]
        # Only "a" has positive weight — after picking it, remaining weights are [0, 0]
        weights = [10.0, 0.0, 0.0]
        random.seed(42)
        result = weighted_sample_without_replacement(items, weights, k=3)
        assert len(result) == 3
        assert result[0] == "a"  # "a" dominates first pick
        assert set(result) == {"a", "b", "c"}

    def test_high_weight_item_selected_more_often(self) -> None:
        """Statistical test: a heavily weighted item should appear in most samples."""
        items = ["rare", "common"]
        weights = [0.001, 1000.0]
        count_common_first = 0
        for seed in range(100):
            random.seed(seed)
            result = weighted_sample_without_replacement(items, weights, k=1)
            if result[0] == "common":
                count_common_first += 1
        # "common" should be selected first in nearly all trials
        assert count_common_first > 95

    def test_no_duplicates_in_result(self) -> None:
        """Ensure without-replacement guarantee: no item appears twice."""
        items = list(range(20))
        weights = [1.0] * 20
        random.seed(123)
        result = weighted_sample_without_replacement(items, weights, k=15)
        assert len(result) == len(set(result))


# ═══════════════════════════════════════════════════════════════════════════
# extract_fitness_values
# ═══════════════════════════════════════════════════════════════════════════


class TestExtractFitnessValues:
    def _make_program(self, metrics: dict) -> Program:
        p = Program(code="pass")
        p.metrics = metrics
        return p

    def test_basic_extraction(self) -> None:
        p = self._make_program({"score": 0.9, "speed": 1.5})
        keys = ["score", "speed"]
        hib = {"score": True, "speed": True}
        result = extract_fitness_values(p, keys, hib)
        assert result == [0.9, 1.5]

    def test_higher_is_better_false_negates(self) -> None:
        """When higher_is_better is False, value is negated."""
        p = self._make_program({"loss": 0.3, "accuracy": 0.95})
        keys = ["loss", "accuracy"]
        hib = {"loss": False, "accuracy": True}
        result = extract_fitness_values(p, keys, hib)
        assert result[0] == pytest.approx(-0.3)
        assert result[1] == pytest.approx(0.95)

    def test_missing_key_raises(self) -> None:
        p = self._make_program({"score": 0.5})
        keys = ["score", "missing"]
        hib = {"score": True, "missing": True}
        with pytest.raises(KeyError, match="missing"):
            extract_fitness_values(p, keys, hib)

    def test_mismatched_keys_raises(self) -> None:
        """fitness_keys and fitness_key_higher_is_better must have same keys."""
        p = self._make_program({"a": 1.0, "b": 2.0})
        keys = ["a", "b"]
        hib = {"a": True, "c": True}  # 'c' doesn't match 'b'
        with pytest.raises(AssertionError):
            extract_fitness_values(p, keys, hib)

    def test_single_objective(self) -> None:
        p = self._make_program({"fitness": 42.0})
        result = extract_fitness_values(p, ["fitness"], {"fitness": True})
        assert result == [42.0]


# ═══════════════════════════════════════════════════════════════════════════
# dominates
# ═══════════════════════════════════════════════════════════════════════════


class TestDominates:
    def test_strict_domination(self) -> None:
        """p strictly greater in all objectives → dominates."""
        assert dominates([3, 4, 5], [1, 2, 3]) is True

    def test_equal_vectors_no_domination(self) -> None:
        """Equal vectors: p >= q in all, but not > in any → does NOT dominate."""
        assert dominates([1, 2, 3], [1, 2, 3]) is False

    def test_partial_domination_insufficient(self) -> None:
        """p better in some, worse in others → does NOT dominate."""
        assert dominates([3, 1], [1, 3]) is False

    def test_one_greater_rest_equal(self) -> None:
        """p >= in all and > in at least one → dominates."""
        assert dominates([2, 2, 3], [2, 2, 2]) is True

    def test_one_less_fails(self) -> None:
        """p < q in one objective → does NOT dominate even if better in others."""
        assert dominates([10, 0, 10], [1, 1, 1]) is False

    def test_single_objective_greater(self) -> None:
        assert dominates([5], [3]) is True

    def test_single_objective_equal(self) -> None:
        assert dominates([5], [5]) is False

    def test_single_objective_less(self) -> None:
        assert dominates([3], [5]) is False

    def test_empty_vectors(self) -> None:
        """Empty vectors: all() on empty is True, any() is False → does NOT dominate."""
        assert dominates([], []) is False

    def test_negative_values(self) -> None:
        assert dominates([-1, 0], [-2, -1]) is True
        assert dominates([-2, 0], [-1, -1]) is False

    def test_float_values(self) -> None:
        assert dominates([1.5, 2.5], [1.0, 2.0]) is True
        assert dominates([1.0, 2.5], [1.5, 2.0]) is False
