"""Edge-case and boundary tests for elite selectors and utility functions.

Focus: double-negation in tournament _rank, Pareto rank on candidates not population,
       weighted_sample edge cases, dominates() corners, fitness normalization boundary.
"""

from __future__ import annotations

import random

import pytest

from gigaevo.evolution.strategies.elite_selectors import (
    FitnessProportionalEliteSelector,
    ParetoTournamentEliteSelector,
    RandomEliteSelector,
    ScalarTournamentEliteSelector,
)
from gigaevo.evolution.strategies.utils import (
    dominates,
    extract_fitness_values,
    weighted_sample_without_replacement,
)
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_prog(fitness: float, prog_id: str | None = None, **extra_metrics) -> Program:
    p = Program(code=f"x={fitness}", state=ProgramState.RUNNING)
    metrics = {"fitness": fitness, **extra_metrics}
    p.add_metrics(metrics)
    if prog_id:
        p._id = prog_id
    return p


def _make_multi_prog(m1: float, m2: float) -> Program:
    p = Program(code=f"x={m1},{m2}", state=ProgramState.RUNNING)
    p.add_metrics({"m1": m1, "m2": m2})
    return p


# ===================================================================
# 1. weighted_sample_without_replacement: zero-weight fallback boundary
# Target: utils.py line 54
# ===================================================================


class TestWeightedSampleZeroWeightFallback:
    def test_all_zero_after_first_selection(self):
        """weights=[1,0,0], k=3: after first pick, remaining all zero → fallback."""
        random.seed(42)
        items = ["a", "b", "c"]
        result = weighted_sample_without_replacement(items, [1.0, 0.0, 0.0], 3)
        assert len(result) == 3
        assert result[0] == "a"  # Only "a" has weight
        assert set(result) == {"a", "b", "c"}

    def test_exact_count_equals_remaining(self):
        """3 items, weights=[1,0,0], k=3: fallback needs exactly 2 from 2."""
        random.seed(99)
        items = [10, 20, 30]
        result = weighted_sample_without_replacement(items, [1.0, 0.0, 0.0], 3)
        assert len(result) == 3
        assert result[0] == 10

    def test_k_larger_than_items(self):
        """k > len(items) → clamped to len(items)."""
        items = ["x", "y"]
        result = weighted_sample_without_replacement(items, [1.0, 1.0], 5)
        assert len(result) == 2

    def test_single_item(self):
        result = weighted_sample_without_replacement(["only"], [1.0], 1)
        assert result == ["only"]

    def test_all_zero_weights(self):
        """All zero from the start → fallback immediately."""
        random.seed(0)
        items = [1, 2, 3]
        result = weighted_sample_without_replacement(items, [0.0, 0.0, 0.0], 2)
        assert len(result) == 2


# ===================================================================
# 2. weighted_sample: .index() with duplicate items
# Target: utils.py line 60
# ===================================================================


class TestWeightedSampleDuplicateItems:
    def test_duplicate_values_removed_correctly(self):
        """Items with same value: .index() returns first occurrence.
        After removing first, second occurrence remains."""
        random.seed(42)
        items = ["a", "a", "b"]
        weights = [10.0, 0.001, 0.001]
        result = weighted_sample_without_replacement(items, weights, 2)
        assert len(result) == 2
        # First pick is "a" (high weight). .index("a") returns 0.
        # After removing index 0, remaining = ["a", "b"]
        # Second pick from remaining. Result has 2 items.

    def test_identical_objects_by_reference(self):
        """Same object reference: .index() still finds the first one."""
        obj = {"key": "val"}
        items = [obj, obj, {"key": "other"}]
        random.seed(42)
        result = weighted_sample_without_replacement(items, [10.0, 0.001, 0.001], 2)
        assert len(result) == 2


# ===================================================================
# 3. ScalarTournament: double-negation in _rank
# Target: elite_selectors.py line 267: -self._rank(p)
#   _rank calls extract_fitness_values which negates if higher_is_better=False.
#   Then line 267 negates AGAIN. So for higher_is_better=False:
#     raw=10 → extract negates → -10 → line 267 negates → 10
#     raw=5 → extract negates → -5 → line 267 negates → 5
#   sort by (10, 5) ascending → 5 wins = program with raw fitness 5
#   With higher_is_better=False, LOWER is better, so 5 winning is CORRECT.
# ===================================================================


class TestScalarTournamentDoubleNegation:
    def test_higher_is_better_true_picks_highest(self):
        """higher_is_better=True: program with fitness=100 should win."""
        random.seed(42)
        progs = [_make_prog(10.0), _make_prog(100.0), _make_prog(50.0)]
        sel = ScalarTournamentEliteSelector(
            fitness_key="fitness",
            fitness_key_higher_is_better=True,
            tournament_size=3,  # All compete
        )
        result = sel(progs, 1)
        assert len(result) == 1
        assert result[0].metrics["fitness"] == 100.0

    def test_higher_is_better_false_picks_lowest(self):
        """higher_is_better=False: double negation → lowest raw fitness wins."""
        random.seed(42)
        progs = [_make_prog(10.0), _make_prog(100.0), _make_prog(50.0)]
        sel = ScalarTournamentEliteSelector(
            fitness_key="fitness",
            fitness_key_higher_is_better=False,
            tournament_size=3,
        )
        result = sel(progs, 1)
        assert len(result) == 1
        assert result[0].metrics["fitness"] == 10.0

    def test_tournament_size_1_picks_random(self):
        """tournament_size=1: winner is the only candidate (random pick)."""
        random.seed(42)
        progs = [_make_prog(float(i)) for i in range(10)]
        sel = ScalarTournamentEliteSelector(
            fitness_key="fitness",
            fitness_key_higher_is_better=True,
            tournament_size=1,
        )
        result = sel(progs, 5)
        assert len(result) == 5
        # With tournament_size=1, each pick is random (any program)

    def test_tournament_size_equals_population_always_picks_best(self):
        """tournament_size=N: always picks the best."""
        random.seed(42)
        progs = [_make_prog(float(i)) for i in range(5)]
        sel = ScalarTournamentEliteSelector(
            fitness_key="fitness",
            fitness_key_higher_is_better=True,
            tournament_size=5,
        )
        result = sel(progs, 1)
        assert result[0].metrics["fitness"] == 4.0  # Highest

    def test_remove_uses_identity_not_equality(self):
        """remaining_programs.remove(winner) uses == which for Program
        compares id. Programs with same fitness are distinct objects."""
        random.seed(42)
        # Three programs with identical fitness
        progs = [_make_prog(50.0) for _ in range(3)]
        sel = ScalarTournamentEliteSelector(
            fitness_key="fitness",
            fitness_key_higher_is_better=True,
            tournament_size=1,
        )
        result = sel(progs, 3)
        # Should get all 3 (each removed by identity)
        assert len(result) == 3


# ===================================================================
# 4. ParetoTournament: rank computed on candidates, not full population
# Target: elite_selectors.py line 325
# ===================================================================


class TestParetoTournamentRankScope:
    def test_dominated_in_full_but_not_in_tournament_can_win(self):
        """A program dominated in full population can win a tournament
        that doesn't include its dominator."""
        random.seed(42)
        # p0 dominates p1 on both metrics. But if p0 is not in the tournament,
        # p1 has rank 0 and can win.
        p0 = _make_multi_prog(10.0, 10.0)  # Dominates p1
        p1 = _make_multi_prog(5.0, 5.0)  # Dominated by p0
        p2 = _make_multi_prog(3.0, 8.0)  # Not dominated by p1

        sel = ParetoTournamentEliteSelector(
            fitness_keys=["m1", "m2"],
            tournament_size=2,
        )
        # With seed, tournament might not include p0
        # Run multiple times to verify p1 CAN win
        won = False
        for seed in range(100):
            random.seed(seed)
            result = sel([p0, p1, p2], 1)
            if result[0] is p1:
                won = True
                break
        assert won, "p1 should be able to win at least once in 100 seeds"

    def test_pareto_requires_at_least_two_keys(self):
        with pytest.raises(ValueError, match="at least two"):
            ParetoTournamentEliteSelector(fitness_keys=["m1"])


# ===================================================================
# 5. FitnessProportional: normalization boundary at exactly 1e-10
# Target: elite_selectors.py line 91
# ===================================================================


class TestFitnessProportionalNormalizationBoundary:
    def test_range_below_threshold_uniform(self):
        """Range < 1e-10 → uniform weights."""
        sel = FitnessProportionalEliteSelector(fitness_key="fitness")
        # All identical → range = 0 < 1e-10 → uniform
        weights = sel._compute_weights([5.0, 5.0, 5.0])
        assert len(weights) == 3
        assert abs(weights[0] - weights[1]) < 1e-10
        assert abs(weights[1] - weights[2]) < 1e-10

    def test_range_at_threshold_not_uniform(self):
        """Range = 1e-10 → NOT < 1e-10 → normalization proceeds."""
        sel = FitnessProportionalEliteSelector(fitness_key="fitness")
        weights = sel._compute_weights([1.0, 1.0 + 1e-10])
        # Should NOT be uniform (normalization produces [0.0, 1.0])
        assert len(weights) == 2
        # The higher value should have higher weight
        assert weights[1] > weights[0]

    def test_single_program_returns_all(self):
        """len(programs) <= total → return all, skip selection."""
        sel = FitnessProportionalEliteSelector(fitness_key="fitness")
        progs = [_make_prog(5.0)]
        result = sel(progs, 5)
        assert result == progs

    def test_non_finite_falls_back_to_uniform(self):
        """inf/nan → uniform fallback."""
        random.seed(42)
        sel = FitnessProportionalEliteSelector(fitness_key="fitness")
        progs = [_make_prog(1.0), _make_prog(float("inf")), _make_prog(3.0)]
        result = sel(progs, 2)
        assert len(result) == 2

    def test_negative_fitness_values(self):
        """Negative fitness should work (normalization handles it)."""
        random.seed(42)
        sel = FitnessProportionalEliteSelector(fitness_key="fitness")
        progs = [_make_prog(-10.0), _make_prog(-5.0), _make_prog(0.0), _make_prog(5.0)]
        result = sel(progs, 2)
        assert len(result) == 2


# ===================================================================
# 6. dominates() corner cases
# Target: utils.py lines 86-90
# ===================================================================


class TestDominatesCorners:
    def test_equal_vectors_not_dominate(self):
        """Equal vectors: all >= True, any > False → False."""
        assert dominates([1, 2, 3], [1, 2, 3]) is False

    def test_strictly_better_dominates(self):
        assert dominates([2, 3, 4], [1, 2, 3]) is True

    def test_strictly_worse_does_not_dominate(self):
        assert dominates([1, 2, 3], [2, 3, 4]) is False

    def test_mixed_does_not_dominate(self):
        """Better in one, worse in another → no domination."""
        assert dominates([3, 1], [1, 3]) is False

    def test_better_in_one_equal_rest(self):
        """Better in one, equal in rest → dominates."""
        assert dominates([2, 2, 3], [2, 2, 2]) is True

    def test_empty_vectors(self):
        """Empty: all([]) = True, any([]) = False → False."""
        assert dominates([], []) is False

    def test_single_dimension_better(self):
        assert dominates([5], [3]) is True

    def test_single_dimension_equal(self):
        assert dominates([3], [3]) is False

    def test_asymmetry(self):
        """If a dominates b, then b does NOT dominate a."""
        a, b = [3, 4], [1, 2]
        assert dominates(a, b) is True
        assert dominates(b, a) is False


# ===================================================================
# 7. extract_fitness_values: assertion on key mismatch
# Target: utils.py line 72
# ===================================================================


class TestExtractFitnessValuesMismatch:
    def test_mismatched_keys_raises_assertion(self):
        prog = _make_prog(5.0)
        with pytest.raises(AssertionError):
            extract_fitness_values(
                prog,
                fitness_keys=["fitness"],
                fitness_key_higher_is_better={"fitness": True, "extra": True},
            )

    def test_missing_metric_raises_key_error(self):
        prog = _make_prog(5.0)
        with pytest.raises(KeyError, match="Missing fitness key"):
            extract_fitness_values(
                prog,
                fitness_keys=["nonexistent"],
                fitness_key_higher_is_better={"nonexistent": True},
            )

    def test_higher_is_better_false_negates(self):
        prog = _make_prog(10.0)
        values = extract_fitness_values(
            prog,
            fitness_keys=["fitness"],
            fitness_key_higher_is_better={"fitness": False},
        )
        assert values == [-10.0]


# ===================================================================
# 8. RandomEliteSelector
# ===================================================================


class TestRandomEliteSelectorEdges:
    def test_fewer_than_requested(self):
        progs = [_make_prog(1.0), _make_prog(2.0)]
        sel = RandomEliteSelector()
        result = sel(progs, 5)
        assert result == progs  # Returns all

    def test_exact_count(self):
        progs = [_make_prog(float(i)) for i in range(3)]
        sel = RandomEliteSelector()
        result = sel(progs, 3)
        assert result == progs

    def test_empty_population(self):
        sel = RandomEliteSelector()
        result = sel([], 5)
        assert result == []

    def test_sample_without_replacement(self):
        random.seed(42)
        progs = [_make_prog(float(i)) for i in range(10)]
        sel = RandomEliteSelector()
        result = sel(progs, 5)
        assert len(result) == 5
        assert len(set(id(p) for p in result)) == 5  # All unique
