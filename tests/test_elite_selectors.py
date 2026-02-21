"""Tests for WeightedEliteSelector and FitnessProportionalEliteSelector temperature."""

from collections import Counter

import pytest

from gigaevo.evolution.strategies.elite_selectors import (
    FitnessProportionalEliteSelector,
    WeightedEliteSelector,
)
from gigaevo.programs.program import Lineage, Program


def _make_program(
    fitness: float,
    fitness_key: str = "score",
    child_count: int = 0,
) -> Program:
    p = Program(code="def solve(): return 1", metrics={fitness_key: fitness})
    p.lineage = Lineage(children=[f"child_{i}" for i in range(child_count)])
    return p


# ---------------------------------------------------------------------------
# WeightedEliteSelector
# ---------------------------------------------------------------------------


class TestWeightedEliteSelector:
    def test_returns_all_when_fewer_than_total(self):
        sel = WeightedEliteSelector(fitness_key="score")
        progs = [_make_program(i) for i in range(3)]
        result = sel(progs, total=5)
        assert result == progs

    def test_correct_count_returned(self):
        sel = WeightedEliteSelector(fitness_key="score")
        progs = [_make_program(float(i)) for i in range(10)]
        result = sel(progs, total=4)
        assert len(result) == 4

    def test_no_duplicates(self):
        sel = WeightedEliteSelector(fitness_key="score")
        progs = [_make_program(float(i)) for i in range(10)]
        result = sel(progs, total=5)
        assert len(set(id(p) for p in result)) == 5

    def test_higher_fitness_preferred(self):
        """Statistical test: above-median programs should be selected much
        more often than below-median ones. With lambda=10, the sigmoid
        saturates so all above-median programs share weight ~equally."""
        sel = WeightedEliteSelector(fitness_key="score", lambda_=10.0)
        progs = [_make_program(float(i)) for i in range(10)]

        counts: Counter = Counter()
        n_trials = 1000
        for _ in range(n_trials):
            result = sel(progs, total=1)
            counts[id(result[0])] += 1

        above_median = sum(counts[id(p)] for p in progs[5:])
        below_median = sum(counts[id(p)] for p in progs[:5])
        assert above_median > below_median * 5

    def test_children_penalty_reduces_selection(self):
        """A program with many children should be selected less often than
        an equally-fit program with no children."""
        sel = WeightedEliteSelector(fitness_key="score", lambda_=10.0)
        no_kids = _make_program(10.0, child_count=0)
        many_kids = _make_program(10.0, child_count=50)
        filler = _make_program(0.0)

        counts_no_kids = 0
        counts_many_kids = 0
        n_trials = 1000
        for _ in range(n_trials):
            result = sel([no_kids, many_kids, filler], total=1)
            if result[0] is no_kids:
                counts_no_kids += 1
            elif result[0] is many_kids:
                counts_many_kids += 1

        assert counts_no_kids > counts_many_kids

    def test_lambda_zero_makes_fitness_uniform(self):
        """With lambda_=0, sigmoid always outputs 0.5 regardless of fitness.
        Selection is then driven only by children counts (all zero here),
        so it should be approximately uniform."""
        sel = WeightedEliteSelector(fitness_key="score", lambda_=0.0)
        progs = [_make_program(float(i) * 100) for i in range(5)]

        counts: Counter = Counter()
        n_trials = 2000
        for _ in range(n_trials):
            result = sel(progs, total=1)
            counts[id(result[0])] += 1

        # Each should get ~20% of selections; check within [10%, 30%]
        for p in progs:
            frac = counts[id(p)] / n_trials
            assert 0.10 < frac < 0.30, f"Expected ~uniform, got {frac:.2%}"

    def test_missing_fitness_key_raises(self):
        sel = WeightedEliteSelector(fitness_key="nonexistent")
        progs = [_make_program(1.0, fitness_key="score") for _ in range(3)]
        with pytest.raises(ValueError, match="Missing fitness key"):
            sel(progs, total=1)

    def test_higher_is_better_false(self):
        """When higher_is_better=False, below-median (low score) programs
        should be strongly preferred."""
        sel = WeightedEliteSelector(
            fitness_key="score", fitness_key_higher_is_better=False, lambda_=10.0
        )
        progs = [_make_program(float(i)) for i in range(10)]

        counts: Counter = Counter()
        n_trials = 1000
        for _ in range(n_trials):
            result = sel(progs, total=1)
            counts[id(result[0])] += 1

        # Low scores (0-4) are preferred when higher_is_better=False
        low_scores = sum(counts[id(p)] for p in progs[:5])
        high_scores = sum(counts[id(p)] for p in progs[5:])
        assert low_scores > high_scores * 5

    def test_all_identical_fitness(self):
        """When all fitness values are identical, sigmoid outputs 0.5 for all,
        so selection should be roughly uniform (given no children)."""
        sel = WeightedEliteSelector(fitness_key="score", lambda_=10.0)
        progs = [_make_program(5.0) for _ in range(5)]
        result = sel(progs, total=3)
        assert len(result) == 3
        assert len(set(id(p) for p in result)) == 3


# ---------------------------------------------------------------------------
# FitnessProportionalEliteSelector — temperature parameter
# ---------------------------------------------------------------------------


class TestFitnessProportionalTemperature:
    def test_no_temperature_preserves_linear_behaviour(self):
        """Default (temperature=None) should favour higher fitness linearly."""
        sel = FitnessProportionalEliteSelector(fitness_key="score")
        progs = [_make_program(1.0), _make_program(100.0)]

        counts: Counter = Counter()
        n_trials = 1000
        for _ in range(n_trials):
            result = sel(progs, total=1)
            counts[id(result[0])] += 1

        # The 100x-higher fitness program should dominate
        assert counts[id(progs[1])] > n_trials * 0.9

    def test_high_temperature_gives_near_uniform(self):
        """Very high temperature should flatten differences → near uniform."""
        sel = FitnessProportionalEliteSelector(fitness_key="score", temperature=1e6)
        progs = [_make_program(float(i) * 100) for i in range(5)]

        counts: Counter = Counter()
        n_trials = 2000
        for _ in range(n_trials):
            result = sel(progs, total=1)
            counts[id(result[0])] += 1

        for p in progs:
            frac = counts[id(p)] / n_trials
            assert 0.10 < frac < 0.30, f"Expected ~uniform, got {frac:.2%}"

    def test_low_temperature_gives_greedy(self):
        """Very low temperature should almost always pick the best program."""
        sel = FitnessProportionalEliteSelector(fitness_key="score", temperature=0.01)
        progs = [_make_program(float(i)) for i in range(10)]
        best = progs[-1]

        counts: Counter = Counter()
        n_trials = 500
        for _ in range(n_trials):
            result = sel(progs, total=1)
            counts[id(result[0])] += 1

        assert counts[id(best)] > n_trials * 0.95

    def test_moderate_temperature_still_prefers_higher(self):
        """Moderate temperature should still favour higher fitness but less
        aggressively than linear proportional."""
        sel = FitnessProportionalEliteSelector(fitness_key="score", temperature=1.0)
        progs = [_make_program(float(i)) for i in range(10)]
        best = progs[-1]

        counts: Counter = Counter()
        n_trials = 1000
        for _ in range(n_trials):
            result = sel(progs, total=1)
            counts[id(result[0])] += 1

        assert counts[id(best)] > counts[id(progs[0])]

    def test_temperature_higher_is_better_false(self):
        """Temperature + higher_is_better=False should favour low scores."""
        sel = FitnessProportionalEliteSelector(
            fitness_key="score",
            fitness_key_higher_is_better=False,
            temperature=0.01,
        )
        progs = [_make_program(float(i)) for i in range(10)]
        lowest = progs[0]  # best when higher_is_better=False

        counts: Counter = Counter()
        n_trials = 500
        for _ in range(n_trials):
            result = sel(progs, total=1)
            counts[id(result[0])] += 1

        assert counts[id(lowest)] > n_trials * 0.95

    def test_temperature_no_duplicates(self):
        sel = FitnessProportionalEliteSelector(fitness_key="score", temperature=1.0)
        progs = [_make_program(float(i)) for i in range(10)]
        result = sel(progs, total=5)
        assert len(set(id(p) for p in result)) == 5

    def test_temperature_correct_count(self):
        sel = FitnessProportionalEliteSelector(fitness_key="score", temperature=1.0)
        progs = [_make_program(float(i)) for i in range(10)]
        result = sel(progs, total=4)
        assert len(result) == 4

    def test_temperature_returns_all_when_fewer(self):
        sel = FitnessProportionalEliteSelector(fitness_key="score", temperature=1.0)
        progs = [_make_program(float(i)) for i in range(3)]
        result = sel(progs, total=5)
        assert result == progs
