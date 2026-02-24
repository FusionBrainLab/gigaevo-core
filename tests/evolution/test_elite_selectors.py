"""Tests for elite selectors including lower-is-better fitness."""

from collections import Counter

import pytest

from gigaevo.evolution.strategies.elite_selectors import (
    FitnessProportionalEliteSelector,
    ParetoTournamentEliteSelector,
    ScalarTournamentEliteSelector,
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
    def test_auto_temperature_favours_higher_fitness(self):
        """Default (temperature=None) auto-computes temperature from stdev,
        giving moderate preference to higher fitness without the extreme
        distortion of raw linear weighting."""
        sel = FitnessProportionalEliteSelector(fitness_key="score")
        progs = [_make_program(1.0), _make_program(100.0)]

        counts: Counter = Counter()
        n_trials = 1000
        for _ in range(n_trials):
            result = sel(progs, total=1)
            counts[id(result[0])] += 1

        # Higher fitness program should be selected more often
        assert counts[id(progs[1])] > counts[id(progs[0])]

    def test_high_temperature_gives_near_uniform(self):
        """Very high temperature in normalized [0,1] space should flatten
        differences → near uniform."""
        sel = FitnessProportionalEliteSelector(fitness_key="score", temperature=10.0)
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
        """Very low temperature in normalized [0,1] space should almost
        always pick the best program."""
        sel = FitnessProportionalEliteSelector(fitness_key="score", temperature=0.001)
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
        sel = FitnessProportionalEliteSelector(fitness_key="score", temperature=0.5)
        progs = [_make_program(float(i)) for i in range(10)]
        best = progs[-1]

        counts: Counter = Counter()
        n_trials = 1000
        for _ in range(n_trials):
            result = sel(progs, total=1)
            counts[id(result[0])] += 1

        assert counts[id(best)] > counts[id(progs[0])]

    def test_temperature_higher_is_better_false(self):
        """Low temperature + higher_is_better=False should favour low scores."""
        sel = FitnessProportionalEliteSelector(
            fitness_key="score",
            fitness_key_higher_is_better=False,
            temperature=0.001,
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
        sel = FitnessProportionalEliteSelector(fitness_key="score", temperature=0.5)
        progs = [_make_program(float(i)) for i in range(10)]
        result = sel(progs, total=5)
        assert len(set(id(p) for p in result)) == 5

    def test_temperature_correct_count(self):
        sel = FitnessProportionalEliteSelector(fitness_key="score", temperature=0.5)
        progs = [_make_program(float(i)) for i in range(10)]
        result = sel(progs, total=4)
        assert len(result) == 4

    def test_temperature_returns_all_when_fewer(self):
        sel = FitnessProportionalEliteSelector(fitness_key="score", temperature=0.5)
        progs = [_make_program(float(i)) for i in range(3)]
        result = sel(progs, total=5)
        assert result == progs

    def test_auto_temperature_higher_is_better_false(self):
        """Auto temperature + higher_is_better=False: low scores preferred."""
        sel = FitnessProportionalEliteSelector(
            fitness_key="score", fitness_key_higher_is_better=False
        )
        progs = [_make_program(1.0), _make_program(100.0)]

        counts: Counter = Counter()
        n_trials = 1000
        for _ in range(n_trials):
            result = sel(progs, total=1)
            counts[id(result[0])] += 1

        # Lower fitness program should be selected more often
        assert counts[id(progs[0])] > counts[id(progs[1])]

    def test_moderate_temperature_higher_is_better_false(self):
        """Moderate temperature + higher_is_better=False: low scores still preferred."""
        sel = FitnessProportionalEliteSelector(
            fitness_key="score",
            fitness_key_higher_is_better=False,
            temperature=0.5,
        )
        progs = [_make_program(float(i)) for i in range(10)]
        lowest = progs[0]

        counts: Counter = Counter()
        n_trials = 1000
        for _ in range(n_trials):
            result = sel(progs, total=1)
            counts[id(result[0])] += 1

        assert counts[id(lowest)] > counts[id(progs[-1])]

    def test_auto_temperature_converged_population_not_greedy(self):
        """When fitnesses are very close, auto-temperature must NOT collapse
        to greedy selection — all programs should have reasonable probability."""
        sel = FitnessProportionalEliteSelector(fitness_key="score")
        progs = [_make_program(0.035990 + i * 0.000001) for i in range(5)]

        counts: Counter = Counter()
        n_trials = 2000
        for _ in range(n_trials):
            result = sel(progs, total=1)
            counts[id(result[0])] += 1

        # No single program should dominate with >60% selection probability.
        # (Before the fix, the best program got ~100% due to greedy collapse;
        # after normalization, the best gets ~49% weight.)
        for p in progs:
            frac = counts[id(p)] / n_trials
            assert frac < 0.60, (
                f"Greedy collapse: one program got {frac:.1%} in converged population"
            )

    def test_auto_temperature_identical_fitnesses_uniform(self):
        """All-identical fitnesses must produce uniform selection."""
        sel = FitnessProportionalEliteSelector(fitness_key="score")
        progs = [_make_program(5.0) for _ in range(4)]

        counts: Counter = Counter()
        n_trials = 2000
        for _ in range(n_trials):
            result = sel(progs, total=1)
            counts[id(result[0])] += 1

        for p in progs:
            frac = counts[id(p)] / n_trials
            assert 0.15 < frac < 0.35, f"Expected ~uniform, got {frac:.2%}"


# ---------------------------------------------------------------------------
# ScalarTournamentEliteSelector — lower-is-better
# ---------------------------------------------------------------------------


class TestScalarTournamentEliteSelector:
    def test_higher_is_better_true_selects_highest(self):
        sel = ScalarTournamentEliteSelector(
            fitness_key="score", fitness_key_higher_is_better=True, tournament_size=3
        )
        progs = [_make_program(float(i)) for i in range(10)]
        best = progs[-1]

        counts: Counter = Counter()
        n_trials = 500
        for _ in range(n_trials):
            result = sel(progs, total=1)
            counts[id(result[0])] += 1

        # Highest-fitness program should be selected most often
        assert counts[id(best)] == max(counts.values())

    def test_higher_is_better_false_selects_lowest(self):
        """When higher_is_better=False, the lowest-fitness program should win
        tournaments most often."""
        sel = ScalarTournamentEliteSelector(
            fitness_key="score", fitness_key_higher_is_better=False, tournament_size=3
        )
        progs = [_make_program(float(i)) for i in range(10)]
        lowest = progs[0]

        counts: Counter = Counter()
        n_trials = 500
        for _ in range(n_trials):
            result = sel(progs, total=1)
            counts[id(result[0])] += 1

        # Lowest-fitness program should be selected most often
        assert counts[id(lowest)] == max(counts.values())

    def test_higher_is_better_false_statistical(self):
        """Lower-fitness programs should collectively dominate selection."""
        sel = ScalarTournamentEliteSelector(
            fitness_key="score", fitness_key_higher_is_better=False, tournament_size=3
        )
        progs = [_make_program(float(i)) for i in range(10)]

        counts: Counter = Counter()
        n_trials = 1000
        for _ in range(n_trials):
            result = sel(progs, total=1)
            counts[id(result[0])] += 1

        low_scores = sum(counts[id(p)] for p in progs[:5])
        high_scores = sum(counts[id(p)] for p in progs[5:])
        assert low_scores > high_scores * 2

    def test_correct_count_returned(self):
        sel = ScalarTournamentEliteSelector(fitness_key="score")
        progs = [_make_program(float(i)) for i in range(10)]
        result = sel(progs, total=4)
        assert len(result) == 4

    def test_no_duplicates(self):
        sel = ScalarTournamentEliteSelector(fitness_key="score")
        progs = [_make_program(float(i)) for i in range(10)]
        result = sel(progs, total=5)
        assert len(set(id(p) for p in result)) == 5

    def test_returns_all_when_fewer_than_total(self):
        sel = ScalarTournamentEliteSelector(fitness_key="score")
        progs = [_make_program(float(i)) for i in range(3)]
        result = sel(progs, total=5)
        assert result == progs


# ---------------------------------------------------------------------------
# ParetoTournamentEliteSelector — lower-is-better
# ---------------------------------------------------------------------------


class TestParetoTournamentEliteSelector:
    def test_higher_is_better_true_dominance(self):
        """Program dominating on both keys should be selected most often."""
        sel = ParetoTournamentEliteSelector(
            fitness_keys=["a", "b"],
            fitness_key_higher_is_better={"a": True, "b": True},
            tournament_size=3,
        )
        dominant = Program(code="x=1", metrics={"a": 10.0, "b": 10.0})
        weak = Program(code="x=2", metrics={"a": 1.0, "b": 1.0})
        mid = Program(code="x=3", metrics={"a": 5.0, "b": 5.0})
        progs = [weak, mid, dominant]

        counts: Counter = Counter()
        n_trials = 500
        for _ in range(n_trials):
            result = sel(progs, total=1)
            counts[id(result[0])] += 1

        assert counts[id(dominant)] > counts[id(weak)]

    def test_higher_is_better_false_dominance(self):
        """When both keys have higher_is_better=False, the program with the
        lowest values on both dimensions should dominate and be selected most."""
        sel = ParetoTournamentEliteSelector(
            fitness_keys=["cost", "error"],
            fitness_key_higher_is_better={"cost": False, "error": False},
            tournament_size=3,
        )
        best = Program(code="x=1", metrics={"cost": 1.0, "error": 1.0})
        worst = Program(code="x=2", metrics={"cost": 10.0, "error": 10.0})
        mid = Program(code="x=3", metrics={"cost": 5.0, "error": 5.0})
        progs = [worst, mid, best]

        counts: Counter = Counter()
        n_trials = 500
        for _ in range(n_trials):
            result = sel(progs, total=1)
            counts[id(result[0])] += 1

        assert counts[id(best)] > counts[id(worst)]

    def test_mixed_higher_is_better(self):
        """One key higher-is-better, one lower-is-better. The program that is
        high on key_a and low on key_b should dominate."""
        sel = ParetoTournamentEliteSelector(
            fitness_keys=["accuracy", "latency"],
            fitness_key_higher_is_better={"accuracy": True, "latency": False},
            tournament_size=3,
        )
        # Best: high accuracy, low latency
        best = Program(code="x=1", metrics={"accuracy": 0.99, "latency": 10.0})
        # Worst: low accuracy, high latency
        worst = Program(code="x=2", metrics={"accuracy": 0.50, "latency": 100.0})
        mid = Program(code="x=3", metrics={"accuracy": 0.80, "latency": 50.0})
        progs = [worst, mid, best]

        counts: Counter = Counter()
        n_trials = 500
        for _ in range(n_trials):
            result = sel(progs, total=1)
            counts[id(result[0])] += 1

        assert counts[id(best)] > counts[id(worst)]

    def test_correct_count_returned(self):
        sel = ParetoTournamentEliteSelector(fitness_keys=["a", "b"])
        progs = [
            Program(code=f"x={i}", metrics={"a": float(i), "b": float(10 - i)})
            for i in range(10)
        ]
        result = sel(progs, total=4)
        assert len(result) == 4

    def test_no_duplicates(self):
        sel = ParetoTournamentEliteSelector(fitness_keys=["a", "b"])
        progs = [
            Program(code=f"x={i}", metrics={"a": float(i), "b": float(10 - i)})
            for i in range(10)
        ]
        result = sel(progs, total=5)
        assert len(set(id(p) for p in result)) == 5
