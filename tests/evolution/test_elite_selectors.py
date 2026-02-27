"""Tests for elite selectors including lower-is-better fitness."""

from collections import Counter
import random

import numpy as np
import pytest

from gigaevo.evolution.strategies.elite_selectors import (
    FitnessProportionalEliteSelector,
    ParetoTournamentEliteSelector,
    RandomEliteSelector,
    ScalarTournamentEliteSelector,
    WeightedEliteSelector,
)
from gigaevo.evolution.strategies.selectors import ParetoFrontSelector
from gigaevo.evolution.strategies.utils import (
    dominates,
    weighted_sample_without_replacement,
)
from gigaevo.programs.program import Lineage, Program
from gigaevo.programs.program_state import ProgramState


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
        n_trials = 2000
        for _ in range(n_trials):
            result = sel(progs, total=1)
            counts[id(result[0])] += 1

        # Highest-fitness program should be selected much more than uniform (1/10)
        best_frac = counts[id(best)] / n_trials
        assert (
            best_frac > 0.15
        )  # uniform would be 0.10; tournament bias pushes well above

    def test_higher_is_better_false_selects_lowest(self):
        """When higher_is_better=False, the lowest-fitness program should win
        tournaments much more than uniform chance."""
        sel = ScalarTournamentEliteSelector(
            fitness_key="score", fitness_key_higher_is_better=False, tournament_size=3
        )
        progs = [_make_program(float(i)) for i in range(10)]
        lowest = progs[0]

        counts: Counter = Counter()
        n_trials = 2000
        for _ in range(n_trials):
            result = sel(progs, total=1)
            counts[id(result[0])] += 1

        # Lowest-fitness program should be selected much more than uniform (1/10)
        lowest_frac = counts[id(lowest)] / n_trials
        assert lowest_frac > 0.15

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


# ---------------------------------------------------------------------------
# Audit Finding 1: ParetoFrontSelector — missing reverse direction
# ---------------------------------------------------------------------------


class TestParetoFrontSelectorReverseDirection:
    """Verify that a strictly worse new point is rejected by ParetoFrontSelector."""

    def test_worse_point_not_added_to_front(self):
        """When new is strictly worse on ALL objectives, it must NOT dominate current."""
        sel = ParetoFrontSelector(fitness_keys=["a", "b"])
        worse_new = Program(
            code="x=1", state=ProgramState.DONE, metrics={"a": 1.0, "b": 1.0}
        )
        better_current = Program(
            code="x=2", state=ProgramState.DONE, metrics={"a": 10.0, "b": 10.0}
        )
        # new is strictly worse => should NOT dominate current
        assert sel(worse_new, better_current) is False

    def test_current_dominates_new_reverse_check(self):
        """The reverse direction: current dominates new. Verify new does NOT replace."""
        sel = ParetoFrontSelector(fitness_keys=["score", "speed"])
        new = Program(
            code="x=1", state=ProgramState.DONE, metrics={"score": 2.0, "speed": 3.0}
        )
        current = Program(
            code="x=2",
            state=ProgramState.DONE,
            metrics={"score": 10.0, "speed": 10.0},
        )
        assert sel(new, current) is False
        # And confirm the reverse IS a domination
        assert sel(current, new) is True

    def test_worse_on_all_with_lower_is_better(self):
        """With higher_is_better=False, a new point with HIGHER values on all
        objectives is strictly worse and should be rejected."""
        sel = ParetoFrontSelector(
            fitness_keys=["error", "latency"],
            fitness_key_higher_is_better=[False, False],
        )
        # For lower-is-better: lower is better; new has higher values = worse
        worse_new = Program(
            code="x=1",
            state=ProgramState.DONE,
            metrics={"error": 10.0, "latency": 100.0},
        )
        better_current = Program(
            code="x=2",
            state=ProgramState.DONE,
            metrics={"error": 1.0, "latency": 10.0},
        )
        assert sel(worse_new, better_current) is False


# ---------------------------------------------------------------------------
# Audit Finding 2: Tournament size never varied
# ---------------------------------------------------------------------------


class TestTournamentSizeVariation:
    """Test tournament selectors with different tournament sizes."""

    def test_scalar_tournament_size_2(self):
        """With tournament_size=2, best should still be selected more than uniform."""
        random.seed(42)
        np.random.seed(42)
        sel = ScalarTournamentEliteSelector(fitness_key="score", tournament_size=2)
        progs = [_make_program(float(i)) for i in range(10)]
        best = progs[-1]

        counts: Counter = Counter()
        for _ in range(2000):
            result = sel(progs, total=1)
            counts[id(result[0])] += 1

        # With size 2, best still wins more than uniform (10%)
        assert counts[id(best)] / 2000 > 0.10

    def test_scalar_tournament_size_5(self):
        """With tournament_size=5, best should be selected more often than with size 2."""
        random.seed(42)
        np.random.seed(42)
        sel_2 = ScalarTournamentEliteSelector(fitness_key="score", tournament_size=2)
        sel_5 = ScalarTournamentEliteSelector(fitness_key="score", tournament_size=5)
        progs = [_make_program(float(i)) for i in range(10)]
        best = progs[-1]

        counts_2: Counter = Counter()
        counts_5: Counter = Counter()
        n_trials = 3000
        for _ in range(n_trials):
            random.seed(_ + 1000)
            r2 = sel_2(progs, total=1)
            counts_2[id(r2[0])] += 1
            random.seed(_ + 2000)
            r5 = sel_5(progs, total=1)
            counts_5[id(r5[0])] += 1

        # Larger tournament -> more selection pressure on best
        assert counts_5[id(best)] > counts_2[id(best)]

    def test_scalar_tournament_size_equals_population(self):
        """With tournament_size == population_size, the best program ALWAYS wins."""
        random.seed(42)
        sel = ScalarTournamentEliteSelector(fitness_key="score", tournament_size=10)
        progs = [_make_program(float(i)) for i in range(10)]
        best = progs[-1]

        for _ in range(100):
            result = sel(progs, total=1)
            assert result[0] is best, (
                "With tournament_size == population_size, best must always win"
            )

    def test_pareto_tournament_size_varies(self):
        """ParetoTournament with tournament_size == population: dominant always wins."""
        random.seed(42)
        sel = ParetoTournamentEliteSelector(
            fitness_keys=["a", "b"],
            fitness_key_higher_is_better={"a": True, "b": True},
            tournament_size=3,  # == population size
        )
        dominant = Program(code="x=1", metrics={"a": 10.0, "b": 10.0})
        weak = Program(code="x=2", metrics={"a": 1.0, "b": 1.0})
        mid = Program(code="x=3", metrics={"a": 5.0, "b": 5.0})
        progs = [weak, mid, dominant]

        for _ in range(50):
            result = sel(progs, total=1)
            assert result[0] is dominant


# ---------------------------------------------------------------------------
# Audit Finding 3: dominates() asymmetry
# ---------------------------------------------------------------------------


class TestDominatesAsymmetry:
    """Verify that dominates(a, b) and dominates(b, a) are NOT both True."""

    def test_dominates_forward_not_reverse(self):
        """If a dominates b, then b must NOT dominate a."""
        a = [10.0, 10.0]
        b = [5.0, 5.0]
        assert dominates(a, b) is True
        assert dominates(b, a) is False

    def test_dominates_reverse_not_forward(self):
        """If b dominates a, then a must NOT dominate b."""
        a = [1.0, 1.0]
        b = [5.0, 5.0]
        assert dominates(a, b) is False
        assert dominates(b, a) is True

    def test_neither_dominates_tradeoff(self):
        """With a trade-off, neither should dominate the other."""
        a = [10.0, 1.0]
        b = [1.0, 10.0]
        assert dominates(a, b) is False
        assert dominates(b, a) is False

    def test_equal_vectors_no_domination(self):
        """Equal vectors: domination requires at least one strict inequality."""
        a = [5.0, 5.0, 5.0]
        b = [5.0, 5.0, 5.0]
        assert dominates(a, b) is False
        assert dominates(b, a) is False

    def test_dominates_partial_equal_one_better(self):
        """a is >= b on all and strictly > on one dimension."""
        a = [5.0, 6.0]
        b = [5.0, 5.0]
        assert dominates(a, b) is True
        assert dominates(b, a) is False

    def test_dominates_three_dimensions(self):
        """Asymmetry check in 3D."""
        a = [10.0, 10.0, 10.0]
        b = [5.0, 5.0, 5.0]
        assert dominates(a, b) is True
        assert dominates(b, a) is False


# ---------------------------------------------------------------------------
# Audit Finding 4: weighted_sample multi-draw distribution
# ---------------------------------------------------------------------------


class TestWeightedSampleMultiDraw:
    """Test weighted_sample_without_replacement with multiple draws."""

    def test_no_duplicates_in_multi_draw(self):
        """Multiple draws should never return duplicates (sampling w/o replacement)."""
        random.seed(42)
        items = list(range(10))
        weights = [1.0] * 10
        result = weighted_sample_without_replacement(items, weights, k=5)
        assert len(result) == 5
        assert len(set(result)) == 5, "Duplicates found in without-replacement sample"

    def test_higher_weight_items_appear_more(self):
        """Over many trials, items with higher weights should appear more often."""
        items = ["A", "B", "C", "D"]
        # A has 10x the weight of others
        weights = [10.0, 1.0, 1.0, 1.0]

        counts: Counter = Counter()
        n_trials = 2000
        for i in range(n_trials):
            random.seed(i)
            result = weighted_sample_without_replacement(items, weights, k=1)
            counts[result[0]] += 1

        # A should appear much more frequently than any other single item
        assert counts["A"] > counts["B"]
        assert counts["A"] > counts["C"]
        assert counts["A"] > counts["D"]
        # A should get majority of selections
        assert counts["A"] > n_trials * 0.5

    def test_draw_all_items(self):
        """Drawing k == len(items) should return all items (no duplicates)."""
        random.seed(42)
        items = ["X", "Y", "Z"]
        weights = [1.0, 2.0, 3.0]
        result = weighted_sample_without_replacement(items, weights, k=3)
        assert len(result) == 3
        assert set(result) == {"X", "Y", "Z"}

    def test_k_larger_than_population(self):
        """k > len(items) is clamped to len(items)."""
        random.seed(42)
        items = [1, 2, 3]
        weights = [1.0, 1.0, 1.0]
        result = weighted_sample_without_replacement(items, weights, k=100)
        assert len(result) == 3
        assert set(result) == {1, 2, 3}


# ---------------------------------------------------------------------------
# Audit Finding 5: Fixed random seeds for determinism
# ---------------------------------------------------------------------------


class TestFixedRandomSeeds:
    """Tests that use explicit random seeds for deterministic, non-flaky behavior."""

    def test_scalar_tournament_deterministic_with_seed(self):
        """Same seed produces same selection result."""
        sel = ScalarTournamentEliteSelector(fitness_key="score", tournament_size=3)
        progs = [_make_program(float(i)) for i in range(10)]

        random.seed(12345)
        result1 = sel(progs, total=3)
        ids1 = [id(p) for p in result1]

        random.seed(12345)
        result2 = sel(progs, total=3)
        ids2 = [id(p) for p in result2]

        assert ids1 == ids2

    def test_weighted_elite_deterministic_with_seed(self):
        """WeightedEliteSelector produces same results with same seed."""
        sel = WeightedEliteSelector(fitness_key="score", lambda_=10.0)
        progs = [_make_program(float(i)) for i in range(10)]

        random.seed(99999)
        result1 = sel(progs, total=3)
        ids1 = [id(p) for p in result1]

        random.seed(99999)
        result2 = sel(progs, total=3)
        ids2 = [id(p) for p in result2]

        assert ids1 == ids2

    def test_fitness_proportional_deterministic_with_seed(self):
        """FitnessProportionalEliteSelector produces same results with same seed."""
        sel = FitnessProportionalEliteSelector(fitness_key="score", temperature=0.5)
        progs = [_make_program(float(i)) for i in range(10)]

        random.seed(54321)
        result1 = sel(progs, total=3)
        ids1 = [id(p) for p in result1]

        random.seed(54321)
        result2 = sel(progs, total=3)
        ids2 = [id(p) for p in result2]

        assert ids1 == ids2

    def test_random_elite_deterministic_with_seed(self):
        """RandomEliteSelector produces same results with same seed."""
        sel = RandomEliteSelector()
        progs = [_make_program(float(i)) for i in range(10)]

        random.seed(77777)
        result1 = sel(progs, total=3)
        ids1 = [id(p) for p in result1]

        random.seed(77777)
        result2 = sel(progs, total=3)
        ids2 = [id(p) for p in result2]

        assert ids1 == ids2


# ---------------------------------------------------------------------------
# Audit Finding 6: Negative fitness values
# ---------------------------------------------------------------------------


class TestNegativeFitnessValues:
    """Selectors should handle negative fitness values correctly."""

    def test_scalar_tournament_negative_fitness(self):
        """Tournament selector works correctly with negative fitness values."""
        random.seed(42)
        sel = ScalarTournamentEliteSelector(
            fitness_key="score",
            tournament_size=10,  # full population
        )
        progs = [_make_program(f) for f in [-10.0, -5.0, 0.0, 5.0, 10.0]]
        best = progs[-1]  # 10.0 is the highest

        # With full population tournament, the best should always win
        for _ in range(50):
            result = sel(progs, total=1)
            assert result[0] is best

    def test_scalar_tournament_negative_lower_is_better(self):
        """With higher_is_better=False and negative values, lowest value wins."""
        random.seed(42)
        sel = ScalarTournamentEliteSelector(
            fitness_key="score",
            fitness_key_higher_is_better=False,
            tournament_size=5,  # full population
        )
        progs = [_make_program(f) for f in [-10.0, -5.0, 0.0, 5.0, 10.0]]
        best = progs[0]  # -10.0 is the lowest

        for _ in range(50):
            result = sel(progs, total=1)
            assert result[0] is best

    def test_weighted_elite_negative_fitness(self):
        """WeightedEliteSelector handles negative fitness values."""
        random.seed(42)
        sel = WeightedEliteSelector(fitness_key="score", lambda_=10.0)
        progs = [_make_program(f) for f in [-10.0, -5.0, 0.0, 5.0, 10.0]]

        # Should not crash and should return correct count
        result = sel(progs, total=2)
        assert len(result) == 2
        assert len(set(id(p) for p in result)) == 2

    def test_fitness_proportional_negative_fitness(self):
        """FitnessProportionalEliteSelector handles negative fitness values."""
        random.seed(42)
        sel = FitnessProportionalEliteSelector(fitness_key="score", temperature=0.001)
        progs = [_make_program(f) for f in [-10.0, -5.0, 0.0, 5.0, 10.0]]
        best = progs[-1]  # 10.0

        counts: Counter = Counter()
        for _ in range(500):
            result = sel(progs, total=1)
            counts[id(result[0])] += 1

        # With low temperature, best should dominate
        assert counts[id(best)] > 400

    def test_pareto_tournament_negative_fitness(self):
        """ParetoTournament handles negative fitness on both dimensions."""
        random.seed(42)
        sel = ParetoTournamentEliteSelector(
            fitness_keys=["a", "b"],
            fitness_key_higher_is_better={"a": True, "b": True},
            tournament_size=3,
        )
        # dominant has highest values even though all are negative or zero
        dominant = Program(code="x=1", metrics={"a": 0.0, "b": 0.0})
        weak = Program(code="x=2", metrics={"a": -10.0, "b": -10.0})
        mid = Program(code="x=3", metrics={"a": -5.0, "b": -5.0})
        progs = [weak, mid, dominant]

        counts: Counter = Counter()
        for _ in range(500):
            result = sel(progs, total=1)
            counts[id(result[0])] += 1

        assert counts[id(dominant)] > counts[id(weak)]

    def test_dominates_with_negative_values(self):
        """dominates() handles negative values correctly."""
        a = [-1.0, -1.0]
        b = [-5.0, -5.0]
        assert dominates(a, b) is True  # -1 > -5
        assert dominates(b, a) is False

    def test_pareto_front_selector_negative_fitness(self):
        """ParetoFrontSelector handles negative fitness values."""
        sel = ParetoFrontSelector(fitness_keys=["a", "b"])
        new = Program(
            code="x=1", state=ProgramState.DONE, metrics={"a": -1.0, "b": -1.0}
        )
        curr = Program(
            code="x=2", state=ProgramState.DONE, metrics={"a": -5.0, "b": -5.0}
        )
        # -1 > -5, so new dominates current
        assert sel(new, curr) is True
        # Reverse: current does NOT dominate new
        assert sel(curr, new) is False


# ---------------------------------------------------------------------------
# Audit Finding 7: Single-element population
# ---------------------------------------------------------------------------


class TestSingleElementPopulation:
    """Selectors should handle a single-element population gracefully."""

    def test_scalar_tournament_single_program(self):
        """Single program: should return it when total >= 1."""
        sel = ScalarTournamentEliteSelector(fitness_key="score", tournament_size=3)
        progs = [_make_program(5.0)]
        result = sel(progs, total=1)
        assert result == progs
        assert result[0] is progs[0]

    def test_scalar_tournament_single_program_total_gt_1(self):
        """Single program with total > 1: should return all (just 1)."""
        sel = ScalarTournamentEliteSelector(fitness_key="score", tournament_size=3)
        progs = [_make_program(5.0)]
        result = sel(progs, total=5)
        assert result == progs

    def test_pareto_tournament_single_program(self):
        """ParetoTournament with single program."""
        sel = ParetoTournamentEliteSelector(
            fitness_keys=["a", "b"],
            tournament_size=3,
        )
        progs = [Program(code="x=1", metrics={"a": 1.0, "b": 2.0})]
        result = sel(progs, total=1)
        assert result == progs

    def test_weighted_elite_single_program(self):
        """WeightedEliteSelector with single program."""
        sel = WeightedEliteSelector(fitness_key="score", lambda_=10.0)
        progs = [_make_program(5.0)]
        result = sel(progs, total=1)
        assert result == progs

    def test_weighted_elite_single_program_total_gt_1(self):
        """WeightedEliteSelector with single program, total > 1."""
        sel = WeightedEliteSelector(fitness_key="score", lambda_=10.0)
        progs = [_make_program(5.0)]
        result = sel(progs, total=3)
        assert result == progs

    def test_fitness_proportional_single_program(self):
        """FitnessProportionalEliteSelector with single program."""
        sel = FitnessProportionalEliteSelector(fitness_key="score")
        progs = [_make_program(5.0)]
        result = sel(progs, total=1)
        assert result == progs

    def test_random_elite_single_program(self):
        """RandomEliteSelector with single program."""
        sel = RandomEliteSelector()
        progs = [_make_program(5.0)]
        result = sel(progs, total=1)
        assert result == progs

    def test_random_elite_single_program_total_gt_1(self):
        """RandomEliteSelector with single program, total > 1."""
        sel = RandomEliteSelector()
        progs = [_make_program(5.0)]
        result = sel(progs, total=5)
        assert result == progs
