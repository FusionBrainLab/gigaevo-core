"""Edge-case and boundary tests for gigaevo/evolution/strategies/elite_selectors.py

Covers:
1. RandomEliteSelector: selection behavior
2. FitnessProportionalEliteSelector: missing fitness key, inf/nan fallback,
   None temperature attribute
3. ParetoTournamentEliteSelector: constructor validation (requires >=2 keys),
   custom tie_breaker, default/custom higher_is_better
4. dominates asymmetry, weighted_sample, negative fitness,
   single-element, seed determinism, ParetoFront reverse direction
"""

from __future__ import annotations

from collections import Counter
import random

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
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState


def _make_program(metrics: dict, child_count: int = 0) -> Program:
    p = Program(code="pass")
    p.metrics = metrics
    for i in range(child_count):
        p.lineage.add_child(f"fake_child_{i}")
    return p


# ═══════════════════════════════════════════════════════════════════════════
# RandomEliteSelector — not covered in test_elite_selectors.py at all
# ═══════════════════════════════════════════════════════════════════════════


class TestRandomEliteSelector:
    def test_returns_all_when_fewer_than_total(self) -> None:
        """When len(programs) <= total, return all programs unchanged."""
        selector = RandomEliteSelector()
        programs = [_make_program({"s": i}) for i in range(3)]
        result = selector(programs, total=5)
        assert result == programs

    def test_returns_all_when_equal_to_total(self) -> None:
        selector = RandomEliteSelector()
        programs = [_make_program({"s": i}) for i in range(3)]
        result = selector(programs, total=3)
        assert result == programs

    def test_selects_subset_when_more_than_total(self) -> None:
        selector = RandomEliteSelector()
        programs = [_make_program({"s": i}) for i in range(10)]
        random.seed(42)
        result = selector(programs, total=3)
        assert len(result) == 3
        assert all(p in programs for p in result)

    def test_no_duplicates(self) -> None:
        selector = RandomEliteSelector()
        programs = [_make_program({"s": i}) for i in range(10)]
        random.seed(0)
        result = selector(programs, total=5)
        assert len(result) == len(set(id(p) for p in result))


# ═══════════════════════════════════════════════════════════════════════════
# FitnessProportionalEliteSelector — edge cases not in test_elite_selectors.py
# ═══════════════════════════════════════════════════════════════════════════


class TestFitnessProportionalEdgeCases:
    def test_missing_fitness_key_raises(self) -> None:
        """Programs missing the fitness key should raise ValueError."""
        selector = FitnessProportionalEliteSelector(
            fitness_key="score", fitness_key_higher_is_better=True
        )
        programs = [
            _make_program({"score": 0.5}),
            _make_program({"wrong_key": 0.8}),  # missing 'score'
        ]
        with pytest.raises(ValueError, match="Missing fitness key"):
            selector(programs, total=1)

    def test_non_finite_fitness_falls_back_to_uniform(self) -> None:
        """Non-finite fitnesses (inf) should fallback to uniform sampling."""
        selector = FitnessProportionalEliteSelector(
            fitness_key="score", fitness_key_higher_is_better=True
        )
        programs = [
            _make_program({"score": 0.5}),
            _make_program({"score": float("inf")}),
            _make_program({"score": 0.3}),
        ]
        random.seed(42)
        result = selector(programs, total=2)
        assert len(result) == 2
        assert all(p in programs for p in result)

    def test_nan_fitness_falls_back_to_uniform(self) -> None:
        """NaN fitness should fallback to uniform sampling."""
        selector = FitnessProportionalEliteSelector(
            fitness_key="score", fitness_key_higher_is_better=True
        )
        programs = [
            _make_program({"score": float("nan")}),
            _make_program({"score": 0.5}),
        ]
        random.seed(0)
        result = selector(programs, total=1)
        assert len(result) == 1

    def test_temperature_none_attribute(self) -> None:
        """Default temperature=None is stored as attribute and auto-computed at call."""
        selector = FitnessProportionalEliteSelector(
            fitness_key="score", fitness_key_higher_is_better=True
        )
        assert selector.temperature is None
        # Should still work (auto-computes temperature from fitness spread)
        programs = [_make_program({"score": float(i)}) for i in range(5)]
        random.seed(42)
        result = selector(programs, total=2)
        assert len(result) == 2


# ═══════════════════════════════════════════════════════════════════════════
# ParetoTournamentEliteSelector — constructor validation and novel features
# ═══════════════════════════════════════════════════════════════════════════


class TestParetoTournamentConstructor:
    def test_requires_at_least_two_keys(self) -> None:
        """Single fitness key should raise ValueError."""
        with pytest.raises(ValueError, match="at least two"):
            ParetoTournamentEliteSelector(fitness_keys=["score"])

    def test_empty_keys_raises(self) -> None:
        with pytest.raises(ValueError, match="at least two"):
            ParetoTournamentEliteSelector(fitness_keys=[])

    def test_default_higher_is_better(self) -> None:
        """Default fitness_key_higher_is_better should be True for all keys."""
        selector = ParetoTournamentEliteSelector(
            fitness_keys=["a", "b"],
        )
        assert selector.higher_is_better == {"a": True, "b": True}

    def test_custom_higher_is_better(self) -> None:
        selector = ParetoTournamentEliteSelector(
            fitness_keys=["score", "loss"],
            fitness_key_higher_is_better={"score": True, "loss": False},
        )
        assert selector.higher_is_better["loss"] is False


class TestParetoTournamentTieBreaker:
    def test_custom_tie_breaker_influences_selection(self) -> None:
        """Custom tie-breaker should influence selection among Pareto-equal programs."""
        # All programs are on the Pareto front (a + b = 10)
        selector = ParetoTournamentEliteSelector(
            fitness_keys=["a", "b"],
            tie_breaker=lambda p: -p.metrics["a"],  # prefer higher 'a'
            tournament_size=3,
        )
        programs = [
            _make_program({"a": float(i), "b": float(10 - i)}) for i in range(10)
        ]
        random.seed(42)
        result = selector(programs, total=3)
        assert len(result) == 3

    def test_returns_all_when_fewer_than_total(self) -> None:
        """Pareto selector with fewer programs than total returns all."""
        selector = ParetoTournamentEliteSelector(
            fitness_keys=["a", "b"],
        )
        programs = [_make_program({"a": 1.0, "b": 2.0})]
        result = selector(programs, total=5)
        assert result == programs


# ═══════════════════════════════════════════════════════════════════════════
# Audit Finding 1: ParetoFrontSelector — strictly worse point is rejected
# ═══════════════════════════════════════════════════════════════════════════


class TestParetoFrontSelectorReverseDominance:
    """A new point that is strictly worse on ALL objectives must be rejected."""

    def test_strictly_worse_new_is_rejected(self) -> None:
        sel = ParetoFrontSelector(fitness_keys=["a", "b"])
        worse = Program(
            code="x=1", state=ProgramState.DONE, metrics={"a": 1.0, "b": 1.0}
        )
        better = Program(
            code="x=2", state=ProgramState.DONE, metrics={"a": 10.0, "b": 10.0}
        )
        assert sel(worse, better) is False

    def test_strictly_worse_with_three_objectives(self) -> None:
        sel = ParetoFrontSelector(fitness_keys=["x", "y", "z"])
        worse = Program(
            code="x=1",
            state=ProgramState.DONE,
            metrics={"x": 0.0, "y": 0.0, "z": 0.0},
        )
        better = Program(
            code="x=2",
            state=ProgramState.DONE,
            metrics={"x": 5.0, "y": 5.0, "z": 5.0},
        )
        assert sel(worse, better) is False
        # And the reverse holds
        assert sel(better, worse) is True

    def test_worse_with_lower_is_better(self) -> None:
        """With higher_is_better=False, a new point with higher values is worse."""
        sel = ParetoFrontSelector(
            fitness_keys=["loss", "error"],
            fitness_key_higher_is_better=[False, False],
        )
        worse = Program(
            code="x=1",
            state=ProgramState.DONE,
            metrics={"loss": 100.0, "error": 50.0},
        )
        better = Program(
            code="x=2",
            state=ProgramState.DONE,
            metrics={"loss": 1.0, "error": 1.0},
        )
        assert sel(worse, better) is False


# ═══════════════════════════════════════════════════════════════════════════
# Audit Finding 2: Tournament size variation
# ═══════════════════════════════════════════════════════════════════════════


class TestExtTournamentSizeVariation:
    def test_full_population_tournament_always_picks_best(self) -> None:
        """tournament_size == len(population) => deterministic best selection."""
        random.seed(42)
        sel = ScalarTournamentEliteSelector(
            fitness_key="score",
            fitness_key_higher_is_better=True,
            tournament_size=5,
        )
        programs = [_make_program({"score": float(i)}) for i in range(5)]
        best = programs[-1]  # score=4.0

        for _ in range(100):
            result = sel(programs, total=1)
            assert result[0] is best

    def test_small_tournament_gives_diversity(self) -> None:
        """tournament_size=2 with a 10-program population should occasionally
        select non-best programs."""
        random.seed(42)
        sel = ScalarTournamentEliteSelector(
            fitness_key="score",
            fitness_key_higher_is_better=True,
            tournament_size=2,
        )
        programs = [_make_program({"score": float(i)}) for i in range(10)]
        best = programs[-1]

        counts: Counter = Counter()
        for _ in range(2000):
            result = sel(programs, total=1)
            counts[id(result[0])] += 1

        # Best should NOT get 100% — some diversity expected with size 2
        assert counts[id(best)] < 2000
        # But should still be most-selected
        for p in programs[:-1]:
            assert counts[id(best)] >= counts[id(p)]


# ═══════════════════════════════════════════════════════════════════════════
# Audit Finding 3: dominates() asymmetry
# ═══════════════════════════════════════════════════════════════════════════


class TestExtDominatesAsymmetry:
    def test_asymmetry_basic(self) -> None:
        """dominates(a, b) True implies dominates(b, a) False."""
        a = [8.0, 9.0]
        b = [3.0, 4.0]
        assert dominates(a, b) is True
        assert dominates(b, a) is False

    def test_tradeoff_neither_dominates(self) -> None:
        a = [10.0, 1.0]
        b = [1.0, 10.0]
        assert dominates(a, b) is False
        assert dominates(b, a) is False

    def test_equal_no_domination_either_direction(self) -> None:
        a = [7.0, 7.0]
        assert dominates(a, a) is False


# ═══════════════════════════════════════════════════════════════════════════
# Audit Finding 4: weighted_sample multi-draw distribution
# ═══════════════════════════════════════════════════════════════════════════


class TestExtWeightedSampleMultiDraw:
    def test_no_duplicates(self) -> None:
        random.seed(42)
        items = list("ABCDE")
        weights = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = weighted_sample_without_replacement(items, weights, k=5)
        assert len(result) == len(set(result))

    def test_heavily_weighted_item_drawn_first(self) -> None:
        """With a heavily weighted item, it should be drawn first very often."""
        counts_first: Counter = Counter()
        n_trials = 1000
        for i in range(n_trials):
            random.seed(i)
            result = weighted_sample_without_replacement(
                ["heavy", "light1", "light2"], [100.0, 0.01, 0.01], k=2
            )
            counts_first[result[0]] += 1

        assert counts_first["heavy"] > n_trials * 0.95

    def test_zero_weight_items_still_drawable_as_fallback(self) -> None:
        """All-zero weights fall back to uniform sampling."""
        random.seed(42)
        items = [1, 2, 3]
        weights = [0.0, 0.0, 0.0]
        result = weighted_sample_without_replacement(items, weights, k=2)
        assert len(result) == 2
        assert all(r in items for r in result)


# ═══════════════════════════════════════════════════════════════════════════
# Audit Finding 5: Fixed random seeds
# ═══════════════════════════════════════════════════════════════════════════


class TestExtFixedRandomSeeds:
    def test_weighted_sample_reproducible(self) -> None:
        items = list(range(20))
        weights = [float(i + 1) for i in range(20)]

        random.seed(42)
        r1 = weighted_sample_without_replacement(items, weights, k=5)

        random.seed(42)
        r2 = weighted_sample_without_replacement(items, weights, k=5)

        assert r1 == r2

    def test_random_elite_reproducible(self) -> None:
        sel = RandomEliteSelector()
        programs = [_make_program({"s": float(i)}) for i in range(10)]

        random.seed(123)
        r1 = sel(programs, total=3)

        random.seed(123)
        r2 = sel(programs, total=3)

        assert [id(p) for p in r1] == [id(p) for p in r2]


# ═══════════════════════════════════════════════════════════════════════════
# Audit Finding 6: Negative fitness values
# ═══════════════════════════════════════════════════════════════════════════


class TestExtNegativeFitness:
    def test_fitness_proportional_with_all_negative(self) -> None:
        """All-negative fitness should still produce valid selection."""
        random.seed(42)
        sel = FitnessProportionalEliteSelector(fitness_key="score")
        programs = [_make_program({"score": f}) for f in [-100.0, -50.0, -10.0, -1.0]]
        result = sel(programs, total=2)
        assert len(result) == 2
        assert len(set(id(p) for p in result)) == 2

    def test_weighted_elite_with_negative_fitness(self) -> None:
        """WeightedEliteSelector with negative fitness should not crash."""
        random.seed(42)
        sel = WeightedEliteSelector(fitness_key="score", lambda_=5.0)
        programs = [_make_program({"score": f}) for f in [-20.0, -10.0, 0.0, 10.0]]
        result = sel(programs, total=2)
        assert len(result) == 2

    def test_dominates_all_negative(self) -> None:
        """dominates works correctly with all-negative vectors."""
        a = [-1.0, -2.0]  # better (less negative)
        b = [-5.0, -10.0]  # worse (more negative)
        assert dominates(a, b) is True
        assert dominates(b, a) is False

    def test_pareto_front_selector_mixed_sign(self) -> None:
        """ParetoFrontSelector with mixed positive/negative fitness."""
        sel = ParetoFrontSelector(fitness_keys=["a", "b"])
        new = Program(
            code="x=1", state=ProgramState.DONE, metrics={"a": 5.0, "b": -1.0}
        )
        curr = Program(
            code="x=2", state=ProgramState.DONE, metrics={"a": -5.0, "b": -10.0}
        )
        # 5 >= -5 (True), -1 >= -10 (True), at least one strict -> dominates
        assert sel(new, curr) is True


# ═══════════════════════════════════════════════════════════════════════════
# Audit Finding 7: Single-element population
# ═══════════════════════════════════════════════════════════════════════════


class TestExtSingleElementPopulation:
    def test_random_elite_single(self) -> None:
        sel = RandomEliteSelector()
        programs = [_make_program({"s": 1.0})]
        result = sel(programs, total=1)
        assert result == programs

    def test_fitness_proportional_single(self) -> None:
        sel = FitnessProportionalEliteSelector(fitness_key="score")
        programs = [_make_program({"score": 42.0})]
        result = sel(programs, total=1)
        assert result == programs

    def test_scalar_tournament_single(self) -> None:
        sel = ScalarTournamentEliteSelector(fitness_key="score", tournament_size=5)
        programs = [_make_program({"score": 99.0})]
        result = sel(programs, total=1)
        assert result == programs

    def test_weighted_elite_single(self) -> None:
        sel = WeightedEliteSelector(fitness_key="score")
        programs = [_make_program({"score": 7.0})]
        result = sel(programs, total=1)
        assert result == programs

    def test_pareto_tournament_single(self) -> None:
        sel = ParetoTournamentEliteSelector(fitness_keys=["a", "b"])
        programs = [_make_program({"a": 1.0, "b": 2.0})]
        result = sel(programs, total=1)
        assert result == programs

    def test_single_program_total_exceeds_population(self) -> None:
        """Requesting more programs than available should return all (just 1)."""
        sel = ScalarTournamentEliteSelector(fitness_key="score")
        programs = [_make_program({"score": 3.14})]
        result = sel(programs, total=10)
        assert len(result) == 1
        assert result[0] is programs[0]
