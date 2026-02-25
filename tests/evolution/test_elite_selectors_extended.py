"""Extended tests for gigaevo/evolution/strategies/elite_selectors.py

Tests ONLY paths not covered by test_elite_selectors.py:
1. RandomEliteSelector: not tested at all in the existing file
2. FitnessProportionalEliteSelector: missing fitness key, inf/nan fallback,
   None temperature attribute
3. ParetoTournamentEliteSelector: constructor validation (requires >=2 keys),
   custom tie_breaker, default/custom higher_is_better
"""

from __future__ import annotations

import random

import pytest

from gigaevo.evolution.strategies.elite_selectors import (
    FitnessProportionalEliteSelector,
    ParetoTournamentEliteSelector,
    RandomEliteSelector,
)
from gigaevo.programs.program import Program


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
