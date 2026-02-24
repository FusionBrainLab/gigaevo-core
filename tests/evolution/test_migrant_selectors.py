"""Tests for gigaevo/evolution/strategies/migrant_selectors.py."""

from __future__ import annotations

from unittest.mock import MagicMock

from gigaevo.evolution.strategies.migrant_selectors import (
    ParetoFrontMigrantSelector,
    RandomMigrantSelector,
    TopFitnessMigrantSelector,
)


def _prog(metrics: dict[str, float], prog_id: str = "p") -> MagicMock:
    p = MagicMock()
    p.metrics = metrics
    p.id = prog_id
    return p


# ---------------------------------------------------------------------------
# RandomMigrantSelector
# ---------------------------------------------------------------------------


class TestRandomMigrantSelector:
    def test_empty_returns_empty(self) -> None:
        sel = RandomMigrantSelector()
        assert sel([], 5) == []

    def test_returns_all_when_count_ge_len(self) -> None:
        sel = RandomMigrantSelector()
        progs = [_prog({"x": 1}, f"p{i}") for i in range(3)]
        result = sel(progs, 10)
        assert result is progs  # returns the same list

    def test_returns_correct_count(self) -> None:
        sel = RandomMigrantSelector()
        progs = [_prog({"x": float(i)}, f"p{i}") for i in range(10)]
        result = sel(progs, 3)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# TopFitnessMigrantSelector
# ---------------------------------------------------------------------------


class TestTopFitnessMigrantSelector:
    def test_returns_top_by_fitness(self) -> None:
        sel = TopFitnessMigrantSelector(fitness_key="score")
        progs = [
            _prog({"score": 10.0}, "low"),
            _prog({"score": 50.0}, "mid"),
            _prog({"score": 90.0}, "high"),
        ]
        result = sel(progs, 2)
        ids = [p.id for p in result]
        assert ids == ["high", "mid"]

    def test_lower_is_better(self) -> None:
        sel = TopFitnessMigrantSelector(
            fitness_key="cost", fitness_key_higher_is_better=False
        )
        progs = [
            _prog({"cost": 100.0}, "expensive"),
            _prog({"cost": 1.0}, "cheap"),
            _prog({"cost": 50.0}, "mid"),
        ]
        result = sel(progs, 2)
        ids = [p.id for p in result]
        assert ids == ["cheap", "mid"]

    def test_empty_returns_empty(self) -> None:
        sel = TopFitnessMigrantSelector(fitness_key="score")
        assert sel([], 5) == []


# ---------------------------------------------------------------------------
# ParetoFrontMigrantSelector
# ---------------------------------------------------------------------------


class TestParetoFrontMigrantSelector:
    def test_returns_pareto_front(self) -> None:
        sel = ParetoFrontMigrantSelector(fitness_keys=["a", "b"])
        # p1 dominates p3 (better in both), p2 is non-dominated
        progs = [
            _prog({"a": 10.0, "b": 10.0}, "p1"),
            _prog({"a": 5.0, "b": 15.0}, "p2"),
            _prog({"a": 3.0, "b": 3.0}, "p3"),  # dominated by p1
        ]
        result = sel(progs, 5)
        ids = {p.id for p in result}
        assert "p1" in ids
        assert "p2" in ids

    def test_fills_from_remaining(self) -> None:
        sel = ParetoFrontMigrantSelector(fitness_keys=["a"])
        # p1 dominates p2
        progs = [
            _prog({"a": 10.0}, "p1"),
            _prog({"a": 5.0}, "p2"),
        ]
        # request 2, pareto front is just p1, so p2 fills
        result = sel(progs, 2)
        assert len(result) == 2

    def test_empty_returns_empty(self) -> None:
        sel = ParetoFrontMigrantSelector(fitness_keys=["a"])
        assert sel([], 5) == []

    def test_custom_higher_is_better(self) -> None:
        sel = ParetoFrontMigrantSelector(
            fitness_keys=["cost"],
            fitness_key_higher_is_better={"cost": False},
        )
        progs = [
            _prog({"cost": 1.0}, "cheap"),
            _prog({"cost": 100.0}, "expensive"),
        ]
        result = sel(progs, 1)
        assert result[0].id == "cheap"
