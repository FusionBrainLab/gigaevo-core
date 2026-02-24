"""Tests for gigaevo/evolution/strategies/selectors.py"""

import pytest

from gigaevo.evolution.strategies.selectors import (
    ParetoFrontSelector,
    SumArchiveSelector,
)
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState


def _prog(metrics=None):
    p = Program(code="def solve(): return 1", state=ProgramState.DONE)
    if metrics:
        p.add_metrics(metrics)
    return p


class TestArchiveSelectorBase:
    def test_empty_keys_raises(self):
        with pytest.raises(ValueError, match="fitness_keys cannot be empty"):
            SumArchiveSelector(fitness_keys=[])

    def test_default_higher_is_better(self):
        sel = SumArchiveSelector(fitness_keys=["a", "b"])
        assert sel.fitness_key_higher_is_better == {"a": True, "b": True}


class TestSumArchiveSelector:
    def test_better_sum_replaces(self):
        sel = SumArchiveSelector(fitness_keys=["score"])
        new = _prog(metrics={"score": 10.0})
        curr = _prog(metrics={"score": 5.0})
        assert sel(new, curr) is True

    def test_worse_sum_keeps(self):
        sel = SumArchiveSelector(fitness_keys=["score"])
        new = _prog(metrics={"score": 3.0})
        curr = _prog(metrics={"score": 5.0})
        assert sel(new, curr) is False

    def test_equal_sum_keeps(self):
        sel = SumArchiveSelector(fitness_keys=["score"])
        new = _prog(metrics={"score": 5.0})
        curr = _prog(metrics={"score": 5.0})
        assert sel(new, curr) is False

    def test_custom_weights(self):
        sel = SumArchiveSelector(
            fitness_keys=["a", "b"],
            weights=[2.0, 1.0],
        )
        # new: 2*3 + 1*1 = 7; curr: 2*1 + 1*10 = 12
        new = _prog(metrics={"a": 3.0, "b": 1.0})
        curr = _prog(metrics={"a": 1.0, "b": 10.0})
        assert sel(new, curr) is False

    def test_higher_is_better_false(self):
        sel = SumArchiveSelector(
            fitness_keys=["error"],
            fitness_key_higher_is_better=[False],
        )
        # higher_is_better=False -> values negated: val => -val
        # new: error=2.0 -> -2.0; curr: error=5.0 -> -5.0
        # new_sum=-2 > curr_sum=-5 => True (lower error wins)
        new = _prog(metrics={"error": 2.0})
        curr = _prog(metrics={"error": 5.0})
        assert sel(new, curr) is True

        # Same error => equal => not strictly better => False
        new_eq = _prog(metrics={"error": 5.0})
        curr_eq = _prog(metrics={"error": 5.0})
        assert sel(new_eq, curr_eq) is False

    def test_score_method(self):
        sel = SumArchiveSelector(
            fitness_keys=["a", "b"],
            weights=[1.0, 2.0],
        )
        p = _prog(metrics={"a": 3.0, "b": 4.0})
        assert sel.score(p) == pytest.approx(3.0 + 8.0)

    def test_missing_key_raises(self):
        sel = SumArchiveSelector(fitness_keys=["missing"])
        new = _prog(metrics={"score": 1.0})
        curr = _prog(metrics={"score": 2.0})
        with pytest.raises(KeyError, match="Missing fitness key"):
            sel(new, curr)

    def test_weights_length_mismatch_truncates(self):
        """If weights has different length than fitness_keys, zip truncates silently."""
        sel = SumArchiveSelector(
            fitness_keys=["a", "b"],
            weights=[2.0, 1.0, 999.0],  # extra weight ignored
        )
        # new: 2*5 + 1*1 = 11; curr: 2*1 + 1*10 = 12
        new = _prog(metrics={"a": 5.0, "b": 1.0})
        curr = _prog(metrics={"a": 1.0, "b": 10.0})
        assert sel(new, curr) is False

    def test_multi_key_with_mixed_higher_is_better(self):
        """3 keys with [True, False, True], verify correct sum."""
        sel = SumArchiveSelector(
            fitness_keys=["reward", "error", "bonus"],
            fitness_key_higher_is_better=[True, False, True],
            weights=[1.0, 1.0, 1.0],
        )
        # new values after sign flip: [10, -1, 5] -> sum=14
        # curr values after sign flip: [5, -10, 3] -> sum=-2
        new = _prog(metrics={"reward": 10.0, "error": 1.0, "bonus": 5.0})
        curr = _prog(metrics={"reward": 5.0, "error": 10.0, "bonus": 3.0})
        assert sel(new, curr) is True


class TestParetoFrontSelector:
    def test_dominates(self):
        sel = ParetoFrontSelector(fitness_keys=["a", "b"])
        new = _prog(metrics={"a": 5.0, "b": 5.0})
        curr = _prog(metrics={"a": 3.0, "b": 3.0})
        assert sel(new, curr) is True

    def test_does_not_dominate(self):
        sel = ParetoFrontSelector(fitness_keys=["a", "b"])
        new = _prog(metrics={"a": 5.0, "b": 1.0})
        curr = _prog(metrics={"a": 3.0, "b": 3.0})
        assert sel(new, curr) is False

    def test_equal_does_not_dominate(self):
        sel = ParetoFrontSelector(fitness_keys=["a", "b"])
        new = _prog(metrics={"a": 3.0, "b": 3.0})
        curr = _prog(metrics={"a": 3.0, "b": 3.0})
        assert sel(new, curr) is False

    def test_mixed_higher_is_better(self):
        sel = ParetoFrontSelector(
            fitness_keys=["reward", "error"],
            fitness_key_higher_is_better=[True, False],
        )
        # reward: higher is better; error: lower is better (negated)
        # new: [10, -1] vs curr: [5, -5]
        # 10>=5 and -1>=-5 and at least one strictly > -> True
        new = _prog(metrics={"reward": 10.0, "error": 1.0})
        curr = _prog(metrics={"reward": 5.0, "error": 5.0})
        assert sel(new, curr) is True

    def test_partial_domination_one_equal(self):
        """new=(5, 3) vs curr=(3, 3): strictly better on one, equal on other -> dominates."""
        sel = ParetoFrontSelector(fitness_keys=["a", "b"])
        new = _prog(metrics={"a": 5.0, "b": 3.0})
        curr = _prog(metrics={"a": 3.0, "b": 3.0})
        assert sel(new, curr) is True

    def test_single_key_pareto(self):
        """Single fitness key: strictly better dominates."""
        sel = ParetoFrontSelector(fitness_keys=["score"])
        new = _prog(metrics={"score": 10.0})
        curr = _prog(metrics={"score": 5.0})
        assert sel(new, curr) is True

        # Equal -> no domination
        new_eq = _prog(metrics={"score": 5.0})
        curr_eq = _prog(metrics={"score": 5.0})
        assert sel(new_eq, curr_eq) is False
