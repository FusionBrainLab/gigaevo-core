"""Tests for gigaevo/evolution/mutation/parent_selector.py."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from gigaevo.evolution.mutation.parent_selector import (
    AllCombinationsParentSelector,
    RandomParentSelector,
)


def _mock_programs(n: int) -> list:
    return [MagicMock(id=f"prog-{i}") for i in range(n)]


# ---------------------------------------------------------------------------
# RandomParentSelector
# ---------------------------------------------------------------------------


class TestRandomParentSelector:
    def test_num_parents_lt_1_raises(self) -> None:
        with pytest.raises(ValueError, match="num_parents must be at least 1"):
            RandomParentSelector(num_parents=0)

    def test_empty_list_yields_nothing(self) -> None:
        sel = RandomParentSelector(num_parents=2)
        results = list(sel.create_parent_iterator([]))
        assert results == []

    def test_yields_correct_count(self) -> None:
        sel = RandomParentSelector(num_parents=2)
        programs = _mock_programs(5)
        it = sel.create_parent_iterator(programs)
        first = next(it)
        assert len(first) == 2

    def test_yields_all_when_fewer_available(self) -> None:
        sel = RandomParentSelector(num_parents=5)
        programs = _mock_programs(2)
        it = sel.create_parent_iterator(programs)
        first = next(it)
        assert len(first) == 2

    def test_infinite_iterator(self) -> None:
        sel = RandomParentSelector(num_parents=1)
        programs = _mock_programs(3)
        it = sel.create_parent_iterator(programs)
        for _ in range(100):
            batch = next(it)
            assert len(batch) == 1
            assert batch[0] in programs


# ---------------------------------------------------------------------------
# AllCombinationsParentSelector
# ---------------------------------------------------------------------------


class TestAllCombinationsParentSelector:
    def test_num_parents_lt_1_raises(self) -> None:
        with pytest.raises(ValueError, match="num_parents must be at least 1"):
            AllCombinationsParentSelector(num_parents=0)

    def test_empty_list_yields_nothing(self) -> None:
        sel = AllCombinationsParentSelector(num_parents=2)
        results = list(sel.create_parent_iterator([]))
        assert results == []

    def test_all_combinations(self) -> None:
        sel = AllCombinationsParentSelector(num_parents=2)
        programs = _mock_programs(3)
        combos = list(sel.create_parent_iterator(programs))
        # C(3,2) = 3
        assert len(combos) == 3
        for combo in combos:
            assert len(combo) == 2

    def test_fewer_parents_than_requested_yields_all_once(self) -> None:
        sel = AllCombinationsParentSelector(num_parents=5)
        programs = _mock_programs(2)
        combos = list(sel.create_parent_iterator(programs))
        assert len(combos) == 1
        assert len(combos[0]) == 2

    def test_finite_iterator(self) -> None:
        sel = AllCombinationsParentSelector(num_parents=1)
        programs = _mock_programs(4)
        combos = list(sel.create_parent_iterator(programs))
        # C(4,1) = 4
        assert len(combos) == 4
