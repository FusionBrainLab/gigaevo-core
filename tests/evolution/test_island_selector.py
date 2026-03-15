"""Tests for IslandSelector variants (island_selector.py).

Covers all three selector strategies, the compatibility mixin,
and the _filter_accepting_islands helper.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from gigaevo.evolution.strategies.island_selector import (
    IslandCompatibilityMixin,
    RandomIslandSelector,
    RoundRobinIslandSelector,
    WeightedIslandSelector,
)
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_program(metrics: dict | None = None) -> Program:
    p = Program(code="def solve(): pass", state=ProgramState.RUNNING)
    if metrics:
        p.add_metrics(metrics)
    return p


def _make_island(
    island_id: str = "island_0",
    behavior_keys: list[str] | None = None,
    size: int = 0,
    elite=None,
    accepts: bool = True,
    is_dynamic: bool = False,
) -> MagicMock:
    """Create a minimal mock island suitable for selector tests.

    When accepts=False, a non-None elite is injected so that
    _can_accept_program actually reaches the archive_selector call
    (otherwise current=None short-circuits to True).
    """
    island = MagicMock()
    island.config.island_id = island_id

    behavior_space = MagicMock()
    # Use explicit check so that an empty list [] is preserved (not falsy-defaulted)
    behavior_space.behavior_keys = (
        behavior_keys if behavior_keys is not None else ["score"]
    )
    behavior_space.get_cell.return_value = (0, 0)

    if is_dynamic:
        from gigaevo.evolution.strategies.models import DynamicBehaviorSpace

        behavior_space.__class__ = DynamicBehaviorSpace
        behavior_space.check_and_expand = MagicMock()
    else:
        behavior_space.__class__ = object  # not DynamicBehaviorSpace

    island.config.behavior_space = behavior_space
    # When accepts=False we need a non-None elite so archive_selector is reached
    effective_elite = (
        elite if elite is not None else (MagicMock() if not accepts else None)
    )
    island.archive_storage.get_elite = AsyncMock(return_value=effective_elite)
    island.config.archive_selector.return_value = accepts
    island.__len__ = AsyncMock(return_value=size)
    return island


# ---------------------------------------------------------------------------
# IslandCompatibilityMixin — _can_accept_program
# ---------------------------------------------------------------------------


class TestCanAcceptProgram:
    async def test_missing_behavior_key_rejects(self):
        """Program missing a required behavior key → False."""
        island = _make_island(behavior_keys=["score", "cost"])
        prog = _make_program(metrics={"score": 1.0})  # missing "cost"
        result = await IslandCompatibilityMixin._can_accept_program(island, prog)
        assert result is False

    async def test_all_keys_present_empty_cell_accepts(self):
        """All keys present and cell empty → True."""
        island = _make_island(behavior_keys=["score"], elite=None)
        prog = _make_program(metrics={"score": 0.5})
        result = await IslandCompatibilityMixin._can_accept_program(island, prog)
        assert result is True

    async def test_occupied_cell_accepts_if_archive_selector_true(self):
        current = _make_program(metrics={"score": 0.3})
        island = _make_island(behavior_keys=["score"], elite=current, accepts=True)
        prog = _make_program(metrics={"score": 0.7})
        result = await IslandCompatibilityMixin._can_accept_program(island, prog)
        assert result is True

    async def test_occupied_cell_rejects_if_archive_selector_false(self):
        current = _make_program(metrics={"score": 0.9})
        island = _make_island(behavior_keys=["score"], elite=current, accepts=False)
        prog = _make_program(metrics={"score": 0.1})
        result = await IslandCompatibilityMixin._can_accept_program(island, prog)
        assert result is False

    async def test_dynamic_behavior_space_calls_check_and_expand(self):
        """DynamicBehaviorSpace.check_and_expand must be called before get_cell."""
        island = _make_island(behavior_keys=["score"], is_dynamic=True)
        prog = _make_program(metrics={"score": 0.5})
        await IslandCompatibilityMixin._can_accept_program(island, prog)
        island.config.behavior_space.check_and_expand.assert_called_once_with(
            prog.metrics
        )

    async def test_no_behavior_keys_required_always_accepts(self):
        """Island with zero behavior keys always accepts (empty set ⊆ any set)."""
        island = _make_island(behavior_keys=[])  # explicitly empty, not the default
        prog = _make_program(metrics={"score": 0.5})
        result = await IslandCompatibilityMixin._can_accept_program(island, prog)
        assert result is True


# ---------------------------------------------------------------------------
# IslandCompatibilityMixin — _filter_accepting_islands
# ---------------------------------------------------------------------------


class TestFilterAcceptingIslands:
    async def test_empty_list_returns_empty(self):
        prog = _make_program(metrics={"score": 1.0})
        result = await IslandCompatibilityMixin._filter_accepting_islands([], prog)
        assert result == []

    async def test_all_accept(self):
        islands = [
            _make_island("i0", behavior_keys=["score"], accepts=True),
            _make_island("i1", behavior_keys=["score"], accepts=True),
        ]
        prog = _make_program(metrics={"score": 0.5})
        result = await IslandCompatibilityMixin._filter_accepting_islands(islands, prog)
        assert len(result) == 2

    async def test_partial_accept(self):
        islands = [
            _make_island("i0", behavior_keys=["score"], accepts=True),
            _make_island("i1", behavior_keys=["score"], accepts=False),
        ]
        prog = _make_program(metrics={"score": 0.5})
        result = await IslandCompatibilityMixin._filter_accepting_islands(islands, prog)
        assert len(result) == 1
        assert result[0].config.island_id == "i0"

    async def test_none_accept(self):
        islands = [
            _make_island("i0", behavior_keys=["score"], accepts=False),
            _make_island("i1", behavior_keys=["score"], accepts=False),
        ]
        prog = _make_program(metrics={"score": 0.5})
        result = await IslandCompatibilityMixin._filter_accepting_islands(islands, prog)
        assert result == []

    async def test_missing_key_always_rejects(self):
        island = _make_island("i0", behavior_keys=["score", "cost"])
        prog = _make_program(metrics={"score": 1.0})
        result = await IslandCompatibilityMixin._filter_accepting_islands(
            [island], prog
        )
        assert result == []


# ---------------------------------------------------------------------------
# WeightedIslandSelector
# ---------------------------------------------------------------------------


class TestWeightedIslandSelector:
    async def test_returns_none_for_empty_islands(self):
        selector = WeightedIslandSelector()
        prog = _make_program(metrics={"score": 1.0})
        assert await selector.select_island(prog, []) is None

    async def test_returns_none_when_no_island_accepts(self):
        island = _make_island(behavior_keys=["score"], accepts=False)
        selector = WeightedIslandSelector()
        prog = _make_program(metrics={"score": 0.5})
        assert await selector.select_island(prog, [island]) is None

    async def test_single_accepting_island_always_returned(self):
        island = _make_island(behavior_keys=["score"], size=5, accepts=True)
        selector = WeightedIslandSelector()
        prog = _make_program(metrics={"score": 0.5})
        result = await selector.select_island(prog, [island])
        assert result is island

    async def test_smaller_island_favored_by_weight(self):
        """Smaller island (size=0 → weight=1.0) > larger island (size=100 → weight≈0.01)."""
        small = _make_island("small", behavior_keys=["score"], size=0, accepts=True)
        large = _make_island("large", behavior_keys=["score"], size=100, accepts=True)
        selector = WeightedIslandSelector()
        prog = _make_program(metrics={"score": 0.5})

        counts: dict[str, int] = {"small": 0, "large": 0}
        for _ in range(300):
            result = await selector.select_island(prog, [small, large])
            counts[result.config.island_id] += 1

        assert counts["small"] > counts["large"] * 5, (
            f"small={counts['small']}, large={counts['large']}: "
            "inverse-size weighting should heavily favour the smaller island"
        )

    async def test_result_is_in_accepting_islands(self):
        islands = [
            _make_island("i0", behavior_keys=["score"], size=1, accepts=True),
            _make_island("i1", behavior_keys=["score"], size=2, accepts=True),
        ]
        selector = WeightedIslandSelector()
        prog = _make_program(metrics={"score": 0.5})
        result = await selector.select_island(prog, islands)
        assert result in islands


# ---------------------------------------------------------------------------
# RoundRobinIslandSelector
# ---------------------------------------------------------------------------


class TestRoundRobinIslandSelector:
    async def test_returns_none_for_empty_islands(self):
        selector = RoundRobinIslandSelector()
        prog = _make_program(metrics={"score": 1.0})
        assert await selector.select_island(prog, []) is None

    async def test_returns_none_when_no_island_accepts(self):
        island = _make_island(behavior_keys=["score"], accepts=False)
        selector = RoundRobinIslandSelector()
        prog = _make_program(metrics={"score": 0.5})
        assert await selector.select_island(prog, [island]) is None

    async def test_cycles_through_two_islands(self):
        i0 = _make_island("i0", behavior_keys=["score"], accepts=True)
        i1 = _make_island("i1", behavior_keys=["score"], accepts=True)
        selector = RoundRobinIslandSelector()
        prog = _make_program(metrics={"score": 0.5})

        r1 = await selector.select_island(prog, [i0, i1])
        r2 = await selector.select_island(prog, [i0, i1])
        r3 = await selector.select_island(prog, [i0, i1])

        # Should cycle i0→i1→i0 (or i1→i0→i1)
        assert r1 is not r2
        assert r3 is r1

    async def test_single_island_always_returned(self):
        island = _make_island(behavior_keys=["score"], accepts=True)
        selector = RoundRobinIslandSelector()
        prog = _make_program(metrics={"score": 0.5})
        for _ in range(4):
            assert await selector.select_island(prog, [island]) is island

    async def test_index_increments_modulo_accepting(self):
        """Index cycles only among accepting islands, not all passed."""
        i0 = _make_island("i0", behavior_keys=["score"], accepts=True)
        i1 = _make_island("i1", behavior_keys=["score"], accepts=True)
        i2 = _make_island("i2", behavior_keys=["score"], accepts=True)
        selector = RoundRobinIslandSelector()
        prog = _make_program(metrics={"score": 0.5})

        seen = set()
        for _ in range(9):
            r = await selector.select_island(prog, [i0, i1, i2])
            seen.add(r.config.island_id)

        assert seen == {"i0", "i1", "i2"}


# ---------------------------------------------------------------------------
# RandomIslandSelector
# ---------------------------------------------------------------------------


class TestRandomIslandSelector:
    async def test_returns_none_for_empty_islands(self):
        selector = RandomIslandSelector()
        prog = _make_program(metrics={"score": 1.0})
        assert await selector.select_island(prog, []) is None

    async def test_returns_none_when_no_island_accepts(self):
        island = _make_island(behavior_keys=["score"], accepts=False)
        selector = RandomIslandSelector()
        prog = _make_program(metrics={"score": 0.5})
        assert await selector.select_island(prog, [island]) is None

    async def test_returns_from_accepting_set(self):
        i0 = _make_island("i0", behavior_keys=["score"], accepts=True)
        i1 = _make_island("i1", behavior_keys=["score"], accepts=True)
        selector = RandomIslandSelector()
        prog = _make_program(metrics={"score": 0.5})
        result = await selector.select_island(prog, [i0, i1])
        assert result in [i0, i1]

    async def test_rejecting_island_never_returned(self):
        accepting = _make_island("good", behavior_keys=["score"], accepts=True)
        rejecting = _make_island("bad", behavior_keys=["score"], accepts=False)
        selector = RandomIslandSelector()
        prog = _make_program(metrics={"score": 0.5})

        for _ in range(20):
            result = await selector.select_island(prog, [accepting, rejecting])
            assert result is accepting

    async def test_all_islands_reachable_over_many_calls(self):
        islands = [
            _make_island(f"i{i}", behavior_keys=["score"], accepts=True)
            for i in range(4)
        ]
        selector = RandomIslandSelector()
        prog = _make_program(metrics={"score": 0.5})

        seen = set()
        for _ in range(200):
            r = await selector.select_island(prog, islands)
            seen.add(r.config.island_id)

        assert len(seen) == 4, "RandomIslandSelector should visit all islands"
