"""Tests for gigaevo/programs/stages/ancestry_selector.py

Covers AncestrySelector.__init__ and AncestrySelector.select() for both
"random" and "best_fitness" strategies, including edge cases such as:
- max_selected clamping
- missing fitness key skipping
- unknown strategy raising ValueError
"""

from __future__ import annotations

import pytest

from gigaevo.programs.metrics.context import MetricsContext, MetricSpec
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState
from gigaevo.programs.stages.ancestry_selector import AncestrySelector

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_metrics_context(*, higher_is_better: bool = True) -> MetricsContext:
    """Build a minimal MetricsContext with a single primary metric."""
    spec = MetricSpec(
        description="Test fitness",
        is_primary=True,
        higher_is_better=higher_is_better,
    )
    return MetricsContext(specs={"fitness": spec})


def _make_program(fitness: float | None = None) -> Program:
    """Create a Program and optionally add a 'fitness' metric."""
    p = Program(code="def solve(): return 1", state=ProgramState.DONE)
    if fitness is not None:
        p.add_metrics({"fitness": fitness})
    return p


# ---------------------------------------------------------------------------
# __init__ — constructor tests
# ---------------------------------------------------------------------------


class TestAncestorySelectorInit:
    def test_default_strategy_is_random(self) -> None:
        ctx = _make_metrics_context()
        sel = AncestrySelector(ctx)
        assert sel.strategy == "random"

    def test_default_max_selected_is_one(self) -> None:
        ctx = _make_metrics_context()
        sel = AncestrySelector(ctx)
        assert sel.max_selected == 1

    def test_max_selected_clamped_to_1_when_zero(self) -> None:
        """max_selected=0 must be clamped to 1 (always select at least one)."""
        ctx = _make_metrics_context()
        sel = AncestrySelector(ctx, max_selected=0)
        assert sel.max_selected == 1

    def test_max_selected_clamped_to_1_when_negative(self) -> None:
        """Negative max_selected must be clamped to 1."""
        ctx = _make_metrics_context()
        sel = AncestrySelector(ctx, max_selected=-5)
        assert sel.max_selected == 1

    def test_max_selected_stored_as_int(self) -> None:
        """max_selected is stored as int even when supplied as float."""
        ctx = _make_metrics_context()
        sel = AncestrySelector(ctx, max_selected=3.9)  # type: ignore[arg-type]
        assert sel.max_selected == 3
        assert isinstance(sel.max_selected, int)

    def test_custom_strategy_stored(self) -> None:
        ctx = _make_metrics_context()
        sel = AncestrySelector(ctx, strategy="best_fitness", max_selected=2)
        assert sel.strategy == "best_fitness"
        assert sel.max_selected == 2


# ---------------------------------------------------------------------------
# select() — "best_fitness" strategy
# ---------------------------------------------------------------------------


class TestBestFitnessStrategy:
    async def test_best_fitness_selects_highest_when_higher_is_better(self) -> None:
        """With higher_is_better=True the program with the largest fitness is returned."""
        ctx = _make_metrics_context(higher_is_better=True)
        sel = AncestrySelector(ctx, strategy="best_fitness", max_selected=1)

        low = _make_program(fitness=1.0)
        high = _make_program(fitness=9.0)
        mid = _make_program(fitness=5.0)

        result = await sel.select([low, high, mid])
        assert len(result) == 1
        assert result[0] is high

    async def test_best_fitness_selects_lowest_when_lower_is_better(self) -> None:
        """With higher_is_better=False the program with the smallest fitness is returned."""
        ctx = _make_metrics_context(higher_is_better=False)
        sel = AncestrySelector(ctx, strategy="best_fitness", max_selected=1)

        low = _make_program(fitness=1.0)
        high = _make_program(fitness=9.0)
        mid = _make_program(fitness=5.0)

        result = await sel.select([low, high, mid])
        assert len(result) == 1
        assert result[0] is low

    async def test_best_fitness_skips_program_missing_key(self) -> None:
        """Programs that lack the primary fitness key must be silently skipped."""
        ctx = _make_metrics_context(higher_is_better=True)
        sel = AncestrySelector(ctx, strategy="best_fitness", max_selected=1)

        no_fitness = _make_program(fitness=None)  # missing key
        has_fitness = _make_program(fitness=7.0)

        result = await sel.select([no_fitness, has_fitness])
        assert result == [has_fitness]

    async def test_best_fitness_all_missing_key_returns_empty(self) -> None:
        """When no program has the fitness key the result must be an empty list."""
        ctx = _make_metrics_context(higher_is_better=True)
        sel = AncestrySelector(ctx, strategy="best_fitness", max_selected=2)

        programs = [_make_program(fitness=None) for _ in range(3)]
        result = await sel.select(programs)
        assert result == []

    async def test_best_fitness_respects_max_selected(self) -> None:
        """Returns at most max_selected programs when list is longer."""
        ctx = _make_metrics_context(higher_is_better=True)
        sel = AncestrySelector(ctx, strategy="best_fitness", max_selected=2)

        programs = [_make_program(fitness=float(i)) for i in range(5)]
        result = await sel.select(programs)
        assert len(result) == 2
        # The two highest fitness values are 4.0 and 3.0
        result_fitnesses = sorted([p.metrics["fitness"] for p in result], reverse=True)
        assert result_fitnesses == [4.0, 3.0]

    async def test_best_fitness_max_selected_larger_than_list(self) -> None:
        """When max_selected exceeds len(programs), all eligible programs returned."""
        ctx = _make_metrics_context(higher_is_better=True)
        sel = AncestrySelector(ctx, strategy="best_fitness", max_selected=10)

        programs = [_make_program(fitness=float(i)) for i in range(3)]
        result = await sel.select(programs)
        assert len(result) == 3

    async def test_best_fitness_empty_list_returns_empty(self) -> None:
        ctx = _make_metrics_context(higher_is_better=True)
        sel = AncestrySelector(ctx, strategy="best_fitness", max_selected=1)
        result = await sel.select([])
        assert result == []


# ---------------------------------------------------------------------------
# select() — "random" strategy
# ---------------------------------------------------------------------------


class TestRandomStrategy:
    async def test_random_returns_correct_count_when_enough_programs(self) -> None:
        """random strategy returns exactly max_selected programs when list is large enough."""
        ctx = _make_metrics_context()
        sel = AncestrySelector(ctx, strategy="random", max_selected=3)

        programs = [_make_program() for _ in range(10)]
        result = await sel.select(programs)
        assert len(result) == 3

    async def test_random_returns_all_when_fewer_than_max(self) -> None:
        """random strategy returns min(max_selected, len(programs)) programs."""
        ctx = _make_metrics_context()
        sel = AncestrySelector(ctx, strategy="random", max_selected=10)

        programs = [_make_program() for _ in range(4)]
        result = await sel.select(programs)
        assert len(result) == 4

    async def test_random_result_is_subset_of_input(self) -> None:
        """All returned programs must come from the original list."""
        ctx = _make_metrics_context()
        sel = AncestrySelector(ctx, strategy="random", max_selected=3)

        programs = [_make_program() for _ in range(6)]
        result = await sel.select(programs)
        for p in result:
            assert p in programs

    async def test_random_returns_single_when_max_selected_is_1(self) -> None:
        ctx = _make_metrics_context()
        sel = AncestrySelector(ctx, strategy="random", max_selected=1)

        programs = [_make_program() for _ in range(5)]
        result = await sel.select(programs)
        assert len(result) == 1
        assert result[0] in programs

    async def test_random_empty_list_returns_empty(self) -> None:
        ctx = _make_metrics_context()
        sel = AncestrySelector(ctx, strategy="random", max_selected=1)
        result = await sel.select([])
        assert result == []

    async def test_random_no_duplicates_in_result(self) -> None:
        """random.sample does not repeat elements."""
        ctx = _make_metrics_context()
        sel = AncestrySelector(ctx, strategy="random", max_selected=5)

        programs = [_make_program() for _ in range(5)]
        result = await sel.select(programs)
        # IDs must be unique
        assert len({p.id for p in result}) == len(result)


# ---------------------------------------------------------------------------
# select() — unknown strategy
# ---------------------------------------------------------------------------


class TestUnknownStrategy:
    async def test_unknown_strategy_raises_value_error(self) -> None:
        """Passing an invalid strategy at construction then calling select raises ValueError."""
        ctx = _make_metrics_context()
        # Bypass Literal type-checking by creating object directly
        sel = AncestrySelector.__new__(AncestrySelector)
        sel.metrics_context = ctx
        sel.strategy = "unsupported_strategy"  # type: ignore[assignment]
        sel.max_selected = 1

        programs = [_make_program()]
        with pytest.raises(ValueError, match="Unknown program selection strategy"):
            await sel.select(programs)
