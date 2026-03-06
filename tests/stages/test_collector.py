"""Tests for collector stages: helper functions and EvolutionaryStatisticsCollector,
ProgramIdsCollector, DescendantProgramIds, AncestorProgramIds."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from gigaevo.programs.metrics.context import MetricsContext, MetricSpec
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState
from gigaevo.programs.stages.ancestry_selector import AncestrySelector
from gigaevo.programs.stages.collector import (
    AncestorProgramIds,
    DescendantProgramIds,
    EvolutionaryStatisticsCollector,
    ProgramIdsCollector,
    _compute_fitness_stats_all_metrics,
    _compute_main_metric_stats,
    _compute_num_children_stats,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ctx(higher_is_better: bool = True) -> MetricsContext:
    return MetricsContext(
        specs={
            "score": MetricSpec(
                description="main score",
                is_primary=True,
                higher_is_better=higher_is_better,
                lower_bound=0.0,
                upper_bound=100.0,
            ),
            "is_valid": MetricSpec(
                description="validity",
                is_primary=False,
                higher_is_better=True,
                lower_bound=0.0,
                upper_bound=1.0,
            ),
        }
    )


def _prog(
    score: float = 50.0,
    is_valid: float = 1.0,
    generation: int = 1,
) -> Program:
    p = Program(code="def solve(): return 42", state=ProgramState.DONE)
    p.add_metrics({"score": score, "is_valid": is_valid})
    p.lineage.generation = generation
    return p


# ---------------------------------------------------------------------------
# TestComputeFitnessStatsAllMetrics
# ---------------------------------------------------------------------------


class TestComputeFitnessStatsAllMetrics:
    def test_empty_list(self):
        """Empty program list → empty dicts, 0.0 valid_rate."""
        best, worst, avg, vr = _compute_fitness_stats_all_metrics([], _ctx())
        assert best == {}
        assert worst == {}
        assert avg == {}
        assert vr == 0.0

    def test_single_valid_program(self):
        """One valid program → best=worst=avg=its fitness."""
        programs = [_prog(score=80.0)]
        best, worst, avg, vr = _compute_fitness_stats_all_metrics(programs, _ctx())
        assert best["score"] == 80.0
        assert worst["score"] == 80.0
        assert avg["score"] == 80.0
        assert vr == 1.0

    def test_multiple_valid_programs(self):
        """Multiple valid programs → correct stats."""
        programs = [_prog(score=60.0), _prog(score=80.0), _prog(score=100.0)]
        best, worst, avg, vr = _compute_fitness_stats_all_metrics(programs, _ctx())
        assert best["score"] == 100.0
        assert worst["score"] == 60.0
        assert avg["score"] == pytest.approx(80.0)
        assert vr == 1.0

    def test_no_valid_programs(self):
        """All invalid → empty dicts, correct valid_rate."""
        programs = [_prog(score=50.0, is_valid=0.0), _prog(score=60.0, is_valid=0.0)]
        best, worst, avg, vr = _compute_fitness_stats_all_metrics(programs, _ctx())
        assert best == {}
        assert worst == {}
        assert avg == {}
        assert vr == 0.0

    def test_mixed_valid_invalid(self):
        """Mix of valid/invalid → stats computed only from valid."""
        programs = [
            _prog(score=90.0, is_valid=1.0),
            _prog(score=10.0, is_valid=0.0),
            _prog(score=50.0, is_valid=1.0),
        ]
        best, worst, avg, vr = _compute_fitness_stats_all_metrics(programs, _ctx())
        assert best["score"] == 90.0
        assert worst["score"] == 50.0
        assert avg["score"] == pytest.approx(70.0)
        assert vr == pytest.approx(2 / 3)

    def test_higher_is_better_false(self):
        """With higher_is_better=False, best=min, worst=max."""
        ctx = _ctx(higher_is_better=False)
        programs = [_prog(score=20.0), _prog(score=80.0)]
        best, worst, avg, vr = _compute_fitness_stats_all_metrics(programs, ctx)
        assert best["score"] == 20.0  # lower is better → min is best
        assert worst["score"] == 80.0

    def test_missing_metric_key_skipped(self):
        """Programs missing a metric key → that key not in result."""
        ctx = MetricsContext(
            specs={
                "score": MetricSpec(
                    description="main",
                    is_primary=True,
                    higher_is_better=True,
                    lower_bound=0.0,
                    upper_bound=100.0,
                ),
                "is_valid": MetricSpec(
                    description="validity",
                    is_primary=False,
                    higher_is_better=True,
                    lower_bound=0.0,
                    upper_bound=1.0,
                ),
                "rare_metric": MetricSpec(
                    description="rarely present",
                    is_primary=False,
                    higher_is_better=True,
                    lower_bound=0.0,
                    upper_bound=10.0,
                ),
            }
        )
        programs = [_prog(score=50.0)]  # has score and is_valid, not rare_metric
        best, worst, avg, vr = _compute_fitness_stats_all_metrics(programs, ctx)
        assert "score" in best
        assert "rare_metric" not in best


# ---------------------------------------------------------------------------
# TestComputeNumChildrenStats
# ---------------------------------------------------------------------------


class TestComputeNumChildrenStats:
    def test_empty_list(self):
        """Empty → (0.0, 0, 0)."""
        avg, mx, count = _compute_num_children_stats([])
        assert avg == 0.0
        assert mx == 0
        assert count == 0

    def test_programs_with_no_children(self):
        """Programs with 0 children each."""
        programs = [_prog(), _prog()]
        avg, mx, count = _compute_num_children_stats(programs)
        assert avg == 0.0
        assert mx == 0
        assert count == 2

    def test_programs_with_children(self):
        """Programs with varying child counts."""
        p1 = _prog()
        p2 = _prog()
        p1.lineage.children = ["c1", "c2"]
        p2.lineage.children = ["c3"]
        programs = [p1, p2]
        avg, mx, count = _compute_num_children_stats(programs)
        assert avg == pytest.approx(1.5)
        assert mx == 2
        assert count == 2


# ---------------------------------------------------------------------------
# TestComputeMainMetricStats
# ---------------------------------------------------------------------------


class TestComputeMainMetricStats:
    def test_empty_list(self):
        """Empty → all None, 0 counts."""
        gm = _compute_main_metric_stats([], "score", True)
        assert gm.best is None
        assert gm.worst is None
        assert gm.average is None
        assert gm.valid_rate == 0.0
        assert gm.program_count == 0

    def test_no_valid_programs(self):
        """All invalid → None fitness, but valid_rate computed."""
        programs = [_prog(score=50.0, is_valid=0.0)]
        gm = _compute_main_metric_stats(programs, "score", True)
        assert gm.best is None
        assert gm.worst is None
        assert gm.average is None
        assert gm.valid_rate == 0.0
        assert gm.program_count == 1

    def test_higher_is_better_true(self):
        """higher_is_better=True: best=max, worst=min."""
        programs = [_prog(score=30.0), _prog(score=70.0)]
        gm = _compute_main_metric_stats(programs, "score", True)
        assert gm.best == 70.0
        assert gm.worst == 30.0
        assert gm.average == pytest.approx(50.0)

    def test_higher_is_better_false(self):
        """higher_is_better=False: best=min, worst=max."""
        programs = [_prog(score=30.0), _prog(score=70.0)]
        gm = _compute_main_metric_stats(programs, "score", False)
        assert gm.best == 30.0
        assert gm.worst == 70.0

    def test_valid_programs_missing_metric(self):
        """Valid program that doesn't have the metric → skipped in stats."""
        p = Program(code="def solve(): return 1", state=ProgramState.DONE)
        p.add_metrics({"is_valid": 1.0})  # no "score" key
        gm = _compute_main_metric_stats([p], "score", True)
        assert gm.best is None
        assert gm.valid_rate == 1.0  # program is valid


# ---------------------------------------------------------------------------
# TestProgramIdsCollector
# ---------------------------------------------------------------------------


class TestProgramIdsCollector:
    async def test_returns_all_program_ids(self):
        """Collector returns IDs of all programs from _collect_programs."""
        storage = AsyncMock()
        p1, p2 = _prog(), _prog()

        class TestCollector(ProgramIdsCollector):
            async def _collect_programs(self, program):
                return [p1, p2]

        stage = TestCollector(storage=storage, timeout=5.0)
        stage.attach_inputs({})
        result = await stage.execute(_prog())

        assert result.status.name == "COMPLETED"
        assert set(result.output.items) == {p1.id, p2.id}

    async def test_empty_returns_empty_list(self):
        """No related programs → empty StringList."""
        storage = AsyncMock()

        class EmptyCollector(ProgramIdsCollector):
            async def _collect_programs(self, program):
                return []

        stage = EmptyCollector(storage=storage, timeout=5.0)
        stage.attach_inputs({})
        result = await stage.execute(_prog())

        assert result.output.items == []


# ---------------------------------------------------------------------------
# TestDescendantProgramIds
# ---------------------------------------------------------------------------


class TestDescendantProgramIds:
    async def test_returns_descendant_ids(self):
        """Returns IDs of selected descendant programs."""
        storage = AsyncMock()
        child1, child2 = _prog(), _prog()
        storage.mget.return_value = [child1, child2]

        selector = AsyncMock(spec=AncestrySelector)
        selector.select.return_value = [child1, child2]

        prog = _prog()
        prog.lineage.children = [child1.id, child2.id]

        stage = DescendantProgramIds(storage=storage, selector=selector, timeout=5.0)
        stage.attach_inputs({})
        result = await stage.execute(prog)

        assert result.status.name == "COMPLETED"
        assert set(result.output.items) == {child1.id, child2.id}

    async def test_no_children_returns_empty(self):
        """Program with no children → empty list."""
        storage = AsyncMock()
        storage.mget.return_value = []

        selector = AsyncMock(spec=AncestrySelector)
        selector.select.return_value = []

        prog = _prog()
        prog.lineage.children = []

        stage = DescendantProgramIds(storage=storage, selector=selector, timeout=5.0)
        stage.attach_inputs({})
        result = await stage.execute(prog)

        assert result.output.items == []


# ---------------------------------------------------------------------------
# TestAncestorProgramIds
# ---------------------------------------------------------------------------


class TestAncestorProgramIds:
    async def test_returns_ancestor_ids(self):
        """Returns IDs of selected ancestor programs."""
        storage = AsyncMock()
        parent = _prog()
        storage.mget.return_value = [parent]

        selector = AsyncMock(spec=AncestrySelector)
        selector.select.return_value = [parent]

        prog = _prog()
        prog.lineage.parents = [parent.id]

        stage = AncestorProgramIds(storage=storage, selector=selector, timeout=5.0)
        stage.attach_inputs({})
        result = await stage.execute(prog)

        assert result.status.name == "COMPLETED"
        assert result.output.items == [parent.id]

    async def test_no_parents_returns_empty(self):
        """Program with no parents → empty list."""
        storage = AsyncMock()
        storage.mget.return_value = []

        selector = AsyncMock(spec=AncestrySelector)
        selector.select.return_value = []

        prog = _prog()
        prog.lineage.parents = []

        stage = AncestorProgramIds(storage=storage, selector=selector, timeout=5.0)
        stage.attach_inputs({})
        result = await stage.execute(prog)

        assert result.output.items == []

    async def test_deleted_parents_filtered(self):
        """storage.mget returns fewer programs than IDs → selector handles it."""
        storage = AsyncMock()
        parent = _prog()
        # One parent exists, one was deleted
        storage.mget.return_value = [parent]

        selector = AsyncMock(spec=AncestrySelector)
        selector.select.return_value = [parent]

        prog = _prog()
        prog.lineage.parents = [parent.id, "deleted-id"]

        stage = AncestorProgramIds(storage=storage, selector=selector, timeout=5.0)
        stage.attach_inputs({})
        result = await stage.execute(prog)

        assert result.output.items == [parent.id]


# ---------------------------------------------------------------------------
# TestEvolutionaryStatisticsCollector
# ---------------------------------------------------------------------------


class TestEvolutionaryStatisticsCollector:
    async def test_basic_stats(self):
        """Collector returns correct global stats for simple case."""
        storage = AsyncMock()
        p1 = _prog(score=60.0, generation=0)
        p2 = _prog(score=80.0, generation=0)
        p3 = _prog(score=40.0, generation=1)
        storage.get_all.return_value = [p1, p2, p3]
        storage.mget.return_value = []  # no ancestors/descendants

        ctx = _ctx()
        stage = EvolutionaryStatisticsCollector(
            storage=storage, metrics_context=ctx, timeout=5.0
        )
        stage.attach_inputs({})
        result = await stage.execute(p1)

        assert result.status.name == "COMPLETED"
        stats = result.output
        assert stats.best_fitness["score"] == 80.0
        assert stats.worst_fitness["score"] == 40.0
        assert stats.average_fitness["score"] == pytest.approx(60.0)
        assert stats.valid_rate == 1.0
        assert stats.total_program_count == 3

    async def test_generation_stats(self):
        """Generation-specific stats computed correctly."""
        storage = AsyncMock()
        p1 = _prog(score=60.0, generation=1)
        p2 = _prog(score=80.0, generation=1)
        p3 = _prog(score=40.0, generation=2)
        storage.get_all.return_value = [p1, p2, p3]
        storage.mget.return_value = []

        ctx = _ctx()
        stage = EvolutionaryStatisticsCollector(
            storage=storage, metrics_context=ctx, timeout=5.0
        )
        stage.attach_inputs({})
        result = await stage.execute(p1)

        stats = result.output
        # p1 is gen 1, gen 1 has p1(60) and p2(80)
        assert stats.best_fitness_in_generation["score"] == 80.0
        assert stats.worst_fitness_in_generation["score"] == 60.0

    async def test_ancestor_stats(self):
        """Ancestor statistics from mget."""
        storage = AsyncMock()
        parent = _prog(score=90.0, generation=1)
        child = _prog(score=50.0, generation=2)
        child.lineage.parents = [parent.id]
        child.lineage.children = []

        storage.get_all.return_value = [parent, child]
        # First mget call is for ancestors (parents), second is for descendants (children)
        storage.mget.side_effect = [[parent], []]

        ctx = _ctx()
        stage = EvolutionaryStatisticsCollector(
            storage=storage, metrics_context=ctx, timeout=5.0
        )
        stage.attach_inputs({})
        result = await stage.execute(child)

        stats = result.output
        assert stats.ancestor_count == 1
        assert stats.best_fitness_in_ancestors["score"] == 90.0
        assert stats.descendant_count == 0

    async def test_generation_history(self):
        """Generation history keyed by generation number."""
        storage = AsyncMock()
        p1 = _prog(score=60.0, generation=1)
        p2 = _prog(score=80.0, generation=2)
        p3 = _prog(score=40.0, generation=2)
        storage.get_all.return_value = [p1, p2, p3]
        storage.mget.return_value = []

        ctx = _ctx()
        stage = EvolutionaryStatisticsCollector(
            storage=storage, metrics_context=ctx, timeout=5.0
        )
        stage.attach_inputs({})
        result = await stage.execute(p1)

        gh = result.output.generation_history
        assert 1 in gh
        assert 2 in gh
        assert gh[1].best == 60.0
        assert gh[2].best == 80.0
        assert gh[2].worst == 40.0

    async def test_no_programs(self):
        """Empty storage → empty stats."""
        storage = AsyncMock()
        storage.get_all.return_value = []
        storage.mget.return_value = []

        # Need a program to execute on even though storage is "empty"
        prog = _prog(score=50.0, generation=0)

        ctx = _ctx()
        stage = EvolutionaryStatisticsCollector(
            storage=storage, metrics_context=ctx, timeout=5.0
        )
        stage.attach_inputs({})
        result = await stage.execute(prog)

        stats = result.output
        assert stats.total_program_count == 0
        assert stats.valid_rate == 0.0

    async def test_iteration_stats_when_present(self):
        """Iteration stats computed when program has iteration metadata."""
        storage = AsyncMock()
        p1 = _prog(score=60.0, generation=0)
        p1.set_metadata("iteration", 1)
        p2 = _prog(score=80.0, generation=0)
        p2.set_metadata("iteration", 1)
        p3 = _prog(score=40.0, generation=0)
        p3.set_metadata("iteration", 2)
        storage.get_all.return_value = [p1, p2, p3]
        storage.mget.return_value = []

        ctx = _ctx()
        stage = EvolutionaryStatisticsCollector(
            storage=storage, metrics_context=ctx, timeout=5.0
        )
        stage.attach_inputs({})
        result = await stage.execute(p1)

        stats = result.output
        assert stats.iteration == 1
        # Iteration stats should be for iteration=1 only (p1, p2)
        assert stats.best_fitness_in_iteration["score"] == 80.0
        assert stats.worst_fitness_in_iteration["score"] == 60.0

    async def test_iteration_stats_none_when_no_iteration(self):
        """No iteration metadata → iteration stats are None."""
        storage = AsyncMock()
        p1 = _prog(score=60.0, generation=0)
        storage.get_all.return_value = [p1]
        storage.mget.return_value = []

        ctx = _ctx()
        stage = EvolutionaryStatisticsCollector(
            storage=storage, metrics_context=ctx, timeout=5.0
        )
        stage.attach_inputs({})
        result = await stage.execute(p1)

        stats = result.output
        assert stats.iteration is None
        assert stats.best_fitness_in_iteration is None
        assert stats.valid_rate_in_iteration is None
