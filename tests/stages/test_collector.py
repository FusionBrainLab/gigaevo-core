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
    _compute_num_children_stats,
    _max_invalid_streak,
    _trend_from_thirds,
)


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
    iteration: int = 0,
) -> Program:
    p = Program(code="def solve(): return 42", state=ProgramState.DONE)
    p.add_metrics({"score": score, "is_valid": is_valid})
    p.lineage.generation = generation
    p.iteration = iteration
    return p


class TestComputeFitnessStatsAllMetrics:
    def test_empty_list(self):
        best, worst, avg, vr = _compute_fitness_stats_all_metrics([], _ctx())
        assert best == {}
        assert worst == {}
        assert avg == {}
        assert vr == 0.0

    def test_single_valid_program(self):
        programs = [_prog(score=80.0)]
        best, worst, avg, vr = _compute_fitness_stats_all_metrics(programs, _ctx())
        assert best["score"] == 80.0
        assert worst["score"] == 80.0
        assert avg["score"] == 80.0
        assert vr == 1.0

    def test_multiple_valid_programs(self):
        programs = [_prog(score=60.0), _prog(score=80.0), _prog(score=100.0)]
        best, worst, avg, vr = _compute_fitness_stats_all_metrics(programs, _ctx())
        assert best["score"] == 100.0
        assert worst["score"] == 60.0
        assert avg["score"] == pytest.approx(80.0)
        assert vr == 1.0

    def test_no_valid_programs(self):
        programs = [_prog(score=50.0, is_valid=0.0), _prog(score=60.0, is_valid=0.0)]
        best, worst, avg, vr = _compute_fitness_stats_all_metrics(programs, _ctx())
        assert best == {}
        assert worst == {}
        assert avg == {}
        assert vr == 0.0

    def test_mixed_valid_invalid(self):
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
        ctx = _ctx(higher_is_better=False)
        programs = [_prog(score=20.0), _prog(score=80.0)]
        best, worst, _, _ = _compute_fitness_stats_all_metrics(programs, ctx)
        assert best["score"] == 20.0
        assert worst["score"] == 80.0


class TestComputeNumChildrenStats:
    def test_empty_list(self):
        assert _compute_num_children_stats([]) == (0.0, 0, 0)

    def test_programs_with_children(self):
        p1, p2 = _prog(), _prog()
        p1.lineage.children = ["c1", "c2"]
        p2.lineage.children = ["c3"]
        avg, mx, count = _compute_num_children_stats([p1, p2])
        assert avg == pytest.approx(1.5)
        assert mx == 2
        assert count == 2


class TestMaxInvalidStreak:
    def test_all_valid(self):
        progs = [_prog(is_valid=1.0) for _ in range(5)]
        assert _max_invalid_streak(progs) == 0

    def test_all_invalid(self):
        progs = [_prog(is_valid=0.0) for _ in range(4)]
        assert _max_invalid_streak(progs) == 4

    def test_mixed_with_break(self):
        progs = [
            _prog(is_valid=0.0),
            _prog(is_valid=0.0),
            _prog(is_valid=1.0),
            _prog(is_valid=0.0),
            _prog(is_valid=0.0),
            _prog(is_valid=0.0),
            _prog(is_valid=1.0),
        ]
        assert _max_invalid_streak(progs) == 3


class TestTrendFromThirds:
    def test_too_few_points(self):
        label, thirds = _trend_from_thirds([1.0, 2.0, 3.0], higher_is_better=True)
        assert label == "flat"
        assert thirds == (None, None, None)

    def test_rising_higher_is_better(self):
        fits = [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]
        label, _ = _trend_from_thirds(fits, higher_is_better=True)
        assert label == "rising"

    def test_falling_higher_is_better(self):
        fits = [3.0, 3.0, 2.0, 2.0, 1.0, 1.0]
        label, _ = _trend_from_thirds(fits, higher_is_better=True)
        assert label == "falling"

    def test_flat_when_signal_below_mad(self):
        """A small directional move is masked when within-window dispersion (MAD)
        is comparable in magnitude — by design the MAD floor adapts to the run's
        own noise scale instead of relying on a fixed 5% ratio."""
        fits = [1.0, 1.05, 1.01, 0.95, 1.02, 1.08]
        label, _ = _trend_from_thirds(fits, higher_is_better=True)
        assert label == "flat"

    def test_rising_lower_is_better(self):
        """Lower-is-better metric going DOWN is reported as 'rising' (= improving)."""
        fits = [3.0, 3.0, 2.0, 2.0, 1.0, 1.0]
        label, _ = _trend_from_thirds(fits, higher_is_better=False)
        assert label == "rising"

    def test_falling_lower_is_better(self):
        fits = [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]
        label, _ = _trend_from_thirds(fits, higher_is_better=False)
        assert label == "falling"

    def test_mad_detects_small_signal_in_low_dispersion(self):
        """When the window has very low dispersion (small MAD), even a modest
        rising trend is detected — this is the cycle-6 regression scenario the
        MAD floor was introduced to catch (legacy 5%-of-max threshold reported
        small absolute deltas as flat at low fitness)."""
        fits = [0.001, 0.001, 0.002, 0.002, 0.003, 0.003]
        label, _ = _trend_from_thirds(fits, higher_is_better=True)
        assert label == "rising"

    def test_mad_masks_signal_in_high_dispersion(self):
        """When the window is noisy (high MAD), the same absolute trend is
        correctly identified as flat — MAD scales with the data."""
        fits = [0.0, 0.6, 0.1, 0.5, 0.2, 0.4]
        label, _ = _trend_from_thirds(fits, higher_is_better=True)
        assert label == "flat"

    def test_zero_dispersion_returns_flat(self):
        """When all window values are identical, MAD=0 but the epsilon floor
        keeps the comparison well-defined and any exact-tie returns flat."""
        fits = [0.5] * 6
        label, _ = _trend_from_thirds(fits, higher_is_better=True)
        assert label == "flat"


class TestProgramIdsCollector:
    async def test_returns_all_program_ids(self):
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
        storage = AsyncMock()

        class EmptyCollector(ProgramIdsCollector):
            async def _collect_programs(self, program):
                return []

        stage = EmptyCollector(storage=storage, timeout=5.0)
        stage.attach_inputs({})
        result = await stage.execute(_prog())
        assert result.output.items == []


class TestDescendantProgramIds:
    async def test_returns_descendant_ids(self):
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


class TestAncestorProgramIds:
    async def test_returns_ancestor_ids(self):
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


class TestEvolutionaryStatisticsCollector:
    async def test_global_stats(self):
        storage = AsyncMock()
        p1 = _prog(score=60.0, generation=0, iteration=1)
        p2 = _prog(score=80.0, generation=0, iteration=2)
        p3 = _prog(score=40.0, generation=1, iteration=3)
        storage.mget.return_value = []
        storage.snapshot.get_all.return_value = [p1, p2, p3]

        stage = EvolutionaryStatisticsCollector(
            storage=storage, metrics_context=_ctx(), timeout=5.0
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

    async def test_ancestor_stats(self):
        storage = AsyncMock()
        parent = _prog(score=90.0, generation=1, iteration=1)
        child = _prog(score=50.0, generation=2, iteration=2)
        child.lineage.parents = [parent.id]
        child.lineage.children = []

        storage.mget.side_effect = [[parent], []]
        storage.snapshot.get_all.return_value = [parent, child]

        stage = EvolutionaryStatisticsCollector(
            storage=storage, metrics_context=_ctx(), timeout=5.0
        )
        stage.attach_inputs({})
        result = await stage.execute(child)

        stats = result.output
        assert stats.ancestor_count == 1
        assert stats.best_fitness_in_ancestors["score"] == 90.0
        assert stats.descendant_count == 0

    async def test_iter_window_disabled_with_zero_radius(self):
        storage = AsyncMock()
        p1 = _prog(score=60.0, iteration=1)
        storage.mget.return_value = []
        storage.snapshot.get_all.return_value = [p1]

        stage = EvolutionaryStatisticsCollector(
            storage=storage,
            metrics_context=_ctx(),
            timeout=5.0,
            iteration_window_radius=0,
        )
        stage.attach_inputs({})
        result = await stage.execute(p1)

        stats = result.output
        assert stats.iter_window_lo is None
        assert stats.iter_window_hi is None
        assert stats.iter_window_programs == 0
        assert stats.iter_window_valid == 0
        assert stats.iter_window_best_fitness is None

    async def test_iter_window_basic_population(self):
        storage = AsyncMock()
        progs = [_prog(score=10.0 * i, iteration=i) for i in range(1, 11)]
        storage.mget.return_value = []
        storage.snapshot.get_all.return_value = progs

        stage = EvolutionaryStatisticsCollector(
            storage=storage,
            metrics_context=_ctx(),
            timeout=5.0,
            iteration_window_radius=2,
        )
        stage.attach_inputs({})
        result = await stage.execute(progs[4])

        stats = result.output
        assert stats.iter_window_lo == 3
        assert stats.iter_window_hi == 7
        assert stats.iter_window_programs == 5
        assert stats.iter_window_valid == 5
        assert stats.iter_window_best_fitness == 70.0
        assert stats.iter_window_best_iter == 7
        assert stats.iter_window_invalid_count == 0
        assert stats.iter_window_invalid_streak_max == 0

    async def test_iter_window_higher_is_better_false(self):
        storage = AsyncMock()
        progs = [_prog(score=10.0 * i, iteration=i) for i in range(1, 11)]
        storage.mget.return_value = []
        storage.snapshot.get_all.return_value = progs

        stage = EvolutionaryStatisticsCollector(
            storage=storage,
            metrics_context=_ctx(higher_is_better=False),
            timeout=5.0,
            iteration_window_radius=2,
        )
        stage.attach_inputs({})
        result = await stage.execute(progs[4])

        stats = result.output
        # window: scores 30,40,50,60,70; lower_is_better → best=30 at iter=3
        assert stats.iter_window_best_fitness == 30.0
        assert stats.iter_window_best_iter == 3

    async def test_iter_window_rank(self):
        storage = AsyncMock()
        p1 = _prog(score=10.0, iteration=1)
        p2 = _prog(score=50.0, iteration=2)
        p3 = _prog(score=30.0, iteration=3)
        storage.mget.return_value = []
        storage.snapshot.get_all.return_value = [p1, p2, p3]

        stage = EvolutionaryStatisticsCollector(
            storage=storage, metrics_context=_ctx(), timeout=5.0
        )
        stage.attach_inputs({})
        result = await stage.execute(p2)

        # p2 has the highest score → rank 1
        assert result.output.iter_window_rank == 1

        stage.attach_inputs({})
        result = await stage.execute(p3)
        # p3 has middle score → rank 2
        assert result.output.iter_window_rank == 2

    async def test_iter_window_median_before_after(self):
        storage = AsyncMock()
        # iters 1..21 with scores = iter
        progs = [_prog(score=float(i), iteration=i) for i in range(1, 22)]
        storage.mget.return_value = []
        storage.snapshot.get_all.return_value = progs

        stage = EvolutionaryStatisticsCollector(
            storage=storage,
            metrics_context=_ctx(),
            timeout=5.0,
            iteration_window_radius=10,
        )
        stage.attach_inputs({})
        focal = progs[10]  # iteration=11
        result = await stage.execute(focal)

        stats = result.output
        # Before: iters 1..10, last 10 → median of 1..10 = 5.5
        assert stats.iter_window_median_before == pytest.approx(5.5)
        # After: iters 12..21, first 10 → median of 12..21 = 16.5
        assert stats.iter_window_median_after == pytest.approx(16.5)

    async def test_iter_window_median_after_none_at_tail(self):
        storage = AsyncMock()
        progs = [_prog(score=float(i), iteration=i) for i in range(1, 6)]
        storage.mget.return_value = []
        storage.snapshot.get_all.return_value = progs

        stage = EvolutionaryStatisticsCollector(
            storage=storage,
            metrics_context=_ctx(),
            timeout=5.0,
            iteration_window_radius=10,
        )
        stage.attach_inputs({})
        result = await stage.execute(progs[-1])

        assert result.output.iter_window_median_after is None
        assert result.output.iter_window_median_before is not None

    async def test_iter_window_invalid_streak(self):
        storage = AsyncMock()
        progs = []
        for i in range(1, 8):
            # iters 3,4 invalid → streak of 2
            valid = 0.0 if i in (3, 4) else 1.0
            progs.append(_prog(score=10.0 * i, is_valid=valid, iteration=i))
        storage.mget.return_value = []
        storage.snapshot.get_all.return_value = progs

        stage = EvolutionaryStatisticsCollector(
            storage=storage,
            metrics_context=_ctx(),
            timeout=5.0,
            iteration_window_radius=10,
        )
        stage.attach_inputs({})
        result = await stage.execute(progs[0])

        stats = result.output
        assert stats.iter_window_invalid_count == 2
        assert stats.iter_window_invalid_streak_max == 2

    async def test_iters_since_last_new_best_zero_at_new_best(self):
        storage = AsyncMock()
        progs = [_prog(score=10.0 * i, iteration=i) for i in range(1, 6)]
        storage.mget.return_value = []
        storage.snapshot.get_all.return_value = progs

        stage = EvolutionaryStatisticsCollector(
            storage=storage, metrics_context=_ctx(), timeout=5.0
        )
        stage.attach_inputs({})
        result = await stage.execute(progs[-1])

        # Best is at the latest iteration → plateau = 0
        assert result.output.iters_since_last_new_best == 0

    async def test_iters_since_last_new_best_after_plateau(self):
        storage = AsyncMock()
        # best achieved at iter=3 (score=100), then no improvement
        scores = [10.0, 20.0, 100.0, 50.0, 60.0, 70.0]
        progs = [_prog(score=s, iteration=i + 1) for i, s in enumerate(scores)]
        storage.mget.return_value = []
        storage.snapshot.get_all.return_value = progs

        stage = EvolutionaryStatisticsCollector(
            storage=storage, metrics_context=_ctx(), timeout=5.0
        )
        stage.attach_inputs({})
        result = await stage.execute(progs[-1])

        # focal at iter=6, last new best at iter=3 → plateau = 3
        assert result.output.iters_since_last_new_best == 3

    async def test_iters_since_last_new_best_lower_is_better(self):
        storage = AsyncMock()
        # lower-is-better: best (lowest) achieved at iter=2 (score=5)
        scores = [10.0, 5.0, 8.0, 9.0, 7.0]
        progs = [_prog(score=s, iteration=i + 1) for i, s in enumerate(scores)]
        storage.mget.return_value = []
        storage.snapshot.get_all.return_value = progs

        stage = EvolutionaryStatisticsCollector(
            storage=storage,
            metrics_context=_ctx(higher_is_better=False),
            timeout=5.0,
        )
        stage.attach_inputs({})
        result = await stage.execute(progs[-1])

        # focal at iter=5, last new best at iter=2 → plateau = 3
        assert result.output.iters_since_last_new_best == 3

    async def test_iter_window_rank_none_when_focal_invalid(self):
        storage = AsyncMock()
        focal = _prog(score=50.0, is_valid=0.0, iteration=2)
        peer = _prog(score=80.0, iteration=1)
        storage.mget.return_value = []
        storage.snapshot.get_all.return_value = [peer, focal]

        stage = EvolutionaryStatisticsCollector(
            storage=storage, metrics_context=_ctx(), timeout=5.0
        )
        stage.attach_inputs({})
        result = await stage.execute(focal)

        assert result.output.iter_window_rank is None

    async def test_iter_window_rank_when_focal_missing_from_snapshot(self):
        """Snapshot-lag regression: focal is in pipeline but not yet in the
        population snapshot. Rank must still be computed against window peers
        + focal; current bug silently sets rank=None via sorted.index ValueError.
        """
        storage = AsyncMock()
        peer1 = _prog(score=10.0, iteration=1)
        peer2 = _prog(score=30.0, iteration=2)
        focal = _prog(score=50.0, iteration=3)
        storage.mget.return_value = []
        storage.snapshot.get_all.return_value = [peer1, peer2]

        stage = EvolutionaryStatisticsCollector(
            storage=storage, metrics_context=_ctx(), timeout=5.0
        )
        stage.attach_inputs({})
        result = await stage.execute(focal)

        stats = result.output
        assert stats.iter_window_rank == 1
        assert stats.iter_window_valid == 3
        assert stats.iter_window_best_fitness == 50.0
        assert stats.iter_window_best_iter == 3

    async def test_iter_window_rank_when_focal_in_snapshot_but_stale_metrics(
        self,
    ):
        """Snapshot contains focal but with VALIDITY_KEY=0 (not yet evaluated
        in the snapshot view). The program object passed to execute() has the
        fresh, valid metrics. Rank must use the fresh metrics.
        """
        storage = AsyncMock()
        peer1 = _prog(score=10.0, iteration=1)
        peer2 = _prog(score=30.0, iteration=2)
        stale_focal = _prog(score=0.0, is_valid=0.0, iteration=3)
        fresh_focal = _prog(score=50.0, iteration=3)
        fresh_focal.id = stale_focal.id
        storage.mget.return_value = []
        storage.snapshot.get_all.return_value = [peer1, peer2, stale_focal]

        stage = EvolutionaryStatisticsCollector(
            storage=storage, metrics_context=_ctx(), timeout=5.0
        )
        stage.attach_inputs({})
        result = await stage.execute(fresh_focal)

        stats = result.output
        assert stats.iter_window_rank == 1
        assert stats.iter_window_valid == 3
        assert stats.iter_window_best_fitness == 50.0

    async def test_iter_window_no_valid_in_window(self):
        storage = AsyncMock()
        p1 = _prog(score=10.0, is_valid=0.0, iteration=1)
        storage.mget.return_value = []
        storage.snapshot.get_all.return_value = [p1]

        stage = EvolutionaryStatisticsCollector(
            storage=storage, metrics_context=_ctx(), timeout=5.0
        )
        stage.attach_inputs({})
        result = await stage.execute(p1)

        stats = result.output
        assert stats.iter_window_lo == -14
        assert stats.iter_window_hi == 16
        assert stats.iter_window_programs == 1
        assert stats.iter_window_valid == 0
        assert stats.iter_window_best_fitness is None
        assert stats.iter_window_trend == "flat"

    def test_iteration_window_radius_must_be_non_negative(self):
        storage = AsyncMock()
        with pytest.raises(ValueError, match="iteration_window_radius"):
            EvolutionaryStatisticsCollector(
                storage=storage,
                metrics_context=_ctx(),
                timeout=5.0,
                iteration_window_radius=-1,
            )
