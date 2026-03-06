"""Tests for gigaevo/evolution/strategies/removers.py"""

from datetime import datetime, timedelta, timezone
import random

import pytest

from gigaevo.evolution.strategies.removers import (
    FitnessArchiveRemover,
    OldestArchiveRemover,
    ParetoFrontArchiveRemover,
    ParetoFrontArchiveRemoverDropOldest,
    RandomArchiveRemover,
)
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState


def _prog(metrics=None, created_at=None):
    p = Program(code="def solve(): return 1", state=ProgramState.DONE)
    if metrics:
        p.add_metrics(metrics)
    if created_at:
        p.created_at = created_at
    return p


class TestOldestArchiveRemover:
    def test_removes_oldest(self):
        now = datetime.now(timezone.utc)
        old = _prog(created_at=now - timedelta(hours=2))
        mid = _prog(created_at=now - timedelta(hours=1))
        new = _prog(created_at=now)

        remover = OldestArchiveRemover()
        removed = remover([old, mid, new], max_size_to_keep=2)
        assert len(removed) == 1
        assert removed[0].id == old.id

    def test_no_removal_under_limit(self):
        progs = [_prog() for _ in range(3)]
        remover = OldestArchiveRemover()
        assert remover(progs, max_size_to_keep=5) == []

    def test_empty_list(self):
        remover = OldestArchiveRemover()
        assert remover([], max_size_to_keep=1) == []

    def test_exact_limit(self):
        progs = [_prog() for _ in range(3)]
        remover = OldestArchiveRemover()
        assert remover(progs, max_size_to_keep=3) == []


class TestRandomArchiveRemover:
    def test_correct_count(self):
        progs = [_prog() for _ in range(5)]
        remover = RandomArchiveRemover()
        removed = remover(progs, max_size_to_keep=3)
        assert len(removed) == 2

    def test_no_removal_under_limit(self):
        progs = [_prog() for _ in range(2)]
        remover = RandomArchiveRemover()
        assert remover(progs, max_size_to_keep=5) == []

    def test_removed_are_subset(self):
        """Every removed program was in the original list."""
        progs = [_prog() for _ in range(5)]
        original_ids = {p.id for p in progs}
        remover = RandomArchiveRemover()
        removed = remover(progs, max_size_to_keep=2)
        assert len(removed) == 3
        for p in removed:
            assert p.id in original_ids

    def test_seeded_random_deterministic(self):
        """With a fixed seed, removal order is deterministic."""
        progs = [_prog() for _ in range(10)]
        remover = RandomArchiveRemover()

        random.seed(42)
        removed1 = remover(progs, max_size_to_keep=5)
        ids1 = [p.id for p in removed1]

        random.seed(42)
        removed2 = remover(progs, max_size_to_keep=5)
        ids2 = [p.id for p in removed2]

        assert ids1 == ids2


class TestFitnessArchiveRemover:
    def test_removes_lowest(self):
        p1 = _prog(metrics={"score": 1.0})
        p2 = _prog(metrics={"score": 5.0})
        p3 = _prog(metrics={"score": 10.0})

        remover = FitnessArchiveRemover(fitness_key="score")
        removed = remover([p1, p2, p3], max_size_to_keep=2)
        assert len(removed) == 1
        assert removed[0].id == p1.id

    def test_higher_is_better_false(self):
        p_low = _prog(metrics={"error": 0.1})
        p_high = _prog(metrics={"error": 10.0})

        remover = FitnessArchiveRemover(
            fitness_key="error", fitness_key_higher_is_better=False
        )
        # When higher_is_better=False, score = -value, so high error gets lowest score
        removed = remover([p_low, p_high], max_size_to_keep=1)
        assert len(removed) == 1
        assert removed[0].id == p_high.id

    def test_missing_key_raises(self):
        p = _prog(metrics={"other": 1.0})
        remover = FitnessArchiveRemover(fitness_key="score")
        with pytest.raises(ValueError, match="Fitness key"):
            remover([p], max_size_to_keep=0)

    def test_under_limit(self):
        p = _prog(metrics={"score": 1.0})
        remover = FitnessArchiveRemover(fitness_key="score")
        assert remover([p], max_size_to_keep=5) == []

    def test_removes_multiple(self):
        """max_size_to_keep=1 with 4 programs: 3 lowest removed."""
        p1 = _prog(metrics={"score": 1.0})
        p2 = _prog(metrics={"score": 2.0})
        p3 = _prog(metrics={"score": 3.0})
        p4 = _prog(metrics={"score": 10.0})

        remover = FitnessArchiveRemover(fitness_key="score")
        removed = remover([p1, p2, p3, p4], max_size_to_keep=1)
        assert len(removed) == 3
        removed_ids = {p.id for p in removed}
        assert p1.id in removed_ids
        assert p2.id in removed_ids
        assert p3.id in removed_ids
        assert p4.id not in removed_ids


class TestParetoFrontArchiveRemover:
    def test_removes_most_dominated(self):
        # p1 is dominated by both p2 and p3
        p1 = _prog(metrics={"a": 1.0, "b": 1.0})
        p2 = _prog(metrics={"a": 5.0, "b": 5.0})
        p3 = _prog(metrics={"a": 3.0, "b": 3.0})

        remover = ParetoFrontArchiveRemover(
            fitness_keys=["a", "b"],
            tie_breaker=lambda p: 0,
        )
        removed = remover([p1, p2, p3], max_size_to_keep=2)
        assert len(removed) == 1
        assert removed[0].id == p1.id

    def test_order_candidates(self):
        p1 = _prog(metrics={"a": 1.0, "b": 1.0})
        p2 = _prog(metrics={"a": 10.0, "b": 10.0})
        p3 = _prog(metrics={"a": 5.0, "b": 5.0})

        remover = ParetoFrontArchiveRemover(
            fitness_keys=["a", "b"],
            tie_breaker=lambda p: 0,
        )
        ordered = remover.order_candidates([p1, p2, p3])
        # p1 is dominated by both p2 and p3 (dominated_count=2)
        # p3 is dominated by p2 (dominated_count=1)
        # p2 is dominated by none (dominated_count=0)
        assert ordered[0].id == p1.id  # worst
        assert ordered[-1].id == p2.id  # best

    def test_empty(self):
        remover = ParetoFrontArchiveRemover(
            fitness_keys=["a", "b"],
            tie_breaker=lambda p: 0,
        )
        assert remover([], max_size_to_keep=1) == []

    def test_under_limit(self):
        progs = [_prog(metrics={"a": 1.0, "b": 1.0})]
        remover = ParetoFrontArchiveRemover(
            fitness_keys=["a", "b"],
            tie_breaker=lambda p: 0,
        )
        assert remover(progs, max_size_to_keep=5) == []

    def test_non_dominated_set_preserved(self):
        """3 programs on Pareto front, max_size_to_keep=3: none removed."""
        p1 = _prog(metrics={"a": 10.0, "b": 1.0})
        p2 = _prog(metrics={"a": 1.0, "b": 10.0})
        p3 = _prog(metrics={"a": 5.0, "b": 5.0})

        remover = ParetoFrontArchiveRemover(
            fitness_keys=["a", "b"],
            tie_breaker=lambda p: 0,
        )
        # None dominates any other -> all on Pareto front
        removed = remover([p1, p2, p3], max_size_to_keep=3)
        assert removed == []

    def test_all_equal_fitness(self):
        """All programs have identical metrics: removal falls to tie-breaker."""
        now = datetime.now(timezone.utc)
        p1 = _prog(metrics={"a": 5.0, "b": 5.0})
        p1.created_at = now - timedelta(hours=3)
        p2 = _prog(metrics={"a": 5.0, "b": 5.0})
        p2.created_at = now - timedelta(hours=1)
        p3 = _prog(metrics={"a": 5.0, "b": 5.0})
        p3.created_at = now

        remover = ParetoFrontArchiveRemover(
            fitness_keys=["a", "b"],
            tie_breaker=lambda p: p.created_at.timestamp(),
        )
        # All have dominated_count=0, so tie_breaker sorts by timestamp asc
        # Sorted worst-to-best by (-dominated_count, tie_break_score)
        # All dominated_count=0 -> sort by timestamp asc -> oldest first
        removed = remover([p1, p2, p3], max_size_to_keep=1)
        assert len(removed) == 2
        # Oldest (p1) and middle (p2) should be removed
        removed_ids = {p.id for p in removed}
        assert p1.id in removed_ids
        assert p2.id in removed_ids


class TestParetoFrontArchiveRemoverDropOldest:
    def test_tie_breaks_by_age(self):
        now = datetime.now(timezone.utc)
        # Same fitness, different ages
        old = _prog(metrics={"a": 5.0, "b": 5.0})
        old.created_at = now - timedelta(hours=2)
        new = _prog(metrics={"a": 5.0, "b": 5.0})
        new.created_at = now

        remover = ParetoFrontArchiveRemoverDropOldest(fitness_keys=["a", "b"])
        removed = remover([old, new], max_size_to_keep=1)
        assert len(removed) == 1
        # Both have dominated_count=0 (neither dominates the other).
        # Tie-break by created_at.timestamp() ascending -> old has lower timestamp.
        assert removed[0].id == old.id
