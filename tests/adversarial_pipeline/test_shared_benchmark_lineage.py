"""Tests for SharedBenchmarkLineageStage and resolvers (v3 Prong 2 lineage fix)."""

from __future__ import annotations

import uuid

import fakeredis.aioredis
import pytest

from gigaevo.adversarial.dg_tracker import DGImprovementTracker
from gigaevo.adversarial.shared_benchmark_lineage import (
    DGTrackerSharedOpponentResolver,
    SharedBenchmarkLineageStage,
)
from gigaevo.programs.program import Program


@pytest.fixture
async def tracker():
    """DGImprovementTracker with fake Redis."""
    t = DGImprovementTracker(host="localhost", port=6379, db=0, prefix="test")
    t._redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    yield t
    await t.close()


@pytest.fixture
async def resolver(tracker):
    """SharedBenchmarkResolver using DG tracker."""
    return DGTrackerSharedOpponentResolver(tracker=tracker)


_PROG_IDS = {
    "d0_prog": str(uuid.uuid4()),
    "d1_prog": str(uuid.uuid4()),
}


def make_program(
    prog_key: str, code: str = "pass", parent_key: str | None = None
) -> Program:
    """Create a minimal valid Program for testing.

    Args:
        prog_key: Key into _PROG_IDS (e.g., 'd0_prog', 'd1_prog').
        code: Program source code.
        parent_key: Parent program key (None = no parent).
    """
    prog_id = _PROG_IDS[prog_key]
    parent_id = _PROG_IDS[parent_key] if parent_key else None
    return Program(id=prog_id, code=code, metadata={"parent_id": parent_id})


# ---------------------------------------------------------------------------
# SharedBenchmarkResolver tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_shared_benchmark_returns_intersection_of_faced_sets(tracker, resolver):
    """shared_benchmark returns g_ids both D's have faced."""
    # d1 faced {g1, g2, g3}
    # d2 faced {g2, g3, g4}
    # intersection: {g2, g3}
    await tracker.record_batch(
        [
            ("d1", "g1", 0.05),
            ("d1", "g2", 0.03),
            ("d1", "g3", -0.01),
            ("d2", "g2", 0.02),
            ("d2", "g3", -0.02),
            ("d2", "g4", 0.01),
        ]
    )
    shared = await resolver.shared_benchmark("d1", "d2")
    assert set(shared) == {"g2", "g3"}


@pytest.mark.asyncio
async def test_shared_benchmark_empty_intersection(tracker, resolver):
    """shared_benchmark returns empty list when no overlap."""
    await tracker.record_batch(
        [
            ("d1", "g1", 0.05),
            ("d2", "g2", 0.03),
        ]
    )
    shared = await resolver.shared_benchmark("d1", "d2")
    assert shared == []


@pytest.mark.asyncio
async def test_shared_benchmark_one_d_never_recorded(tracker, resolver):
    """shared_benchmark returns empty when one D has never been recorded."""
    await tracker.record_batch([("d1", "g1", 0.05)])
    shared = await resolver.shared_benchmark("d1", "d_nobody")
    assert shared == []


# ---------------------------------------------------------------------------
# SharedBenchmarkLineageStage tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lineage_stage_returns_trend_on_sufficient_shared(tracker, resolver):
    """SharedBenchmarkLineageStage computes child-vs-parent trend on shared benchmark."""
    # child (d1): g1->+0.10, g2->-0.02 (mean = +0.04)
    # parent (d0): g1->+0.05, g2->+0.01 (mean = +0.03)
    # trend = 0.04 - 0.03 = +0.01
    await tracker.record_batch(
        [
            ("d0", "g1", 0.05),
            ("d0", "g2", 0.01),
            ("d1", "g1", 0.10),
            ("d1", "g2", -0.02),
        ]
    )

    stage = SharedBenchmarkLineageStage(resolver=resolver, min_shared=2, timeout=30.0)

    async def mock_load_parent(prog):
        if prog.id == _PROG_IDS["d1_prog"]:
            return make_program("d0_prog")
        return None

    stage._load_parent = mock_load_parent
    stage._parent_id_to_d_id = {
        _PROG_IDS["d0_prog"]: "d0",
        _PROG_IDS["d1_prog"]: "d1",
    }

    child = make_program("d1_prog", parent_key="d0_prog")
    output = await stage.compute(child)
    assert output.trend is not None
    assert output.trend == pytest.approx(0.01, abs=1e-6)
    assert output.n_shared == 2


@pytest.mark.asyncio
async def test_lineage_stage_returns_none_when_no_parent(resolver):
    """SharedBenchmarkLineageStage returns None-trend when program has no parent."""
    stage = SharedBenchmarkLineageStage(resolver=resolver, min_shared=2, timeout=30.0)

    async def mock_load_parent(prog):
        return None

    stage._load_parent = mock_load_parent
    child = make_program("d1_prog")
    output = await stage.compute(child)
    assert output.trend is None
    assert output.n_shared is None


@pytest.mark.asyncio
async def test_lineage_stage_returns_none_when_insufficient_shared(tracker, resolver):
    """SharedBenchmarkLineageStage returns None-trend when shared benchmark < min_shared."""
    # Only one shared g_id, but min_shared=2
    await tracker.record_batch(
        [
            ("d0", "g1", 0.05),
            ("d1", "g1", 0.10),
        ]
    )

    stage = SharedBenchmarkLineageStage(resolver=resolver, min_shared=2, timeout=30.0)
    stage._parent_id_to_d_id = {
        _PROG_IDS["d0_prog"]: "d0",
        _PROG_IDS["d1_prog"]: "d1",
    }

    async def mock_load_parent(prog):
        if prog.id == _PROG_IDS["d1_prog"]:
            return make_program("d0_prog")
        return None

    stage._load_parent = mock_load_parent
    child = make_program("d1_prog", parent_key="d0_prog")
    output = await stage.compute(child)
    assert output.trend is None
    assert output.n_shared is None


@pytest.mark.asyncio
async def test_lineage_stage_negative_trend(tracker, resolver):
    """SharedBenchmarkLineageStage correctly computes negative trend (regression)."""
    # child worse than parent on average
    await tracker.record_batch(
        [
            ("d0", "g1", 0.10),
            ("d0", "g2", 0.08),
            ("d1", "g1", 0.05),
            ("d1", "g2", 0.03),
        ]
    )

    stage = SharedBenchmarkLineageStage(resolver=resolver, min_shared=2, timeout=30.0)
    stage._parent_id_to_d_id = {
        _PROG_IDS["d0_prog"]: "d0",
        _PROG_IDS["d1_prog"]: "d1",
    }

    async def mock_load_parent(prog):
        if prog.id == _PROG_IDS["d1_prog"]:
            return make_program("d0_prog")
        return None

    stage._load_parent = mock_load_parent
    child = make_program("d1_prog", parent_key="d0_prog")
    output = await stage.compute(child)
    assert output.trend is not None
    assert output.trend < 0  # regression


@pytest.mark.asyncio
async def test_lineage_stage_mixed_outcomes_trend(tracker, resolver):
    """SharedBenchmarkLineageStage averages across mixed-sign deltas."""
    # child: +0.10 on g1, -0.05 on g2 -> mean = +0.025
    # parent: +0.05 on g1, +0.02 on g2 -> mean = +0.035
    # trend = 0.025 - 0.035 = -0.01
    await tracker.record_batch(
        [
            ("d0", "g1", 0.05),
            ("d0", "g2", 0.02),
            ("d1", "g1", 0.10),
            ("d1", "g2", -0.05),
        ]
    )

    stage = SharedBenchmarkLineageStage(resolver=resolver, min_shared=2, timeout=30.0)
    stage._parent_id_to_d_id = {
        _PROG_IDS["d0_prog"]: "d0",
        _PROG_IDS["d1_prog"]: "d1",
    }

    async def mock_load_parent(prog):
        if prog.id == _PROG_IDS["d1_prog"]:
            return make_program("d0_prog")
        return None

    stage._load_parent = mock_load_parent
    child = make_program("d1_prog", parent_key="d0_prog")
    output = await stage.compute(child)
    assert output.trend is not None
    assert output.trend == pytest.approx(-0.01, abs=1e-6)
