"""Tests for tracker coverage stages (v3 BD axis y computation)."""

from __future__ import annotations

import uuid

import fakeredis.aioredis
import pytest

from gigaevo.adversarial.dg_tracker import DGImprovementTracker
from gigaevo.adversarial.tracker_coverage_stages import (
    ComputeDWinsCountStage,
    ComputeGResistedCountStage,
)
from gigaevo.programs.program import Program

_PROG_IDS = {
    "d1": str(uuid.uuid4()),
    "d2": str(uuid.uuid4()),
    "d3": str(uuid.uuid4()),
    "g1": str(uuid.uuid4()),
}


@pytest.fixture
async def tracker():
    """DGImprovementTracker with fake Redis."""
    t = DGImprovementTracker(host="localhost", port=6379, db=0, prefix="test")
    t._redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    yield t
    await t.close()


def make_program(prog_key: str) -> Program:
    """Create a minimal Program for testing."""
    prog_id = _PROG_IDS.get(prog_key, str(uuid.uuid4()))
    return Program(id=prog_id, code="pass")


@pytest.mark.asyncio
async def test_compute_d_wins_count_stage_records_count(tracker):
    """ComputeDWinsCountStage records the count of G's beaten by D."""
    # Simulate D beating g1, g2, g3 (3 wins)
    d1_id = _PROG_IDS["d1"]
    g_ids = [_PROG_IDS["g1"], str(uuid.uuid4()), str(uuid.uuid4())]
    await tracker.record_batch(
        [
            (d1_id, g_ids[0], {"delta": 0.05, "is_valid": 1.0}),
            (d1_id, g_ids[1], {"delta": 0.03, "is_valid": 1.0}),
            (d1_id, g_ids[2], {"delta": 0.01, "is_valid": 1.0}),
        ]
    )

    stage = ComputeDWinsCountStage(dg_tracker=tracker, timeout=30.0)
    prog = make_program("d1")
    await stage.compute(prog)

    assert prog.metrics["wins"] == 3


@pytest.mark.asyncio
async def test_compute_d_wins_count_stage_zero_wins(tracker):
    """ComputeDWinsCountStage records 0 when D has beaten no one."""
    stage = ComputeDWinsCountStage(dg_tracker=tracker, timeout=30.0)
    prog = make_program("d1")
    await stage.compute(prog)

    assert prog.metrics["wins"] == 0


@pytest.mark.asyncio
async def test_compute_d_wins_count_stage_ignores_losses(tracker):
    """ComputeDWinsCountStage counts only positive deltas (wins)."""
    # Mix of wins and losses for d1
    d1_id = _PROG_IDS["d1"]
    g_ids = [_PROG_IDS["g1"], str(uuid.uuid4()), str(uuid.uuid4())]
    await tracker.record_batch(
        [
            (d1_id, g_ids[0], {"delta": 0.05, "is_valid": 1.0}),  # win
            (d1_id, g_ids[1], {"delta": -0.02, "is_valid": 1.0}),  # loss
            (d1_id, g_ids[2], {"delta": 0.01, "is_valid": 1.0}),  # win
        ]
    )

    stage = ComputeDWinsCountStage(dg_tracker=tracker, timeout=30.0)
    prog = make_program("d1")
    await stage.compute(prog)

    assert prog.metrics["wins"] == 2


@pytest.mark.asyncio
async def test_compute_g_resisted_count_stage_records_count(tracker):
    """ComputeGResistedCountStage records the count of D's resisted by G."""
    # Simulate G resisting d1, d2 (2 resists via non-positive deltas)
    g1_id = _PROG_IDS["g1"]
    d_ids = [_PROG_IDS["d1"], _PROG_IDS["d2"], _PROG_IDS["d3"]]
    await tracker.record_batch(
        [
            (d_ids[0], g1_id, {"delta": -0.05, "is_valid": 1.0}),  # resisted
            (d_ids[1], g1_id, {"delta": 0.0, "is_valid": 1.0}),  # resisted (zero)
            (d_ids[2], g1_id, {"delta": 0.03, "is_valid": 1.0}),  # not resisted
        ]
    )

    stage = ComputeGResistedCountStage(dg_tracker=tracker, timeout=30.0)
    prog = make_program("g1")
    await stage.compute(prog)

    assert prog.metrics["wins"] == 2


@pytest.mark.asyncio
async def test_compute_g_resisted_count_stage_zero_resists(tracker):
    """ComputeGResistedCountStage records 0 when G resists no one."""
    stage = ComputeGResistedCountStage(dg_tracker=tracker, timeout=30.0)
    prog = make_program("g1")
    await stage.compute(prog)

    assert prog.metrics["wins"] == 0


@pytest.mark.asyncio
async def test_compute_g_resisted_count_stage_includes_zero_delta(tracker):
    """ComputeGResistedCountStage includes zero-delta entries (draw = resisted)."""
    g1_id = _PROG_IDS["g1"]
    d_ids = [_PROG_IDS["d1"], _PROG_IDS["d2"], _PROG_IDS["d3"]]
    await tracker.record_batch(
        [
            (d_ids[0], g1_id, {"delta": -0.05, "is_valid": 1.0}),  # resisted
            (d_ids[1], g1_id, {"delta": 0.0, "is_valid": 1.0}),  # resisted (zero)
            (d_ids[2], g1_id, {"delta": 0.0, "is_valid": 1.0}),  # resisted (zero)
        ]
    )

    stage = ComputeGResistedCountStage(dg_tracker=tracker, timeout=30.0)
    prog = make_program("g1")
    await stage.compute(prog)

    assert prog.metrics["wins"] == 3
