"""Tests for DGImprovementTracker — per-(D, G) improvement pair tracking."""

from __future__ import annotations

import fakeredis.aioredis
import pytest

from gigaevo.adversarial.dg_tracker import DGImprovementTracker


@pytest.fixture
async def tracker():
    t = DGImprovementTracker(host="localhost", port=6379, db=0, prefix="test")
    t._redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    yield t
    await t.close()


@pytest.mark.asyncio
async def test_record_improvement_stores_in_sorted_set(tracker):
    """record_improvement(d_id, g_id, delta) stores pair in Redis sorted set keyed by g_id."""
    await tracker.record_improvement(d_id="d1", g_id="g1", delta=0.05)
    key = tracker._key("g1")
    score = await tracker._redis.zscore(key, "d1")
    assert score == pytest.approx(0.05)


@pytest.mark.asyncio
async def test_record_improvement_ignores_non_positive_delta(tracker):
    """record_improvement with delta <= 0 is silently ignored."""
    await tracker.record_improvement(d_id="d1", g_id="g1", delta=0.0)
    await tracker.record_improvement(d_id="d2", g_id="g1", delta=-0.5)
    key = tracker._key("g1")
    count = await tracker._redis.zcard(key)
    assert count == 0


@pytest.mark.asyncio
async def test_get_best_d_for_g_returns_highest_delta(tracker):
    """get_best_d_for_g returns the d_id with the highest delta for that G."""
    await tracker.record_improvement(d_id="d1", g_id="g1", delta=0.05)
    result = await tracker.get_best_d_for_g("g1")
    assert result is not None
    d_id, delta = result
    assert d_id == "d1"
    assert delta == pytest.approx(0.05)


@pytest.mark.asyncio
async def test_get_best_d_for_g_returns_none_when_no_data(tracker):
    """get_best_d_for_g returns None when no improvement data exists for g_id."""
    result = await tracker.get_best_d_for_g("nonexistent_g")
    assert result is None


@pytest.mark.asyncio
async def test_multiple_d_programs_returns_best(tracker):
    """Multiple D programs improving same G -- returns the one with highest delta."""
    await tracker.record_improvement(d_id="d1", g_id="g1", delta=0.03)
    await tracker.record_improvement(d_id="d2", g_id="g1", delta=0.08)
    await tracker.record_improvement(d_id="d3", g_id="g1", delta=0.05)
    result = await tracker.get_best_d_for_g("g1")
    assert result is not None
    d_id, delta = result
    assert d_id == "d2"
    assert delta == pytest.approx(0.08)


@pytest.mark.asyncio
async def test_same_d_same_g_highest_delta_wins(tracker):
    """Same D improving same G with different deltas -- highest delta wins (ZADD GT)."""
    await tracker.record_improvement(d_id="d1", g_id="g1", delta=0.03)
    await tracker.record_improvement(d_id="d1", g_id="g1", delta=0.10)
    result = await tracker.get_best_d_for_g("g1")
    assert result is not None
    d_id, delta = result
    assert d_id == "d1"
    assert delta == pytest.approx(0.10)


@pytest.mark.asyncio
async def test_get_best_d_for_g_returns_tuple(tracker):
    """get_best_d_for_g returns (d_id, delta) as a tuple."""
    await tracker.record_improvement(d_id="d1", g_id="g1", delta=0.07)
    result = await tracker.get_best_d_for_g("g1")
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], str)
    assert isinstance(result[1], float)


@pytest.mark.asyncio
async def test_record_batch_stores_multiple_pairs(tracker):
    """record_batch stores multiple pairs efficiently via pipeline."""
    pairs = [
        ("d1", "g1", 0.05),
        ("d2", "g1", 0.10),
        ("d3", "g2", 0.03),
        ("d4", "g3", -0.01),  # negative -- should be filtered
    ]
    count = await tracker.record_batch(pairs)
    assert count == 3  # only positive deltas

    result_g1 = await tracker.get_best_d_for_g("g1")
    assert result_g1 is not None
    assert result_g1[0] == "d2"

    result_g2 = await tracker.get_best_d_for_g("g2")
    assert result_g2 is not None
    assert result_g2[0] == "d3"

    result_g3 = await tracker.get_best_d_for_g("g3")
    assert result_g3 is None  # negative delta was filtered
