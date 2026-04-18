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
        ("d4", "g3", -0.01),  # negative -- should be filtered from per-G sorted set
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


# ---------------------------------------------------------------------------
# v3 inverted-index extension — dg_d_wins, dg_g_resisted, dg_delta
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_record_batch_dual_writes_d_wins(tracker):
    """Positive deltas populate dg_d_wins:{d_id} SET with the beaten g_ids.

    D's BD y-axis (tracker_coverage_count) = SCARD of this set.
    """
    pairs = [
        ("d1", "g1", 0.05),
        ("d1", "g2", 0.10),
        ("d2", "g1", 0.03),
        ("d2", "g2", -0.01),  # non-positive -- NOT a win for d2
    ]
    await tracker.record_batch(pairs)

    d1_wins = await tracker._redis.smembers(tracker._d_wins_key("d1"))
    assert d1_wins == {"g1", "g2"}

    d2_wins = await tracker._redis.smembers(tracker._d_wins_key("d2"))
    assert d2_wins == {"g1"}  # only the positive delta


@pytest.mark.asyncio
async def test_record_batch_dual_writes_g_resisted(tracker):
    """Non-positive deltas populate dg_g_resisted:{g_id} SET with the resisted d_ids.

    G's fallback BD y-axis (g_tracker_coverage_count) = SCARD of this set.
    """
    pairs = [
        ("d1", "g1", -0.01),  # D made it worse -> G resisted D
        ("d2", "g1", 0.0),  # D changed nothing -> G resisted D
        ("d3", "g1", 0.05),  # D improved -> G did NOT resist D
    ]
    await tracker.record_batch(pairs)

    g1_resisted = await tracker._redis.smembers(tracker._g_resisted_key("g1"))
    assert g1_resisted == {"d1", "d2"}


@pytest.mark.asyncio
async def test_record_batch_writes_d_delta_hash_for_all_signs(tracker):
    """record_batch writes every pair (pos, neg, zero) into dg_delta:{d_id} hash.

    SharedBenchmarkLineageStage reads this hash to compare child-D vs parent-D
    on the intersection of G's they have both been evaluated against.
    """
    pairs = [
        ("d1", "g1", 0.05),
        ("d1", "g2", -0.01),
        ("d1", "g3", 0.0),
    ]
    await tracker.record_batch(pairs)

    g_ids = await tracker._redis.hkeys(tracker._d_delta_key("d1"))
    assert set(g_ids) == {"g1", "g2", "g3"}

    deltas = await tracker._redis.hmget(tracker._d_delta_key("d1"), ["g1", "g2", "g3"])
    assert float(deltas[0]) == pytest.approx(0.05)
    assert float(deltas[1]) == pytest.approx(-0.01)
    assert float(deltas[2]) == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_count_g_beaten_by_d(tracker):
    """count_g_beaten_by_d returns SCARD of dg_d_wins:{d_id}."""
    await tracker.record_batch(
        [
            ("d1", "g1", 0.05),
            ("d1", "g2", 0.10),
            ("d1", "g3", -0.01),  # NOT a win
        ]
    )
    assert await tracker.count_g_beaten_by_d("d1") == 2


@pytest.mark.asyncio
async def test_count_g_beaten_by_d_empty(tracker):
    """count_g_beaten_by_d returns 0 when no wins recorded."""
    assert await tracker.count_g_beaten_by_d("d_nobody") == 0


@pytest.mark.asyncio
async def test_count_d_resisted_by_g(tracker):
    """count_d_resisted_by_g returns SCARD of dg_g_resisted:{g_id}."""
    await tracker.record_batch(
        [
            ("d1", "g1", -0.01),
            ("d2", "g1", 0.0),
            ("d3", "g1", 0.05),  # NOT resisted
        ]
    )
    assert await tracker.count_d_resisted_by_g("g1") == 2


@pytest.mark.asyncio
async def test_count_d_resisted_by_g_empty(tracker):
    """count_d_resisted_by_g returns 0 when no resistance recorded."""
    assert await tracker.count_d_resisted_by_g("g_nobody") == 0


@pytest.mark.asyncio
async def test_faced_by_d_returns_all_g_ids(tracker):
    """faced_by_d returns HKEYS of dg_delta:{d_id} (all G's this D has been tested against)."""
    await tracker.record_batch(
        [
            ("d1", "g1", 0.05),  # win
            ("d1", "g2", -0.01),  # loss
            ("d1", "g3", 0.0),  # neutral
        ]
    )
    faced = await tracker.faced_by_d("d1")
    assert faced == {"g1", "g2", "g3"}


@pytest.mark.asyncio
async def test_faced_by_d_empty_returns_empty_set(tracker):
    """faced_by_d returns empty set when D has never been recorded."""
    assert await tracker.faced_by_d("d_nobody") == set()


@pytest.mark.asyncio
async def test_get_deltas_against_returns_paired_deltas_in_input_order(tracker):
    """get_deltas_against(d_a, d_b, g_ids) returns [(delta_a, delta_b), ...] for shared G's."""
    await tracker.record_batch(
        [
            ("d_child", "g1", 0.05),
            ("d_child", "g2", -0.02),
            ("d_parent", "g1", 0.03),
            ("d_parent", "g2", 0.01),
        ]
    )
    pairs = await tracker.get_deltas_against("d_child", "d_parent", ["g1", "g2"])
    assert len(pairs) == 2
    assert pairs[0][0] == pytest.approx(0.05)
    assert pairs[0][1] == pytest.approx(0.03)
    assert pairs[1][0] == pytest.approx(-0.02)
    assert pairs[1][1] == pytest.approx(0.01)


@pytest.mark.asyncio
async def test_get_deltas_against_skips_missing_pairs(tracker):
    """get_deltas_against skips g_ids where either side has no recorded delta."""
    await tracker.record_batch(
        [
            ("d_child", "g1", 0.05),
            ("d_parent", "g1", 0.03),
            ("d_child", "g2", 0.02),
            # d_parent has no delta for g2 -> skip
        ]
    )
    pairs = await tracker.get_deltas_against("d_child", "d_parent", ["g1", "g2"])
    assert len(pairs) == 1
    assert pairs[0][0] == pytest.approx(0.05)
    assert pairs[0][1] == pytest.approx(0.03)


@pytest.mark.asyncio
async def test_get_deltas_against_empty_g_ids_returns_empty(tracker):
    """get_deltas_against returns [] when g_ids is empty (short-circuit)."""
    pairs = await tracker.get_deltas_against("d1", "d2", [])
    assert pairs == []


@pytest.mark.asyncio
async def test_record_batch_ttl_refreshed_on_new_keys(tracker):
    """record_batch sets TTL on dg_d_wins, dg_g_resisted, and dg_delta keys."""
    await tracker.record_batch(
        [
            ("d1", "g1", 0.05),
            ("d1", "g2", -0.01),
        ]
    )

    assert await tracker._redis.ttl(tracker._d_wins_key("d1")) > 0
    assert await tracker._redis.ttl(tracker._g_resisted_key("g2")) > 0
    assert await tracker._redis.ttl(tracker._d_delta_key("d1")) > 0


@pytest.mark.asyncio
async def test_record_batch_emits_tracker_write_log(tracker):
    """record_batch emits [TRACKER_WRITE] canonical event via emit() seam."""
    import json
    import re

    from loguru import logger

    captured: list[str] = []

    def sink(message):
        captured.append(str(message))

    sink_id = logger.add(sink, level="DEBUG", format="{message}")
    try:
        await tracker.record_batch(
            [
                ("d1", "g1", 0.05),
                ("d1", "g2", -0.01),
                ("d2", "g1", 0.03),
            ],
            gen=7,
        )
    finally:
        logger.remove(sink_id)

    tracker_write_lines = [line for line in captured if "[TRACKER_WRITE]" in line]
    assert len(tracker_write_lines) >= 1

    match = re.search(r"\[TRACKER_WRITE\]\s+(\{.*\})", tracker_write_lines[0])
    assert match, f"unexpected shape: {tracker_write_lines[0]!r}"
    payload = json.loads(match.group(1))
    assert payload["event"] == "TRACKER_WRITE"
    assert payload["gen"] == 7
    assert payload["pairs_count"] == 3
    assert payload["positive_count"] == 2
    assert payload["d_wins_added"] == 2  # d1->g1, d2->g1
    assert payload["g_resisted_added"] == 1  # d1 resisted by g2
    assert payload["d_faced_added"] == 3  # 2 for d1, 1 for d2


@pytest.mark.asyncio
async def test_record_batch_empty_is_noop(tracker):
    """record_batch([]) returns 0 and writes nothing."""
    count = await tracker.record_batch([])
    assert count == 0
