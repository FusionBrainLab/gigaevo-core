"""Tests for Redis query functions -- canonical queries with fakeredis."""

from __future__ import annotations

import json

import fakeredis
import pytest

from gigaevo.monitoring.redis_queries import (
    collect_snapshot,
    get_frontier_metrics,
    get_generation,
    get_program_counts,
    get_status_counts,
    get_validator_duration,
)
from gigaevo.monitoring.run_spec import RunSpec

PREFIX = "chains/hotpotqa/static"


def _make_redis() -> fakeredis.FakeRedis:
    return fakeredis.FakeRedis(decode_responses=True)


def _make_spec() -> RunSpec:
    return RunSpec(prefix=PREFIX, db=4, label="O")


def _metric_entry(step: int, value: float, ts: int = 123) -> str:
    return json.dumps({"s": step, "v": value, "t": ts, "k": "scalar"})


# ---------------------------------------------------------------------------
# 1. get_generation tests
# ---------------------------------------------------------------------------


def test_get_generation_normal() -> None:
    r = _make_redis()
    r.hset(f"{PREFIX}:run_state", "engine:total_generations", "42")
    assert get_generation(r, PREFIX) == 42


def test_get_generation_missing_key() -> None:
    r = _make_redis()
    assert get_generation(r, PREFIX) is None


def test_get_generation_non_numeric() -> None:
    r = _make_redis()
    r.hset(f"{PREFIX}:run_state", "engine:total_generations", "not_a_number")
    assert get_generation(r, PREFIX) is None


# ---------------------------------------------------------------------------
# 2. get_frontier_metrics tests
# ---------------------------------------------------------------------------


def test_get_frontier_metrics_single() -> None:
    r = _make_redis()
    key = f"{PREFIX}:metrics:history:program_metrics:valid_frontier_fitness"
    r.rpush(key, _metric_entry(1, 0.50))
    r.rpush(key, _metric_entry(2, 0.76))
    result = get_frontier_metrics(r, PREFIX, ["fitness"])
    assert result == {"fitness": 0.76}


def test_get_frontier_metrics_multiple() -> None:
    r = _make_redis()
    r.rpush(
        f"{PREFIX}:metrics:history:program_metrics:valid_frontier_fitness",
        _metric_entry(1, 0.76),
    )
    r.rpush(
        f"{PREFIX}:metrics:history:program_metrics:valid_frontier_prompt_length",
        _metric_entry(1, 299.0),
    )
    result = get_frontier_metrics(r, PREFIX, ["fitness", "prompt_length"])
    assert result == {"fitness": 0.76, "prompt_length": 299.0}


def test_get_frontier_metrics_empty_list() -> None:
    r = _make_redis()
    result = get_frontier_metrics(r, PREFIX, ["fitness"])
    assert result == {"fitness": None}


def test_get_frontier_metrics_malformed_json() -> None:
    r = _make_redis()
    key = f"{PREFIX}:metrics:history:program_metrics:valid_frontier_fitness"
    r.rpush(key, "NOT_JSON")
    result = get_frontier_metrics(r, PREFIX, ["fitness"])
    assert result == {"fitness": None}


# ---------------------------------------------------------------------------
# 3. get_program_counts tests
# ---------------------------------------------------------------------------


def test_get_program_counts_normal() -> None:
    r = _make_redis()
    r.rpush(
        f"{PREFIX}:metrics:history:program_metrics:programs_total_count",
        _metric_entry(1, 100),
    )
    r.rpush(
        f"{PREFIX}:metrics:history:program_metrics:programs_valid_count",
        _metric_entry(1, 85),
    )
    total, valid = get_program_counts(r, PREFIX)
    assert total == 100
    assert valid == 85


def test_get_program_counts_missing() -> None:
    r = _make_redis()
    total, valid = get_program_counts(r, PREFIX)
    assert total is None
    assert valid is None


# ---------------------------------------------------------------------------
# 4. get_validator_duration tests
# ---------------------------------------------------------------------------


def test_get_validator_duration_normal() -> None:
    r = _make_redis()
    key = (
        f"{PREFIX}:metrics:history:dag_runner:dag:internals:"
        "CallValidatorFunction:stage_duration"
    )
    # Push 25 entries; function should read only last 20
    for i in range(25):
        r.rpush(key, _metric_entry(i, float(10 + i)))
    mean, mx = get_validator_duration(r, PREFIX)
    # Last 20 entries: values 15..34 (i=5..24 -> 10+5=15 .. 10+24=34)
    expected_values = [float(10 + i) for i in range(5, 25)]
    assert mean == pytest.approx(sum(expected_values) / len(expected_values))
    assert mx == pytest.approx(max(expected_values))


def test_get_validator_duration_empty() -> None:
    r = _make_redis()
    mean, mx = get_validator_duration(r, PREFIX)
    assert mean is None
    assert mx is None


def test_get_validator_duration_few_entries() -> None:
    r = _make_redis()
    key = (
        f"{PREFIX}:metrics:history:dag_runner:dag:internals:"
        "CallValidatorFunction:stage_duration"
    )
    r.rpush(key, _metric_entry(1, 5.0))
    r.rpush(key, _metric_entry(2, 10.0))
    mean, mx = get_validator_duration(r, PREFIX)
    assert mean == pytest.approx(7.5)
    assert mx == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# 5. get_status_counts tests
# ---------------------------------------------------------------------------


def test_get_status_counts_normal() -> None:
    r = _make_redis()
    r.sadd(f"{PREFIX}:status:RUNNING", "id1", "id2", "id3")
    r.sadd(f"{PREFIX}:status:QUEUED", "id4", "id5")
    r.sadd(f"{PREFIX}:status:DONE", "id6")
    counts = get_status_counts(r, PREFIX)
    assert counts["RUNNING"] == 3
    assert counts["QUEUED"] == 2
    assert counts["DONE"] == 1
    assert counts["DISCARDED"] == 0


def test_get_status_counts_empty() -> None:
    r = _make_redis()
    counts = get_status_counts(r, PREFIX)
    assert counts == {"DONE": 0, "QUEUED": 0, "RUNNING": 0, "DISCARDED": 0}


# ---------------------------------------------------------------------------
# 6. collect_snapshot tests
# ---------------------------------------------------------------------------


def test_collect_snapshot_complete() -> None:
    r = _make_redis()
    spec = _make_spec()

    # Populate generation
    r.hset(f"{PREFIX}:run_state", "engine:total_generations", "5")

    # Populate frontier fitness
    r.rpush(
        f"{PREFIX}:metrics:history:program_metrics:valid_frontier_fitness",
        _metric_entry(5, 0.76),
    )

    # Populate program counts
    r.rpush(
        f"{PREFIX}:metrics:history:program_metrics:programs_total_count",
        _metric_entry(1, 100),
    )
    r.rpush(
        f"{PREFIX}:metrics:history:program_metrics:programs_valid_count",
        _metric_entry(1, 85),
    )

    # Populate status sets
    r.sadd(f"{PREFIX}:status:RUNNING", "id1", "id2")
    r.sadd(f"{PREFIX}:status:QUEUED", "id3")
    r.sadd(f"{PREFIX}:status:DONE", "id4", "id5", "id6")

    # Populate validator duration
    val_key = (
        f"{PREFIX}:metrics:history:dag_runner:dag:internals:"
        "CallValidatorFunction:stage_duration"
    )
    r.rpush(val_key, _metric_entry(1, 12.0))
    r.rpush(val_key, _metric_entry(2, 18.0))

    snap = collect_snapshot(r, spec, metric_names=["fitness"])

    assert snap.run_spec == spec
    assert snap.generation == 5
    assert snap.metrics == {"fitness": 0.76}
    assert snap.total_programs == 100
    assert snap.valid_programs == 85
    assert snap.running_programs == 2
    assert snap.queued_programs == 1
    assert snap.done_programs == 3
    assert snap.validator_mean_s == pytest.approx(15.0)
    assert snap.validator_max_s == pytest.approx(18.0)
    assert snap.total_keys is not None
    assert snap.error is None


def test_collect_snapshot_redis_error() -> None:
    """Redis connection error returns empty snapshot with error."""
    spec = _make_spec()

    class BrokenRedis:
        def hget(self, *a, **kw):
            raise ConnectionError("Connection refused")

    snap = collect_snapshot(BrokenRedis(), spec, metric_names=["fitness"])  # type: ignore[arg-type]
    assert snap.run_spec == spec
    assert snap.error is not None
    assert "Connection refused" in snap.error


def test_collect_snapshot_default_metric_names() -> None:
    """When metric_names is None, defaults to ['fitness']."""
    r = _make_redis()
    spec = _make_spec()
    r.rpush(
        f"{PREFIX}:metrics:history:program_metrics:valid_frontier_fitness",
        _metric_entry(1, 0.5),
    )
    snap = collect_snapshot(r, spec)
    assert "fitness" in snap.metrics
