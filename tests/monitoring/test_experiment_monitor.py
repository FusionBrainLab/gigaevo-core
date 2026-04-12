"""Tests for ExperimentMonitor -- collects snapshots for all runs."""

from __future__ import annotations

import json

import fakeredis

from gigaevo.monitoring.experiment_monitor import ExperimentMonitor, RunConfig
from gigaevo.monitoring.run_spec import RunSpec


def _metric_entry(step: int, value: float, ts: int = 123) -> str:
    return json.dumps({"s": step, "v": value, "t": ts, "k": "scalar"})


def _populate_run(
    server: fakeredis.FakeServer,
    db: int,
    prefix: str,
    generation: int,
    fitness: float,
    total: int,
    valid: int,
) -> None:
    """Populate a fakeredis DB with standard run data."""
    r = fakeredis.FakeRedis(server=server, db=db, decode_responses=True)
    r.hset(f"{prefix}:run_state", "engine:total_generations", str(generation))
    r.rpush(
        f"{prefix}:metrics:history:program_metrics:valid_frontier_fitness",
        _metric_entry(generation, fitness),
    )
    r.rpush(
        f"{prefix}:metrics:history:program_metrics:programs_total_count",
        _metric_entry(1, total),
    )
    r.rpush(
        f"{prefix}:metrics:history:program_metrics:programs_valid_count",
        _metric_entry(1, valid),
    )
    r.sadd(f"{prefix}:status:RUNNING", "id1")
    r.sadd(f"{prefix}:status:DONE", "id2", "id3")


# ---------------------------------------------------------------------------
# 1. collect with multiple runs
# ---------------------------------------------------------------------------


def test_collect_multiple_runs() -> None:
    server = fakeredis.FakeServer()
    _populate_run(
        server, 4, "prefix_a", generation=10, fitness=0.76, total=100, valid=85
    )
    _populate_run(
        server, 5, "prefix_b", generation=20, fitness=0.82, total=200, valid=190
    )
    _populate_run(server, 6, "prefix_c", generation=5, fitness=0.50, total=50, valid=40)

    monitor = ExperimentMonitor(
        redis_factory=lambda db: fakeredis.FakeRedis(
            server=server, db=db, decode_responses=True
        )
    )

    runs = [
        RunConfig(RunSpec(prefix="prefix_a", db=4, label="A")),
        RunConfig(RunSpec(prefix="prefix_b", db=5, label="B")),
        RunConfig(RunSpec(prefix="prefix_c", db=6, label="C")),
    ]

    snapshots = monitor.collect(runs)
    assert len(snapshots) == 3

    assert snapshots[0].generation == 10
    assert snapshots[0].metrics["fitness"] == 0.76
    assert snapshots[0].total_programs == 100

    assert snapshots[1].generation == 20
    assert snapshots[1].metrics["fitness"] == 0.82
    assert snapshots[1].total_programs == 200

    assert snapshots[2].generation == 5
    assert snapshots[2].metrics["fitness"] == 0.50
    assert snapshots[2].total_programs == 50


# ---------------------------------------------------------------------------
# 2. collect with metric_names per run
# ---------------------------------------------------------------------------


def test_collect_with_different_metric_names() -> None:
    server = fakeredis.FakeServer()
    r4 = fakeredis.FakeRedis(server=server, db=4, decode_responses=True)
    r4.hset("prefix_a:run_state", "engine:total_generations", "10")
    r4.rpush(
        "prefix_a:metrics:history:program_metrics:valid_frontier_fitness",
        _metric_entry(10, 0.76),
    )

    r5 = fakeredis.FakeRedis(server=server, db=5, decode_responses=True)
    r5.hset("prefix_b:run_state", "engine:total_generations", "20")
    r5.rpush(
        "prefix_b:metrics:history:program_metrics:valid_frontier_fitness",
        _metric_entry(20, 0.82),
    )
    r5.rpush(
        "prefix_b:metrics:history:program_metrics:valid_frontier_prompt_length",
        _metric_entry(20, 299.0),
    )

    monitor = ExperimentMonitor(
        redis_factory=lambda db: fakeredis.FakeRedis(
            server=server, db=db, decode_responses=True
        )
    )

    runs = [
        RunConfig(
            RunSpec(prefix="prefix_a", db=4, label="A"),
            metric_names=["fitness"],
        ),
        RunConfig(
            RunSpec(prefix="prefix_b", db=5, label="B"),
            metric_names=["fitness", "prompt_length"],
        ),
    ]

    snapshots = monitor.collect(runs)
    assert set(snapshots[0].metrics.keys()) == {"fitness"}
    assert set(snapshots[1].metrics.keys()) == {"fitness", "prompt_length"}
    assert snapshots[1].metrics["prompt_length"] == 299.0


# ---------------------------------------------------------------------------
# 3. collect with one run failing
# ---------------------------------------------------------------------------


def test_collect_one_run_fails() -> None:
    server = fakeredis.FakeServer()
    _populate_run(
        server, 4, "prefix_a", generation=10, fitness=0.76, total=100, valid=85
    )
    _populate_run(server, 6, "prefix_c", generation=5, fitness=0.50, total=50, valid=40)

    def failing_factory(db: int) -> fakeredis.FakeRedis:
        if db == 5:
            raise ConnectionError("Connection refused")
        return fakeredis.FakeRedis(server=server, db=db, decode_responses=True)

    monitor = ExperimentMonitor(redis_factory=failing_factory)

    runs = [
        RunConfig(RunSpec(prefix="prefix_a", db=4, label="A")),
        RunConfig(RunSpec(prefix="prefix_b", db=5, label="B")),
        RunConfig(RunSpec(prefix="prefix_c", db=6, label="C")),
    ]

    snapshots = monitor.collect(runs)
    assert len(snapshots) == 3

    # First and third succeed
    assert snapshots[0].generation == 10
    assert snapshots[0].error is None

    # Second has error
    assert snapshots[1].error is not None
    assert "Connection refused" in snapshots[1].error

    # Third still succeeds despite second failing
    assert snapshots[2].generation == 5
    assert snapshots[2].error is None


# ---------------------------------------------------------------------------
# 4. collect empty runs list
# ---------------------------------------------------------------------------


def test_collect_empty_runs() -> None:
    monitor = ExperimentMonitor(
        redis_factory=lambda db: fakeredis.FakeRedis(decode_responses=True)
    )
    snapshots = monitor.collect([])
    assert snapshots == []


# ---------------------------------------------------------------------------
# 5. collect with PID info
# ---------------------------------------------------------------------------


def test_collect_with_pid() -> None:
    server = fakeredis.FakeServer()
    _populate_run(
        server, 4, "prefix_a", generation=10, fitness=0.76, total=100, valid=85
    )

    monitor = ExperimentMonitor(
        redis_factory=lambda db: fakeredis.FakeRedis(
            server=server, db=db, decode_responses=True
        )
    )

    runs = [
        RunConfig(
            RunSpec(prefix="prefix_a", db=4, label="A"),
            pid=12345,
        ),
    ]

    snapshots = monitor.collect(runs)
    assert len(snapshots) == 1
    assert snapshots[0].pid == 12345
    assert isinstance(snapshots[0].pid_alive, bool)
