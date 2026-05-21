"""Tests for RunSnapshot -- immutable point-in-time run state."""

from __future__ import annotations

import pytest

from gigaevo.monitoring.run_spec import RunSpec
from gigaevo.monitoring.snapshot import RunSnapshot


def _make_spec(label: str = "O") -> RunSpec:
    return RunSpec(prefix="chains/synthetic/static", db=4, label=label)


# ---------------------------------------------------------------------------
# 1. Construction tests
# ---------------------------------------------------------------------------


def test_full_construction() -> None:
    spec = _make_spec()
    snap = RunSnapshot(
        run_spec=spec,
        generation=42,
        metrics={"fitness": 0.76, "prompt_length": 299.0},
        total_programs=100,
        valid_programs=85,
        running_programs=3,
        queued_programs=2,
        done_programs=95,
        validator_mean_s=12.5,
        validator_max_s=45.0,
        total_keys=157,
        pid=49341,
        pid_alive=True,
        error=None,
    )
    assert snap.run_spec == spec
    assert snap.generation == 42
    assert snap.metrics["fitness"] == 0.76
    assert snap.total_programs == 100
    assert snap.valid_programs == 85
    assert snap.running_programs == 3
    assert snap.queued_programs == 2
    assert snap.done_programs == 95
    assert snap.validator_mean_s == 12.5
    assert snap.validator_max_s == 45.0
    assert snap.total_keys == 157
    assert snap.pid == 49341
    assert snap.pid_alive is True
    assert snap.error is None


def test_construction_with_defaults() -> None:
    spec = _make_spec()
    snap = RunSnapshot(run_spec=spec)
    assert snap.generation is None
    assert snap.metrics == {}
    assert snap.total_programs is None
    assert snap.valid_programs is None
    assert snap.running_programs is None
    assert snap.queued_programs is None
    assert snap.done_programs is None
    assert snap.validator_mean_s is None
    assert snap.validator_max_s is None
    assert snap.total_keys is None
    assert snap.pid is None
    assert snap.pid_alive is None
    assert snap.error is None


def test_frozen() -> None:
    spec = _make_spec()
    snap = RunSnapshot(run_spec=spec, generation=10)
    with pytest.raises(AttributeError):
        snap.generation = 20  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 2. Computed property tests
# ---------------------------------------------------------------------------


def test_invalid_rate_normal() -> None:
    snap = RunSnapshot(run_spec=_make_spec(), total_programs=100, valid_programs=85)
    assert snap.invalid_rate == pytest.approx(0.15)


def test_invalid_rate_zero_total() -> None:
    snap = RunSnapshot(run_spec=_make_spec(), total_programs=0, valid_programs=0)
    assert snap.invalid_rate == 0.0


def test_invalid_rate_none_when_missing_total() -> None:
    snap = RunSnapshot(run_spec=_make_spec(), valid_programs=85)
    assert snap.invalid_rate is None


def test_invalid_rate_none_when_missing_valid() -> None:
    snap = RunSnapshot(run_spec=_make_spec(), total_programs=100)
    assert snap.invalid_rate is None


def test_has_error_true() -> None:
    snap = RunSnapshot(run_spec=_make_spec(), error="Connection refused")
    assert snap.has_error is True


def test_has_error_false() -> None:
    snap = RunSnapshot(run_spec=_make_spec(), error=None)
    assert snap.has_error is False


def test_is_stalled_true() -> None:
    spec = _make_spec()
    prev = RunSnapshot(
        run_spec=spec, generation=10, running_programs=0, total_programs=50
    )
    curr = RunSnapshot(
        run_spec=spec, generation=10, running_programs=0, total_programs=50
    )
    assert curr.is_stalled(prev) is True


def test_is_stalled_false_generation_advanced() -> None:
    spec = _make_spec()
    prev = RunSnapshot(
        run_spec=spec, generation=10, running_programs=0, total_programs=50
    )
    curr = RunSnapshot(
        run_spec=spec, generation=11, running_programs=0, total_programs=50
    )
    assert curr.is_stalled(prev) is False


def test_is_stalled_false_programs_running() -> None:
    spec = _make_spec()
    prev = RunSnapshot(
        run_spec=spec, generation=10, running_programs=0, total_programs=50
    )
    curr = RunSnapshot(
        run_spec=spec, generation=10, running_programs=2, total_programs=50
    )
    assert curr.is_stalled(prev) is False


def test_is_stalled_false_new_submissions() -> None:
    spec = _make_spec()
    prev = RunSnapshot(
        run_spec=spec, generation=10, running_programs=0, total_programs=50
    )
    curr = RunSnapshot(
        run_spec=spec, generation=10, running_programs=0, total_programs=52
    )
    assert curr.is_stalled(prev) is False


def test_is_stalled_false_when_generation_none() -> None:
    spec = _make_spec()
    prev = RunSnapshot(run_spec=spec, generation=None)
    curr = RunSnapshot(run_spec=spec, generation=10)
    assert curr.is_stalled(prev) is False


# ---------------------------------------------------------------------------
# 3. Factory function tests
# ---------------------------------------------------------------------------


def test_empty_factory() -> None:
    spec = _make_spec()
    snap = RunSnapshot.empty(spec)
    assert snap.run_spec == spec
    assert snap.generation is None
    assert snap.metrics == {}
    assert snap.total_programs is None
    assert snap.valid_programs is None
    assert snap.error is None
