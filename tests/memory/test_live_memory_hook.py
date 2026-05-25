"""Tests for LiveMemoryRefreshHook bounded-sweep behaviour."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
import uuid

import pytest

from gigaevo.memory.live_memory_hook import LiveMemoryRefreshHook
from gigaevo.programs.program import Program


class _StubStorage:
    def __init__(self, programs: list[Program]) -> None:
        self._programs = list(programs)

    async def get_all(self, *, exclude=None):  # type: ignore[no-untyped-def]
        return list(self._programs)


class _RecordingTracker:
    def __init__(self) -> None:
        self.calls: list[list[Program]] = []

    async def run_increment(self, programs):  # type: ignore[no-untyped-def]
        self.calls.append(list(programs))


def _make_program(idx: int, created_at: datetime) -> Program:
    # Deterministic UUID5 so test assertions can compare on id; code field
    # required by Program schema.
    pid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"layer1-test-{idx}"))
    return Program(id=pid, code=f"# program {idx}", created_at=created_at)


@pytest.fixture
def five_programs() -> list[Program]:
    base = datetime(2026, 5, 24, 0, 0, 0, tzinfo=UTC)
    return [_make_program(i, base + timedelta(seconds=i)) for i in range(5)]


@pytest.mark.asyncio
async def test_unbounded_default_passes_all_programs(five_programs):
    """Default (max_programs_per_sweep=None) preserves legacy behaviour."""
    tracker = _RecordingTracker()
    storage = _StubStorage(five_programs)
    hook = LiveMemoryRefreshHook(tracker=tracker, storage=storage, refresh_every=1)

    await hook()

    assert len(tracker.calls) == 1
    assert {p.id for p in tracker.calls[0]} == {p.id for p in five_programs}


@pytest.mark.asyncio
async def test_bounded_sweep_passes_only_newest_n(five_programs):
    """max_programs_per_sweep=2 should pass the 2 NEWEST programs (by created_at)."""
    tracker = _RecordingTracker()
    storage = _StubStorage(five_programs)
    hook = LiveMemoryRefreshHook(
        tracker=tracker,
        storage=storage,
        refresh_every=1,
        max_programs_per_sweep=2,
    )

    await hook()

    assert len(tracker.calls) == 1
    passed_ids = {p.id for p in tracker.calls[0]}
    expected_newest = {five_programs[3].id, five_programs[4].id}
    assert passed_ids == expected_newest


@pytest.mark.asyncio
async def test_bounded_sweep_larger_than_pool_passes_all(five_programs):
    """max_programs_per_sweep > pool size returns the full pool, no error."""
    tracker = _RecordingTracker()
    storage = _StubStorage(five_programs)
    hook = LiveMemoryRefreshHook(
        tracker=tracker, storage=storage, refresh_every=1, max_programs_per_sweep=100
    )

    await hook()

    assert {p.id for p in tracker.calls[0]} == {p.id for p in five_programs}


@pytest.mark.asyncio
async def test_cadence_gate_unchanged_by_bound():
    """Bounded hook still respects refresh_every cadence."""
    tracker = _RecordingTracker()
    storage = _StubStorage([_make_program(0, datetime.now(UTC))])
    hook = LiveMemoryRefreshHook(
        tracker=tracker, storage=storage, refresh_every=3, max_programs_per_sweep=10
    )

    await hook()
    await hook()
    assert tracker.calls == []
    await hook()
    assert len(tracker.calls) == 1


@pytest.mark.asyncio
async def test_empty_storage_skips_without_error():
    """Empty storage + bounded sweep skips cleanly, no slice on empty list."""
    tracker = _RecordingTracker()
    storage = _StubStorage([])
    hook = LiveMemoryRefreshHook(
        tracker=tracker, storage=storage, refresh_every=1, max_programs_per_sweep=5
    )

    await hook()

    assert tracker.calls == []
