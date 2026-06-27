"""Behavior tests for the IdeaTracker consolidation trigger.

The write path counts cards written across sweeps and schedules exactly one
background consolidation pass per ``consolidation_every_n``. We drive the counter
and the dispatch in isolation (no LLM, no backend) and assert on scheduling, the
off-by-one boundary, and that dispatch does not block.
"""

from __future__ import annotations

import asyncio

import pytest

from gigaevo.memory.ideas_tracker.ideas_tracker import IdeaTracker


def _tracker(n: int) -> IdeaTracker:
    return IdeaTracker(memory_write_enabled=False, consolidation_every_n=n)


def test_scheduled_once_when_threshold_reached_then_counter_resets() -> None:
    tracker = _tracker(3)
    calls: list[int] = []
    tracker._schedule_consolidation = lambda: calls.append(1)  # type: ignore[method-assign]
    tracker._note_writes_and_maybe_consolidate(2)
    assert calls == []
    tracker._note_writes_and_maybe_consolidate(1)
    assert calls == [1]
    assert tracker._writes_since_consolidation == 0


def test_threshold_reached_in_a_single_write_triggers() -> None:
    # Kills the >= -> > off-by-one: exactly N must fire.
    tracker = _tracker(3)
    calls: list[int] = []
    tracker._schedule_consolidation = lambda: calls.append(1)  # type: ignore[method-assign]
    tracker._note_writes_and_maybe_consolidate(3)
    assert calls == [1]


def test_below_threshold_does_not_trigger() -> None:
    tracker = _tracker(3)
    calls: list[int] = []
    tracker._schedule_consolidation = lambda: calls.append(1)  # type: ignore[method-assign]
    tracker._note_writes_and_maybe_consolidate(2)
    assert calls == []


def test_disabled_when_every_n_is_zero() -> None:
    tracker = _tracker(0)
    calls: list[int] = []
    tracker._schedule_consolidation = lambda: calls.append(1)  # type: ignore[method-assign]
    tracker._note_writes_and_maybe_consolidate(100)
    assert calls == []


@pytest.mark.asyncio
async def test_schedule_dispatches_non_blocking_background_task(monkeypatch) -> None:
    tracker = _tracker(1)
    ran: dict[str, bool] = {}

    async def fake_consolidate(**kwargs) -> int:  # noqa: ANN003
        ran["called"] = True
        return 0

    monkeypatch.setattr(
        "gigaevo.memory.ideas_tracker.ideas_tracker.consolidate", fake_consolidate
    )
    tracker._store = object()
    tracker._gate = object()
    tracker._consolidation_neighbors = object()
    tracker._consolidation_agent = object()

    tracker._schedule_consolidation()
    assert isinstance(tracker._consolidation_task, asyncio.Task)
    assert not tracker._consolidation_task.done()  # dispatch did not block

    await tracker._consolidation_task
    assert ran.get("called") is True


@pytest.mark.asyncio
async def test_no_overlapping_consolidation_while_one_is_in_flight(
    monkeypatch,
) -> None:
    tracker = _tracker(1)
    gate = asyncio.Event()

    async def slow_consolidate(**kwargs) -> int:  # noqa: ANN003
        await gate.wait()
        return 0

    monkeypatch.setattr(
        "gigaevo.memory.ideas_tracker.ideas_tracker.consolidate", slow_consolidate
    )
    tracker._store = object()
    tracker._gate = object()
    tracker._consolidation_neighbors = object()
    tracker._consolidation_agent = object()

    tracker._schedule_consolidation()
    first = tracker._consolidation_task
    tracker._schedule_consolidation()
    assert tracker._consolidation_task is first  # not replaced while in-flight

    gate.set()
    await first
