"""Dispatcher acquires _producer_sema (not _buffer_sema) before spawning.

These tests use a fake engine surface so dispatcher_loop's semaphore
interaction can be observed without spinning up Redis / strategies /
hooks. We pin three properties:
  1. Each iteration acquires _producer_sema BEFORE create_task.
  2. _buffer_sema is NOT touched by the dispatcher.
  3. Early-stop (engine._running=False after acquire) releases _producer_sema.
"""

from __future__ import annotations

import asyncio

import pytest

from gigaevo.evolution.engine.dispatcher import dispatcher_loop


class _FakeDispatcherEngine:
    """Minimal surface dispatcher_loop reads. No real engine wiring."""

    def __init__(self, max_in_flight: int = 3) -> None:
        self._running = True
        self._producer_sema = asyncio.Semaphore(max_in_flight)
        self._buffer_sema = asyncio.Semaphore(max_in_flight)
        self._max = max_in_flight
        self._spawn_count = 0
        self._reached = False

    def _reached_mutant_cap(self) -> bool:
        return self._reached

    async def _select_parents_for_mutation(self):  # never called in these tests
        return []


@pytest.mark.asyncio
async def test_dispatcher_acquires_producer_sema(monkeypatch) -> None:
    engine = _FakeDispatcherEngine(max_in_flight=2)

    spawned: list[int] = []

    async def fake_run_one_mutant(eng, task_id: int) -> None:
        spawned.append(task_id)
        await asyncio.sleep(0.5)  # hold the slot

    monkeypatch.setattr(
        "gigaevo.evolution.engine.dispatcher.run_one_mutant", fake_run_one_mutant
    )

    task = asyncio.create_task(dispatcher_loop(engine))
    await asyncio.sleep(0.05)  # let dispatcher spawn up to capacity

    # Both initial slots taken from _producer_sema; _buffer_sema untouched.
    assert engine._producer_sema._value == 0
    assert engine._buffer_sema._value == 2
    assert len(spawned) == 2

    engine._running = False
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_dispatcher_early_stop_releases_producer_sema(monkeypatch) -> None:
    """If engine stops between acquire and spawn, the producer slot is returned."""
    engine = _FakeDispatcherEngine(max_in_flight=1)

    async def fake_run_one_mutant(eng, task_id: int) -> None:  # pragma: no cover
        raise AssertionError("should never spawn after early-stop")

    monkeypatch.setattr(
        "gigaevo.evolution.engine.dispatcher.run_one_mutant", fake_run_one_mutant
    )

    # Patch _reached_mutant_cap so the post-acquire check fires immediately
    # for the first iteration but leaves the loop guard simple.
    engine._reached = True

    task = asyncio.create_task(dispatcher_loop(engine))
    await asyncio.sleep(0.05)

    # acquire fired once, post-check tripped, release fired — back to full.
    assert engine._producer_sema._value == 1
    assert engine._buffer_sema._value == 1

    engine._running = False
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
