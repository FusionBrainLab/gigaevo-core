"""Behavior tests for the ConsolidationScheduler.

The write path counts cards written across sweeps and schedules exactly one
background consolidation pass per ``every_n``. We drive the counter and the
dispatch in isolation (a fake write stack, no LLM, no backend) and assert on
scheduling, the off-by-one boundary, non-blocking dispatch, and that the pass
runs under the shared write lock so it can never interleave with a live sweep.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from gigaevo.memory.core.events import memory_event_context
from gigaevo.memory.ideas_tracker.consolidation_scheduler import ConsolidationScheduler


class _FakeStack:
    """A built write stack: every component the consolidation pass reads is
    present and non-None so ``schedule`` does not short-circuit."""

    def __init__(self) -> None:
        self.store = object()
        self.gate = object()
        self.neighbors = object()
        self.consolidation_agent = object()


class _UnbuiltStack:
    """An un-built write stack: ``schedule`` must refuse to dispatch."""

    store = None
    gate = None
    neighbors = None
    consolidation_agent = None


def _scheduler(
    n: int,
    *,
    run_lock: asyncio.Lock | None = None,
    eps: float = 0.05,
    k: int = 5,
    stack: object | None = None,
) -> ConsolidationScheduler:
    return ConsolidationScheduler(
        stack=stack if stack is not None else _FakeStack(),
        run_lock=run_lock if run_lock is not None else asyncio.Lock(),
        every_n=n,
        eps=eps,
        k=k,
    )


def _dispatched_spy(calls: list[int]):  # noqa: ANN202
    """A ``schedule`` stand-in that reports a successful dispatch (returns True),
    so ``note_writes`` resets the cadence counter."""

    def _spy() -> bool:
        calls.append(1)
        return True

    return _spy


def test_scheduled_once_when_threshold_reached_then_counter_resets() -> None:
    sched = _scheduler(3)
    calls: list[int] = []
    sched.schedule = _dispatched_spy(calls)  # type: ignore[method-assign]
    sched.note_writes(2)
    assert calls == []
    sched.note_writes(1)
    assert calls == [1]
    assert sched.writes_since == 0


def test_threshold_reached_in_a_single_write_triggers() -> None:
    # Kills the >= -> > off-by-one: exactly N must fire.
    sched = _scheduler(3)
    calls: list[int] = []
    sched.schedule = _dispatched_spy(calls)  # type: ignore[method-assign]
    sched.note_writes(3)
    assert calls == [1]


def test_below_threshold_does_not_trigger() -> None:
    sched = _scheduler(3)
    calls: list[int] = []
    sched.schedule = _dispatched_spy(calls)  # type: ignore[method-assign]
    sched.note_writes(2)
    assert calls == []


def test_disabled_when_every_n_is_zero() -> None:
    sched = _scheduler(0)
    calls: list[int] = []
    sched.schedule = _dispatched_spy(calls)  # type: ignore[method-assign]
    sched.note_writes(100)
    assert calls == []


def test_cadence_retained_when_dispatch_cannot_run() -> None:
    # Finding 3: a failed dispatch (unbuilt stack, in-flight pass, or no running
    # loop) must NOT consume the cadence counter, or a persistently failing
    # dispatch silently disables consolidation forever.
    sched = _scheduler(3)
    sched.schedule = lambda: False  # type: ignore[method-assign]
    sched.note_writes(3)
    assert sched.writes_since == 3
    sched.note_writes(1)
    assert sched.writes_since == 4


def test_schedule_refuses_to_dispatch_when_stack_unbuilt() -> None:
    sched = _scheduler(1, stack=_UnbuiltStack())
    assert sched.schedule() is False
    assert sched.task is None


@pytest.mark.asyncio
async def test_schedule_dispatches_non_blocking_background_task(monkeypatch) -> None:
    sched = _scheduler(1)
    ran: dict[str, bool] = {}

    async def fake_consolidate(**kwargs) -> int:  # noqa: ANN003
        ran["called"] = True
        return 0

    monkeypatch.setattr(
        "gigaevo.memory.ideas_tracker.consolidation_scheduler.consolidate",
        fake_consolidate,
    )

    sched.schedule()
    assert isinstance(sched.task, asyncio.Task)
    assert not sched.task.done()  # dispatch did not block

    await sched.task
    assert ran.get("called") is True


@pytest.mark.asyncio
async def test_no_overlapping_consolidation_while_one_is_in_flight(
    monkeypatch,
) -> None:
    sched = _scheduler(1)
    gate = asyncio.Event()

    async def slow_consolidate(**kwargs) -> int:  # noqa: ANN003
        await gate.wait()
        return 0

    monkeypatch.setattr(
        "gigaevo.memory.ideas_tracker.consolidation_scheduler.consolidate",
        slow_consolidate,
    )

    sched.schedule()
    first = sched.task
    sched.schedule()
    assert sched.task is first  # not replaced while in-flight

    gate.set()
    await first


@pytest.mark.asyncio
async def test_consolidate_receives_policy_eps_and_k(monkeypatch) -> None:
    # The dedup thresholds are config, not hardcoded: whatever eps/k the
    # scheduler was built with reach the consolidation pass verbatim.
    captured: dict = {}

    async def capturing_consolidate(**kwargs) -> int:  # noqa: ANN003
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(
        "gigaevo.memory.ideas_tracker.consolidation_scheduler.consolidate",
        capturing_consolidate,
    )
    sched = _scheduler(1, eps=0.2, k=9)
    sched.schedule()
    await sched.task
    assert captured["eps"] == 0.2
    assert captured["k"] == 9


@pytest.mark.asyncio
async def test_successful_pass_emits_event_and_no_failure(
    tmp_path, monkeypatch
) -> None:
    async def ok_consolidate(**kwargs) -> int:  # noqa: ANN003
        return 2

    monkeypatch.setattr(
        "gigaevo.memory.ideas_tracker.consolidation_scheduler.consolidate",
        ok_consolidate,
    )
    sched = _scheduler(1)
    path = tmp_path / "memory_events.jsonl"
    with memory_event_context(event_path=path):
        sched.schedule()
        await sched.task
    assert sched.failures == 0
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    passes = [r for r in rows if r["event_type"] == "consolidation.pass"]
    assert passes and passes[0]["payload"]["merged"] == 2


@pytest.mark.asyncio
async def test_failed_pass_emits_event_and_counts_failure(
    tmp_path, monkeypatch
) -> None:
    # Finding 3: a consolidation pass that keeps throwing must be observable, not
    # a silent warning, so a wedged dedup surface can be detected from the event
    # log alone.
    async def boom_consolidate(**kwargs) -> int:  # noqa: ANN003
        raise RuntimeError("consolidate blew up")

    monkeypatch.setattr(
        "gigaevo.memory.ideas_tracker.consolidation_scheduler.consolidate",
        boom_consolidate,
    )
    sched = _scheduler(1)
    path = tmp_path / "memory_events.jsonl"
    with memory_event_context(event_path=path):
        sched.schedule()
        await sched.task
    assert sched.failures == 1
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    assert any(r["event_type"] == "consolidation.failed" for r in rows)


@pytest.mark.asyncio
async def test_drain_awaits_inflight_pass_to_completion(monkeypatch) -> None:
    # A pass scheduled by the final post-run sweep must finish before the event
    # loop is torn down (asyncio.run cancels pending tasks on exit), or the last
    # consolidation is silently lost. drain() blocks until the pass completes.
    sched = _scheduler(1)
    gate = asyncio.Event()
    done = {"finished": False}

    async def slow_consolidate(**kwargs) -> int:  # noqa: ANN003
        await gate.wait()
        done["finished"] = True
        return 0

    monkeypatch.setattr(
        "gigaevo.memory.ideas_tracker.consolidation_scheduler.consolidate",
        slow_consolidate,
    )
    sched.schedule()
    assert not sched.task.done()
    gate.set()
    await sched.drain(timeout=5.0)
    assert done["finished"] is True
    assert sched.task.done()


@pytest.mark.asyncio
async def test_drain_is_noop_when_nothing_scheduled() -> None:
    sched = _scheduler(0)
    await sched.drain(timeout=1.0)  # must not raise
    assert sched.task is None


@pytest.mark.asyncio
async def test_drain_cancels_a_stalled_pass(monkeypatch) -> None:
    # A hung memory-LLM call in the final pass must not hang engine teardown:
    # drain bounds the wait and cancels the pass instead of blocking forever.
    sched = _scheduler(1)
    never = asyncio.Event()

    async def hung_consolidate(**kwargs) -> int:  # noqa: ANN003
        await never.wait()
        return 0

    monkeypatch.setattr(
        "gigaevo.memory.ideas_tracker.consolidation_scheduler.consolidate",
        hung_consolidate,
    )
    sched.schedule()
    await sched.drain(timeout=0.05)
    assert sched.task.done()


@pytest.mark.asyncio
async def test_drain_returns_when_pass_suppresses_cancel(monkeypatch) -> None:
    # A final consolidation pass that swallows CancelledError must not wedge
    # engine teardown. drain must bound the wait with asyncio.wait (NOT
    # asyncio.wait_for, which awaits the cancel to be honoured and so blocks
    # indefinitely on a pass that ignores it), then abandon the orphan.
    sched = _scheduler(1)
    started = asyncio.Event()
    never = asyncio.Event()
    release = asyncio.Event()

    async def stubborn_consolidate(**kwargs) -> int:  # noqa: ANN003
        started.set()
        try:
            await never.wait()
        except asyncio.CancelledError:
            await release.wait()  # ignore the first cancel; keep running
        return 0

    monkeypatch.setattr(
        "gigaevo.memory.ideas_tracker.consolidation_scheduler.consolidate",
        stubborn_consolidate,
    )
    sched.schedule()
    await started.wait()
    try:
        # Outer bound: a wedged drain fails the test instead of hanging it.
        await asyncio.wait_for(sched.drain(timeout=0.05), timeout=3.0)
        assert not sched.task.done()  # orphan abandoned, not awaited forever
    finally:
        release.set()
        await asyncio.gather(sched.task, return_exceptions=True)


@pytest.mark.asyncio
async def test_consolidation_runs_under_the_write_lock(monkeypatch) -> None:
    # The pass rewrites the bank, so it must hold the same lock a live sweep
    # takes — never interleaved with one — for its whole duration.
    lock = asyncio.Lock()
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_consolidate(**kwargs) -> int:  # noqa: ANN003
        started.set()
        await release.wait()
        return 0

    monkeypatch.setattr(
        "gigaevo.memory.ideas_tracker.consolidation_scheduler.consolidate",
        slow_consolidate,
    )
    sched = _scheduler(1, run_lock=lock)

    sched.note_writes(1)
    task = sched.task
    await started.wait()
    assert lock.locked()  # consolidation holds the write lock while running

    release.set()
    await task
    assert not lock.locked()  # ...and releases it when the pass finishes
