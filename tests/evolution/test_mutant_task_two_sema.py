"""Two-sema accounting on every exit path of run_one_mutant.

Each test holds one producer_sema slot at entry (caller protocol — the
dispatcher acquires it before spawning) and verifies the post-condition:

  producer_sema: always released (no transfer semantics)
  buffer_sema  : transferred to ingestor only when slot_transferred=True
  ticket       : transferred only when slot_transferred=True
  _in_flight   : contains new_id iff slot_transferred=True
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from gigaevo.evolution.engine.mutant_task import run_one_mutant
from gigaevo.evolution.engine.refresh import ParentRefreshTicket
from gigaevo.evolution.mutation.parent_selector import RandomParentSelector
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState


def _make_parent() -> Program:
    return Program(code="def solve(): return 42", state=ProgramState.DONE)


class _FakeEngine:
    """Minimal engine surface used by run_one_mutant under the two-sema model."""

    def __init__(self, parent: Program, *, max_in_flight: int = 3) -> None:
        self.storage = AsyncMock()
        self.state = AsyncMock()
        self.mutation_operator = AsyncMock()
        self._in_flight: set[str] = set()
        self._inflight_tickets: dict[str, ParentRefreshTicket] = {}
        self._in_flight_lock = asyncio.Lock()
        self._producer_sema = asyncio.Semaphore(max_in_flight)
        self._buffer_sema = asyncio.Semaphore(max_in_flight)
        # LLM occupancy counter — incremented/decremented by run_one_mutant
        # around generate_one_mutation. Required attribute since the steady-
        # state engine started sampling per-LLM occupancy for backpressure.
        self._llm_active: int = 0

        self.metrics = type("M", (), {})()
        self.metrics.iteration = 0
        self.metrics.mutations_created = 0
        self.metrics.submitted_for_refresh = 0

        cfg = type("C", (), {})()
        cfg.loop_interval = 0.01
        cfg.parent_selector = RandomParentSelector(num_parents=1)
        self.config = cfg

        refresher = type("R", (), {})()

        async def _refresh_with_ticket(parents):
            return ParentRefreshTicket(refreshed=parents, _locks=[])

        refresher.refresh_with_ticket = _refresh_with_ticket
        self._parent_refresher = refresher
        self._parent = parent

    async def _select_parents_for_mutation(self):
        return [self._parent]

    async def _write_snapshot(self, **_kwargs) -> None:
        return None


async def _hold_producer_slot(engine: _FakeEngine) -> None:
    """Mirror the dispatcher contract: caller holds one producer slot."""
    await engine._producer_sema.acquire()


@pytest.mark.asyncio
async def test_success_path_transfers_buffer_and_ticket(monkeypatch) -> None:
    engine = _FakeEngine(_make_parent(), max_in_flight=3)
    await _hold_producer_slot(engine)

    async def fake_gen(**_k):
        return "new-id-1"

    monkeypatch.setattr(
        "gigaevo.evolution.engine.mutant_task.generate_one_mutation", fake_gen
    )

    result = await run_one_mutant(engine, task_id=0)

    assert result == "new-id-1"
    # producer slot: released
    assert engine._producer_sema._value == 3
    # buffer slot: held (transferred to ingestor)
    assert engine._buffer_sema._value == 2
    # in-flight & ticket: transferred
    assert "new-id-1" in engine._in_flight
    assert "new-id-1" in engine._inflight_tickets


@pytest.mark.asyncio
async def test_refresh_failure_releases_producer_no_buffer(monkeypatch) -> None:
    engine = _FakeEngine(_make_parent(), max_in_flight=3)
    await _hold_producer_slot(engine)

    async def boom(_parents):
        raise ValueError("refresh boom")

    engine._parent_refresher.refresh_with_ticket = boom

    async def fake_gen(**_k):  # pragma: no cover
        raise AssertionError("must not reach generate_one_mutation")

    monkeypatch.setattr(
        "gigaevo.evolution.engine.mutant_task.generate_one_mutation", fake_gen
    )

    result = await run_one_mutant(engine, task_id=0)

    assert result is None
    assert engine._producer_sema._value == 3
    # buffer never acquired
    assert engine._buffer_sema._value == 3
    assert not engine._in_flight


@pytest.mark.asyncio
async def test_llm_returns_none_releases_producer_no_buffer(monkeypatch) -> None:
    engine = _FakeEngine(_make_parent(), max_in_flight=2)
    await _hold_producer_slot(engine)

    async def fake_gen(**_k):
        return None

    monkeypatch.setattr(
        "gigaevo.evolution.engine.mutant_task.generate_one_mutation", fake_gen
    )

    result = await run_one_mutant(engine, task_id=0)

    assert result is None
    assert engine._producer_sema._value == 2
    assert engine._buffer_sema._value == 2  # untouched
    assert not engine._in_flight


@pytest.mark.asyncio
async def test_cancel_blocked_on_buffer_releases_producer(monkeypatch) -> None:
    """Cancel while producer is waiting on _buffer_sema.acquire().

    Sets up: buffer fully drained so the next acquire blocks. Cancel the
    task while it's parked. Both semaphores must end at their pre-test
    counts (producer back to full, buffer still drained).
    """
    engine = _FakeEngine(_make_parent(), max_in_flight=2)
    # Drain buffer to zero so the producer's acquire blocks.
    await engine._buffer_sema.acquire()
    await engine._buffer_sema.acquire()
    assert engine._buffer_sema._value == 0

    await _hold_producer_slot(engine)

    async def fake_gen(**_k):
        return "drift-id-1"

    monkeypatch.setattr(
        "gigaevo.evolution.engine.mutant_task.generate_one_mutation", fake_gen
    )

    task = asyncio.create_task(run_one_mutant(engine, task_id=0))
    await asyncio.sleep(0.05)  # let it park at _buffer_sema.acquire()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # producer: released. buffer: still zero (we hold both externally).
    assert engine._producer_sema._value == 2
    assert engine._buffer_sema._value == 0
    # _in_flight not populated; persist is the user-visible orphan we
    # acknowledge in the spec's cancellation matrix.
    assert "drift-id-1" not in engine._in_flight
