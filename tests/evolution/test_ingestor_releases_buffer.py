"""Ingestor releases _buffer_sema on DONE/DISCARDED, never _producer_sema.

The producer task already released _producer_sema in its finally when the
mutant entered _in_flight; the ingestor's job is to release the buffer slot
the producer transferred under _in_flight_lock.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock
import uuid

import pytest

from gigaevo.evolution.engine.ingestor import poll_and_ingest
from gigaevo.evolution.engine.refresh import ParentRefreshTicket
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState


class _FakeIngestorEngine:
    def __init__(self, max_in_flight: int = 3) -> None:
        self._in_flight: set[str] = set()
        self._inflight_tickets: dict[str, ParentRefreshTicket] = {}
        self._in_flight_lock = asyncio.Lock()
        self._producer_sema = asyncio.Semaphore(max_in_flight)
        self._buffer_sema = asyncio.Semaphore(max_in_flight)

        self.storage = AsyncMock()
        self.strategy = AsyncMock()

        # config surface
        cfg = type("C", (), {})()
        cfg.loop_interval = 0.01
        cfg.program_acceptor = type("A", (), {})()
        cfg.program_acceptor.is_accepted = lambda _p: True
        cfg.post_step_hook_timeout_s = 1.0
        cfg.post_step_hook_cancel_grace_s = 0.5
        self.config = cfg

        self._post_step_hook = None

        # metrics surface
        self.metrics = type("M", (), {})()
        self.metrics.programs_processed = 0

        def _record(_a, _v, _s):
            return None

        self.metrics.record_ingestion_metrics = _record

        async def _notify(_p, _o):
            return None

        self._notify_hook = _notify

        async def _write_snapshot(**_k):
            return None

        self._write_snapshot = _write_snapshot

    async def _add_in_flight(self, pid: str) -> None:
        # Mirror what the producer does on transfer: acquire buffer_sema,
        # then atomically register under _in_flight_lock with a ticket.
        await self._buffer_sema.acquire()
        async with self._in_flight_lock:
            self._in_flight.add(pid)
            self._inflight_tickets[pid] = ParentRefreshTicket(refreshed=[], _locks=[])


@pytest.mark.asyncio
async def test_ingestor_done_releases_buffer_not_producer() -> None:
    engine = _FakeIngestorEngine(max_in_flight=3)
    pid = str(uuid.uuid4())
    await engine._add_in_flight(pid)
    assert engine._buffer_sema._value == 2  # one buffer slot held by producer
    assert engine._producer_sema._value == 3  # producer slot already returned

    # Strategy.add returns True so the program is accepted (no DISCARDED transition).
    engine.strategy.add.return_value = True

    done_prog = Program(
        id=pid, code="def f(): pass", state=ProgramState.DONE, metrics={}
    )
    engine.storage.mget.return_value = [done_prog]

    handled = await poll_and_ingest(engine)

    assert handled == 1
    # Buffer slot returned to pool; producer pool untouched.
    assert engine._buffer_sema._value == 3
    assert engine._producer_sema._value == 3
    assert not engine._in_flight
    assert not engine._inflight_tickets


@pytest.mark.asyncio
async def test_ingestor_discarded_releases_buffer_not_producer() -> None:
    engine = _FakeIngestorEngine(max_in_flight=3)
    pid = str(uuid.uuid4())
    await engine._add_in_flight(pid)

    discarded_prog = Program(
        id=pid, code="def f(): pass", state=ProgramState.DISCARDED, metrics={}
    )
    engine.storage.mget.return_value = [discarded_prog]

    handled = await poll_and_ingest(engine)

    assert handled == 1
    assert engine._buffer_sema._value == 3
    assert engine._producer_sema._value == 3
    assert not engine._in_flight


@pytest.mark.asyncio
async def test_ingestor_vanished_program_releases_buffer() -> None:
    engine = _FakeIngestorEngine(max_in_flight=3)
    pid = str(uuid.uuid4())
    await engine._add_in_flight(pid)

    # storage.mget returns no entries — id leaked.
    engine.storage.mget.return_value = []

    handled = await poll_and_ingest(engine)

    assert handled == 1
    assert engine._buffer_sema._value == 3
    assert engine._producer_sema._value == 3
    assert not engine._in_flight
