"""End-to-end tests for `coalesce_refresh` mode in mutant_task.

These tests stub the LLM call and exercise the real `ParentRefresher`,
real `_inflight_tickets`, real `_buffer_sema`/`_producer_sema` plumbing,
and a `FakeDag` standing in for the DAG runner.

Contract being verified:
  1. With `coalesce_refresh=True`, `refresh_if_stale` is the entry point
     and no ticket lands in `_inflight_tickets`.
  2. `submitted_for_refresh` increments by `stale_count` (NOT by the
     full parents list — fresh-skip parents do not count).
  3. Two concurrent mutations of the same parent coalesce: only one
     DAG flip happens. The second mutation's mutant_task still
     completes.
  4. After one mutation completes (DONE or DISCARDED) and the ingestor
     sweeps, the parent's `_fresh` entry is dropped and the next
     mutation triggers a fresh flip.
  5. With `coalesce_refresh=False`, the legacy ticket path is exercised
     unchanged (regression anchor).
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock
import uuid

import pytest

from gigaevo.evolution.engine.config import SteadyStateEngineConfig
from gigaevo.evolution.engine.ingestor import poll_and_ingest
from gigaevo.evolution.engine.mutant_task import run_one_mutant
from gigaevo.evolution.engine.refresh import ParentRefresher
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState
from tests.evolution._fake_dag import FakeDag


class _FakeEngine:
    """Minimal engine surface for `run_one_mutant`."""

    def __init__(
        self,
        *,
        storage,
        parents: list[Program],
        coalesce_refresh: bool,
    ) -> None:
        self.storage = storage
        self._ss_config = SteadyStateEngineConfig(
            coalesce_refresh=coalesce_refresh,
            max_in_flight=3,
        )
        self.config = self._ss_config
        self._parent_refresher = ParentRefresher(storage=storage, poll_interval=0.01)
        self._in_flight: set[str] = set()
        self._inflight_tickets: dict = {}
        self._in_flight_lock = asyncio.Lock()
        self._producer_sema = asyncio.Semaphore(3)
        self._buffer_sema = asyncio.Semaphore(3)
        self._llm_active = 0

        self.metrics = type("M", (), {})()
        self.metrics.submitted_for_refresh = 0
        self.metrics.iteration = 0
        self.metrics.mutations_created = 0
        self._parents_to_select = parents

        async def _write_snapshot(**_k):
            return None

        self._write_snapshot = _write_snapshot

        async def _select(*_args, **_kwargs):
            return list(self._parents_to_select)

        self._select_parents_for_mutation = _select

        self.mutation_operator = AsyncMock()
        self.state = AsyncMock()

    async def acquire_producer(self) -> None:
        await self._producer_sema.acquire()


@pytest.fixture
def mock_generate_one_mutation(monkeypatch):
    """Patch generate_one_mutation to skip the LLM call and just return an id."""

    async def _fake(parents, mutator, storage, state_manager, iteration, task_id):
        new_id = str(uuid.uuid4())
        prog = Program(
            id=new_id,
            code="def m(): pass",
            state=ProgramState.QUEUED,
        )
        prog.lineage.parents = [p.id for p in parents]
        await storage.add(prog)
        return new_id

    monkeypatch.setattr(
        "gigaevo.evolution.engine.mutant_task.generate_one_mutation",
        _fake,
    )
    return _fake


@pytest.mark.asyncio
async def test_coalesce_mode_uses_refresh_if_stale_no_ticket(
    fakeredis_storage, mock_generate_one_mutation
):
    """coalesce_refresh=True → refresh_if_stale path → no ticket in _inflight_tickets."""
    fake_dag = FakeDag(fakeredis_storage)
    fake_dag.start()
    try:
        p1 = await fake_dag.add_program("p1")
        engine = _FakeEngine(
            storage=fakeredis_storage,
            parents=[p1],
            coalesce_refresh=True,
        )
        await engine.acquire_producer()
        new_id = await run_one_mutant(engine, task_id=0)
        assert new_id is not None
        assert engine._inflight_tickets == {}
        assert new_id in engine._in_flight
        assert p1.id in engine._parent_refresher._fresh
        assert engine.metrics.submitted_for_refresh == 1
    finally:
        await fake_dag.stop()


@pytest.mark.asyncio
async def test_coalesce_mode_skips_flip_when_fresh(
    fakeredis_storage, mock_generate_one_mutation
):
    """Two sequential mutations from the same parent — the second sees
    the parent fresh and skips the flip."""
    fake_dag = FakeDag(fakeredis_storage)
    fake_dag.start()
    try:
        p1 = await fake_dag.add_program("p1")
        engine = _FakeEngine(
            storage=fakeredis_storage,
            parents=[p1],
            coalesce_refresh=True,
        )
        await engine.acquire_producer()
        await run_one_mutant(engine, task_id=0)
        assert fake_dag.flip_count_for(p1.id) == 1
        assert engine.metrics.submitted_for_refresh == 1

        await engine.acquire_producer()
        await run_one_mutant(engine, task_id=1)
        assert fake_dag.flip_count_for(p1.id) == 1
        assert engine.metrics.submitted_for_refresh == 1
    finally:
        await fake_dag.stop()


@pytest.mark.asyncio
async def test_coalesce_mode_two_concurrent_mutations_coalesce_to_one_flip(
    fakeredis_storage, mock_generate_one_mutation
):
    """Two producers picking the same stale parent run concurrently.
    Exactly one DAG flip is paid; both producers complete and register
    distinct children in `_in_flight`."""
    fake_dag = FakeDag(fakeredis_storage)
    fake_dag.start()
    try:
        p1 = await fake_dag.add_program("p1")
        engine = _FakeEngine(
            storage=fakeredis_storage,
            parents=[p1],
            coalesce_refresh=True,
        )
        await engine.acquire_producer()
        await engine.acquire_producer()

        new_id_a, new_id_b = await asyncio.gather(
            run_one_mutant(engine, task_id=0),
            run_one_mutant(engine, task_id=1),
        )
        assert new_id_a is not None and new_id_b is not None
        assert new_id_a != new_id_b
        assert fake_dag.flip_count_for(p1.id) == 1
        assert {new_id_a, new_id_b} <= engine._in_flight
        assert engine._inflight_tickets == {}
        assert engine.metrics.submitted_for_refresh == 1
    finally:
        await fake_dag.stop()


@pytest.mark.asyncio
async def test_coalesce_mode_child_done_invalidates_parent_for_next_pick(
    fakeredis_storage, mock_generate_one_mutation
):
    """End-to-end through the ingestor: one mutation completes, the
    ingestor sweep invalidates the parent, the NEXT mutation triggers
    a fresh flip."""
    fake_dag = FakeDag(fakeredis_storage)
    fake_dag.start()
    try:
        p1 = await fake_dag.add_program("p1")
        engine = _FakeEngine(
            storage=fakeredis_storage,
            parents=[p1],
            coalesce_refresh=True,
        )
        engine.strategy = AsyncMock()
        engine.strategy.add.return_value = True
        engine.config.program_acceptor = type("A", (), {})()
        engine.config.program_acceptor.is_accepted = lambda _p: True
        engine.config.post_step_hook_timeout_s = 1.0
        engine.config.post_step_hook_cancel_grace_s = 0.5
        engine._post_step_hook = None
        engine.storage.snapshot = type("S", (), {})()
        engine.storage.snapshot.bump = lambda **_k: None

        def _record(_a, _v, _s):
            return None

        engine.metrics.record_ingestion_metrics = _record
        engine.metrics.programs_processed = 0

        async def _notify(_p, _o):
            return None

        engine._notify_hook = _notify

        await engine.acquire_producer()
        new_id = await run_one_mutant(engine, task_id=0)
        assert fake_dag.flip_count_for(p1.id) == 1
        assert p1.id in engine._parent_refresher._fresh

        for _ in range(200):
            child = await fakeredis_storage.get(new_id)
            if child is not None and child.state == ProgramState.DONE:
                break
            await asyncio.sleep(0.005)
        else:
            pytest.fail("child DAG never reached DONE")

        await poll_and_ingest(engine)
        assert p1.id not in engine._parent_refresher._fresh

        await engine.acquire_producer()
        await run_one_mutant(engine, task_id=1)
        assert fake_dag.flip_count_for(p1.id) == 2
        assert engine.metrics.submitted_for_refresh == 2
    finally:
        await fake_dag.stop()


@pytest.mark.asyncio
async def test_legacy_mode_still_uses_ticket_path(
    fakeredis_storage, mock_generate_one_mutation
):
    """Regression anchor: with coalesce_refresh=False, the legacy path
    is unchanged — a ticket lands in `_inflight_tickets` keyed by the
    new mutant id."""
    fake_dag = FakeDag(fakeredis_storage)
    fake_dag.start()
    try:
        p1 = await fake_dag.add_program("p1")
        engine = _FakeEngine(
            storage=fakeredis_storage,
            parents=[p1],
            coalesce_refresh=False,
        )
        await engine.acquire_producer()
        new_id = await run_one_mutant(engine, task_id=0)
        assert new_id is not None
        assert new_id in engine._in_flight
        assert new_id in engine._inflight_tickets
        assert engine.metrics.submitted_for_refresh == 1
        assert engine._parent_refresher._fresh == set()
    finally:
        await fake_dag.stop()
