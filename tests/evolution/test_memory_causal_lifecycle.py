from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gigaevo.evolution.engine.core import EvolutionEngine
from gigaevo.evolution.engine.dispatcher import dispatcher_loop
from gigaevo.evolution.engine.ingestor import _ingest_batch
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState


class _MissingDecisionLifecycleSink:
    def __init__(self) -> None:
        self.edges: list[dict] = []

    def has_attempt_decision(self, _attempt_id: str) -> bool:
        return False

    def record_attempt_failure(self, **_kwargs) -> bool:
        return False

    def link_attempt_child(self, **_kwargs) -> bool:
        return False

    def record_missing_child(self, *_args, **_kwargs) -> bool:
        return False

    def reconcile_unlinked_attempts(self, **_kwargs) -> int:
        return 0

    def pending_child_ids(self) -> tuple[str, ...]:
        return ()

    def record_mutation_edge(self, **payload) -> None:
        self.edges.append(payload)

    def record_mutation_outcome(self, _event) -> None:
        return None

    def record_archive_disposition(self, _child_id, *, accepted: bool) -> None:
        del accepted

    def pending_archive_child_ids(self) -> tuple[str, ...]:
        return ()


def test_memory_free_child_records_topology_without_a_decision_link() -> None:
    engine = EvolutionEngine.__new__(EvolutionEngine)
    sink = _MissingDecisionLifecycleSink()
    engine._memory_outcome_sink = sink

    engine._link_memory_attempt_child(
        attempt_id="attempt-without-decision",
        child_id="child-id",
        parent_id="parent-id",
        island_id="main",
        completion_ordinal=7,
    )

    assert sink.edges == [
        {
            "parent_id": "parent-id",
            "child_id": "child-id",
            "island_id": "main",
            "completion_ordinal": 7,
        }
    ]


@pytest.mark.asyncio
async def test_startup_ingestion_aborts_when_causal_outcome_is_not_durable() -> None:
    child = Program(code="def child(): return 1", state=ProgramState.DONE)
    outcome_error = RuntimeError("ledger unavailable")
    storage = SimpleNamespace(
        get_ids_by_status=AsyncMock(return_value=[child.id]),
        mget=AsyncMock(return_value=[child]),
        batch_transition_by_ids=AsyncMock(),
    )
    strategy = SimpleNamespace(
        get_program_ids=AsyncMock(return_value=[]),
        add=AsyncMock(return_value=True),
    )
    engine = SimpleNamespace(
        _resumed=False,
        storage=storage,
        strategy=strategy,
        metrics=SimpleNamespace(iteration=0),
        _record_memory_outcome=AsyncMock(side_effect=outcome_error),
    )

    with pytest.raises(
        RuntimeError, match=f"startup causal outcome write failed for {child.id}"
    ) as raised:
        await EvolutionEngine._ingest_completed_programs(engine)

    assert raised.value.__cause__ is outcome_error
    assert child.state == ProgramState.DONE
    engine._record_memory_outcome.assert_awaited_once_with(child)
    strategy.add.assert_not_awaited()
    storage.batch_transition_by_ids.assert_not_awaited()


@pytest.mark.asyncio
async def test_outcome_sink_failure_retains_done_child_for_retry() -> None:
    child = Program(code="def child(): return 1", state=ProgramState.DONE)
    storage = SimpleNamespace(mget=AsyncMock(return_value=[child]))
    outcome = AsyncMock(side_effect=[RuntimeError("ledger unavailable"), None])
    strategy = SimpleNamespace(add=AsyncMock(return_value=True))
    metrics = SimpleNamespace(
        programs_processed=0,
        record_ingestion_metrics=Mock(),
    )
    engine = SimpleNamespace(
        storage=storage,
        strategy=strategy,
        metrics=metrics,
        _outcome_failure_counts={},
        config=SimpleNamespace(
            program_acceptor=SimpleNamespace(is_accepted=Mock(return_value=True)),
            causal_outcome_max_consecutive_failures=3,
        ),
        _record_memory_outcome=outcome,
        _record_memory_archive_disposition=Mock(),
        _notify_hook=AsyncMock(),
    )

    added, handled = await _ingest_batch(engine, [child.id])

    assert (added, handled) == (0, [])
    assert child.state == ProgramState.DONE
    assert metrics.programs_processed == 0
    strategy.add.assert_not_awaited()

    added, handled = await _ingest_batch(engine, [child.id])

    assert (added, handled) == (1, [child.id])
    assert metrics.programs_processed == 1
    strategy.add.assert_awaited_once_with(child)
    assert outcome.await_count == 2


@pytest.mark.asyncio
async def test_persistent_outcome_sink_failure_trips_circuit_breaker() -> None:
    child = Program(code="def child(): return 1", state=ProgramState.DONE)
    engine = SimpleNamespace(
        storage=SimpleNamespace(mget=AsyncMock(return_value=[child])),
        strategy=SimpleNamespace(add=AsyncMock()),
        metrics=SimpleNamespace(
            programs_processed=0,
            record_ingestion_metrics=Mock(),
        ),
        _outcome_failure_counts={},
        config=SimpleNamespace(
            program_acceptor=SimpleNamespace(is_accepted=Mock(return_value=True)),
            causal_outcome_max_consecutive_failures=2,
        ),
        _record_memory_outcome=AsyncMock(side_effect=RuntimeError("ledger down")),
        _notify_hook=AsyncMock(),
    )

    assert await _ingest_batch(engine, [child.id]) == (0, [])
    with pytest.raises(RuntimeError, match="2 consecutive times"):
        await _ingest_batch(engine, [child.id])

    assert child.state == ProgramState.DONE
    engine.strategy.add.assert_not_awaited()


@pytest.mark.asyncio
async def test_dispatcher_surfaces_mutant_task_exception(monkeypatch) -> None:
    engine = SimpleNamespace(
        _running=True,
        _producer_sema=asyncio.Semaphore(1),
        _can_dispatch_mutant=Mock(return_value=True),
        config=SimpleNamespace(max_consecutive_mutation_failures=3),
    )

    async def failing_mutant(active_engine, _task_id):
        active_engine._producer_sema.release()
        raise RuntimeError("pre-exposure configuration failure")

    monkeypatch.setattr(
        "gigaevo.evolution.engine.dispatcher.run_one_mutant", failing_mutant
    )

    with pytest.raises(RuntimeError, match="pre-exposure configuration failure"):
        await asyncio.wait_for(dispatcher_loop(engine), timeout=1.0)


@pytest.mark.asyncio
async def test_dispatcher_bounds_persistent_empty_mutations(monkeypatch) -> None:
    engine = SimpleNamespace(
        _running=True,
        _producer_sema=asyncio.Semaphore(1),
        _can_dispatch_mutant=Mock(return_value=True),
        config=SimpleNamespace(max_consecutive_mutation_failures=3),
    )

    async def empty_mutant(active_engine, _task_id):
        active_engine._producer_sema.release()
        return None

    monkeypatch.setattr(
        "gigaevo.evolution.engine.dispatcher.run_one_mutant", empty_mutant
    )

    with pytest.raises(RuntimeError, match="3 consecutive times"):
        await asyncio.wait_for(dispatcher_loop(engine), timeout=1.0)


@pytest.mark.asyncio
async def test_dispatcher_reservations_make_success_cap_exact(monkeypatch) -> None:
    limit = 5
    engine = SimpleNamespace(
        _running=True,
        _producer_sema=asyncio.Semaphore(3),
        created=0,
        config=SimpleNamespace(max_consecutive_mutation_failures=3),
    )
    engine._can_dispatch_mutant = lambda *, reserved: engine.created + reserved < limit
    spawned: list[int] = []

    async def successful_mutant(active_engine, task_id):
        spawned.append(task_id)
        await asyncio.sleep(0)
        active_engine.created += 1
        active_engine._producer_sema.release()
        return f"child-{task_id}"

    monkeypatch.setattr(
        "gigaevo.evolution.engine.dispatcher.run_one_mutant", successful_mutant
    )

    await asyncio.wait_for(dispatcher_loop(engine), timeout=1.0)

    assert engine.created == limit
    assert spawned == list(range(limit))


@pytest.mark.asyncio
async def test_dispatcher_replaces_failed_reserved_attempt_until_success_cap(
    monkeypatch,
) -> None:
    limit = 4
    engine = SimpleNamespace(
        _running=True,
        _producer_sema=asyncio.Semaphore(2),
        _terminal_stop_decision=None,
        created=0,
        config=SimpleNamespace(max_consecutive_mutation_failures=3),
    )
    engine._can_dispatch_mutant = lambda *, reserved: engine.created + reserved < limit
    spawned: list[int] = []

    async def mutant_with_one_failure(active_engine, task_id):
        spawned.append(task_id)
        await asyncio.sleep(0)
        active_engine._producer_sema.release()
        if task_id == 1:
            return None
        active_engine.created += 1
        return f"child-{task_id}"

    monkeypatch.setattr(
        "gigaevo.evolution.engine.dispatcher.run_one_mutant", mutant_with_one_failure
    )

    await asyncio.wait_for(dispatcher_loop(engine), timeout=1.0)

    assert engine.created == limit
    assert spawned == [0, 1, 2, 3, 4]
