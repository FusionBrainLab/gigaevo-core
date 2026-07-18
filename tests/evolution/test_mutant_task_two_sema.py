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
from gigaevo.evolution.engine.mutation import MutationFailure
from gigaevo.evolution.engine.refresh import ParentRefreshTicket
from gigaevo.evolution.mutation.constants import (
    MUTATION_MEMORY_ASSIGNMENT_METADATA_KEY,
    MUTATION_MEMORY_CANDIDATE_SLATE_METADATA_KEY,
    MUTATION_MEMORY_DECISION_ID_METADATA_KEY,
    MUTATION_MEMORY_NO_CARD_CONTROL_METADATA_KEY,
    MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY,
)
from gigaevo.evolution.mutation.parent_selector import RandomParentSelector
from gigaevo.memory.selection_leases import InFlightSelectionRegistry
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
        self._selection_leases = None
        self.memory_attempt_has_decision: bool | None = None
        self.memory_failures: list[dict] = []
        self.memory_child_links: list[dict] = []
        self.memory_topology_failures: list[dict] = []

        self.metrics = type("M", (), {})()
        self.metrics.iteration = 0
        self.metrics.mutations_created = 0
        self.metrics.submitted_for_refresh = 0

        cfg = type("C", (), {})()
        cfg.loop_interval = 0.01
        cfg.parent_selector = RandomParentSelector(num_parents=1)
        cfg.coalesce_refresh = False
        cfg.max_in_flight = max_in_flight
        self.config = cfg
        self._ss_config = cfg

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

    def _memory_attempt_has_decision(self, _attempt_id: str | None) -> bool | None:
        return self.memory_attempt_has_decision

    def _record_memory_attempt_failure(self, _parents, **payload) -> int:
        self.memory_failures.append(payload)
        return 1

    def _link_memory_attempt_child(self, **payload) -> None:
        self.memory_child_links.append(payload)

    def _record_memory_topology_failure(self, child_id: str, **payload) -> None:
        self.memory_topology_failures.append({"child_id": child_id, **payload})


async def _hold_producer_slot(engine: _FakeEngine) -> None:
    """Mirror the dispatcher contract: caller holds one producer slot."""
    await engine._producer_sema.acquire()


@pytest.mark.asyncio
async def test_success_path_transfers_buffer_and_ticket(monkeypatch) -> None:
    engine = _FakeEngine(_make_parent(), max_in_flight=3)
    await _hold_producer_slot(engine)

    async def fake_gen(*, parents, child_observer, **_kwargs):
        child_observer("new-id-1", parents[0].id, "main")
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
    assert engine.memory_child_links[0]["child_id"] == "new-id-1"


@pytest.mark.asyncio
async def test_gate_skipped_memory_decision_mutates_without_stale_selection(
    monkeypatch,
) -> None:
    parent = _make_parent()
    parent.set_metadata(MUTATION_MEMORY_CANDIDATE_SLATE_METADATA_KEY, [{"old": True}])
    parent.set_metadata(MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY, ["stale-card"])
    parent.set_metadata(MUTATION_MEMORY_NO_CARD_CONTROL_METADATA_KEY, True)
    parent.set_metadata(MUTATION_MEMORY_DECISION_ID_METADATA_KEY, "old-decision")
    parent.set_metadata(
        MUTATION_MEMORY_ASSIGNMENT_METADATA_KEY,
        {"decision_id": "old-decision"},
    )
    engine = _FakeEngine(parent, max_in_flight=2)
    engine._selection_leases = InFlightSelectionRegistry()
    engine.memory_attempt_has_decision = False
    await _hold_producer_slot(engine)

    async def fake_gen(*, parents, child_observer, **_kwargs):
        refreshed = parents[0]
        assert (
            refreshed.get_metadata(MUTATION_MEMORY_CANDIDATE_SLATE_METADATA_KEY) == []
        )
        assert refreshed.get_metadata(MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY) == []
        assert (
            refreshed.get_metadata(MUTATION_MEMORY_NO_CARD_CONTROL_METADATA_KEY)
            is False
        )
        assert refreshed.get_metadata(MUTATION_MEMORY_DECISION_ID_METADATA_KEY) == ""
        assert refreshed.get_metadata(MUTATION_MEMORY_ASSIGNMENT_METADATA_KEY) is None
        assert child_observer is not None
        child_observer("memory-free-child", refreshed.id, "main")
        return "memory-free-child"

    monkeypatch.setattr(
        "gigaevo.evolution.engine.mutant_task.generate_one_mutation", fake_gen
    )

    result = await run_one_mutant(engine, task_id=234)

    assert result == "memory-free-child"
    assert "memory-free-child" in engine._in_flight
    assert engine.metrics.mutations_created == 1
    assert engine.memory_child_links == [
        {
            "attempt_id": engine.memory_child_links[0]["attempt_id"],
            "child_id": "memory-free-child",
            "parent_id": parent.id,
            "island_id": "main",
            "completion_ordinal": 0,
        }
    ]
    assert engine._selection_leases.attempts_for_parent(parent.id) == ()


@pytest.mark.asyncio
async def test_memory_free_persistence_failure_closes_topology(monkeypatch) -> None:
    parent = _make_parent()
    engine = _FakeEngine(parent, max_in_flight=2)
    engine.memory_attempt_has_decision = False
    await _hold_producer_slot(engine)

    async def fake_gen(
        *,
        child_observer,
        failure_observer,
        **_kwargs,
    ):
        child_observer("unpersisted-child", parent.id, "main")
        failure_observer(
            MutationFailure(status="censored", stage="mutation_persistence")
        )
        return None

    monkeypatch.setattr(
        "gigaevo.evolution.engine.mutant_task.generate_one_mutation",
        fake_gen,
    )

    assert await run_one_mutant(engine, task_id=235) is None
    assert engine.memory_topology_failures == [
        {
            "child_id": "unpersisted-child",
            "status": "censored",
            "failure_stage": "mutation_persistence",
        }
    ]


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
    assert engine.memory_failures[0]["status"] == "invalid"
    assert engine.memory_failures[0]["failure_stage"] == "parent_refresh"


@pytest.mark.asyncio
async def test_pre_exposure_refresh_failure_propagates(monkeypatch) -> None:
    engine = _FakeEngine(_make_parent(), max_in_flight=3)
    await _hold_producer_slot(engine)

    async def boom(_parents):
        raise ValueError("provider configuration invalid")

    engine._parent_refresher.refresh_with_ticket = boom
    engine._record_memory_attempt_failure = lambda *_args, **_kwargs: 0
    monkeypatch.setattr(
        "gigaevo.evolution.engine.mutant_task.generate_one_mutation",
        AsyncMock(side_effect=AssertionError("must not mutate")),
    )

    with pytest.raises(ValueError, match="provider configuration invalid"):
        await run_one_mutant(engine, task_id=0)

    assert engine._producer_sema._value == 3
    assert engine._buffer_sema._value == 3
    assert not engine._in_flight


@pytest.mark.asyncio
async def test_noncausal_refresh_failure_remains_attempt_local(monkeypatch) -> None:
    engine = _FakeEngine(_make_parent(), max_in_flight=2)
    await _hold_producer_slot(engine)

    async def boom(_parents):
        raise ValueError("parent became invalid")

    engine._parent_refresher.refresh_with_ticket = boom
    # None means the configured sink has no causal attempt-lifecycle API.
    engine._record_memory_attempt_failure = lambda *_args, **_kwargs: None
    monkeypatch.setattr(
        "gigaevo.evolution.engine.mutant_task.generate_one_mutation",
        AsyncMock(side_effect=AssertionError("must not mutate")),
    )

    assert await run_one_mutant(engine, task_id=0) is None
    assert engine._producer_sema._value == 2
    assert engine._buffer_sema._value == 2


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
async def test_cancel_mid_llm_releases_attempt_selection_lease(monkeypatch) -> None:
    parent = _make_parent()
    engine = _FakeEngine(parent, max_in_flight=2)
    registry = InFlightSelectionRegistry()
    engine._selection_leases = registry
    await _hold_producer_slot(engine)
    llm_started = asyncio.Event()
    stay_in_llm = asyncio.Event()

    async def refresh_with_selection(parents):
        registry.attach_cards_for_parent(parents[0].id, ("card-a",))
        return ParentRefreshTicket(refreshed=parents, _locks=[])

    engine._parent_refresher.refresh_with_ticket = refresh_with_selection

    async def fake_gen(**_k):
        assert registry.is_leased("card-a")
        llm_started.set()
        await stay_in_llm.wait()

    monkeypatch.setattr(
        "gigaevo.evolution.engine.mutant_task.generate_one_mutation", fake_gen
    )
    task = asyncio.create_task(run_one_mutant(engine, task_id=0))
    await llm_started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert not registry.is_leased("card-a")
    assert engine.memory_failures[0]["status"] == "censored"


@pytest.mark.asyncio
async def test_cancel_blocked_on_buffer_completes_persisted_child_handoff(
    monkeypatch,
) -> None:
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
    await asyncio.sleep(0)
    # The ingestor releases capacity while the producer defers cancellation.
    engine._buffer_sema.release()
    with pytest.raises(asyncio.CancelledError):
        await task

    # Producer is released; the freed buffer slot transferred to the ingestor.
    assert engine._producer_sema._value == 2
    assert engine._buffer_sema._value == 0
    assert "drift-id-1" in engine._in_flight
    assert "drift-id-1" in engine._inflight_tickets
    assert engine.metrics.mutations_created == 1


@pytest.mark.asyncio
async def test_concurrent_mutants_get_unique_iterations(monkeypatch) -> None:
    """Race regression: N concurrent producers must each receive a unique iteration.

    The original bug: ``iteration=engine.metrics.iteration`` was read BEFORE
    the LLM await, with the increment AFTER the await. With max_in_flight=8,
    up to 8 concurrent tasks read the same pre-increment value and all child
    programs ended up sharing one iteration ordinal — producing the
    "long vertical lines at iteration 0" plot symptom.
    """
    N = 8
    engine = _FakeEngine(_make_parent(), max_in_flight=N)
    for _ in range(N):
        await engine._producer_sema.acquire()

    captured: list[int] = []

    async def fake_gen(*, iteration, task_id, **_k):
        captured.append(iteration)
        # Yield to event loop so concurrent tasks interleave; this is what
        # exposes the race in the pre-fix code.
        await asyncio.sleep(0.01)
        return f"id-{task_id}"

    monkeypatch.setattr(
        "gigaevo.evolution.engine.mutant_task.generate_one_mutation", fake_gen
    )

    tasks = [asyncio.create_task(run_one_mutant(engine, task_id=i)) for i in range(N)]
    results = await asyncio.gather(*tasks)

    assert all(r is not None for r in results), f"Some tasks failed: {results}"
    assert len(set(captured)) == N, (
        f"Iteration uniqueness violated under concurrency: got {captured}"
    )
