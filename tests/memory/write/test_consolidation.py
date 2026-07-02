"""Batch near-duplicate consolidation and its background scheduler."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from gigaevo.llm.agents.reconcile import LibrarianCard
from gigaevo.memory.cards import CardKind
from gigaevo.memory.events import MemoryConsolidationPass
from gigaevo.memory.storage.base import ScoredCard
from gigaevo.memory.write.admission import CardAdmissionGate
from gigaevo.memory.write.consolidation import ConsolidationScheduler, consolidate
from gigaevo.memory.write.eviction import NullEvictor


class FakeConsolidateAgent:
    def __init__(self, *, merge: bool = True) -> None:
        self.merge = merge
        self.calls: list[tuple[str, str]] = []

    async def arun(self, *, card_a, card_b):
        self.calls.append((card_a.id, card_b.id))
        if not self.merge:
            return SimpleNamespace(merge=False, card=None)
        return SimpleNamespace(
            merge=True,
            card=LibrarianCard(
                description="union prose",
                explanation_summary="union why",
                keywords=["union"],
            ),
        )


def pair_neighbors(store, a, b):
    """Symmetric nearest: each card's sole neighbor is the other."""

    def nearest(text, k, kind=None):
        if text == a.description:
            return [ScoredCard(card=b, distance=0.1)]
        if text == b.description:
            return [ScoredCard(card=a, distance=0.1)]
        return []

    store.nearest = nearest


async def run_consolidate(store, agent, **overrides):
    params = {
        "store": store,
        "gate": CardAdmissionGate(store=store, evictor=NullEvictor()),
        "neighbors": store,
        "agent": agent,
        "k": 5,
    }
    params.update(overrides)
    return await consolidate(**params)


async def test_folds_near_pair_and_deletes_partner(store, make_card):
    a = make_card(programs=("p1",))
    b = make_card(programs=("p2",), absorbed_ids=("dead-0",))
    store.save(a)
    store.save(b)
    pair_neighbors(store, a, b)

    merged = await run_consolidate(store, FakeConsolidateAgent())

    assert merged == 1
    survivor, partner = (a, b) if store.get(a.id) is not None else (b, a)
    assert store.get(partner.id) is None
    banked = store.get(survivor.id)
    assert banked.description == "union prose"
    assert banked.explanation_summary == "union why"
    assert set(banked.programs) == {"p1", "p2"}
    assert partner.id in banked.absorbed_ids
    assert "dead-0" in banked.absorbed_ids


async def test_abstain_reviews_symmetric_pair_once(store, make_card):
    a = make_card()
    b = make_card()
    store.save(a)
    store.save(b)
    pair_neighbors(store, a, b)
    agent = FakeConsolidateAgent(merge=False)

    merged = await run_consolidate(store, agent)

    assert merged == 0
    assert len(agent.calls) == 1
    assert store.get(a.id) is not None
    assert store.get(b.id) is not None


async def test_program_cards_never_offered(store, make_card):
    exemplar = make_card(
        kind=CardKind.PROGRAM, program_id="p1", code="x = 1", fitness=0.5
    )
    store.save(exemplar)
    agent = FakeConsolidateAgent()
    merged = await run_consolidate(store, agent)
    assert merged == 0
    assert agent.calls == []


async def test_declined_pair_remembered_across_passes(store, make_card):
    a = make_card()
    b = make_card()
    store.save(a)
    store.save(b)
    pair_neighbors(store, a, b)
    agent = FakeConsolidateAgent(merge=False)
    reviewed: set = set()

    assert await run_consolidate(store, agent, reviewed=reviewed) == 0
    assert await run_consolidate(store, agent, reviewed=reviewed) == 0
    assert len(agent.calls) == 1


async def test_declined_pair_rereviewed_after_content_change(store, make_card):
    a = make_card()
    b = make_card()
    store.save(a)
    store.save(b)
    pair_neighbors(store, a, b)
    agent = FakeConsolidateAgent(merge=False)
    reviewed: set = set()
    await run_consolidate(store, agent, reviewed=reviewed)

    reworded = b.model_copy(update={"description": "sharpened prose"})
    store.save(reworded)
    pair_neighbors(store, a, reworded)
    await run_consolidate(store, agent, reviewed=reviewed)
    assert len(agent.calls) == 2


async def test_distant_hits_still_reach_arbiter(store, make_card):
    a = make_card()
    b = make_card()
    store.save(a)
    store.save(b)
    store.hits = [ScoredCard(card=b, distance=0.9), ScoredCard(card=a, distance=0.9)]
    agent = FakeConsolidateAgent(merge=False)
    merged = await run_consolidate(store, agent)
    assert merged == 0
    assert len(agent.calls) == 1
    assert store.get(a.id) is not None
    assert store.get(b.id) is not None


def make_scheduler(store, agent, **overrides):
    stack = SimpleNamespace(
        store=store,
        gate=CardAdmissionGate(store=store, evictor=NullEvictor()),
        neighbors=store,
        consolidation_agent=agent,
    )
    params = {
        "stack": stack,
        "run_lock": asyncio.Lock(),
        "every_n": 2,
        "k": 5,
    }
    params.update(overrides)
    return ConsolidationScheduler(**params)


async def test_scheduler_remembers_declines_across_passes(store, make_card):
    a = make_card()
    b = make_card()
    store.save(a)
    store.save(b)
    pair_neighbors(store, a, b)
    agent = FakeConsolidateAgent(merge=False)
    scheduler = make_scheduler(store, agent)
    assert scheduler.schedule() is True
    await scheduler.drain()
    assert scheduler.schedule() is True
    await scheduler.drain()
    assert len(agent.calls) == 1


async def test_note_writes_consumes_counter_only_on_dispatch(store):
    scheduler = make_scheduler(store, FakeConsolidateAgent())
    scheduler.note_writes(1)
    assert scheduler.writes_since == 1
    assert scheduler.task is None
    scheduler.note_writes(1)
    assert scheduler.writes_since == 0
    assert scheduler.task is not None
    await scheduler.drain()


async def test_unbuilt_stack_leaves_writes_pending(store):
    scheduler = make_scheduler(store, FakeConsolidateAgent())
    scheduler._stack.store = None
    scheduler.note_writes(5)
    assert scheduler.task is None
    assert scheduler.writes_since == 5


async def test_no_double_dispatch_while_in_flight(store):
    release = asyncio.Event()

    class BlockingAgent(FakeConsolidateAgent):
        async def arun(self, *, card_a, card_b):
            await release.wait()
            return await super().arun(card_a=card_a, card_b=card_b)

    scheduler = make_scheduler(store, BlockingAgent())
    lock = scheduler._run_lock
    async with lock:
        scheduler.note_writes(2)
        first = scheduler.task
        assert first is not None
        scheduler.note_writes(2)
        assert scheduler.task is first
        assert scheduler.writes_since == 2
    release.set()
    await scheduler.drain()
    assert first.done()


async def test_run_failure_counts_and_emits_failed_pass(store, monkeypatch):
    emitted: list = []
    monkeypatch.setattr(
        "gigaevo.memory.write.consolidation.emit_memory_event", emitted.append
    )

    def broken_snapshot():
        raise RuntimeError("bank corrupt")

    monkeypatch.setattr(store, "snapshot", broken_snapshot)
    scheduler = make_scheduler(store, FakeConsolidateAgent())
    assert scheduler.schedule() is True
    await scheduler.drain()
    assert scheduler.failures == 1
    passes = [e for e in emitted if isinstance(e, MemoryConsolidationPass)]
    assert len(passes) == 1
    assert passes[0].outcome == "failed"
    assert "bank corrupt" in passes[0].error


async def test_successful_pass_emits_ok(store, make_card, monkeypatch):
    emitted: list = []
    monkeypatch.setattr(
        "gigaevo.memory.write.consolidation.emit_memory_event", emitted.append
    )
    a = make_card()
    b = make_card()
    store.save(a)
    store.save(b)
    pair_neighbors(store, a, b)
    scheduler = make_scheduler(store, FakeConsolidateAgent())
    assert scheduler.schedule() is True
    await scheduler.drain()
    passes = [e for e in emitted if isinstance(e, MemoryConsolidationPass)]
    assert len(passes) == 1
    assert passes[0].outcome == "ok"
    assert passes[0].merged == 1


def test_schedule_without_running_loop_returns_false(store):
    scheduler = make_scheduler(store, FakeConsolidateAgent())
    assert scheduler.schedule() is False
