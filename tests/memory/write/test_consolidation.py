"""Batch near-duplicate consolidation and its background scheduler."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from loguru import logger
import pytest

from gigaevo.llm.agents.reconcile import LibrarianCard
from gigaevo.memory.cards import CardKind
from gigaevo.memory.events import MemoryConsolidationPass
from gigaevo.memory.selection_leases import InFlightSelectionRegistry
from gigaevo.memory.storage.base import ScoredCard
from gigaevo.memory.write.admission import CardAdmissionGate
from gigaevo.memory.write.consolidation import ConsolidationScheduler, consolidate
from gigaevo.memory.write.eviction import NullEvictor


class MarkingEvictor:
    def __init__(self, harmful: set[str]) -> None:
        self._harmful = harmful

    def should_evict(self, card) -> bool:
        return card.id in self._harmful

    def eviction_reason(self, card) -> str:
        return "test evictor"

    def sweep(self, cards) -> list[str]:
        return [card.id for card in cards if self.should_evict(card)]


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


async def test_consolidation_proposal_and_survivor_keep_target_task_key(
    store, make_card
):
    target = make_card(id="card-a", task_key="authoring-task")
    partner = make_card(id="card-b", task_key="other-task")
    store.save(target)
    store.save(partner)
    pair_neighbors(store, target, partner)

    class CapturingGate:
        def __init__(self):
            self.inner = CardAdmissionGate(store=store, evictor=NullEvictor())
            self.incoming = None

        def merge(self, target_id, incoming):
            self.incoming = incoming
            return self.inner.merge(target_id, incoming)

    gate = CapturingGate()

    assert await run_consolidate(store, FakeConsolidateAgent(), gate=gate) == 1
    assert gate.incoming.task_key == "authoring-task"
    assert store.get(target.id).task_key == "authoring-task"


async def test_leased_partner_is_not_folded_or_retired(store, make_card):
    survivor = make_card(id="card-a")
    partner = make_card(id="card-b")
    store.save(survivor)
    store.save(partner)
    pair_neighbors(store, survivor, partner)
    registry = InFlightSelectionRegistry()
    lease = registry.open_attempt("attempt-1", "parent-1")
    lease.attach_cards((partner.id,))
    gate = CardAdmissionGate(
        store=store, evictor=NullEvictor(), selection_leases=registry
    )

    merged = await run_consolidate(store, FakeConsolidateAgent(), gate=gate)

    assert merged == 0
    assert store.get(survivor.id) == survivor
    assert store.get(partner.id) == partner


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


async def test_harmful_consolidation_removes_both_banked_cards(store, make_card):
    a = make_card()
    b = make_card()
    store.save(a)
    store.save(b)
    pair_neighbors(store, a, b)
    gate = CardAdmissionGate(store=store, evictor=MarkingEvictor({a.id}))

    merged = await run_consolidate(store, FakeConsolidateAgent(), gate=gate)

    assert merged == 0
    assert store.get(a.id) is None
    assert store.get(b.id) is None
    assert gate.is_tombstoned(a.id)
    assert gate.is_tombstoned(b.id)


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


async def test_cancel_mid_pass_logs_progress_and_keeps_committed_folds(
    store, make_card
):
    a = make_card(description="pair-one alpha")
    b = make_card(description="pair-one beta")
    c = make_card(description="pair-two gamma")
    d = make_card(description="pair-two delta")
    for card in (a, b, c, d):
        store.save(card)

    def nearest(text, k, kind=None):
        table = {
            a.description: [ScoredCard(card=b, distance=0.1)],
            b.description: [ScoredCard(card=a, distance=0.1)],
            c.description: [ScoredCard(card=d, distance=0.1)],
            d.description: [ScoredCard(card=c, distance=0.1)],
        }
        return table.get(text, [])

    store.nearest = nearest

    class CancelledMidPassAgent(FakeConsolidateAgent):
        async def arun(self, *, card_a, card_b):
            if self.calls:
                raise asyncio.CancelledError
            return await super().arun(card_a=card_a, card_b=card_b)

    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        with pytest.raises(asyncio.CancelledError):
            await run_consolidate(store, CancelledMidPassAgent())
    finally:
        logger.remove(sink)

    assert store.get(b.id) is None
    assert store.get(a.id).description == "union prose"
    assert store.get(c.id) is not None
    assert store.get(d.id) is not None
    assert any(
        "cancelled at card 3/4" in m and "1 committed merge" in m for m in messages
    )


async def test_cancel_before_first_merge_logs_zero_committed(store, make_card):
    a = make_card(description="pair alpha")
    b = make_card(description="pair beta")
    for card in (a, b):
        store.save(card)
    pair_neighbors(store, a, b)

    class CancelImmediatelyAgent(FakeConsolidateAgent):
        async def arun(self, *, card_a, card_b):
            raise asyncio.CancelledError

    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        with pytest.raises(asyncio.CancelledError):
            await run_consolidate(store, CancelImmediatelyAgent())
    finally:
        logger.remove(sink)

    assert store.get(a.id) is not None
    assert store.get(b.id) is not None
    assert any(
        "cancelled at card 1/2" in m and "0 committed merge" in m for m in messages
    )


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


async def test_subset_restricts_queries_but_folds_against_full_bank(store, make_card):
    # The intra-batch pass queries only the batch's ids, yet neighbors rank over
    # the whole bank — so a batch card folds an OLDER twin, while a duplicate pair
    # outside the subset is never queried and stays untouched.
    batch = make_card(description="batch card")
    old = make_card(description="older twin")
    other_a = make_card(description="other a")
    other_b = make_card(description="other b")
    for card in (batch, old, other_a, other_b):
        store.save(card)

    def nearest(text, k, kind=None):
        table = {
            batch.description: [ScoredCard(card=old, distance=0.1)],
            old.description: [ScoredCard(card=batch, distance=0.1)],
            other_a.description: [ScoredCard(card=other_b, distance=0.1)],
            other_b.description: [ScoredCard(card=other_a, distance=0.1)],
        }
        return table.get(text, [])

    store.nearest = nearest
    merged = await run_consolidate(store, FakeConsolidateAgent(), subset={batch.id})

    assert merged == 1
    assert store.get(batch.id) is not None
    assert store.get(old.id) is None
    assert store.get(other_a.id) is not None
    assert store.get(other_b.id) is not None


async def test_consolidate_written_folds_same_batch_pair(store, make_card):
    a = make_card()
    b = make_card()
    store.save(a)
    store.save(b)
    pair_neighbors(store, a, b)
    scheduler = make_scheduler(store, FakeConsolidateAgent())

    merged = await scheduler.consolidate_written({a.id, b.id})

    assert merged == 1
    survivors = [c for c in (a, b) if store.get(c.id) is not None]
    assert len(survivors) == 1


async def test_consolidate_written_noop_when_disabled(store, make_card):
    a = make_card()
    b = make_card()
    store.save(a)
    store.save(b)
    pair_neighbors(store, a, b)
    agent = FakeConsolidateAgent()
    scheduler = make_scheduler(store, agent, every_n=0)

    assert await scheduler.consolidate_written({a.id, b.id}) == 0
    assert agent.calls == []
    assert store.get(a.id) is not None
    assert store.get(b.id) is not None


async def test_consolidate_written_noop_on_empty_ids(store):
    agent = FakeConsolidateAgent()
    scheduler = make_scheduler(store, agent)
    assert await scheduler.consolidate_written(set()) == 0
    assert agent.calls == []


async def test_consolidate_written_shares_reviewed_memo(store, make_card):
    a = make_card()
    b = make_card()
    store.save(a)
    store.save(b)
    pair_neighbors(store, a, b)
    agent = FakeConsolidateAgent(merge=False)
    scheduler = make_scheduler(store, agent)

    assert await scheduler.consolidate_written({a.id, b.id}) == 0
    assert len(agent.calls) == 1
    # the periodic whole-bank pass must not re-pay the arbiter for a pair the
    # inline pass already declined — both share the scheduler's reviewed memo
    assert scheduler.schedule() is True
    await scheduler.drain()
    assert len(agent.calls) == 1


async def test_consolidate_written_timeout_degrades_to_skip(store, make_card):
    a = make_card()
    b = make_card()
    store.save(a)
    store.save(b)
    pair_neighbors(store, a, b)

    class SlowAgent(FakeConsolidateAgent):
        async def arun(self, *, card_a, card_b):
            await asyncio.sleep(1)
            return await super().arun(card_a=card_a, card_b=card_b)

    scheduler = make_scheduler(store, SlowAgent())

    assert await scheduler.consolidate_written({a.id, b.id}, timeout=0.01) == 0
    assert scheduler.failures == 1
    assert store.get(a.id) is not None
    assert store.get(b.id) is not None
