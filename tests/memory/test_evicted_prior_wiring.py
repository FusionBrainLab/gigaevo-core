from __future__ import annotations

import json

from gigaevo.memory.cards import (
    Card,
    ContextualGain,
    DecisionContext,
    EvidenceAttribution,
)
from gigaevo.memory.events import MemoryPriorCohort
from gigaevo.memory.prior_evidence import JsonlEvictedEvidence
from gigaevo.memory.read.prior import EmpiricalBayesMemoryPrior
from gigaevo.memory.storage.base import MemoryStore, MergeRetireResult, ResearchResult
from gigaevo.memory.write.writer import LibrarianWriteStack


def _card(card_id: str, gain: float) -> Card:
    return Card(
        id=card_id,
        description=card_id,
        gain_events=(
            ContextualGain(
                context=DecisionContext(parent_id="parent"),
                gain=gain,
                attribution=EvidenceAttribution(
                    source_child_id=f"child-{card_id}", credit_weight=1.0
                ),
            ),
        ),
    )


class _Store(MemoryStore):
    def __init__(self, cards=()) -> None:
        self._cards = {card.id: card for card in cards}

    def snapshot(self):
        return tuple(self._cards.values())

    @property
    def is_ready(self):
        return True

    def get(self, card_id):
        return self._cards.get(card_id)

    def save(self, card):
        self._cards[card.id] = card
        return card.id

    def update(self, card_id, transform):
        current = self._cards.get(card_id)
        if current is None:
            return None
        replacement = transform(current)
        if replacement is None:
            del self._cards[card_id]
            return current
        self._cards[card_id] = replacement
        return replacement

    def delete(self, card_id):
        return self._cards.pop(card_id, None) is not None

    def merge_retire(self, target_id, partner_id, fold):
        target = self._cards.get(target_id)
        if target is None:
            return MergeRetireResult(outcome="target_missing")
        replacement = fold(target, self._cards.get(partner_id))
        if replacement is None:
            self._cards.pop(target_id, None)
            self._cards.pop(partner_id, None)
            return MergeRetireResult(outcome="retired")
        self._cards[target_id] = replacement
        self._cards.pop(partner_id, None)
        return MergeRetireResult(outcome="merged")

    def nearest(self, text, k, kind=None):
        del text, k, kind
        return []

    async def research(self, request):
        del request
        return ResearchResult()

    def rebuild(self):
        return None

    def close(self):
        return None


class _EvictNegative:
    def should_evict(self, card):
        return any(event.gain < 0 for event in card.gain_events)

    def eviction_reason(self, card):
        del card
        return "negative evidence"

    def sweep(self, cards):
        return [card.id for card in cards if self.should_evict(card)]


def test_writer_eviction_feeds_same_bounded_store_used_by_cold_prior(
    tmp_path, monkeypatch
) -> None:
    evidence = JsonlEvictedEvidence(tmp_path / "evicted_evidence.jsonl", max_cards=2)
    survivor = _card("survivor", 1.0)
    harmful = _card("harmful", -1.0)
    store = _Store((survivor, harmful))
    for factory in (
        "create_reconcile_agent",
        "create_program_author_agent",
        "create_consolidate_agent",
    ):
        monkeypatch.setattr(
            f"gigaevo.memory.write.writer.{factory}", lambda *args, **kwargs: object()
        )
    stack = LibrarianWriteStack(
        llm=object(),
        evictor=_EvictNegative(),
        store=store,
        checkpoint_dir=tmp_path,
        evicted_evidence=evidence,
    )
    stack._build("")

    assert stack.require_gate()._evicted_evidence_sink is evidence
    assert stack.require_gate().sweep() == ["harmful"]
    assert store.get("harmful") is None
    assert evidence.cards() == (harmful,)

    emitted = []
    monkeypatch.setattr("gigaevo.memory.read.prior.emit_memory_event", emitted.append)
    prior = EmpiricalBayesMemoryPrior(
        store=store,
        evicted_evidence=evidence,
        levels=(),
        shrink_events=0.0,
        n_ref=1.0,
    ).cold_card_prior(Card(id="query", description="query"))

    assert prior.support_n == 2.0
    assert prior.alpha == prior.beta
    (cohort_event,) = emitted
    assert isinstance(cohort_event, MemoryPriorCohort)
    assert cohort_event.live_card_count == 1
    assert cohort_event.evicted_card_count == 1

    bounded = JsonlEvictedEvidence(tmp_path / "bounded.jsonl", max_cards=2)
    for index in range(3):
        bounded.record(_card(f"card-{index}", float(index - 1)))
    assert [card.id for card in bounded.cards()] == ["card-1", "card-2"]
    rows = [
        json.loads(line)
        for line in (tmp_path / "bounded.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 2
    assert {row["schema_version"] for row in rows} == {"prior_evidence.v1"}


def test_evicted_evidence_is_recorded_before_the_card_leaves_the_store(
    tmp_path, monkeypatch
) -> None:
    # The cold prior samples the live bank plus the evicted cohort. If eviction
    # deleted the card and only then recorded it, a prior sampled during that
    # window would miss the card from BOTH cohorts — survivorship bias. The sink
    # must see the card while it is still live in the store.
    class _OrderingSpy:
        def __init__(self, store: _Store) -> None:
            self._store = store
            self.present_at_record: dict[str, bool] = {}
            self.recorded: list[str] = []

        def record(self, card: Card) -> None:
            self.present_at_record[card.id] = self._store.get(card.id) is not None
            self.recorded.append(card.id)

    harmful = _card("harmful", -1.0)
    store = _Store((_card("survivor", 1.0), harmful))
    spy = _OrderingSpy(store)
    for factory in (
        "create_reconcile_agent",
        "create_program_author_agent",
        "create_consolidate_agent",
    ):
        monkeypatch.setattr(
            f"gigaevo.memory.write.writer.{factory}", lambda *args, **kwargs: object()
        )
    stack = LibrarianWriteStack(
        llm=object(),
        evictor=_EvictNegative(),
        store=store,
        checkpoint_dir=tmp_path,
        evicted_evidence=spy,
    )
    stack._build("")

    assert stack.require_gate().sweep() == ["harmful"]
    assert spy.recorded == ["harmful"]
    assert spy.present_at_record["harmful"] is True
    assert store.get("harmful") is None
