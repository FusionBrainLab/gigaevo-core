"""HarmEvictor delegates to the CardScorer; NullEvictor never evicts."""

from __future__ import annotations

import pytest

from gigaevo.memory.cards import Card, CardStatsBlock, DecisionContext
from gigaevo.memory.events import MemoryEvictionSweep
from gigaevo.memory.write.eviction import HarmEvictor, NullEvictor


class MarkedScorer:
    """Flags cards whose id is in ``harmful`` as confidently harmful."""

    def __init__(self, harmful: set[str]) -> None:
        self._harmful = harmful
        self.scored: list[str] = []

    def card_stats(
        self, card: Card, context: DecisionContext | None = None
    ) -> CardStatsBlock | None:
        self.scored.append(card.id)
        if card.id in self._harmful:
            return CardStatsBlock(efficacy_confident=True)
        return None

    def is_confidently_harmful(self, block: CardStatsBlock | None) -> bool:
        return block is not None and bool(block.efficacy_confident)


@pytest.fixture
def captured_events(monkeypatch):
    events: list = []
    monkeypatch.setattr(
        "gigaevo.memory.write.eviction.emit_memory_event", events.append
    )
    return events


def test_should_evict_delegates_through_scorer(make_card):
    bad = make_card()
    good = make_card()
    evictor = HarmEvictor(MarkedScorer({bad.id}))
    assert evictor.should_evict(bad) is True
    assert evictor.should_evict(good) is False


def test_sweep_returns_harmful_ids_and_emits_event(make_card, captured_events):
    bad = make_card()
    good = make_card()
    evictor = HarmEvictor(MarkedScorer({bad.id}))
    assert evictor.sweep([good, bad]) == [bad.id]
    assert len(captured_events) == 1
    event = captured_events[0]
    assert isinstance(event, MemoryEvictionSweep)
    assert event.bank_count == 2
    assert event.evicted_ids == (bad.id,)


def test_sweep_without_evictions_emits_nothing(make_card, captured_events):
    evictor = HarmEvictor(MarkedScorer(set()))
    assert evictor.sweep([make_card(), make_card()]) == []
    assert captured_events == []


def test_null_evictor_never_evicts(make_card):
    evictor = NullEvictor()
    card = make_card()
    assert evictor.should_evict(card) is False
    assert evictor.sweep([card]) == []
