"""HarmEvictor delegates to the CardScorer; NullEvictor never evicts."""

from __future__ import annotations

import pytest

from gigaevo.memory.cards import Card, CardStatsBlock, DecisionContext
from gigaevo.memory.events import MemoryEvictionSweep
from gigaevo.memory.read.reputation import BetaBinomialReputation
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


def test_founding_events_never_trigger_eviction(make_card, make_event):
    """Founding evidence seeds the auction bid, not the harm verdict: a card is
    never evicted on the origin deltas it was distilled from, before use-
    attribution ever credits it. Three losing founding events would trip the harm
    gate if counted; stripped, the card carries no usage evidence and survives.
    """
    evictor = HarmEvictor(BetaBinomialReputation())
    founding_only = make_card(
        gain_events=tuple(make_event(-0.5, founding=True) for _ in range(3))
    )
    use_only = make_card(gain_events=tuple(make_event(-0.5) for _ in range(3)))
    assert evictor.should_evict(founding_only) is False
    assert evictor.should_evict(use_only) is True


def test_founding_events_do_not_lower_the_harm_bar(make_card, make_event):
    """Founding events must not count toward ``harm_min_events``: a card with two
    losing use events and one losing founding event is judged on the two use
    events (below the bar), not evicted as if it had three."""
    evictor = HarmEvictor(BetaBinomialReputation())
    mixed = make_card(
        gain_events=(
            make_event(-0.5),
            make_event(-0.5),
            make_event(-0.5, founding=True),
        )
    )
    assert evictor.should_evict(mixed) is False
