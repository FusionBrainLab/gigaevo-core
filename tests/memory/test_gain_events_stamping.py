from gigaevo.memory.context import ContextualGain, DecisionContext
from gigaevo.memory.efficacy.stamping import CardStatsStamper
from gigaevo.memory.shared_memory.models import MemoryCard


def _gain():
    return ContextualGain(context=DecisionContext(parent_metrics={"r2": 0.8}), gain=0.1)


def test_card_gain_events_default_absent():
    card = MemoryCard(id="card-y")
    assert card.gain_events is None
    assert "gain_events" in card.model_dump()


def test_stamp_gain_events_attaches_list():
    card = MemoryCard(id="card-y")
    stamped = CardStatsStamper().stamp_gain_events(card, {"card-y": [_gain()]})
    assert len(stamped.gain_events) == 1
    assert stamped.gain_events[0].gain == 0.1


def test_stamp_gain_events_none_when_card_uncredited():
    card = MemoryCard(id="other")
    stamped = CardStatsStamper().stamp_gain_events(card, {"card-y": [_gain()]})
    assert stamped.gain_events is None


def test_stamp_gain_events_clears_stale_when_card_uncredited():
    card = MemoryCard(id="other", gain_events=[_gain()])
    stamped = CardStatsStamper().stamp_gain_events(card, {"card-y": [_gain()]})
    assert stamped.gain_events is None
