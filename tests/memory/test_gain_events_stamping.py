from gigaevo.memory.context import ContextualGain, DecisionContext
from gigaevo.memory.efficacy.stamping import CardStatsStamper
from gigaevo.memory.shared_memory.models import EvolutionStatistics, MemoryCard


def _gain():
    return ContextualGain(context=DecisionContext(parent_metrics={"r2": 0.8}), gain=0.1)


def test_evolution_statistics_gain_events_defaults_absent():
    es = EvolutionStatistics()
    assert es.gain_events is None
    assert "gain_events" not in es.model_dump()


def test_stamp_gain_events_attaches_list():
    card = MemoryCard(id="card-y")
    stamped = CardStatsStamper().stamp_gain_events(card, {"card-y": [_gain()]})
    assert len(stamped.evolution_statistics.gain_events) == 1
    assert stamped.evolution_statistics.gain_events[0].gain == 0.1


def test_stamp_gain_events_passthrough_when_absent():
    card = MemoryCard(id="other")
    stamped = CardStatsStamper().stamp_gain_events(card, {"card-y": [_gain()]})
    assert stamped.evolution_statistics.gain_events is None
