from gigaevo.memory.context import ContextualGain, DecisionContext
from gigaevo.memory.efficacy.stamping import CardStatsStamper
from gigaevo.memory.shared_memory.models import MemoryCard


def _gain():
    return ContextualGain(context=DecisionContext(parent_metrics={"r2": 0.8}), gain=0.1)


def _gain_v(value: float):
    return ContextualGain(
        context=DecisionContext(parent_metrics={"r2": value}), gain=value
    )


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


def test_stamp_folds_absorbed_id_events_onto_survivor():
    # A survivor that absorbed partner "mem-P" must inherit the gain events the
    # program pool still attributes to the absorbed id (its children's frozen
    # card_ids_used point at mem-P, which no longer exists in the bank).
    survivor = MemoryCard(id="mem-S", absorbed_ids=["mem-P"])
    stamped = CardStatsStamper().stamp_gain_events(survivor, {"mem-P": [_gain_v(0.1)]})
    assert stamped.gain_events == [_gain_v(0.1)]


def test_stamp_dedups_survivor_and_absorbed_events():
    # A child crediting both the survivor's id and an absorbed id earns the SAME
    # event object under each (the pool appends one ContextualGain per child to
    # every credited id); folding dedups by that trial identity so the merged
    # idea is counted once, not twice.
    shared = _gain_v(0.1)
    survivor = MemoryCard(id="mem-S", absorbed_ids=["mem-P"])
    stamped = CardStatsStamper().stamp_gain_events(
        survivor, {"mem-S": [shared], "mem-P": [shared, _gain_v(0.2)]}
    )
    assert stamped.gain_events == [_gain_v(0.1), _gain_v(0.2)]


def test_stamp_unaffected_when_no_absorbed_ids():
    # The common case (no merges) is unchanged: only the card's own id is read.
    survivor = MemoryCard(id="mem-S")
    stamped = CardStatsStamper().stamp_gain_events(
        survivor, {"mem-S": [_gain_v(0.3)], "mem-P": [_gain_v(0.9)]}
    )
    assert stamped.gain_events == [_gain_v(0.3)]


def test_stamp_preserves_own_id_event_multiplicity():
    # Distinct trials can emit identical events — every invalid child of one base
    # parent yields the same forced-harm event. The own-id list is the trial count
    # the harm gate reads (intro_events), so it must be kept verbatim, NOT deduped.
    inv = ContextualGain(
        context=DecisionContext(parent_metrics={"r2": 0.5}), gain=0.0, invalid=True
    )
    card = MemoryCard(id="mem-A")
    stamped = CardStatsStamper().stamp_gain_events(card, {"mem-A": [inv, inv, inv]})
    assert stamped.gain_events == [inv, inv, inv]


def _inv():
    return ContextualGain(
        context=DecisionContext(parent_metrics={"r2": 0.5}), gain=0.0, invalid=True
    )


def test_stamp_preserves_absorbed_id_event_multiplicity():
    # The absorbed-id fold dedups so a single child crediting both the survivor
    # and an absorbed id is counted once — but distinct invalid children of one
    # base parent emit value-identical forced-harm events, and those are separate
    # trials. The pool gives each child its own event object; the fold must dedup
    # by trial identity, not value, or a merged card's harm count collapses and
    # the harm gate (intro_events >= harm_min_events) can never fire.
    inv1, inv2, inv3 = _inv(), _inv(), _inv()
    survivor = MemoryCard(id="mem-S", absorbed_ids=["mem-P"])
    stamped = CardStatsStamper().stamp_gain_events(
        survivor, {"mem-P": [inv1, inv2, inv3]}
    )
    assert stamped.gain_events == [inv1, inv2, inv3]
