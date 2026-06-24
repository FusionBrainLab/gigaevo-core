from gigaevo.memory.context import DecisionContext
from gigaevo.memory.shared_memory.injection_posterior import (
    InjectionOutcome,
    compute_contextual_gains,
)


def _child(**over):
    base = dict(
        id="child",
        parents=["p1", "p2"],
        fitness=0.90,
        base_selected_ids=["card-y"],
        base_metrics={"r2": 0.80},
        base_fitness=0.80,
        card_ids_used=["card-y"],
    )
    base.update(over)
    return InjectionOutcome(**base)


def test_used_and_base_selected_card_earns_one_event():
    events = compute_contextual_gains([_child()])
    assert set(events) == {"card-y"}
    (g,) = events["card-y"]
    assert g.gain == 0.90 - 0.80
    assert g.context == DecisionContext(parent_metrics={"r2": 0.80})
    assert g.invalid is False


def test_donor_card_used_but_not_base_selected_is_not_credited():
    # card-x was selected for the OTHER parent; declared used but not base-selected.
    events = compute_contextual_gains([_child(card_ids_used=["card-x", "card-y"])])
    assert set(events) == {"card-y"}


def test_selected_but_not_declared_used_earns_nothing():
    events = compute_contextual_gains([_child(card_ids_used=[])])
    assert events == {}


def test_hallucinated_id_is_dropped():
    events = compute_contextual_gains([_child(card_ids_used=["ghost"])])
    assert events == {}


def test_invalid_child_emits_forced_harm_event_gain_zero():
    child = _child(fitness=None, invalid=True)
    events = compute_contextual_gains([child])
    (g,) = events["card-y"]
    assert g.invalid is True
    assert g.gain == 0.0


def test_missing_base_metadata_yields_no_events():
    legacy = InjectionOutcome(id="c", parents=["p1"], fitness=0.9)
    assert compute_contextual_gains([legacy]) == {}


def test_lower_is_better_flips_gain_sign():
    child = _child(fitness=0.70, base_fitness=0.80)
    events = compute_contextual_gains([child], higher_is_better=False)
    (g,) = events["card-y"]
    assert g.gain == 0.80 - 0.70
