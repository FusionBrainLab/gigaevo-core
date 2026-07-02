"""Use-attributed gain computation, stamping, and the restamp sweep."""

from __future__ import annotations

import pytest

from gigaevo.evolution.mutation.constants import (
    MUTATION_MEMORY_BASE_ID_METADATA_KEY,
    MUTATION_MEMORY_BASE_METRICS_METADATA_KEY,
    MUTATION_MEMORY_BASE_SELECTED_IDS_METADATA_KEY,
    MUTATION_OUTPUT_METADATA_KEY,
)
from gigaevo.memory.cards import CardKind
from gigaevo.memory.events import MemoryGainRestamp
from gigaevo.memory.write.admission import CardAdmissionGate
from gigaevo.memory.write.eviction import NullEvictor
from gigaevo.memory.write.stats import (
    CardStatsStamper,
    CardStatsUpdater,
    InjectionOutcome,
    card_gain_events_from_programs,
    compute_contextual_gains,
)
from gigaevo.programs.metrics.context import VALIDITY_KEY


def outcome(**overrides) -> InjectionOutcome:
    params = {
        "id": "child-1",
        "fitness": 0.7,
        "base_selected_ids": ("card-a",),
        "base_metrics": {VALIDITY_KEY: 1.0, "fitness": 0.5},
        "base_id": "parent-1",
        "base_fitness": 0.5,
        "card_ids_used": ("card-a",),
    }
    params.update(overrides)
    return InjectionOutcome(**params)


def base_meta(
    *, selected: list[str], used: list[str], base_fitness: float = 0.5
) -> dict:
    return {
        MUTATION_MEMORY_BASE_SELECTED_IDS_METADATA_KEY: selected,
        MUTATION_MEMORY_BASE_METRICS_METADATA_KEY: {
            VALIDITY_KEY: 1.0,
            "fitness": base_fitness,
        },
        MUTATION_MEMORY_BASE_ID_METADATA_KEY: "parent-1",
        MUTATION_OUTPUT_METADATA_KEY: {"card_ids_used": used},
    }


def test_credits_only_the_selected_and_used_intersection():
    events = compute_contextual_gains(
        [outcome(base_selected_ids=("a", "b"), card_ids_used=("b", "hallucinated"))]
    )
    assert set(events) == {"b"}
    assert events["b"][0].gain == pytest.approx(0.2)


def test_invalid_child_emits_one_forced_harm_event():
    events = compute_contextual_gains([outcome(fitness=None, invalid=True)])
    (event,) = events["card-a"]
    assert event.gain == 0.0
    assert event.invalid is True


def test_missing_fitness_without_invalid_flag_is_skipped():
    assert compute_contextual_gains([outcome(fitness=None)]) == {}


def test_no_baseline_contributes_nothing():
    assert compute_contextual_gains([outcome(base_fitness=None)]) == {}
    assert compute_contextual_gains([outcome(base_selected_ids=())]) == {}


def test_delta_orientation_follows_direction():
    higher = compute_contextual_gains([outcome()], higher_is_better=True)
    lower = compute_contextual_gains([outcome()], higher_is_better=False)
    assert higher["card-a"][0].gain == pytest.approx(0.2)
    assert lower["card-a"][0].gain == pytest.approx(-0.2)


def test_one_child_shares_one_event_object_across_credited_cards():
    events = compute_contextual_gains(
        [outcome(base_selected_ids=("a", "b"), card_ids_used=("a", "b"))]
    )
    assert events["a"][0] is events["b"][0]


def test_gain_events_from_programs_carry_base_context(make_program, metrics_context):
    prog = make_program(
        fitness=0.7, metadata=base_meta(selected=["card-a"], used=["card-a"])
    )
    events = card_gain_events_from_programs(
        [prog],
        fitness_key="fitness",
        higher_is_better=True,
        metrics_context=metrics_context,
    )
    (event,) = events["card-a"]
    assert event.gain == pytest.approx(0.2)
    assert event.context.parent_id == "parent-1"
    assert event.context.timestamp == prog.created_at


def test_sentinel_base_fitness_yields_no_events(make_program, metrics_context):
    prog = make_program(
        fitness=0.7,
        metadata=base_meta(selected=["card-a"], used=["card-a"], base_fitness=-1e5),
    )
    events = card_gain_events_from_programs(
        [prog],
        fitness_key="fitness",
        higher_is_better=True,
        metrics_context=metrics_context,
    )
    assert events == {}


def test_stamper_folds_absorbed_events_deduping_by_identity(make_card, make_event):
    shared = make_event(0.1)
    own = make_event(0.3)
    absorbed_twin = make_event(0.3)
    assert own == absorbed_twin and own is not absorbed_twin
    card = make_card(absorbed_ids=("dead-1",))
    events = {card.id: [shared, own], "dead-1": [shared, absorbed_twin]}
    stamped = CardStatsStamper().stamp_gain_events(card, events)
    assert stamped.gain_events == (shared, own, absorbed_twin)
    assert stamped.gain_events[1] is own
    assert stamped.gain_events[2] is absorbed_twin


def test_stamper_clears_stale_events_when_uncredited(make_card, make_event):
    card = make_card(gain_events=(make_event(0.5),))
    stamped = CardStatsStamper().stamp_gain_events(card, {})
    assert stamped.gain_events == ()


def test_stamper_skips_absorbed_fold_for_program_cards(make_card, make_event):
    card = make_card(
        kind=CardKind.PROGRAM,
        program_id="p1",
        code="x = 1",
        fitness=0.5,
        absorbed_ids=("dead-1",),
    )
    own = make_event(0.1)
    events = {card.id: [own], "dead-1": [make_event(0.2)]}
    stamped = CardStatsStamper().stamp_gain_events(card, events)
    assert stamped.gain_events == (own,)


def test_updater_restamps_changed_cards_and_sweeps(
    store, make_card, make_program, make_event, metrics_context, monkeypatch
):
    emitted: list = []
    monkeypatch.setattr("gigaevo.memory.write.stats.emit_memory_event", emitted.append)
    credited = make_card(id="card-a")
    stale = make_card(id="card-b", gain_events=(make_event(0.5),))
    untouched = make_card(id="card-c")
    for card in (credited, stale, untouched):
        store.save(card)
    saves_before = len(store.saved_ids)

    pool = [
        make_program(
            fitness=0.7, metadata=base_meta(selected=["card-a"], used=["card-a"])
        )
    ]
    updater = CardStatsUpdater(
        fitness_key="fitness", higher_is_better=True, metrics_context=metrics_context
    )
    gate = CardAdmissionGate(store=store, evictor=NullEvictor())
    updater.update(pool, store=store, gate=gate)

    restamps = [e for e in emitted if isinstance(e, MemoryGainRestamp)]
    assert len(restamps) == 1
    assert restamps[0].credited_card_count == 1
    assert restamps[0].event_count_by_card_id == {"card-a": 1}

    assert len(store.get("card-a").gain_events) == 1
    assert store.get("card-b").gain_events == ()
    assert store.saved_ids[saves_before:] == ["card-a", "card-b"]


def test_updater_sweep_removes_harmful_cards(store, make_card, metrics_context):
    class EvictAll:
        def should_evict(self, card) -> bool:
            return True

        def sweep(self, cards) -> list[str]:
            return [card.id for card in cards]

    store.save(make_card(id="card-doomed"))
    updater = CardStatsUpdater(
        fitness_key="fitness", higher_is_better=True, metrics_context=metrics_context
    )
    gate = CardAdmissionGate(store=store, evictor=EvictAll())
    updater.update([], store=store, gate=gate)
    assert store.get("card-doomed") is None
