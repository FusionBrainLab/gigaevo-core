"""Use-attributed gain computation, stamping, and the restamp sweep."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from loguru import logger
import numpy as np
import pytest

from gigaevo.evolution.mutation.constants import (
    MUTATION_MEMORY_BASE_EVALUATION_MEASUREMENTS_METADATA_KEY,
    MUTATION_MEMORY_BASE_ID_METADATA_KEY,
    MUTATION_MEMORY_BASE_METRICS_METADATA_KEY,
    MUTATION_MEMORY_BASE_SCORES_METADATA_KEY,
    MUTATION_MEMORY_BASE_SELECTED_IDS_METADATA_KEY,
    MUTATION_MEMORY_INJECTED_IDS_METADATA_KEY,
    MUTATION_MEMORY_NO_CARD_CONTROL_METADATA_KEY,
    MUTATION_OUTPUT_METADATA_KEY,
)
from gigaevo.evolution.strategies.models import BehaviorSpace, LinearBinning
from gigaevo.memory.cards import CardKind, EvidenceAttribution, EvidenceSource
from gigaevo.memory.context import BDCellMemoryContext
from gigaevo.memory.events import MemoryGainRestamp
from gigaevo.memory.selection_leases import InFlightSelectionRegistry
from gigaevo.memory.storage.config import StoreConfig
from gigaevo.memory.storage.local import LocalMemoryStore
from gigaevo.memory.write.admission import CardAdmissionGate
from gigaevo.memory.write.crediting import PairedEffectEstimator, PointEffectEstimator
from gigaevo.memory.write.eviction import NullEvictor
from gigaevo.memory.write.stats import (
    CardStatsStamper,
    CardStatsUpdater,
    GlobalNoCardBaseline,
    InjectionOutcome,
    card_gain_events_from_programs,
    compute_contextual_gains,
    founding_gain_event,
    injection_outcomes_from_programs,
)
from gigaevo.programs.metrics.context import VALIDITY_KEY
from gigaevo.programs.metrics.evaluation import EVALUATION_MEASUREMENTS_METADATA_KEY
from gigaevo.programs.metrics.paired import PER_SAMPLE_SCORES_KEY
from tests.fakes.embedding import FakeEmbeddingFunction


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


def no_card_outcome(**overrides) -> InjectionOutcome:
    params = {
        "id": "no-card",
        "fitness": 0.5,
        "base_selected_ids": (),
        "base_metrics": {VALIDITY_KEY: 1.0, "fitness": 0.5},
        "base_id": "parent-control",
        "base_fitness": 0.5,
        "card_ids_used": (),
        "no_card_control": True,
    }
    params.update(overrides)
    return InjectionOutcome(**params)


def base_meta(
    *,
    selected: list[str],
    used: list[str],
    base_fitness: float = 0.5,
    no_card_control: bool = False,
) -> dict:
    return {
        MUTATION_MEMORY_BASE_SELECTED_IDS_METADATA_KEY: selected,
        MUTATION_MEMORY_BASE_METRICS_METADATA_KEY: {
            VALIDITY_KEY: 1.0,
            "fitness": base_fitness,
        },
        MUTATION_MEMORY_BASE_ID_METADATA_KEY: "parent-1",
        MUTATION_MEMORY_NO_CARD_CONTROL_METADATA_KEY: no_card_control,
        MUTATION_OUTPUT_METADATA_KEY: {"card_ids_used": used},
    }


def no_card_program(make_program, *, fitness: float = 0.5):
    return make_program(
        fitness=fitness,
        metadata=base_meta(selected=[], used=[], no_card_control=True),
    )


def test_non_finite_no_card_fitness_does_not_poison_global_baseline():
    events = compute_contextual_gains(
        [
            no_card_outcome(id="nan-control", fitness=float("nan")),
            no_card_outcome(id="finite-control", fitness=0.6),
            outcome(fitness=0.8),
        ]
    )
    assert events["card-a"][0].gain == pytest.approx(0.2)


def test_used_cards_and_unused_exposures_are_attributed_separately():
    events = compute_contextual_gains(
        [
            no_card_outcome(),
            outcome(base_selected_ids=("a", "b"), card_ids_used=("b", "hallucinated")),
        ]
    )
    assert set(events) == {"a", "b"}
    assert events["a"][0].gain == 0.0
    assert events["a"][0].unused is True
    assert events["b"][0].gain == pytest.approx(0.2)
    assert events["b"][0].unused is False


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
    assert compute_contextual_gains([outcome()]) == {}


def test_delta_orientation_follows_direction():
    higher = compute_contextual_gains(
        [no_card_outcome(), outcome()], higher_is_better=True
    )
    lower = compute_contextual_gains(
        [no_card_outcome(), outcome()], higher_is_better=False
    )
    assert higher["card-a"][0].gain == pytest.approx(0.2)
    assert lower["card-a"][0].gain == pytest.approx(-0.2)


def test_used_card_gain_subtracts_global_no_card_baseline():
    events = compute_contextual_gains(
        [
            outcome(
                id="no-card",
                fitness=0.6,
                base_selected_ids=(),
                card_ids_used=(),
            ),
            outcome(id="with-card", fitness=0.7),
        ]
    )
    assert events["card-a"][0].gain == pytest.approx(0.1)


def test_no_card_baseline_prefers_randomized_controls_over_policy_abstentions():
    events = compute_contextual_gains(
        [
            no_card_outcome(id="control", fitness=0.55),
            no_card_outcome(
                id="policy-abstention", fitness=0.95, no_card_control=False
            ),
            outcome(id="with-card", fitness=0.7),
        ]
    )

    assert events["card-a"][0].gain == pytest.approx(0.15)


def test_used_card_gain_accepts_custom_no_card_estimator():
    class FixedBaseline:
        has_evidence = True

        def fit_no_card_baseline(self, outcomes, *, higher_is_better):
            del outcomes, higher_is_better
            return self

        def baseline_for(self, outcome):
            del outcome
            return 0.15

        def baseline_se_for(self, outcome):
            del outcome
            return None

    events = compute_contextual_gains(
        [outcome(id="with-card", fitness=0.7)],
        baseline_estimator=FixedBaseline(),
    )
    assert events["card-a"][0].gain == pytest.approx(0.05)


def test_global_no_card_baseline_exposes_sem_for_multiple_deltas():
    controls = [
        no_card_outcome(id="control-low", fitness=0.5),
        no_card_outcome(id="control-high", fitness=0.7),
    ]
    fitted = GlobalNoCardBaseline().fit_no_card_baseline(
        controls, higher_is_better=True
    )
    expected = float(np.std([0.0, 0.2], ddof=1) / np.sqrt(len(controls)))

    assert fitted.baseline_se_for(controls[0]) == pytest.approx(expected)


def test_global_no_card_baseline_single_delta_has_unknown_sem():
    control = no_card_outcome()
    fitted = GlobalNoCardBaseline().fit_no_card_baseline(
        [control], higher_is_better=True
    )

    assert fitted.baseline_se_for(control) is None


def test_used_card_gain_subtracts_same_bd_cell_no_card_baseline():
    space = BehaviorSpace(
        bins={"x": LinearBinning(min_val=0.0, max_val=1.0, num_bins=2)}
    )
    events = compute_contextual_gains(
        [
            outcome(
                id="no-card-low",
                fitness=0.9,
                base_selected_ids=(),
                base_metrics={VALIDITY_KEY: 1.0, "fitness": 0.5, "x": 0.25},
                card_ids_used=(),
            ),
            outcome(
                id="no-card-high",
                fitness=0.6,
                base_selected_ids=(),
                base_metrics={VALIDITY_KEY: 1.0, "fitness": 0.5, "x": 0.75},
                card_ids_used=(),
            ),
            outcome(
                id="with-card-high",
                fitness=0.8,
                base_metrics={VALIDITY_KEY: 1.0, "fitness": 0.5, "x": 0.75},
            ),
        ],
        baseline_estimator=BDCellMemoryContext(behavior_space=space),
    )
    assert events["card-a"][0].gain == pytest.approx(0.2)


def test_credit_intersection_strips_whitespace_padded_ids():
    # The stamper looks cards up under card.id.strip(); a mutator that echoes
    # " mem-x" while selection recorded "mem-x" must still intersect — else
    # real credit silently orphans on formatting noise.
    events = compute_contextual_gains(
        [
            no_card_outcome(),
            outcome(base_selected_ids=(" mem-x",), card_ids_used=("mem-x ", "  ")),
        ]
    )
    assert set(events) == {"mem-x"}
    assert events["mem-x"][0].gain == pytest.approx(0.2)


def test_one_child_shares_one_event_object_across_credited_cards():
    events = compute_contextual_gains(
        [
            no_card_outcome(),
            outcome(base_selected_ids=("a", "b"), card_ids_used=("a", "b")),
        ]
    )
    assert events["a"][0] is events["b"][0]
    assert events["a"][0].gain == pytest.approx(0.2)
    assert events["a"][0].attribution.credit_weight == pytest.approx(0.5)


def test_selected_unused_gets_no_use_event_without_child_fitness():
    events = compute_contextual_gains(
        [
            no_card_outcome(),
            outcome(
                fitness=None,
                base_selected_ids=("used", "unused"),
                card_ids_used=("used",),
            ),
        ]
    )
    assert set(events) == {"unused"}
    assert events["unused"][0].unused is True
    assert events["unused"][0].gain == 0.0


def test_valid_child_without_baseline_skips_unused_exposures_too():
    events = compute_contextual_gains(
        [
            outcome(
                base_selected_ids=("unused",),
                card_ids_used=(),
            )
        ]
    )

    assert events == {}


def test_bundled_use_keeps_full_delta_and_splits_credit_weight():
    # One child, one unit of causal evidence: the delta atom stays whole and
    # only the credit weight is split — splitting both would price a bundled
    # win at 1/k**2 of a solo win in the EV mean.
    events = compute_contextual_gains(
        [
            no_card_outcome(),
            outcome(base_selected_ids=("a", "b", "c"), card_ids_used=("a", "b")),
        ]
    )
    assert events["a"][0] is events["b"][0]
    assert events["a"][0].gain == pytest.approx(0.2)
    assert events["a"][0].attribution.credit_weight == pytest.approx(0.5)
    assert events["b"][0].gain == pytest.approx(0.2)
    assert events["c"][0].unused is True


def test_unused_exposure_weight_diluted_by_slate_size():
    # An ignore inside a crowded prompt is weak evidence: the mutator's
    # attention was split across the slate, so the exposure failure carries
    # 1/len(selected), mirroring the 1/k credit split for bundled use.
    events = compute_contextual_gains(
        [
            no_card_outcome(),
            outcome(base_selected_ids=("a", "b", "c"), card_ids_used=()),
        ]
    )
    assert events["a"][0].unused is True
    assert events["a"][0].attribution.credit_weight == pytest.approx(1.0 / 3.0)


def test_solo_unused_exposure_keeps_full_weight():
    events = compute_contextual_gains(
        [
            no_card_outcome(),
            outcome(base_selected_ids=("a",), card_ids_used=()),
        ]
    )
    assert events["a"][0].unused is True
    assert events["a"][0].attribution.credit_weight == pytest.approx(1.0)


def test_invalid_no_card_rows_are_excluded_from_the_baseline():
    # A crashed no-card child has no honest progress magnitude; counting it as
    # delta 0.0 drags the baseline median toward zero and inflates card gains.
    events = compute_contextual_gains(
        [
            no_card_outcome(id="valid-control", fitness=0.6),
            no_card_outcome(id="crashed-control", fitness=None, invalid=True),
            outcome(id="with-card", fitness=0.7),
        ]
    )
    assert events["card-a"][0].gain == pytest.approx(0.1)


def test_invalid_child_harms_used_cards_but_marks_unused_exposures():
    events = compute_contextual_gains(
        [
            outcome(
                fitness=None,
                invalid=True,
                base_selected_ids=("used", "unused"),
                card_ids_used=("used",),
            )
        ]
    )
    assert events["used"][0].invalid is True
    assert events["used"][0].unused is False
    assert events["unused"][0].unused is True
    assert events["unused"][0].invalid is False


def test_gain_events_from_programs_carry_base_context(make_program, metrics_context):
    prog = make_program(
        fitness=0.7, metadata=base_meta(selected=["card-a"], used=["card-a"])
    )
    events = card_gain_events_from_programs(
        [no_card_program(make_program), prog],
        fitness_key="fitness",
        higher_is_better=True,
        metrics_context=metrics_context,
    )
    (event,) = events["card-a"]
    assert event.gain == pytest.approx(0.2)
    assert event.context.parent_id == "parent-1"
    assert event.context.timestamp == prog.created_at


def test_founding_gain_event_carries_reported_uncertainty(
    make_program, metrics_context
):
    metadata = base_meta(selected=[], used=[])
    metadata[EVALUATION_MEASUREMENTS_METADATA_KEY] = {
        "fitness": {
            "value": 0.7,
            "sample_sd": 0.12,
            "n": 4,
            "method": "cross_validation",
        }
    }
    metadata[MUTATION_MEMORY_BASE_EVALUATION_MEASUREMENTS_METADATA_KEY] = {
        "fitness": {
            "value": 0.5,
            "sample_sd": 0.08,
            "n": 4,
            "method": "cross_validation",
        }
    }
    event = founding_gain_event(
        make_program(fitness=0.7, metadata=metadata),
        fitness_key="fitness",
        higher_is_better=True,
        metrics_context=metrics_context,
    )

    assert event is not None
    assert event.gain_se == pytest.approx(np.hypot(0.12 / 2, 0.08 / 2))


def test_gain_events_from_programs_credit_full_injected_slate(
    make_program, metrics_context
):
    metadata = base_meta(selected=["base-card"], used=["donor-card"])
    metadata[MUTATION_MEMORY_INJECTED_IDS_METADATA_KEY] = ["base-card", "donor-card"]
    prog = make_program(fitness=0.7, metadata=metadata)

    events = card_gain_events_from_programs(
        [no_card_program(make_program), prog],
        fitness_key="fitness",
        higher_is_better=True,
        metrics_context=metrics_context,
    )

    assert set(events) == {"base-card", "donor-card"}
    assert events["base-card"][0].unused is True
    assert events["donor-card"][0].gain == pytest.approx(0.2)


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


def test_founding_gain_event_signed_positive_delta(make_program, metrics_context):
    prog = make_program(
        fitness=0.7, metadata=base_meta(selected=["card-a"], used=["card-a"])
    )
    event = founding_gain_event(
        prog,
        fitness_key="fitness",
        higher_is_better=True,
        metrics_context=metrics_context,
    )
    assert event is not None
    assert event.gain == pytest.approx(0.2)
    assert event.gain_se is None
    assert event.founding is True
    assert event.invalid is False
    assert event.context.parent_id == "parent-1"
    assert event.context.timestamp == prog.created_at


def test_founding_gain_event_stamps_task_key(make_program, metrics_context):
    prog = make_program(
        fitness=0.7, metadata=base_meta(selected=["card-a"], used=["card-a"])
    )

    event = founding_gain_event(
        prog,
        fitness_key="fitness",
        higher_is_better=True,
        metrics_context=metrics_context,
        task_key="heilbronn",
    )

    assert event.context.task_key == "heilbronn"


def test_outcome_events_stamp_task_key():
    events = compute_contextual_gains(
        [no_card_outcome(), outcome()], task_key="heilbronn"
    )

    assert events["card-a"][0].context.task_key == "heilbronn"


def test_founding_gain_event_carries_true_negative_delta(make_program, metrics_context):
    prog = make_program(
        fitness=0.3,
        metadata=base_meta(selected=["card-a"], used=["card-a"], base_fitness=0.5),
    )
    event = founding_gain_event(
        prog,
        fitness_key="fitness",
        higher_is_better=True,
        metrics_context=metrics_context,
    )
    assert event is not None
    assert event.gain == pytest.approx(-0.2)
    assert event.founding is True


def test_founding_gain_event_none_without_base_baseline(make_program, metrics_context):
    prog = make_program(fitness=0.7, parents=["p"], metadata={})
    assert (
        founding_gain_event(
            prog,
            fitness_key="fitness",
            higher_is_better=True,
            metrics_context=metrics_context,
        )
        is None
    )


def test_founding_gain_event_none_when_base_sentinel(make_program, metrics_context):
    prog = make_program(
        fitness=0.7,
        metadata=base_meta(selected=["card-a"], used=["card-a"], base_fitness=-1e5),
    )
    assert (
        founding_gain_event(
            prog,
            fitness_key="fitness",
            higher_is_better=True,
            metrics_context=metrics_context,
        )
        is None
    )


def test_founding_gain_event_respects_minimize_direction(make_program, metrics_context):
    prog = make_program(
        fitness=0.7, metadata=base_meta(selected=["card-a"], used=["card-a"])
    )
    event = founding_gain_event(
        prog,
        fitness_key="fitness",
        higher_is_better=False,
        metrics_context=metrics_context,
    )
    assert event is not None
    assert event.gain == pytest.approx(-0.2)


def test_stamper_preserves_founding_event_when_uncredited(make_card, make_event):
    founding = make_event(0.2, founding=True, parent_id="parent-1")
    card = make_card(gain_events=(founding,))
    stamped = CardStatsStamper().stamp_gain_events(card, {})
    assert stamped.gain_events == (founding,)
    assert stamped.gain_events[0].founding is True


def test_stamper_layers_recomputed_use_events_over_founding(make_card, make_event):
    founding = make_event(0.2, founding=True, parent_id="parent-1")
    use_event = make_event(0.3)
    card = make_card(gain_events=(founding,))
    stamped = CardStatsStamper().stamp_gain_events(card, {card.id: [use_event]})
    assert stamped.gain_events == (founding, use_event)


def test_stamper_still_clears_stale_founding_absent_use_events(make_card, make_event):
    # A founding event is preserved; a stale NON-founding use event unioned onto
    # the card (e.g. by a prior merge) is cleared and recomputed from the pool.
    founding = make_event(0.2, founding=True, parent_id="parent-1")
    stale_use = make_event(0.9)
    card = make_card(gain_events=(founding, stale_use))
    stamped = CardStatsStamper().stamp_gain_events(card, {})
    assert stamped.gain_events == (founding,)


def test_stamper_preserves_both_founding_events_across_merge(make_card, make_event):
    # A survivor that absorbed a near-duplicate carries BOTH its own founding
    # event and the partner's (merge unions gain_events). The from-scratch
    # restamp preserves both, layers this sweep's recomputed use events, and
    # re-aliases the absorbed id's use events — with no founding double-count,
    # since the recomputed pool never contains founding events.
    own_founding = make_event(0.2, founding=True, parent_id="parent-own")
    absorbed_founding = make_event(0.15, founding=True, parent_id="parent-absorbed")
    stale_use = make_event(0.9)
    survivor = make_card(
        gain_events=(own_founding, stale_use, absorbed_founding),
        absorbed_ids=("dead-1",),
    )
    own_use = make_event(0.3)
    aliased_use = make_event(0.4)
    events = {survivor.id: [own_use], "dead-1": [aliased_use]}
    stamped = CardStatsStamper().stamp_gain_events(survivor, events)
    assert stamped.gain_events == (
        own_founding,
        absorbed_founding,
        own_use,
        aliased_use,
    )
    assert sum(1 for e in stamped.gain_events if e.founding) == 2


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


def test_stamper_preserves_unattributed_event_with_external_preservation(
    make_card, make_event
):
    founding = make_event(0.1, founding=True)
    unattributed = make_event(0.2)
    in_pool = make_event(0.3).model_copy(
        update={
            "attribution": EvidenceAttribution(
                source=EvidenceSource.DIRECT,
                source_child_id="current-child",
            )
        }
    )
    replacement = make_event(0.4)
    card = make_card(gain_events=(founding, unattributed, in_pool))

    stamped = CardStatsStamper().stamp_gain_events(
        card,
        {card.id: [replacement]},
        preserve_events_outside_child_ids={"current-child"},
    )

    assert stamped.gain_events == (founding, unattributed, replacement)


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
    child = make_program(
        fitness=0.7, metadata=base_meta(selected=["card-a"], used=["card-a"])
    )
    stale_event = make_event(0.5).model_copy(
        update={
            "attribution": EvidenceAttribution(
                source=EvidenceSource.DIRECT,
                source_child_id=child.id,
            )
        }
    )
    credited = make_card(id="card-a")
    stale = make_card(id="card-b", gain_events=(stale_event,))
    untouched = make_card(id="card-c")
    for card in (credited, stale, untouched):
        store.save(card)
    saves_before = len(store.saved_ids)

    pool = [
        no_card_program(make_program),
        child,
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


def test_updater_preserves_events_from_other_program_pools(
    store, make_card, make_program, make_event, metrics_context
):
    external = make_event(0.4).model_copy(
        update={
            "attribution": EvidenceAttribution(
                source=EvidenceSource.DIRECT,
                source_child_id="external-child",
            )
        }
    )
    store.save(make_card(id="card-a", gain_events=(external,)))
    child = make_program(
        fitness=0.7, metadata=base_meta(selected=["card-a"], used=["card-a"])
    )
    pool = [no_card_program(make_program), child]
    updater = CardStatsUpdater(
        fitness_key="fitness", higher_is_better=True, metrics_context=metrics_context
    )
    gate = CardAdmissionGate(store=store, evictor=NullEvictor())

    updater.update(pool, store=store, gate=gate)

    events = store.get("card-a").gain_events
    assert external in events
    assert {
        event.attribution.source_child_id
        for event in events
        if event.attribution is not None
    } == {"external-child", child.id}


def test_restamp_preserves_write_interleaved_after_snapshot(
    store, make_card, make_event, metrics_context, monkeypatch
):
    stale = make_event(0.1).model_copy(
        update={
            "attribution": EvidenceAttribution(
                source=EvidenceSource.DIRECT,
                source_child_id="current-child",
            )
        }
    )
    replacement = make_event(0.2).model_copy(
        update={
            "attribution": EvidenceAttribution(
                source=EvidenceSource.DIRECT,
                source_child_id="current-child",
            )
        }
    )
    concurrent = make_event(0.3).model_copy(
        update={
            "attribution": EvidenceAttribution(
                source=EvidenceSource.DIRECT,
                source_child_id="other-run-child",
            )
        }
    )
    card = make_card(id="card-a", gain_events=(stale,))
    store.save(card)
    original_snapshot = store.snapshot
    interleaved = False

    def snapshot_with_concurrent_write():
        nonlocal interleaved
        snapshot = original_snapshot()
        if not interleaved:
            interleaved = True
            store.save(card.model_copy(update={"gain_events": (stale, concurrent)}))
        return snapshot

    monkeypatch.setattr(store, "snapshot", snapshot_with_concurrent_write)
    updater = CardStatsUpdater(
        fitness_key="fitness", higher_is_better=True, metrics_context=metrics_context
    )
    gate = CardAdmissionGate(store=store, evictor=NullEvictor())

    updater.restamp_and_sweep(
        {card.id: [replacement]},
        store=store,
        gate=gate,
        preserve_events_outside_child_ids={"current-child"},
    )

    assert store.get(card.id).gain_events == (concurrent, replacement)


def test_restamp_skips_card_vanished_before_atomic_update(
    store, make_card, make_event, metrics_context, monkeypatch
):
    stale = make_event(0.1)
    replacement = make_event(0.2)
    card = make_card(gain_events=(stale,))
    store.save(card)
    original_update = store.update

    def vanish_then_update(card_id, transform):
        store.delete(card_id)
        return original_update(card_id, transform)

    monkeypatch.setattr(store, "update", vanish_then_update)
    updater = CardStatsUpdater(
        fitness_key="fitness", higher_is_better=True, metrics_context=metrics_context
    )
    gate = CardAdmissionGate(store=store, evictor=NullEvictor())
    records: list = []
    handler = logger.add(records.append, level="DEBUG")
    try:
        updater.restamp_and_sweep({card.id: [replacement]}, store=store, gate=gate)
    finally:
        logger.remove(handler)

    assert store.get(card.id) is None
    assert any(
        card.id in str(record) and "evicted" in str(record) for record in records
    )
    assert any("redirected=0 dropped=1" in str(record) for record in records)


def test_restamp_redirects_vanished_card_event_to_merge_survivor(
    tmp_path, make_card, make_event, metrics_context, monkeypatch
):
    monkeypatch.setattr(
        "gigaevo.memory.storage.index.SentenceTransformerEmbeddingFunction",
        FakeEmbeddingFunction,
    )
    config = StoreConfig(path=tmp_path / "shared-store")
    store_a = LocalMemoryStore(config)
    store_b = LocalMemoryStore(config)
    current_child = "current-child"
    external = make_event(0.7).model_copy(
        update={
            "attribution": EvidenceAttribution(
                source=EvidenceSource.DIRECT,
                source_child_id="other-task-child",
            )
        }
    )
    stale = make_event(-0.1).model_copy(
        update={
            "attribution": EvidenceAttribution(
                source=EvidenceSource.DIRECT,
                source_child_id=current_child,
            )
        }
    )
    replacement = make_event(0.4).model_copy(
        update={
            "attribution": EvidenceAttribution(
                source=EvidenceSource.DIRECT,
                source_child_id=current_child,
            )
        }
    )
    absorbed = make_card(id="card-c", gain_events=(stale,))
    survivor = make_card(id="card-s", gain_events=(external,))
    store_a.save(absorbed)
    store_a.save(survivor)
    original_update = store_a.update
    merged = False

    def merge_before_update(card_id, transform):
        nonlocal merged
        if card_id == absorbed.id and not merged:
            merged = True
            updated = store_b.update(
                survivor.id,
                lambda target: target.model_copy(
                    update={
                        "absorbed_ids": (*target.absorbed_ids, absorbed.id),
                        "gain_events": (
                            *target.gain_events,
                            *absorbed.gain_events,
                        ),
                    }
                ),
            )
            assert updated is not None
            assert store_b.delete(absorbed.id)
        return original_update(card_id, transform)

    monkeypatch.setattr(store_a, "update", merge_before_update)
    updater = CardStatsUpdater(
        fitness_key="fitness", higher_is_better=True, metrics_context=metrics_context
    )
    gate = CardAdmissionGate(store=store_a, evictor=NullEvictor())
    try:
        updater.restamp_and_sweep(
            {absorbed.id: [replacement]},
            store=store_a,
            gate=gate,
            preserve_events_outside_child_ids={current_child},
        )

        landed = store_a.get(survivor.id)
        assert landed is not None
        assert absorbed.id in landed.absorbed_ids
        assert landed.gain_events == (external, replacement)
        assert store_a.get(absorbed.id) is None
    finally:
        store_a.close()
        store_b.close()


def test_restamp_cleanly_drops_vanished_card_without_absorber(
    tmp_path, make_card, make_event, metrics_context, monkeypatch
):
    monkeypatch.setattr(
        "gigaevo.memory.storage.index.SentenceTransformerEmbeddingFunction",
        FakeEmbeddingFunction,
    )
    config = StoreConfig(path=tmp_path / "shared-store")
    store_a = LocalMemoryStore(config)
    store_b = LocalMemoryStore(config)
    stale = make_event(-0.1)
    replacement = make_event(0.4)
    card = make_card(id="card-c", gain_events=(stale,))
    store_a.save(card)
    original_update = store_a.update
    deleted = False

    def delete_before_update(card_id, transform):
        nonlocal deleted
        if card_id == card.id and not deleted:
            deleted = True
            assert store_b.delete(card.id)
        return original_update(card_id, transform)

    monkeypatch.setattr(store_a, "update", delete_before_update)
    updater = CardStatsUpdater(
        fitness_key="fitness", higher_is_better=True, metrics_context=metrics_context
    )
    gate = CardAdmissionGate(store=store_a, evictor=NullEvictor())
    records: list = []
    handler = logger.add(records.append, level="DEBUG")
    try:
        updater.restamp_and_sweep({card.id: [replacement]}, store=store_a, gate=gate)

        assert store_a.get(card.id) is None
        assert any(
            f"evicted card {card.id} with no absorbing survivor" in str(record)
            for record in records
        )
        assert any("redirected=0 dropped=1" in str(record) for record in records)
    finally:
        logger.remove(handler)
        store_a.close()
        store_b.close()


def test_terminal_child_lease_releases_only_after_restamp_sweep(
    store, make_card, metrics_context
):
    registry = InFlightSelectionRegistry()
    card = make_card(id="card-a")
    store.save(card)
    lease = registry.open_attempt("attempt-1", "parent-1")
    lease.attach_cards((card.id,))
    lease.transfer_to_child("child-1", (card.id,))

    class ObserveSweepLease:
        observed_leased = False

        def should_evict(self, card) -> bool:
            del card
            return False

        def eviction_reason(self, card) -> str:
            del card
            return ""

        def sweep(self, cards) -> list[str]:
            del cards
            self.observed_leased = registry.is_leased(card.id)
            return []

    evictor = ObserveSweepLease()
    gate = CardAdmissionGate(store=store, evictor=evictor, selection_leases=registry)
    updater = CardStatsUpdater(
        fitness_key="fitness",
        higher_is_better=True,
        metrics_context=metrics_context,
        selection_leases=registry,
    )

    updater.restamp_and_sweep(
        {},
        store=store,
        gate=gate,
        preserve_events_outside_child_ids={"child-1"},
    )

    assert evictor.observed_leased
    assert not registry.is_leased(card.id)


def test_threads_hammer_restamp_update_against_evict_sweep(
    store, make_card, make_event, metrics_context
):
    class StaleCandidateEvictor:
        def should_evict(self, card) -> bool:
            return not card.gain_events

        def eviction_reason(self, card) -> str:
            del card
            return "no gain evidence"

        def sweep(self, cards) -> list[str]:
            return [card.id for card in cards]

    events = [make_event(gain) for gain in (0.1, 0.2, 0.3, 0.4)]
    card = make_card(gain_events=(events[0],))
    store.save(card)
    gate = CardAdmissionGate(store=store, evictor=StaleCandidateEvictor())
    updater = CardStatsUpdater(
        fitness_key="fitness", higher_is_better=True, metrics_context=metrics_context
    )

    def restamp_many():
        for event in events * len(events):
            updater.restamp_and_sweep({card.id: [event]}, store=store, gate=gate)

    def evict_many():
        for _ in events * len(events):
            gate.sweep()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(restamp_many), executor.submit(evict_many)]
        for future in futures:
            future.result()

    survived = store.get(card.id)
    assert survived is not None
    assert len(survived.gain_events) == 1
    assert not gate.is_tombstoned(card.id)


def test_updater_drops_unresolvable_credit_from_restamp(
    store, make_card, make_program, metrics_context, monkeypatch
):
    # "ghost-1" was evicted after the child froze its base selection: the pool
    # still credits it, but nothing in the bank (id or absorbed alias) can
    # receive the events — the restamp event must not report phantom credit.
    emitted: list = []
    monkeypatch.setattr("gigaevo.memory.write.stats.emit_memory_event", emitted.append)
    store.save(make_card(id="card-a"))
    store.save(make_card(id="card-s", absorbed_ids=("dead-1",)))

    pool = [
        no_card_program(make_program),
        make_program(
            fitness=0.7,
            metadata=base_meta(
                selected=["card-a", "dead-1", "ghost-1"],
                used=["card-a", "dead-1", "ghost-1"],
            ),
        ),
    ]
    updater = CardStatsUpdater(
        fitness_key="fitness", higher_is_better=True, metrics_context=metrics_context
    )
    gate = CardAdmissionGate(store=store, evictor=NullEvictor())
    updater.update(pool, store=store, gate=gate)

    (restamp,) = [e for e in emitted if isinstance(e, MemoryGainRestamp)]
    assert set(restamp.event_count_by_card_id) == {"card-a", "dead-1"}
    assert restamp.credited_card_count == 2
    assert len(store.get("card-a").gain_events) == 1
    assert len(store.get("card-s").gain_events) == 1


def test_updater_logs_each_orphan_once_at_debug(
    store, make_card, make_program, metrics_context
):
    # The orphan drop repeats every sweep for as long as the crediting child
    # stays in the pool — a per-sweep info line for the same evicted id is
    # noise, not signal. Log a fresh id once, at debug.
    from loguru import logger

    records: list = []
    handler = logger.add(records.append, level="DEBUG")
    try:
        store.save(make_card(id="card-a"))
        pool = [
            no_card_program(make_program),
            make_program(
                fitness=0.7,
                metadata=base_meta(
                    selected=["card-a", "ghost-1"], used=["card-a", "ghost-1"]
                ),
            ),
        ]
        updater = CardStatsUpdater(
            fitness_key="fitness",
            higher_is_better=True,
            metrics_context=metrics_context,
        )
        gate = CardAdmissionGate(store=store, evictor=NullEvictor())
        updater.update(pool, store=store, gate=gate)
        updater.update(pool, store=store, gate=gate)
    finally:
        logger.remove(handler)

    orphan_logs = [r for r in records if "not resolvable in the bank" in str(r)]
    assert len(orphan_logs) == 1
    assert orphan_logs[0].record["level"].name == "DEBUG"


def test_updater_sweep_removes_harmful_cards(store, make_card, metrics_context):
    class EvictAll:
        def should_evict(self, card) -> bool:
            return True

        def eviction_reason(self, card) -> str:
            return "test evictor"

        def sweep(self, cards) -> list[str]:
            return [card.id for card in cards]

    store.save(make_card(id="card-doomed"))
    updater = CardStatsUpdater(
        fitness_key="fitness", higher_is_better=True, metrics_context=metrics_context
    )
    gate = CardAdmissionGate(store=store, evictor=EvictAll())
    updater.update([], store=store, gate=gate)
    assert store.get("card-doomed") is None


def test_updater_records_no_card_evidence(
    store, make_card, make_program, metrics_context
):
    class Recorder:
        def __init__(self):
            self.calls = []

        def record_outcomes(self, outcomes, *, higher_is_better, task_key=""):
            self.calls.append((tuple(outcomes), higher_is_better, task_key))

    recorder = Recorder()
    store.save(make_card(id="card-a"))
    parent = no_card_program(make_program)
    child = make_program(
        fitness=0.7,
        metadata=base_meta(selected=["card-a"], used=["card-a"]),
    )
    updater = CardStatsUpdater(
        fitness_key="fitness",
        higher_is_better=True,
        metrics_context=metrics_context,
        no_card_recorder=recorder,
        task_key="heilbronn",
    )
    gate = CardAdmissionGate(store=store, evictor=NullEvictor())

    updater.update([parent, child], store=store, gate=gate)

    [(outcomes, higher_is_better, task_key)] = recorder.calls
    assert higher_is_better is True
    assert task_key == "heilbronn"
    assert {outcome.id for outcome in outcomes} == {parent.id, child.id}


def test_updater_stamps_task_key_on_outcome_events(
    store, make_card, make_program, metrics_context
):
    store.save(make_card(id="card-a"))
    child = make_program(
        fitness=0.7,
        metadata=base_meta(selected=["card-a"], used=["card-a"]),
    )
    updater = CardStatsUpdater(
        fitness_key="fitness",
        higher_is_better=True,
        metrics_context=metrics_context,
        task_key="heilbronn",
    )
    gate = CardAdmissionGate(store=store, evictor=NullEvictor())

    updater.update([no_card_program(make_program), child], store=store, gate=gate)

    assert store.get("card-a").gain_events[0].context.task_key == "heilbronn"


def test_updater_continues_when_no_card_recorder_fails(
    store, make_card, make_program, metrics_context
):
    class BrokenRecorder:
        def record_outcomes(self, outcomes, *, higher_is_better, task_key=""):
            del outcomes, higher_is_better, task_key
            raise OSError("disk full")

    store.save(make_card(id="card-a"))
    child = make_program(
        fitness=0.7,
        metadata=base_meta(selected=["card-a"], used=["card-a"]),
    )
    updater = CardStatsUpdater(
        fitness_key="fitness",
        higher_is_better=True,
        metrics_context=metrics_context,
        no_card_recorder=BrokenRecorder(),
    )
    gate = CardAdmissionGate(store=store, evictor=NullEvictor())

    updater.update([no_card_program(make_program), child], store=store, gate=gate)

    assert len(store.get("card-a").gain_events) == 1


BASE_VEC = tuple([0.4, 0.6] * 150)  # mean 0.5, matches the factory base_fitness
CHILD_VEC = tuple([0.75, 0.65] * 150)  # mean 0.7, matches the factory fitness


def test_default_crediting_keeps_gains_exact():
    (event,) = compute_contextual_gains([no_card_outcome(), outcome()])["card-a"]
    assert event.gain_se == 0.0


def test_point_estimator_matches_default_bit_exact():
    rows = [no_card_outcome(), outcome()]
    assert compute_contextual_gains(rows) == compute_contextual_gains(
        rows, effect_estimator=PointEffectEstimator()
    )


def test_paired_estimator_prices_gain_se_without_moving_the_gain():
    rows = [
        no_card_outcome(),
        outcome(base_scores=BASE_VEC, child_scores=CHILD_VEC),
    ]
    point = compute_contextual_gains(rows)
    paired = compute_contextual_gains(rows, effect_estimator=PairedEffectEstimator())
    assert paired["card-a"][0].gain == point["card-a"][0].gain
    assert point["card-a"][0].gain_se == 0.0
    assert paired["card-a"][0].gain_se > 0.0


def test_direct_stamping_combines_measured_and_global_baseline_uncertainty():
    controls = [
        no_card_outcome(id="control-low", fitness=0.5),
        no_card_outcome(id="control-high", fitness=0.7),
    ]
    used = outcome(base_scores=BASE_VEC, child_scores=CHILD_VEC)
    rows = [*controls, used]
    estimator = PairedEffectEstimator()
    measured = estimator.estimate(used, higher_is_better=True)
    fitted = GlobalNoCardBaseline().fit_no_card_baseline(rows, higher_is_better=True)
    baseline_se = fitted.baseline_se_for(used)

    assert measured.se is not None and measured.se > 0.0
    assert baseline_se is not None
    paired_event = compute_contextual_gains(rows, effect_estimator=estimator)["card-a"][
        0
    ]
    point_event = compute_contextual_gains(
        rows, effect_estimator=PointEffectEstimator()
    )["card-a"][0]
    degraded_event = compute_contextual_gains(
        [*controls, used.model_copy(update={"child_scores": None})],
        effect_estimator=PairedEffectEstimator(),
    )["card-a"][0]

    assert paired_event.gain_se == pytest.approx(np.hypot(measured.se, baseline_se))
    assert paired_event.gain_se > measured.se
    assert point_event.gain_se == 0.0
    assert degraded_event.gain_se is None


def test_invalid_and_unused_events_stay_exact_under_paired():
    rows = [
        no_card_outcome(),
        outcome(id="crashed", fitness=None, invalid=True, base_scores=BASE_VEC),
        outcome(
            id="mixed",
            base_selected_ids=("card-a", "card-b"),
            card_ids_used=("card-a",),
            base_scores=BASE_VEC,
            child_scores=CHILD_VEC,
        ),
    ]
    events = compute_contextual_gains(rows, effect_estimator=PairedEffectEstimator())
    invalid_events = [e for e in events["card-a"] if e.invalid]
    assert invalid_events and all(e.gain_se == 0.0 for e in invalid_events)
    assert all(e.unused and e.gain_se == 0.0 for e in events["card-b"])
    used = [e for e in events["card-a"] if not e.invalid]
    assert used and all(e.gain_se > 0.0 for e in used)


def test_injection_outcomes_stamp_score_vectors(make_program, metrics_context):
    meta = base_meta(selected=["card-a"], used=["card-a"])
    meta[MUTATION_MEMORY_BASE_SCORES_METADATA_KEY] = [0.4, 0.6]
    meta[PER_SAMPLE_SCORES_KEY] = [0.6, 0.8]
    prog = make_program(fitness=0.7, metadata=meta)

    (row,) = injection_outcomes_from_programs(
        [prog], fitness_key="fitness", metrics_context=metrics_context
    )

    assert row.base_scores == (0.4, 0.6)
    assert row.child_scores == (0.6, 0.8)


def test_gain_events_from_programs_price_noise_when_vectors_flow(
    make_program, metrics_context
):
    meta = base_meta(selected=["card-a"], used=["card-a"])
    meta[MUTATION_MEMORY_BASE_SCORES_METADATA_KEY] = list(BASE_VEC)
    meta[PER_SAMPLE_SCORES_KEY] = list(CHILD_VEC)
    prog = make_program(fitness=0.7, metadata=meta)

    events = card_gain_events_from_programs(
        [no_card_program(make_program), prog],
        fitness_key="fitness",
        higher_is_better=True,
        metrics_context=metrics_context,
        effect_estimator=PairedEffectEstimator(),
    )

    (event,) = events["card-a"]
    assert event.gain == pytest.approx(0.2)
    assert event.gain_se > 0.0
