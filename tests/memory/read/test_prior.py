from __future__ import annotations

from datetime import datetime, timedelta

from pydantic import ValidationError
import pytest

from gigaevo.evolution.strategies.models import BehaviorSpace, LinearBinning
from gigaevo.memory.cards import DecisionContext, EvidenceAttribution
from gigaevo.memory.context import BDCellMemoryContext
from gigaevo.memory.read.prior import (
    EmpiricalBayesMemoryPrior,
    _first_non_founding_exposure,
)
from gigaevo.memory.storage.base import MemoryStore


class _Store(MemoryStore):
    def __init__(self, cards):
        self._cards = tuple(cards)

    @property
    def is_ready(self):
        return True

    def save(self, card):
        return card.id

    def update(self, card_id, transform):
        card = self.get(card_id)
        return None if card is None else transform(card)

    def get(self, card_id):
        return next((card for card in self._cards if card.id == card_id), None)

    def delete(self, card_id):
        return False

    def snapshot(self):
        return self._cards

    def merge_retire(self, target_id, partner_id, fold):
        del target_id, partner_id, fold
        raise NotImplementedError

    def nearest(self, text, k, kind=None):
        del text, k, kind
        return []

    async def research(self, request):
        from gigaevo.memory.storage.base import ResearchResult

        del request
        return ResearchResult()

    def rebuild(self):
        return None

    def close(self):
        return None


def _for_task(event, task_key: str):
    return event.model_copy(
        update={"context": event.context.model_copy(update={"task_key": task_key})}
    )


def test_first_non_founding_exposure_uses_earliest_timestamp(make_event):
    earlier_at = datetime.min
    later_at = earlier_at + timedelta.resolution
    earlier_failure = make_event(-1.0)
    earlier_failure = earlier_failure.model_copy(
        update={
            "context": earlier_failure.context.model_copy(
                update={"timestamp": earlier_at}
            )
        }
    )
    later_success = make_event(1.0)
    later_success = later_success.model_copy(
        update={
            "context": later_success.context.model_copy(update={"timestamp": later_at})
        }
    )

    assert _first_non_founding_exposure([later_success, earlier_failure]) == (
        1.0,
        False,
    )
    assert _first_non_founding_exposure([earlier_failure, later_success]) == (
        1.0,
        False,
    )


def test_empirical_bayes_cold_prior_keeps_most_specific_nonempty_cohort(
    make_card, make_event
):
    target_evidence = make_card(category="target", gain_events=(make_event(1.0),))
    global_failures = tuple(
        make_card(category="other", gain_events=(make_event(-1.0),)) for _ in range(4)
    )
    query = make_card(category="target")
    prior = EmpiricalBayesMemoryPrior(
        store=_Store((target_evidence, *global_failures)),
        shrink_events=0.0,
        n_ref=1.0,
    ).cold_card_prior(query)

    assert prior.source == "eb_kind_category"
    assert prior.alpha > prior.beta


def test_empirical_bayes_ignores_unused_only_exposures(make_card, make_event):
    # An ignored card says nothing about whether its advice helps when acted
    # on; counting ignores as failures makes the cold prior learn the mutator
    # use-rate instead of P(help | used) and starves exploration.
    ignored = make_card(category="target", gain_events=(make_event(0.0, unused=True),))
    query = make_card(category="target")

    prior = EmpiricalBayesMemoryPrior(
        store=_Store((ignored,)),
        shrink_events=0.0,
        n_ref=1.0,
    ).cold_card_prior(query)

    assert prior.source == "eb_seed"


def test_empirical_bayes_first_exposure_skips_ignores_to_first_real_outcome(
    make_card, make_event
):
    card = make_card(
        category="target",
        gain_events=(make_event(0.0, unused=True), make_event(1.0)),
    )
    query = make_card(category="target")

    prior = EmpiricalBayesMemoryPrior(
        store=_Store((card,)),
        shrink_events=0.0,
        n_ref=1.0,
    ).cold_card_prior(query)

    # A one-card bank yields identical counts at every level, so the deepest
    # informative cohort is the global one.
    assert prior.source == "eb_global"
    assert prior.alpha > prior.beta


def test_empirical_bayes_invalid_first_exposure_counts_as_failure(
    make_card, make_event
):
    crasher = make_card(category="target", gain_events=(make_event(0.0, invalid=True),))
    query = make_card(category="target")

    prior = EmpiricalBayesMemoryPrior(
        store=_Store((crasher,)),
        shrink_events=0.0,
        n_ref=1.0,
    ).cold_card_prior(query)

    assert prior.source == "eb_global"
    assert prior.beta > prior.alpha


@pytest.mark.parametrize("first_exposure_only", [True, False])
def test_empirical_bayes_zero_gain_is_weighted_failure_in_both_counting_modes(
    make_card, make_event, first_exposure_only
):
    exposure_weight = 0.25
    event = make_event(0.0).model_copy(
        update={
            "attribution": EvidenceAttribution(credit_weight=exposure_weight),
        }
    )
    card = make_card(gain_events=(event,))
    model = EmpiricalBayesMemoryPrior(
        store=_Store((card,)), first_exposure_only=first_exposure_only
    )

    counts = model._card_counts(card, None, local=False, task_local=False)

    assert counts == pytest.approx((0.0, exposure_weight))


@pytest.mark.parametrize("first_exposure_only", [True, False])
@pytest.mark.parametrize(
    ("gain", "expected"),
    [(1.0, (1.0, 0.0)), (-1.0, (0.0, 1.0))],
)
def test_empirical_bayes_nonzero_gain_sign_counts_are_unchanged(
    make_card, make_event, first_exposure_only, gain, expected
):
    card = make_card(gain_events=(make_event(gain),))
    model = EmpiricalBayesMemoryPrior(
        store=_Store((card,)), first_exposure_only=first_exposure_only
    )

    counts = model._card_counts(card, None, local=False, task_local=False)

    assert counts == expected


def test_empirical_bayes_bd_local_prior_does_not_leak_other_cells(
    make_card, make_event
):
    space = BehaviorSpace(
        bins={"x": LinearBinning(min_val=0.0, max_val=1.0, num_bins=2)}
    )
    local_success = make_card(
        category="target", gain_events=(make_event(1.0, metrics={"x": 0.2}),)
    )
    other_cell_failures = tuple(
        make_card(
            category="target",
            gain_events=(make_event(-1.0, metrics={"x": 0.8}),),
        )
        for _ in range(4)
    )
    query = make_card(category="target")

    prior = EmpiricalBayesMemoryPrior(
        store=_Store((local_success, *other_cell_failures)),
        context_model=BDCellMemoryContext(behavior_space=space),
        shrink_events=0.0,
        n_ref=1.0,
    ).cold_card_prior(query, DecisionContext(parent_metrics={"x": 0.2}))

    # The context partition is where the counts first change (all cards share
    # kind and category), so the ladder stops refining there.
    assert prior.source == "eb_context"
    assert prior.support_n == 1.0
    assert prior.alpha > prior.beta


def test_context_cohort_excludes_foreign_metrics_but_global_keeps_sign(
    make_card, make_event
):
    space = BehaviorSpace(
        bins={"x": LinearBinning(min_val=0.0, max_val=1.0, num_bins=2)}
    )
    foreign = make_card(
        gain_events=(_for_task(make_event(1e300, metrics={"x": 0.2}), "task-b"),)
    )
    context = DecisionContext(task_key="task-a", parent_metrics={"x": 0.2})
    model = EmpiricalBayesMemoryPrior(
        store=_Store((foreign,)),
        context_model=BDCellMemoryContext(behavior_space=space),
        levels=("context",),
        shrink_events=0.0,
        n_ref=1.0,
    )

    assert model._card_counts(foreign, context, local=True, task_local=False) == (
        0.0,
        0.0,
    )
    assert model._card_counts(foreign, context, local=False, task_local=False) == (
        1.0,
        0.0,
    )
    prior = model.cold_card_prior(make_card(), context)
    assert (prior.source, prior.support_n) == ("eb_global", 1.0)
    assert prior.alpha > prior.beta


def test_identical_cohort_counts_shrink_once_not_per_level(make_card, make_event):
    # Four same-kind/same-category failures appear with identical counts in the
    # global, kind, and kind+category cohorts; re-applying the same counts at
    # every level compounds the shrinkage toward the cohort rate. A level that
    # adds no new information over the previously applied one must be skipped.
    bank = tuple(
        make_card(category="same", gain_events=(make_event(-1.0),)) for _ in range(4)
    )
    query = make_card(category="same")

    ladder = EmpiricalBayesMemoryPrior(store=_Store(bank)).cold_card_prior(query)
    global_only = EmpiricalBayesMemoryPrior(
        store=_Store(bank), levels=()
    ).cold_card_prior(query)

    assert ladder.source == "eb_global"
    assert (ladder.alpha, ladder.beta) == (global_only.alpha, global_only.beta)


def test_cohort_levels_are_config_driven(make_card, make_event):
    same_cat = make_card(category="target", gain_events=(make_event(1.0),))
    other_cat = tuple(
        make_card(category="other", gain_events=(make_event(-1.0),)) for _ in range(4)
    )
    query = make_card(category="target")

    prior = EmpiricalBayesMemoryPrior(
        store=_Store((same_cat, *other_cat)),
        levels=("category",),
        shrink_events=0.0,
        n_ref=1.0,
    ).cold_card_prior(query)

    assert prior.source == "eb_category"
    assert prior.support_n == 1.0
    assert prior.alpha > prior.beta


def test_unknown_cohort_level_token_rejected():
    with pytest.raises(ValueError):
        EmpiricalBayesMemoryPrior(store=_Store(()), levels=("kind+flavor",))


def test_duplicate_cohort_level_rejected():
    with pytest.raises(ValueError):
        EmpiricalBayesMemoryPrior(
            store=_Store(()), levels=("kind+category", "category+kind")
        )


def test_sibling_cohort_levels_rejected():
    # "category" after "kind" is a sibling slice, not a refinement: chaining
    # parent_mu across siblings would shrink one cohort toward the other's rate.
    with pytest.raises(ValueError):
        EmpiricalBayesMemoryPrior(store=_Store(()), levels=("kind", "category"))


def test_coarser_level_after_finer_rejected():
    with pytest.raises(ValueError):
        EmpiricalBayesMemoryPrior(store=_Store(()), levels=("kind+category", "kind"))


def test_global_level_after_context_level_rejected():
    with pytest.raises(ValueError):
        EmpiricalBayesMemoryPrior(store=_Store(()), levels=("context", "kind"))


def test_default_ladder_passes_refinement_validation():
    prior = EmpiricalBayesMemoryPrior(store=_Store(()))
    assert prior.levels == (
        "kind",
        "kind+category",
        "task+kind+category",
        "context",
        "context+kind",
        "context+kind+category",
        "task+context+kind+category",
    )


def test_empirical_bayes_concentration_range_is_monotonic():
    with pytest.raises(ValidationError, match=r"k_max.*2\.0.*k_min.*6\.0"):
        EmpiricalBayesMemoryPrior(store=_Store(()), k_min=6.0, k_max=2.0)

    fixed = EmpiricalBayesMemoryPrior(store=_Store(()), k_min=6.0, k_max=6.0)
    default = EmpiricalBayesMemoryPrior(store=_Store(()))

    assert fixed.k_min == fixed.k_max == 6.0
    assert default.k_min <= default.k_max


def test_task_cohort_counts_only_native_events(make_card, make_event):
    native_success = make_card(
        category="target",
        gain_events=(_for_task(make_event(1.0), "task-a"),),
    )
    foreign_failures = tuple(
        make_card(
            category="target",
            gain_events=(_for_task(make_event(-1.0), "task-b"),),
        )
        for _ in range(4)
    )

    prior = EmpiricalBayesMemoryPrior(
        store=_Store((native_success, *foreign_failures)),
        shrink_events=0.0,
        n_ref=1.0,
    ).cold_card_prior(make_card(category="target"), DecisionContext(task_key="task-a"))

    assert prior.source == "eb_task_kind_category"
    assert prior.support_n == 1.0
    assert prior.alpha > prior.beta


def test_context_without_task_key_skips_task_levels(make_card, make_event):
    card = make_card(
        category="target", gain_events=(_for_task(make_event(1.0), "task-a"),)
    )
    prior = EmpiricalBayesMemoryPrior(
        store=_Store((card,)),
        levels=("task+kind+category",),
        shrink_events=0.0,
        n_ref=1.0,
    ).cold_card_prior(make_card(category="target"), DecisionContext())

    assert prior.source == "eb_global"


def test_legacy_unstamped_default_prior_matches_pre_task_ladder(make_card, make_event):
    bank = (
        make_card(category="target", gain_events=(make_event(1.0),)),
        make_card(category="other", gain_events=(make_event(-1.0),)),
    )
    query = make_card(category="target")
    context = DecisionContext()

    current = EmpiricalBayesMemoryPrior(store=_Store(bank)).cold_card_prior(
        query, context
    )
    legacy = EmpiricalBayesMemoryPrior(
        store=_Store(bank),
        levels=(
            "kind",
            "kind+category",
            "context",
            "context+kind",
            "context+kind+category",
        ),
    ).cold_card_prior(query, context)

    assert current.model_dump_json() == legacy.model_dump_json()


def test_evicted_evidence_none_is_byte_identical(make_card, make_event):
    bank = tuple(
        make_card(category="target", gain_events=(make_event(gain),))
        for gain in (1.0, -1.0)
    )
    query = make_card(category="target")

    implicit = EmpiricalBayesMemoryPrior(store=_Store(bank)).cold_card_prior(query)
    explicit = EmpiricalBayesMemoryPrior(
        store=_Store(bank), evicted_evidence=None
    ).cold_card_prior(query)

    assert (implicit.alpha, implicit.beta, implicit.source, implicit.support_n) == (
        1.125,
        1.125,
        "eb_global",
        2.0,
    )
    assert explicit == implicit


def test_evicted_evidence_lowers_survivor_biased_cold_prior(make_card, make_event):
    class Evidence:
        def __init__(self, cards):
            self._cards = tuple(cards)

        def cards(self):
            return self._cards

    survivors = tuple(
        make_card(category="target", gain_events=(make_event(1.0),)) for _ in range(3)
    )
    evicted = make_card(category="target", gain_events=(make_event(-1.0),))
    query = make_card(category="target")
    kwargs = {
        "store": _Store(survivors),
        "levels": (),
        "shrink_events": 0.0,
        "n_ref": 1.0,
    }

    snapshot_only = EmpiricalBayesMemoryPrior(**kwargs).cold_card_prior(query)
    corrected = EmpiricalBayesMemoryPrior(
        **kwargs, evicted_evidence=Evidence((evicted,))
    ).cold_card_prior(query)

    snapshot_mu = snapshot_only.alpha / (snapshot_only.alpha + snapshot_only.beta)
    corrected_mu = corrected.alpha / (corrected.alpha + corrected.beta)
    assert corrected_mu < snapshot_mu
    assert corrected.support_n > snapshot_only.support_n


def test_evicted_evidence_source_failure_degrades_to_snapshot(make_card, make_event):
    class FailingEvidence:
        def cards(self):
            raise OSError("unavailable evidence")

    bank = (make_card(gain_events=(make_event(1.0),)),)
    query = make_card()
    snapshot_only = EmpiricalBayesMemoryPrior(store=_Store(bank)).cold_card_prior(query)

    degraded = EmpiricalBayesMemoryPrior(
        store=_Store(bank), evicted_evidence=FailingEvidence()
    ).cold_card_prior(query)

    assert degraded == snapshot_only
