from __future__ import annotations

import pytest

from gigaevo.evolution.strategies.models import BehaviorSpace, LinearBinning
from gigaevo.memory.cards import DecisionContext
from gigaevo.memory.context import BDCellMemoryContext
from gigaevo.memory.read.prior import EmpiricalBayesMemoryPrior
from gigaevo.memory.storage.base import MemoryStore


class _Store(MemoryStore):
    def __init__(self, cards):
        self._cards = tuple(cards)

    @property
    def is_ready(self):
        return True

    def save(self, card):
        return card.id

    def get(self, card_id):
        return next((card for card in self._cards if card.id == card_id), None)

    def delete(self, card_id):
        return False

    def snapshot(self):
        return self._cards

    def apply_merges(self, merged):
        del merged
        return []

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
    assert prior.levels[0] == "kind"
