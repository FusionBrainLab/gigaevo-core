from __future__ import annotations

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
