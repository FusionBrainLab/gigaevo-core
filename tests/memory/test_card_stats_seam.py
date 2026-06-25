"""The `card_stats` read seam: the single authority every card-statistic view
resolves through. It returns the block computed from the card's stored gain
events, so `card_posterior`, `card_magnitude`, eviction, and rendering all agree
on one source instead of each recomputing from the raw events independently."""

from __future__ import annotations

from gigaevo.memory.context import ContextualGain, DecisionContext
from gigaevo.memory.core.reputation import BetaBinomialReputation
from gigaevo.memory.shared_memory.models import (
    AnyCard,
    CardStatsBlock,
    MemoryCard,
)


def _events(gains: list[float]) -> list[ContextualGain]:
    return [
        ContextualGain(
            context=DecisionContext(parent_metrics={"min_area": 0.5}), gain=g
        )
        for g in gains
    ]


# Six wins and one loss: the MAD noise band collapses to 0, so exactly k_harm = 1
# of n = 7 events fall below threshold -> Beta(7, 2) downside posterior.
CONFIDENT_EVENTS = _events([0.01] * 6 + [-0.5])


class _FixedStatsReputation(BetaBinomialReputation):
    """Resolves every card to one fixed block regardless of its stored events —
    lets a test prove the per-card views read through `card_stats`, not the raw
    `gain_events`."""

    fixed: CardStatsBlock | None = None

    def card_stats(
        self, card: AnyCard, context: DecisionContext | None = None
    ) -> CardStatsBlock | None:
        return self.fixed


class TestCardStatsSeam:
    def test_card_stats_returns_block_from_gain_events(self):
        rep = BetaBinomialReputation()
        card = MemoryCard(id="idea-1", gain_events=CONFIDENT_EVENTS)
        block = rep.card_stats(card)
        assert block is not None
        assert (block.posterior_a, block.posterior_b) == (7.0, 2.0)
        assert block.intro_events == 7
        assert rep.card_stats(card, None) == block

    def test_card_stats_returns_none_without_events(self):
        rep = BetaBinomialReputation()
        assert rep.card_stats(MemoryCard(id="cold"), None) is None

    def test_card_stats_returns_none_on_empty_events(self):
        rep = BetaBinomialReputation()
        assert rep.card_stats(MemoryCard(id="empty", gain_events=[]), None) is None

    def test_posterior_and_magnitude_view_card_stats(self):
        stamped = MemoryCard(id="stamped", gain_events=CONFIDENT_EVENTS)
        rep = BetaBinomialReputation()
        block = rep.card_stats(stamped)
        assert rep.card_posterior(stamped, None) == (
            block.posterior_a,
            block.posterior_b,
        )
        assert rep.card_magnitude(stamped, None) == block.IntroGain_best_median

        block = CardStatsBlock(
            posterior_a=8.0, posterior_b=2.0, IntroGain_best_median=0.05
        )
        rerouted = _FixedStatsReputation(fixed=block)
        cold = MemoryCard(id="cold")
        assert rerouted.card_posterior(cold, None) == (8.0, 2.0)
        assert rerouted.card_magnitude(cold, None) == 0.05
