"""Reward-definition strategy (AOS credit assignment) for the EV-bid magnitude.

``RewardDefinition`` turns a card's stamped ``CardStatsBlock`` into the auction's
EV magnitude. The default ``AdjustedMedianReward`` reproduces today's
cohort-adjusted read (``IntroGain_best_adj_median``); ``AbsoluteMedianReward``
reads the absolute child-minus-parent median (``IntroGain_best_median``), so a
card that never beats its parent cannot carry a positive bid — closing the §L1
loophole (``docs/audits/heilbron_decision_pipeline_integrity_2026-06-24.md``)
without touching the auctioneer.
"""

from __future__ import annotations

import numpy as np

from gigaevo.memory.core.auctioneer import AuctionCandidate, EVThompsonAuctioneer
from gigaevo.memory.core.reputation import BetaBinomialReputation
from gigaevo.memory.core.reward import (
    AbsoluteMedianReward,
    AdjustedMedianReward,
    RewardWeightedReputation,
)
from gigaevo.memory.shared_memory.models import (
    CardStatsBlock,
    MemoryCard,
    MemoryCardExplanation,
)


def _block(
    *,
    adj: float | None = None,
    absolute: float | None = None,
    posterior_a: float | None = None,
    posterior_b: float | None = None,
) -> CardStatsBlock:
    fields: dict = {}
    if adj is not None:
        fields["IntroGain_best_adj_median"] = adj
    if absolute is not None:
        fields["IntroGain_best_median"] = absolute
    if posterior_a is not None:
        fields["posterior_a"] = posterior_a
    if posterior_b is not None:
        fields["posterior_b"] = posterior_b
    return CardStatsBlock(**fields)


def _card(
    *,
    adj: float | None = None,
    absolute: float | None = None,
    posterior_a: float | None = None,
    posterior_b: float | None = None,
    has_all: bool = True,
) -> MemoryCard:
    es = (
        {
            "ALL": _block(
                adj=adj,
                absolute=absolute,
                posterior_a=posterior_a,
                posterior_b=posterior_b,
            )
        }
        if has_all
        else {}
    )
    return MemoryCard(
        id="m1",
        description="d",
        keywords=[],
        evolution_statistics=es,
        explanation=MemoryCardExplanation(summary=""),
    )


class TestRewardDefinitions:
    def test_adjusted_reads_cohort_adjusted_median(self) -> None:
        assert AdjustedMedianReward().magnitude(_block(adj=0.0123)) == 0.0123

    def test_absolute_reads_raw_child_minus_parent_median(self) -> None:
        assert AbsoluteMedianReward().magnitude(_block(absolute=-0.002)) == -0.002

    def test_adjusted_none_when_field_absent(self) -> None:
        assert AdjustedMedianReward().magnitude(_block(absolute=0.01)) is None

    def test_absolute_none_when_field_absent(self) -> None:
        assert AbsoluteMedianReward().magnitude(_block(adj=0.01)) is None


class TestRewardWeightedReputation:
    def test_default_reward_preserves_beta_binomial_behaviour(self) -> None:
        card = _card(adj=0.0123, absolute=-0.002)
        assert (
            RewardWeightedReputation().card_magnitude(card)
            == BetaBinomialReputation().card_magnitude(card)
            == 0.0123
        )

    def test_absolute_reward_returns_raw_not_adjusted(self) -> None:
        card = _card(adj=0.0123, absolute=-0.002)
        rep = RewardWeightedReputation(reward=AbsoluteMedianReward())
        assert rep.card_magnitude(card) == -0.002

    def test_no_all_block_is_cold_none(self) -> None:
        rep = RewardWeightedReputation(reward=AbsoluteMedianReward())
        assert rep.card_magnitude(_card(has_all=False)) is None

    def test_inherits_card_posterior_unchanged(self) -> None:
        card = _card(adj=0.0123, absolute=-0.002)
        assert RewardWeightedReputation(reward=AbsoluteMedianReward()).card_posterior(
            card
        ) == BetaBinomialReputation().card_posterior(card)


class TestL1LoopholeClosedAtUnmodifiedAuctioneer:
    """A card cohort-positive (adj +) but absolutely-negative (abs −) is selected
    on under the default reward but abstained on under the absolute reward — using
    the auctioneer's existing ``ev_floor``, with zero edits to the auctioneer."""

    def _candidate(self, rep, card) -> AuctionCandidate:
        a, b = rep.card_posterior(card)
        return AuctionCandidate(
            card_id=card.id,
            posterior_a=a,
            posterior_b=b,
            magnitude=rep.card_magnitude(card),
        )

    def test_absolute_reward_card_never_wins_when_absolutely_negative(self) -> None:
        card = _card(adj=0.0123, absolute=-0.002, posterior_a=31.0, posterior_b=2.0)
        auctioneer = EVThompsonAuctioneer()

        absolute_rep = RewardWeightedReputation(reward=AbsoluteMedianReward())
        cand = self._candidate(absolute_rep, card)
        for seed in range(50):
            winners, _ = auctioneer.run([cand], np.random.default_rng(seed))
            assert winners == [], f"absolute-negative card won at seed {seed}"

    def test_default_reward_card_can_win(self) -> None:
        card = _card(adj=0.0123, absolute=-0.002, posterior_a=31.0, posterior_b=2.0)
        auctioneer = EVThompsonAuctioneer()
        default_rep = RewardWeightedReputation()
        cand = self._candidate(default_rep, card)
        assert any(
            auctioneer.run([cand], np.random.default_rng(seed))[0] == [card.id]
            for seed in range(50)
        )
