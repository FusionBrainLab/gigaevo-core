from __future__ import annotations

import pytest

from gigaevo.memory.context.no_card import NoCardGateSummary
from gigaevo.memory.read.prior import BetaPrior
from gigaevo.memory.read.projection import AuctionCandidateProjector
from gigaevo.memory.read.reputation import BetaBinomialReputation


class _Prior:
    def cold_card_prior(self, card, context=None):
        return BetaPrior(alpha=1.0, beta=9.0, source="test_prior", support_n=4.0)


class _NoCard:
    def summary_for(self, context=None):
        return NoCardGateSummary(
            prior=BetaPrior(alpha=5.0, beta=7.0, source="test_no_card", support_n=8.0),
            baseline=0.12,
            evidence_n=8.0,
            source="test_no_card",
        )


def test_default_projector_preserves_reputation_cold_prior(make_card):
    rep = BetaBinomialReputation(cold_prior=(2.0, 5.0))
    candidate = AuctionCandidateProjector().project(
        card=make_card(),
        block=None,
        reputation=rep,
        context=None,
    )

    assert (candidate.posterior_a, candidate.posterior_b) == (2.0, 5.0)
    assert candidate.prior_source == "reputation"


def test_projector_can_override_cold_prior_and_no_card_gate(make_card):
    rep = BetaBinomialReputation(cold_prior=(2.0, 5.0))
    candidate = AuctionCandidateProjector(
        prior=_Prior(), no_card_evidence=_NoCard()
    ).project(card=make_card(), block=None, reputation=rep, context=None)

    assert (candidate.posterior_a, candidate.posterior_b) == (1.0, 9.0)
    assert candidate.prior_source == "test_prior"
    assert (candidate.baseline_a, candidate.baseline_b) == (5.0, 7.0)
    assert candidate.baseline_source == "test_no_card"
    assert candidate.no_card_baseline == pytest.approx(0.12)
    assert candidate.no_card_n == pytest.approx(8.0)


def test_projector_keeps_warm_observed_posterior(make_card, make_event):
    rep = BetaBinomialReputation()
    card = make_card(gain_events=(make_event(0.2), make_event(-0.1)))
    block = rep.card_stats(card)
    candidate = AuctionCandidateProjector(prior=_Prior()).project(
        card=card,
        block=block,
        reputation=rep,
        context=None,
    )

    assert (candidate.posterior_a, candidate.posterior_b) == (2.0, 2.0)
    assert candidate.prior_source == "reputation"
