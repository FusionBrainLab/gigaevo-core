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


def test_projector_can_override_cold_prior(make_card):
    rep = BetaBinomialReputation(cold_prior=(2.0, 5.0))
    candidate = AuctionCandidateProjector(
        prior=_Prior(), no_card_evidence=_NoCard()
    ).project(card=make_card(), block=None, reputation=rep, context=None)

    assert (candidate.posterior_a, candidate.posterior_b) == (1.0, 9.0)
    assert candidate.prior_source == "test_prior"


def test_decision_baseline_resolves_no_card_summary():
    summary = AuctionCandidateProjector(no_card_evidence=_NoCard()).decision_baseline(
        None
    )

    assert (summary.prior.alpha, summary.prior.beta) == (5.0, 7.0)
    assert summary.source == "test_no_card"
    assert summary.baseline == pytest.approx(0.12)
    assert summary.evidence_n == pytest.approx(8.0)


def test_decision_baseline_none_without_evidence():
    assert AuctionCandidateProjector().decision_baseline(None) is None


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


def test_projector_use_count_is_nonfounding_event_count(make_card, make_event):
    card = make_card(
        gain_events=(
            make_event(0.1),
            make_event(-0.05, invalid=True),
            make_event(0.0, unused=True),
            make_event(0.2, founding=True),
        )
    )
    candidate = AuctionCandidateProjector().project(
        card=card,
        block=None,
        reputation=BetaBinomialReputation(cold_prior=(2.0, 5.0)),
        context=None,
    )
    # Every injection-produced event counts (helpful, invalid, unused alike);
    # the founding birth credit is not an injection.
    assert candidate.use_count == 3


def test_projector_use_count_zero_without_events(make_card):
    candidate = AuctionCandidateProjector().project(
        card=make_card(),
        block=None,
        reputation=BetaBinomialReputation(cold_prior=(2.0, 5.0)),
        context=None,
    )
    assert candidate.use_count == 0
