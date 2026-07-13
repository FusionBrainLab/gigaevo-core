from __future__ import annotations

from datetime import datetime

import pytest

from gigaevo.memory.cards import ContextualGain, DecisionContext
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


def test_projector_pending_count_defaults_to_byte_identical_zero(make_card):
    card = make_card()
    projector = AuctionCandidateProjector()
    kwargs = {
        "card": card,
        "block": None,
        "reputation": BetaBinomialReputation(cold_prior=(2.0, 5.0)),
        "context": None,
    }

    omitted = projector.project(**kwargs)
    explicit_none = projector.project(**kwargs, pending_counts=None)

    assert omitted.pending_count == 0
    assert explicit_none.pending_count == 0
    assert explicit_none.model_dump_json() == omitted.model_dump_json()


def test_projector_maps_pending_count_by_card_id(make_card):
    card = make_card(id="pending-card")

    candidate = AuctionCandidateProjector().project(
        card=card,
        block=None,
        reputation=BetaBinomialReputation(cold_prior=(2.0, 5.0)),
        context=None,
        pending_counts={card.id: 4, "other-card": 9},
    )

    assert candidate.pending_count == 4


def test_projector_use_count_is_native_to_decision_task(make_card, make_event):
    def for_task(event, task_key):
        return event.model_copy(
            update={"context": event.context.model_copy(update={"task_key": task_key})}
        )

    card = make_card(
        gain_events=(
            for_task(make_event(0.1), "task-a"),
            for_task(make_event(10.0), "task-b"),
        )
    )
    candidate = AuctionCandidateProjector().project(
        card=card,
        block=None,
        reputation=BetaBinomialReputation(),
        context=DecisionContext(task_key="task-a"),
    )

    assert candidate.use_count == 1


def test_projector_folds_per_event_staleness_into_candidate_weights(
    make_card, make_event
):
    class PerEventStaleReputation(BetaBinomialReputation):
        def staleness_weights(self, card, context=None):
            del card, context
            return (0.25, 1.0)

    card = make_card(gain_events=(make_event(0.2), make_event(-0.1)))
    rep = PerEventStaleReputation()

    candidate = AuctionCandidateProjector().project(
        card=card,
        block=rep.card_stats(card),
        reputation=rep,
        context=None,
    )

    assert rep.event_weights(card) == (1.0, 1.0)
    assert candidate.delta_weights == (0.25, 1.0)
    assert candidate.staleness_weight == 1.0


def test_projector_exposes_unstaled_support_and_staleness_bite(make_card, make_event):
    class PerEventStaleReputation(BetaBinomialReputation):
        def staleness_weights(self, card, context=None):
            del card, context
            return (0.25, 1.0)

    card = make_card(gain_events=(make_event(0.2), make_event(-0.1)))
    rep = PerEventStaleReputation()

    candidate = AuctionCandidateProjector().project(
        card=card,
        block=rep.card_stats(card),
        reputation=rep,
        context=None,
    )

    # Raw per-event credit before staleness: 1.0 + 1.0.
    assert candidate.support_n_unstaled == pytest.approx(2.0)
    # Staled weights the auction reduces to support_n: 0.25 + 1.0.
    staled = sum(candidate.delta_weights)
    assert staled == pytest.approx(1.25)
    # The S4 aging bite is auditable as staled / unstaled straight off the bid.
    assert staled / candidate.support_n_unstaled == pytest.approx(0.625)


def test_projector_gain_se_is_latest_native_event_se(make_card):
    early = ContextualGain(
        context=DecisionContext(timestamp=datetime(2026, 1, 1)),
        gain=0.2,
        gain_se=0.5,
    )
    late = ContextualGain(
        context=DecisionContext(timestamp=datetime(2026, 6, 1)),
        gain=0.3,
        gain_se=0.9,
    )
    # List order puts the LATER-timestamp event first to prove the audit field
    # tracks temporal recency, not list-order-last.
    card = make_card(gain_events=(late, early))
    rep = BetaBinomialReputation()

    candidate = AuctionCandidateProjector().project(
        card=card,
        block=rep.card_stats(card),
        reputation=rep,
        context=None,
    )

    assert candidate.gain_se == pytest.approx(0.9)


def test_projector_gain_se_none_without_events(make_card):
    rep = BetaBinomialReputation()
    candidate = AuctionCandidateProjector().project(
        card=make_card(),
        block=None,
        reputation=rep,
        context=None,
    )

    assert candidate.gain_se is None
    assert candidate.support_n_unstaled == pytest.approx(0.0)
