"""EV-bid Thompson auction + bid-ranked budgeter (the `thompson_ev` arm).

``EVThompsonAuctioneer`` keeps the no-card Thompson gate (a card is selected
iff a fresh posterior draw beats a no-card baseline draw — the abstain arm),
but additionally bids ``theta_bid x magnitude`` so the downstream budgeter can
rank winners by expected gain rather than raw help-probability. Cold cards
(``magnitude is None``) bid against an optimistic ``prior_magnitude`` so the
auction explores them instead of starving them at zero.

``TopBidBudgeter`` caps the winners to ``max_cards`` by realized bid (the EV
bid), not by the gate's theta.

Draw order is pinned for seed-exact replay parity with the offline reference
(``rerank_arm.py``): one bid draw per candidate first, then per candidate the
gate draw followed by the baseline draw.
"""

from __future__ import annotations

from contextlib import contextmanager

from loguru import logger
import numpy as np

from gigaevo.memory.core.auctioneer import (
    AuctionBid,
    AuctionCandidate,
    EVThompsonAuctioneer,
)
from gigaevo.memory.core.budgeter import TopBidBudgeter


@contextmanager
def _captured_event_levels():
    """Record (event_type, level) for every canonical memory event emitted."""
    seen: list[tuple[str, str]] = []

    def sink(message):
        event = message.record["extra"].get("memory_event")
        if event is not None:
            seen.append((event["event_type"], message.record["level"].name))

    sink_id = logger.add(sink, level="DEBUG")
    try:
        yield seen
    finally:
        logger.remove(sink_id)


def _cand(
    card_id: str,
    posterior_a: float,
    posterior_b: float,
    magnitude: float | None = None,
) -> AuctionCandidate:
    return AuctionCandidate(
        card_id=card_id,
        posterior_a=posterior_a,
        posterior_b=posterior_b,
        magnitude=magnitude,
    )


class TestEVThompsonAuction:
    def test_slate_carries_bid_and_magnitude(self) -> None:
        rng = np.random.default_rng(20260604)
        auc = EVThompsonAuctioneer(prior_magnitude=0.1)
        _, slate = auc.run([_cand("c", 5.0, 5.0, magnitude=0.2)], rng)
        bid = slate[0]
        assert bid.magnitude == 0.2
        assert bid.bid is not None
        # bid = theta_bid * magnitude, so 0 <= bid <= magnitude for theta in [0,1]
        assert 0.0 <= bid.bid <= 0.2

    def test_cold_magnitude_uses_prior(self) -> None:
        rng = np.random.default_rng(20260604)
        auc = EVThompsonAuctioneer(prior_magnitude=0.3)
        _, slate = auc.run([_cand("cold", 5.0, 5.0, magnitude=None)], rng)
        assert slate[0].magnitude == 0.3
        assert 0.0 <= slate[0].bid <= 0.3

    def test_higher_magnitude_bids_higher_for_equal_posterior(self) -> None:
        # Equal posterior => same bid-draw distribution; averaged over seeds the
        # higher-magnitude card must out-bid the lower one.
        auc = EVThompsonAuctioneer(prior_magnitude=0.1)
        big = small = 0.0
        for s in range(200):
            rng = np.random.default_rng(s)
            _, slate = auc.run(
                [_cand("big", 5.0, 5.0, 0.5), _cand("small", 5.0, 5.0, 0.05)], rng
            )
            by_id = {b.card_id: b.bid for b in slate}
            big += by_id["big"]
            small += by_id["small"]
        assert big > small

    def test_gate_is_thompson_baseline_cold_is_roughly_fifty_fifty(self) -> None:
        rng = np.random.default_rng(7)
        auc = EVThompsonAuctioneer(prior_magnitude=0.1)
        cold = [_cand("cold", 1.0, 1.0, magnitude=None)]
        selected = sum(auc.run(cold, rng)[0] == ["cold"] for _ in range(2000))
        assert 0.4 < selected / 2000 < 0.6

    def test_proven_beats_suspect_gate(self) -> None:
        rng = np.random.default_rng(20260604)
        auc = EVThompsonAuctioneer(prior_magnitude=0.1)
        winners, _ = auc.run(
            [_cand("proven", 50.0, 1.0, 0.2), _cand("suspect", 1.0, 50.0, 0.9)], rng
        )
        # The gate is help-probability, not magnitude: the suspect's big
        # magnitude does NOT buy it through the gate.
        assert winners == ["proven"]

    def test_empty_candidates_yields_empty(self) -> None:
        winners, slate = EVThompsonAuctioneer(prior_magnitude=0.1).run(
            [], np.random.default_rng(0)
        )
        assert winners == []
        assert slate == []

    def test_negative_magnitude_card_never_wins_even_if_gate_passes(self) -> None:
        # A near-certain-safe posterior clears the Thompson gate every draw, but
        # its expected gain is negative => EV must abstain on it (never inject a
        # card you expect to hurt).
        auc = EVThompsonAuctioneer(prior_magnitude=0.1)
        ever_won = any(
            auc.run(
                [_cand("harmful", 200.0, 1.0, magnitude=-0.05)],
                np.random.default_rng(s),
            )[0]
            for s in range(200)
        )
        assert not ever_won

    def test_all_negative_pool_abstains(self) -> None:
        rng = np.random.default_rng(20260604)
        auc = EVThompsonAuctioneer(prior_magnitude=0.1)
        winners, slate = auc.run(
            [_cand("a", 200.0, 1.0, -0.05), _cand("b", 200.0, 1.0, -0.2)], rng
        )
        assert winners == []
        assert all(not b.selected for b in slate)

    def test_cold_card_clears_ev_floor(self) -> None:
        # Cold magnitude => positive prior bid, so the EV floor never strands a
        # fresh card; it competes purely on the safety gate.
        auc = EVThompsonAuctioneer(prior_magnitude=0.1)
        won = any(
            auc.run(
                [_cand("cold", 200.0, 1.0, magnitude=None)], np.random.default_rng(s)
            )[0]
            for s in range(50)
        )
        assert won


def _bid(card_id: str, bid: float, theta: float) -> AuctionBid:
    return AuctionBid(
        card_id=card_id,
        posterior_a=5.0,
        posterior_b=5.0,
        theta=theta,
        baseline_a=3.0,
        baseline_b=3.0,
        baseline_theta=0.5,
        selected=True,
        magnitude=bid,
        bid=bid,
    )


class TestTopBidBudgeter:
    def test_caps_to_max_cards_by_bid_not_theta(self) -> None:
        # 'lo_ev' has the higher theta but the lower EV bid; budget=1 must keep
        # the higher-bid card.
        winners = ["hi_ev", "lo_ev"]
        slate = [_bid("hi_ev", bid=0.9, theta=0.4), _bid("lo_ev", bid=0.2, theta=0.95)]
        assert TopBidBudgeter().cap(winners, slate, max_cards=1) == ["hi_ev"]

    def test_within_budget_preserves_order(self) -> None:
        winners = ["a", "b"]
        slate = [_bid("a", 0.1, 0.5), _bid("b", 0.9, 0.5)]
        assert TopBidBudgeter().cap(winners, slate, max_cards=2) == ["a", "b"]

    def test_empty_winners(self) -> None:
        assert TopBidBudgeter().cap([], [], max_cards=1) == []


class TestCriticalDecisionsLogAtInfo:
    """Critical memory decisions must reach the log file even at INFO level
    (not only the always-on JSONL), so a run's selection trail is auditable
    without DEBUG logging."""

    def test_ev_auction_run_emits_at_info(self) -> None:
        rng = np.random.default_rng(20260604)
        auc = EVThompsonAuctioneer(prior_magnitude=0.1)
        with _captured_event_levels() as seen:
            auc.run([_cand("c", 5.0, 5.0, magnitude=0.2)], rng)
        assert ("auction.run", "INFO") in seen

    def test_top_bid_budget_cap_emits_at_info(self) -> None:
        winners = ["hi_ev", "lo_ev"]
        slate = [_bid("hi_ev", bid=0.9, theta=0.4), _bid("lo_ev", bid=0.2, theta=0.95)]
        with _captured_event_levels() as seen:
            TopBidBudgeter().cap(winners, slate, max_cards=1)
        assert ("budget.cap", "INFO") in seen
