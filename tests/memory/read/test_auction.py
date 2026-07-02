"""Thompson auctions: seed-exact draw order, EV gate + floor, budget caps."""

from __future__ import annotations

import numpy as np
import pytest

from gigaevo.memory.events import MemoryAuctionRun, MemoryBudgetCap
from gigaevo.memory.read.auction import (
    AuctionBid,
    AuctionCandidate,
    EVThompsonAuctioneer,
    ThompsonAuctioneer,
    TopBidBudgeter,
    TopThetaBudgeter,
)


def _candidate(
    card_id: str, a: float = 2.0, b: float = 2.0, magnitude: float | None = None
) -> AuctionCandidate:
    return AuctionCandidate(
        card_id=card_id, posterior_a=a, posterior_b=b, magnitude=magnitude
    )


def _bid(card_id: str, *, theta: float = 0.5, bid: float | None = None) -> AuctionBid:
    return AuctionBid(
        card_id=card_id,
        posterior_a=2.0,
        posterior_b=2.0,
        theta=theta,
        baseline_a=3.0,
        baseline_b=3.0,
        baseline_theta=0.5,
        selected=True,
        bid=bid,
    )


class TestThompsonAuctioneer:
    def test_empty_candidates(self, captured_events):
        winners, slate = ThompsonAuctioneer().run([], np.random.default_rng(0))
        assert winners == []
        assert slate == []
        assert captured_events == []

    def test_draw_order_is_seed_exact(self, captured_events):
        auctioneer = ThompsonAuctioneer(baseline_prior=(3.0, 3.0))
        candidates = [_candidate(f"c{i}", a=1.0 + i, b=2.0) for i in range(5)]
        winners, slate = auctioneer.run(candidates, np.random.default_rng(42))

        replay = np.random.default_rng(42)
        for candidate, bid in zip(candidates, slate):
            theta = float(replay.beta(candidate.posterior_a, candidate.posterior_b))
            base = float(replay.beta(3.0, 3.0))
            assert bid.theta == theta
            assert bid.baseline_theta == base
            assert bid.selected is (theta > base)
        assert winners == [bid.card_id for bid in slate if bid.selected]

    def test_strong_posterior_dominates(self):
        auctioneer = ThompsonAuctioneer()
        rng = np.random.default_rng(7)
        candidates = [
            _candidate("hot", a=500.0, b=1.0),
            _candidate("bad", a=1.0, b=500.0),
        ]
        wins = {"hot": 0, "bad": 0}
        for _ in range(50):
            winners, _ = auctioneer.run(candidates, rng)
            for w in winners:
                wins[w] += 1
        assert wins["hot"] >= 45
        assert wins["bad"] == 0

    def test_safety_bids_carry_no_ev_fields(self):
        _, slate = ThompsonAuctioneer().run(
            [_candidate("c0", magnitude=0.4)], np.random.default_rng(0)
        )
        assert slate[0].magnitude is None
        assert slate[0].bid is None

    def test_emits_auction_event(self, captured_events):
        winners, slate = ThompsonAuctioneer().run(
            [_candidate("c0"), _candidate("c1")], np.random.default_rng(3)
        )
        (event,) = captured_events
        assert isinstance(event, MemoryAuctionRun)
        assert event.auction == "thompson"
        assert event.candidate_count == 2
        assert event.winner_ids == tuple(winners)
        assert len(event.bids) == 2


class TestEVThompsonAuctioneer:
    def test_draw_order_bids_first_then_gate_then_baseline(self):
        auctioneer = EVThompsonAuctioneer(
            baseline_prior=(3.0, 3.0), prior_magnitude=0.1, ev_floor=0.0
        )
        candidates = [
            _candidate("c0", a=4.0, b=2.0, magnitude=0.5),
            _candidate("c1", a=2.0, b=4.0, magnitude=None),
        ]
        winners, slate = auctioneer.run(candidates, np.random.default_rng(42))

        replay = np.random.default_rng(42)
        expected_bids = []
        for candidate in candidates:
            theta_bid = float(replay.beta(candidate.posterior_a, candidate.posterior_b))
            mag = candidate.magnitude if candidate.magnitude is not None else 0.1
            expected_bids.append((mag, theta_bid * mag))
        for candidate, (mag, bid_value), bid in zip(candidates, expected_bids, slate):
            theta = float(replay.beta(candidate.posterior_a, candidate.posterior_b))
            base = float(replay.beta(3.0, 3.0))
            assert bid.theta == theta
            assert bid.baseline_theta == base
            assert bid.magnitude == mag
            assert bid.bid == bid_value
            assert bid.selected is (theta > base and bid_value > 0.0)
        assert winners == [bid.card_id for bid in slate if bid.selected]

    def test_negative_magnitude_never_clears_default_floor(self):
        auctioneer = EVThompsonAuctioneer()
        rng = np.random.default_rng(0)
        for _ in range(30):
            winners, _ = auctioneer.run(
                [_candidate("harmful", a=50.0, b=1.0, magnitude=-0.2)], rng
            )
            assert winners == []

    def test_cold_card_bids_prior_magnitude(self):
        auctioneer = EVThompsonAuctioneer(prior_magnitude=0.25)
        _, slate = auctioneer.run(
            [_candidate("cold", magnitude=None)], np.random.default_rng(1)
        )
        assert slate[0].magnitude == 0.25

    def test_ev_floor_rejects_small_bids(self):
        auctioneer = EVThompsonAuctioneer(ev_floor=10.0)
        rng = np.random.default_rng(0)
        winners, slate = auctioneer.run(
            [_candidate("c0", a=50.0, b=1.0, magnitude=0.5)], rng
        )
        assert winners == []
        assert slate[0].selected is False

    def test_emits_ev_auction_event(self, captured_events):
        EVThompsonAuctioneer(prior_magnitude=0.3, ev_floor=0.05).run(
            [_candidate("c0", magnitude=0.4)], np.random.default_rng(3)
        )
        (event,) = captured_events
        assert isinstance(event, MemoryAuctionRun)
        assert event.auction == "thompson_ev"
        assert event.prior_magnitude == 0.3
        assert event.ev_floor == 0.05


class TestTopThetaBudgeter:
    def test_within_budget_preserves_auction_order(self, captured_events):
        slate = [_bid("a", theta=0.1), _bid("b", theta=0.9)]
        assert TopThetaBudgeter().cap(["a", "b"], slate, max_cards=2) == ["a", "b"]
        assert captured_events == []

    def test_over_budget_keeps_top_theta(self, captured_events):
        slate = [_bid("a", theta=0.2), _bid("b", theta=0.9), _bid("c", theta=0.5)]
        kept = TopThetaBudgeter().cap(["a", "b", "c"], slate, max_cards=2)
        assert kept == ["b", "c"]
        (event,) = captured_events
        assert isinstance(event, MemoryBudgetCap)
        assert event.rank_key == "theta"
        assert event.kept_ids == ("b", "c")
        assert event.dropped_ids == ("a",)

    def test_zero_budget_empties(self):
        slate = [_bid("a")]
        assert TopThetaBudgeter().cap(["a"], slate, max_cards=0) == []


class TestTopBidBudgeter:
    def test_over_budget_keeps_top_bid(self, captured_events):
        slate = [
            _bid("a", theta=0.9, bid=0.01),
            _bid("b", theta=0.1, bid=0.30),
            _bid("c", theta=0.5, bid=0.20),
        ]
        kept = TopBidBudgeter().cap(["a", "b", "c"], slate, max_cards=2)
        assert kept == ["b", "c"]
        (event,) = captured_events
        assert event.rank_key == "bid"
        assert event.rank_by_card_id["b"] == pytest.approx(0.30)

    def test_missing_bid_sorts_as_zero(self):
        slate = [_bid("a", bid=None), _bid("b", bid=0.1)]
        assert TopBidBudgeter().cap(["a", "b"], slate, max_cards=1) == ["b"]

    def test_within_budget_preserves_auction_order(self):
        slate = [_bid("a", bid=0.1), _bid("b", bid=0.9)]
        assert TopBidBudgeter().cap(["a", "b"], slate, max_cards=5) == ["a", "b"]
