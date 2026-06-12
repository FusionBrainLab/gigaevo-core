"""Thompson auction over candidate cards.

``ThompsonAuctioneer`` draws each card's posterior (Beta(a, b)) against a
no-card baseline arm (Beta(3, 3)); a card is selected iff its draw beats the
baseline draw. Winners are an emergent 0..N subset.
"""

from __future__ import annotations

import numpy as np

from gigaevo.memory.core.auctioneer import (
    AuctionCandidate,
    ThompsonAuctioneer,
)


def _candidate(
    card_id: str, posterior_a: float, posterior_b: float
) -> AuctionCandidate:
    return AuctionCandidate(
        card_id=card_id, posterior_a=posterior_a, posterior_b=posterior_b
    )


class TestThompsonAuction:
    def test_proven_wins_suspect_and_cold_lose_with_seed(self) -> None:
        rng = np.random.default_rng(20260604)
        candidates = [
            _candidate("proven", 50.0, 1.0),
            _candidate("suspect", 1.0, 50.0),
            _candidate("cold", 1.0, 1.0),
        ]
        winners, slate = ThompsonAuctioneer().run(candidates, rng)
        assert winners == ["proven"]
        assert [bid.card_id for bid in slate] == ["proven", "suspect", "cold"]
        assert [bid.selected for bid in slate] == [True, False, False]

    def test_bids_carry_draws_and_baseline_arm(self) -> None:
        rng = np.random.default_rng(20260604)
        winners, slate = ThompsonAuctioneer().run(
            [_candidate("proven", 50.0, 1.0)], rng
        )
        bid = slate[0]
        assert bid.card_id == "proven"
        assert bid.posterior_a == 50.0
        assert bid.posterior_b == 1.0
        assert bid.baseline_a == 3.0
        assert bid.baseline_b == 3.0
        assert 0.0 <= bid.theta <= 1.0
        assert 0.0 <= bid.baseline_theta <= 1.0
        assert bid.selected == (bid.theta > bid.baseline_theta)
        assert winners == ["proven"]

    def test_winners_match_selected_bids(self) -> None:
        rng = np.random.default_rng(11)
        candidates = [
            _candidate(f"c{i}", float(1 + i), float(50 - i)) for i in range(8)
        ]
        winners, slate = ThompsonAuctioneer().run(candidates, rng)
        assert winners == [bid.card_id for bid in slate if bid.selected]

    def test_empty_candidates_yields_empty(self) -> None:
        winners, slate = ThompsonAuctioneer().run([], np.random.default_rng(0))
        assert winners == []
        assert slate == []

    def test_baseline_arm_makes_cold_roughly_fifty_fifty(self) -> None:
        rng = np.random.default_rng(7)
        auctioneer = ThompsonAuctioneer()
        cold = [_candidate("cold", 1.0, 1.0)]
        selected = sum(auctioneer.run(cold, rng)[0] == ["cold"] for _ in range(2000))
        assert 0.4 < selected / 2000 < 0.6

    def test_custom_baseline_arm_is_respected(self) -> None:
        # A demanding baseline (Beta(50, 1) ~ 0.98) starves all but near-certain cards.
        rng = np.random.default_rng(20260604)
        auctioneer = ThompsonAuctioneer(baseline_prior=(50.0, 1.0))
        winners, slate = auctioneer.run([_candidate("midling", 5.0, 5.0)], rng)
        assert slate[0].baseline_a == 50.0
        assert slate[0].baseline_b == 1.0
        assert winners == []
