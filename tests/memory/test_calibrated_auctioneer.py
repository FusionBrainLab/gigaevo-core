"""Slate-calibrated cold-card prior for the EV Thompson auction.

A cold card (magnitude=None) must bid the cold_quantile (median by default) of
the proven magnitudes ON ITS OWN SLATE, not a fixed prior_magnitude — and the
behaviour must be byte-for-byte identical to EVThompsonAuctioneer when no cold
card is present (delegation preserves the seed-exact draw order).
"""

from __future__ import annotations

import numpy as np

from gigaevo.memory.core.auctioneer import AuctionCandidate, EVThompsonAuctioneer
from gigaevo.memory.core.calibrated_auctioneer import CalibratedColdPriorAuctioneer


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


def test_cold_card_bids_slate_median_not_flat_prior() -> None:
    cands = [
        _cand("p1", 5.0, 5.0, magnitude=0.002),
        _cand("p2", 5.0, 5.0, magnitude=0.004),
        _cand("cold", 5.0, 5.0, magnitude=None),
    ]
    _, slate = CalibratedColdPriorAuctioneer().run(cands, np.random.default_rng(1))
    cold = next(b for b in slate if b.card_id == "cold")
    assert cold.magnitude == 0.003  # median(0.002, 0.004), NOT 0.1


def test_cold_quantile_configurable() -> None:
    cands = [
        _cand("p1", 5.0, 5.0, magnitude=0.001),
        _cand("p2", 5.0, 5.0, magnitude=0.002),
        _cand("p3", 5.0, 5.0, magnitude=0.004),
        _cand("cold", 5.0, 5.0, magnitude=None),
    ]
    auc = CalibratedColdPriorAuctioneer(cold_quantile=0.0)  # 0th pct = min present
    _, slate = auc.run(cands, np.random.default_rng(3))
    cold = next(b for b in slate if b.card_id == "cold")
    assert cold.magnitude == 0.001


def test_all_cold_slate_uses_floor() -> None:
    cands = [_cand("c1", 5.0, 5.0, None), _cand("c2", 5.0, 5.0, None)]
    auc = CalibratedColdPriorAuctioneer(cold_floor=1e-6)
    _, slate = auc.run(cands, np.random.default_rng(2))
    assert all(b.magnitude == 1e-6 for b in slate)


def test_nonpositive_quantile_clamped_to_floor() -> None:
    cands = [_cand("p1", 5.0, 5.0, magnitude=-0.005), _cand("cold", 5.0, 5.0, None)]
    auc = CalibratedColdPriorAuctioneer(cold_floor=1e-6)
    _, slate = auc.run(cands, np.random.default_rng(4))
    cold = next(b for b in slate if b.card_id == "cold")
    assert cold.magnitude == 1e-6  # max(median([-0.005]), 1e-6)


def test_parity_with_ev_auction_when_no_cold_cards() -> None:
    cands = [
        _cand("a", 5.0, 5.0, magnitude=0.002),
        _cand("b", 4.0, 6.0, magnitude=0.004),
    ]
    seed = 20260625
    w1, s1 = CalibratedColdPriorAuctioneer().run(cands, np.random.default_rng(seed))
    w2, s2 = EVThompsonAuctioneer().run(cands, np.random.default_rng(seed))
    assert w1 == w2
    assert [b.model_dump() for b in s1] == [b.model_dump() for b in s2]


def test_thompson_ev_calibrated_yaml_instantiates() -> None:
    from pathlib import Path

    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    cfg_path = (
        Path(__file__).resolve().parents[2]
        / "config"
        / "memory"
        / "reader"
        / "auction"
        / "thompson_ev_calibrated.yaml"
    )
    auc = instantiate(OmegaConf.load(cfg_path))
    assert isinstance(auc, CalibratedColdPriorAuctioneer)
    assert auc.cold_quantile == 0.5
    assert auc.cold_floor == 1e-6
