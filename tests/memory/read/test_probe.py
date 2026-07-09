from __future__ import annotations

import numpy as np

from gigaevo.memory.read.auction import AuctionBid
from gigaevo.memory.read.probe import ColdProbePolicy, NoColdProbePolicy


def _bid(
    card_id: str,
    *,
    selected: bool = False,
    support_kind: str = "cold_prior",
    support_n: float = 0.0,
    bid: float = 0.1,
    theta: float = 0.5,
) -> AuctionBid:
    return AuctionBid(
        card_id=card_id,
        posterior_a=3.0,
        posterior_b=3.0,
        theta=theta,
        baseline_a=3.0,
        baseline_b=3.0,
        baseline_theta=0.5,
        selected=selected,
        support_kind=support_kind,
        support_n=support_n,
        bid=bid,
    )


def test_no_probe_policy_is_inert():
    slate = [_bid("cold")]
    budgeted, out = NoColdProbePolicy().apply(
        budgeted_ids=[],
        slate=slate,
        max_cards=1,
        rng=np.random.default_rng(0),
    )

    assert budgeted == []
    assert out == slate


def test_empty_selection_can_probe_best_cold_bid():
    slate = [_bid("low", bid=0.1), _bid("high", bid=0.3)]
    budgeted, out = ColdProbePolicy(empty_selection_probe_rate=1.0).apply(
        budgeted_ids=[],
        slate=slate,
        max_cards=1,
        rng=np.random.default_rng(0),
    )

    assert budgeted == ["high"]
    high = next(bid for bid in out if bid.card_id == "high")
    assert high.selected is True
    assert high.probe_selected is True
    assert high.selection_reason == "cold_probe_empty"


def test_warm_winner_not_replaced_when_override_rate_zero():
    slate = [
        _bid("warm", selected=True, support_kind="ev_rewards", support_n=3.0),
        _bid("cold", selected=False, support_kind="cold_prior", support_n=0.0),
    ]
    budgeted, out = ColdProbePolicy(warm_override_probe_rate=0.0).apply(
        budgeted_ids=["warm"],
        slate=slate,
        max_cards=1,
        rng=np.random.default_rng(0),
    )

    assert budgeted == ["warm"]
    cold = next(bid for bid in out if bid.card_id == "cold")
    assert cold.probe_eligible is True
    assert cold.probe_selected is False


def test_warm_winner_can_be_replaced_by_explicit_override():
    slate = [
        _bid("warm", selected=True, support_kind="ev_rewards", support_n=3.0),
        _bid("cold", selected=False, support_kind="cold_prior", support_n=0.0),
    ]
    budgeted, out = ColdProbePolicy(warm_override_probe_rate=1.0).apply(
        budgeted_ids=["warm"],
        slate=slate,
        max_cards=1,
        rng=np.random.default_rng(0),
    )

    assert budgeted == ["cold"]
    warm = next(bid for bid in out if bid.card_id == "warm")
    cold = next(bid for bid in out if bid.card_id == "cold")
    assert warm.selected is False
    assert cold.probe_selected is True
    assert cold.selected is True
    assert cold.selection_reason == "cold_probe_override"
