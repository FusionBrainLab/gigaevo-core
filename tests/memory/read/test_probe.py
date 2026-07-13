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


def test_sub_threshold_support_card_is_probe_eligible_regardless_of_kind():
    # R1 re-entry: a card whose only evidence is diluted unused/invalid exposure
    # never reaches cold_prior kind again, so gating the probe lane on kind made
    # one ignored exposure an absorbing death.
    slate = [_bid("zombie", support_kind="zero_support", support_n=0.5, bid=0.0)]
    budgeted, out = ColdProbePolicy(empty_selection_probe_rate=1.0).apply(
        budgeted_ids=[],
        slate=slate,
        max_cards=1,
        rng=np.random.default_rng(0),
    )

    assert budgeted == ["zombie"]
    zombie = next(bid for bid in out if bid.card_id == "zombie")
    assert zombie.probe_selected is True
    assert zombie.probe_eligible is True


def test_exposure_diluted_ev_rewards_card_can_probe():
    slate = [_bid("diluted", support_kind="ev_rewards", support_n=0.66, bid=0.0)]
    budgeted, _ = ColdProbePolicy(empty_selection_probe_rate=1.0).apply(
        budgeted_ids=[],
        slate=slate,
        max_cards=1,
        rng=np.random.default_rng(0),
    )

    assert budgeted == ["diluted"]


def test_solo_ignored_card_keeps_probe_path_at_default_threshold():
    # One ignored solo injection carries full weight 1.0 under max_cards=1: it
    # bids 0.0 (fails the sign gate) and sits below the eviction evidence
    # floor, so the probe lane is its only route anywhere. The default
    # threshold must therefore sit at that same eviction floor — a 1.0
    # threshold made a single solo ignore an absorbing death.
    slate = [_bid("ignored", support_kind="ev_rewards", support_n=1.0, bid=0.0)]
    budgeted, _ = ColdProbePolicy(empty_selection_probe_rate=1.0).apply(
        budgeted_ids=[],
        slate=slate,
        max_cards=1,
        rng=np.random.default_rng(0),
    )

    assert budgeted == ["ignored"]


def test_card_at_eviction_floor_support_never_probes():
    # At support_n == the eviction evidence floor the card is adjudicable:
    # either it wins auctions on merit or PolicyNonViable can evict it, so the
    # probe lane hands over exactly at the threshold.
    slate = [_bid("adjudicable", support_kind="ev_rewards", support_n=3.0, bid=0.0)]
    budgeted, _ = ColdProbePolicy(empty_selection_probe_rate=1.0).apply(
        budgeted_ids=[],
        slate=slate,
        max_cards=1,
        rng=np.random.default_rng(0),
    )

    assert budgeted == []


def test_unreported_support_is_never_probe_eligible():
    # Auctioneers that do not report support leave support_kind empty; the
    # probe lane must fail safe (no probes) rather than treat every warm loser
    # as cold via the support_n=0.0 field default.
    slate = [_bid("unknown", support_kind="", support_n=0.0, bid=0.2)]
    budgeted, out = ColdProbePolicy(empty_selection_probe_rate=1.0).apply(
        budgeted_ids=[],
        slate=slate,
        max_cards=1,
        rng=np.random.default_rng(0),
    )

    assert budgeted == []
    assert all(bid.probe_eligible is False for bid in out)


def test_probe_eligibility_is_recomputed_for_every_row_without_decision_change():
    slate = [
        _bid(
            "selected-under-floor",
            selected=True,
            support_kind="ev_rewards",
            support_n=1.0,
        ),
        _bid(
            "unselected-at-floor",
            support_kind="ev_rewards",
            support_n=3.0,
        ).model_copy(update={"probe_eligible": True}),
    ]
    rng = np.random.default_rng(17)
    expected_next_draw = float(np.random.default_rng(17).random())

    budgeted, out = ColdProbePolicy(warm_override_probe_rate=1.0).apply(
        budgeted_ids=["selected-under-floor"],
        slate=slate,
        max_cards=1,
        rng=rng,
    )

    assert budgeted == ["selected-under-floor"]
    assert [bid.selected for bid in out] == [True, False]
    assert [bid.probe_eligible for bid in out] == [True, False]
    assert float(rng.random()) == expected_next_draw


def test_probe_threshold_config_extends_re_entry_to_low_support_losers():
    slate = [_bid("loser", support_kind="ev_rewards", support_n=4.0, bid=-0.1)]
    budgeted, _ = ColdProbePolicy(
        empty_selection_probe_rate=1.0, probe_until_effective_events=5.0
    ).apply(
        budgeted_ids=[],
        slate=slate,
        max_cards=1,
        rng=np.random.default_rng(0),
    )

    assert budgeted == ["loser"]


def test_true_cold_card_outranks_zero_support_zombie_in_probe_lane():
    slate = [
        _bid("zombie", support_kind="zero_support", support_n=0.5, bid=0.0),
        _bid("cold", support_kind="cold_prior", support_n=0.0, bid=0.2),
    ]
    budgeted, _ = ColdProbePolicy(empty_selection_probe_rate=1.0).apply(
        budgeted_ids=[],
        slate=slate,
        max_cards=1,
        rng=np.random.default_rng(0),
    )

    assert budgeted == ["cold"]


def test_warm_override_adds_probe_when_budget_has_room():
    # Overriding exists to buy exploration when the budget is saturated; with a
    # free slot the probe must join the proven winner, not displace it.
    slate = [
        _bid("warm", selected=True, support_kind="ev_rewards", support_n=3.0),
        _bid("cold", selected=False, support_kind="cold_prior", support_n=0.0),
    ]
    budgeted, out = ColdProbePolicy(warm_override_probe_rate=1.0).apply(
        budgeted_ids=["warm"],
        slate=slate,
        max_cards=2,
        rng=np.random.default_rng(0),
    )

    assert budgeted == ["warm", "cold"]
    warm = next(bid for bid in out if bid.card_id == "warm")
    cold = next(bid for bid in out if bid.card_id == "cold")
    assert warm.selected is True
    assert cold.probe_selected is True


def test_warm_override_displaces_weakest_budgeted_card():
    budgeted_ids = ["lower", "higher"]
    slate = [
        _bid(
            "lower",
            selected=True,
            support_kind="ev_rewards",
            support_n=3.0,
            bid=0.1,
            theta=0.2,
        ),
        _bid(
            "higher",
            selected=True,
            support_kind="ev_rewards",
            support_n=3.0,
            bid=0.9,
            theta=0.8,
        ),
        _bid("cold", support_kind="cold_prior", support_n=0.0),
    ]

    budgeted, out = ColdProbePolicy(
        enabled=True,
        warm_override_probe_rate=1.0,
        max_probe_cards_per_decision=1,
    ).apply(
        budgeted_ids=budgeted_ids,
        slate=slate,
        max_cards=2,
        rng=type("ZeroRng", (), {"random": lambda self: 0.0})(),
    )

    # The former tail-drop kept the weaker bid.
    assert budgeted_ids[:-1] == ["lower"]
    assert budgeted == ["higher", "cold"]
    assert next(bid for bid in out if bid.card_id == "cold").probe_selected is True


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
