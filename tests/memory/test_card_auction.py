"""Thompson auction over candidate cards (Fix B core).

``run_card_auction`` draws each card's posterior (Beta(a, b)) against a no-card
baseline arm (Beta(3, 3)); a card is selected iff its draw beats the baseline
draw. Winners are an emergent 0..N subset — replaces the fixed ``top[:max_cards]``.
"""

from __future__ import annotations

import numpy as np

from gigaevo.memory.shared_memory.card_search import run_card_auction


class TestRunCardAuction:
    def test_proven_wins_suspect_and_cold_lose_with_seed(self) -> None:
        rng = np.random.default_rng(20260604)
        candidates = [
            ("proven", 50.0, 1.0),
            ("suspect", 1.0, 50.0),
            ("cold", 1.0, 1.0),
        ]
        winners, records = run_card_auction(candidates, rng)
        assert winners == ["proven"]
        assert [r["card_id"] for r in records] == ["proven", "suspect", "cold"]
        assert [r["selected"] for r in records] == [True, False, False]

    def test_records_carry_draws_and_baseline_arm(self) -> None:
        rng = np.random.default_rng(20260604)
        winners, records = run_card_auction([("proven", 50.0, 1.0)], rng)
        rec = records[0]
        assert rec["card_id"] == "proven"
        assert rec["a"] == 50.0
        assert rec["b"] == 1.0
        assert rec["baseline_a"] == 3.0
        assert rec["baseline_b"] == 3.0
        assert 0.0 <= rec["theta"] <= 1.0
        assert 0.0 <= rec["baseline_theta"] <= 1.0
        assert rec["selected"] == (rec["theta"] > rec["baseline_theta"])
        assert winners == ["proven"]

    def test_winners_match_selected_records(self) -> None:
        rng = np.random.default_rng(11)
        candidates = [(f"c{i}", float(1 + i), float(50 - i)) for i in range(8)]
        winners, records = run_card_auction(candidates, rng)
        assert winners == [r["card_id"] for r in records if r["selected"]]

    def test_empty_candidates_yields_empty(self) -> None:
        winners, records = run_card_auction([], np.random.default_rng(0))
        assert winners == []
        assert records == []

    def test_baseline_arm_makes_cold_roughly_fifty_fifty(self) -> None:
        rng = np.random.default_rng(7)
        cold = [("cold", 1.0, 1.0)]
        selected = sum(run_card_auction(cold, rng)[0] == ["cold"] for _ in range(2000))
        assert 0.4 < selected / 2000 < 0.6

    def test_custom_baseline_arm_is_respected(self) -> None:
        # A demanding baseline (Beta(50, 1) ~ 0.98) starves all but near-certain cards.
        rng = np.random.default_rng(20260604)
        candidates = [("midling", 5.0, 5.0)]
        winners, records = run_card_auction(candidates, rng, baseline=(50.0, 1.0))
        assert records[0]["baseline_a"] == 50.0
        assert records[0]["baseline_b"] == 1.0
        assert winners == []
