"""Unit + contract tests for the modular efficacy core.

BetaBinomialReputation must agree with the standalone posterior function
(beta_binomial_posterior) under default config, ThompsonAuctioneer's draw order
(theta then baseline, per candidate) is pinned seed-exact, and every threshold
is exposed as a constructor parameter. The harm predicate has its semantics
pinned directly here over ``CardStatsBlock`` rows.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import yaml

from gigaevo.memory.core.auctioneer import AuctionCandidate, ThompsonAuctioneer
from gigaevo.memory.core.reputation import BetaBinomialReputation
from gigaevo.memory.efficacy import beta_binomial_posterior
from gigaevo.memory.shared_memory.models import CardStatsBlock


class TestBetaBinomialReputation:
    def test_posterior_matches_legacy(self):
        rep = BetaBinomialReputation()
        for gains, threshold in [
            ([0.1, -0.2, 0.05], 0.0),
            ([0.1, -0.2, 0.05, float("nan")], -0.01),
            ([], 0.0),
            ([-1.0, -2.0], 0.0),
        ]:
            got = rep.posterior(gains, threshold=threshold).model_dump()
            want = beta_binomial_posterior(gains, threshold=threshold).model_dump()
            for key, val in want.items():
                if isinstance(val, float) and math.isnan(val):
                    assert math.isnan(got[key])
                else:
                    assert got[key] == val, key

    def test_harm_predicate_pinned_semantics(self):
        rep = BetaBinomialReputation()
        cases = [
            (CardStatsBlock(intro_events=3, posterior_a=1.0, posterior_b=4.0), True),
            (CardStatsBlock(intro_events=3, posterior_a=4.0, posterior_b=1.0), False),
            (CardStatsBlock(intro_events=2, posterior_a=1.0, posterior_b=3.0), False),
            (CardStatsBlock(intro_events=5, posterior_a=2.0, posterior_b=5.0), True),
            (CardStatsBlock(), False),
            (None, False),
        ]
        for block, expected in cases:
            assert rep.is_confidently_harmful(block) is expected, block

    def test_harm_min_events_configurable(self):
        harmful = CardStatsBlock(intro_events=2, posterior_a=1.0, posterior_b=4.0)
        assert not BetaBinomialReputation().is_confidently_harmful(harmful)
        assert BetaBinomialReputation(harm_min_events=2).is_confidently_harmful(harmful)

    def test_harm_predicate_guards_non_positive_posterior_params(self):
        # Without the positivity guard beta.ppf(q, 0, 0) yields nan and the
        # predicate silently reads "never harmful" through a nan comparison.
        rep = BetaBinomialReputation()
        for a, b in [(0.0, 0.0), (0.0, 4.0), (-1.0, 4.0), (4.0, -1.0)]:
            block = CardStatsBlock(intro_events=5, posterior_a=a, posterior_b=b)
            assert rep.is_confidently_harmful(block) is False, (a, b)

    def test_cold_prior_configurable(self):
        assert BetaBinomialReputation().cold_prior == (1.0, 1.0)
        assert BetaBinomialReputation(cold_prior=(2.0, 3.0)).cold_prior == (2.0, 3.0)


class TestThompsonAuctioneer:
    def test_draw_order_is_seed_exact(self):
        candidates = [
            AuctionCandidate(card_id="a", posterior_a=4.0, posterior_b=1.0),
            AuctionCandidate(card_id="b", posterior_a=1.0, posterior_b=1.0),
            AuctionCandidate(card_id="c", posterior_a=2.0, posterior_b=5.0),
        ] * 50
        winners, slate = ThompsonAuctioneer().run(candidates, np.random.default_rng(7))
        replay = np.random.default_rng(7)
        for candidate, bid in zip(candidates, slate, strict=True):
            assert bid.theta == float(
                replay.beta(candidate.posterior_a, candidate.posterior_b)
            )
            assert bid.baseline_theta == float(replay.beta(3.0, 3.0))
            assert bid.selected == (bid.theta > bid.baseline_theta)
        assert winners == [bid.card_id for bid in slate if bid.selected]

    def test_baseline_prior_configurable(self):
        auct = ThompsonAuctioneer(baseline_prior=(1.0, 1000.0))
        winners, slate = auct.run(
            [AuctionCandidate(card_id="a", posterior_a=2.0, posterior_b=2.0)],
            np.random.default_rng(0),
        )
        assert winners == ["a"]
        assert slate[0].baseline_a == 1.0
        assert slate[0].baseline_b == 1000.0

    def test_empty_candidates(self):
        winners, records = ThompsonAuctioneer().run([], np.random.default_rng(0))
        assert winners == []
        assert records == []


def test_beta_binomial_yaml_surfaces_every_reputation_knob() -> None:
    """Every tunable BetaBinomialReputation field must be surfaced in the
    shipped config so an experimenter can tune it without reading source."""
    repo_root = Path(__file__).resolve().parents[2]
    cfg = yaml.safe_load(
        (repo_root / "config/memory/reputation/beta_binomial.yaml").read_text()
    )
    yaml_keys = {key for key in cfg if not key.startswith("_")}
    assert set(BetaBinomialReputation.model_fields) <= yaml_keys
