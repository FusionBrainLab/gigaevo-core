"""Equivalence + unit tests for the modular efficacy core.

BetaBinomialReputation / ThompsonAuctioneer must reproduce the legacy functions
(beta_binomial_posterior / compute_injection_posterior / run_card_auction)
exactly under default config, and expose every threshold as a constructor
parameter. The admitters and the harm predicate have no legacy twin anymore —
their semantics are pinned directly here over ``IdeaStats`` rows.
"""

from __future__ import annotations

import math

import numpy as np

from gigaevo.memory.core.admitter import (
    PermissiveAdmitter,
    SignBasedAdmitter,
    TieredAdmitter,
)
from gigaevo.memory.core.auctioneer import ThompsonAuctioneer
from gigaevo.memory.core.idea_stats import IdeaStats
from gigaevo.memory.core.reputation import BetaBinomialReputation
from gigaevo.memory.shared_memory.card_search import run_card_auction
from gigaevo.memory.shared_memory.injection_posterior import (
    beta_binomial_posterior,
    compute_injection_posterior,
)


def make_stats_block(
    *,
    intro_events: int = 3,
    posterior_a: float = 4.0,
    posterior_b: float = 1.0,
    median: float = 0.05,
    rel_median: float = 0.05,
    downside: float = 0.0,
    sibling_win_allgens: float = 1.0,
    p10: float = 0.01,
    born_in_elite: float = 1.0,
    p_help_lo20: float = 0.6,
    efficacy_confident: bool = True,
) -> dict:
    return {
        "intro_events": intro_events,
        "posterior_a": posterior_a,
        "posterior_b": posterior_b,
        "IntroGain_best_median": median,
        "IntroGain_best_rel_median": rel_median,
        "DownsideRate_best": downside,
        "SiblingWinRate_allgens": sibling_win_allgens,
        "IntroGain_best_p10": p10,
        "BornInElite_rate": born_in_elite,
        "p_help_lo20": p_help_lo20,
        "efficacy_confident": efficacy_confident,
    }


def make_idea_stats(rows: list[dict]) -> list[IdeaStats]:
    flat = []
    for row in rows:
        rec = dict(row["block"])
        rec["idea_id"] = row["idea_id"]
        rec["quartile"] = row.get("quartile", "ALL")
        rec["description"] = row.get("description", "")
        flat.append(IdeaStats.model_validate(rec))
    return flat


class TestBetaBinomialReputation:
    def test_posterior_matches_legacy(self):
        rep = BetaBinomialReputation()
        for gains, threshold in [
            ([0.1, -0.2, 0.05], 0.0),
            ([0.1, -0.2, 0.05, float("nan")], -0.01),
            ([], 0.0),
            ([-1.0, -2.0], 0.0),
        ]:
            got = rep.posterior(gains, threshold=threshold)
            want = beta_binomial_posterior(gains, threshold=threshold)
            for key, val in want.items():
                if isinstance(val, float) and math.isnan(val):
                    assert math.isnan(got[key])
                else:
                    assert got[key] == val, key

    def test_harm_predicate_pinned_semantics(self):
        rep = BetaBinomialReputation()
        cases = [
            (
                {
                    "ALL": make_stats_block(
                        intro_events=3, posterior_a=1.0, posterior_b=4.0
                    )
                },
                True,
            ),
            (
                {
                    "ALL": make_stats_block(
                        intro_events=3, posterior_a=4.0, posterior_b=1.0
                    )
                },
                False,
            ),
            (
                {
                    "ALL": make_stats_block(
                        intro_events=2, posterior_a=1.0, posterior_b=3.0
                    )
                },
                False,
            ),
            (
                {
                    "ALL": make_stats_block(
                        intro_events=5, posterior_a=2.0, posterior_b=5.0
                    )
                },
                True,
            ),
            ({"Q4": make_stats_block()}, False),
            ({}, False),
            (None, False),
            (
                {"ALL": {"intro_events": 3, "posterior_a": "bad", "posterior_b": 1.0}},
                False,
            ),
        ]
        for stats, expected in cases:
            assert rep.is_confidently_harmful(stats) is expected, stats

    def test_harm_min_events_configurable(self):
        harmful = {
            "ALL": make_stats_block(intro_events=2, posterior_a=1.0, posterior_b=4.0)
        }
        assert not BetaBinomialReputation().is_confidently_harmful(harmful)
        assert BetaBinomialReputation(harm_min_events=2).is_confidently_harmful(harmful)

    def test_injection_posteriors_match_legacy(self):
        programs = [
            {"id": "p1", "fitness": 0.5, "parents": [], "selected_ids": []},
            {"id": "p2", "fitness": 0.6, "parents": ["p1"], "selected_ids": ["c1"]},
            {"id": "p3", "fitness": 0.4, "parents": ["p1"], "selected_ids": ["c2"]},
            {
                "id": "p4",
                "fitness": 0.7,
                "parents": ["p2"],
                "selected_ids": ["c1", "c3"],
            },
            {"id": "p5", "fitness": 0.3, "parents": ["p3"], "selected_ids": ["c2"]},
        ]
        got = BetaBinomialReputation().compute_injection_posteriors(programs)
        want = compute_injection_posterior(programs)
        assert got == want

    def test_injection_posteriors_lower_is_better(self):
        programs = [
            {"id": "p1", "fitness": 0.5, "parents": [], "selected_ids": []},
            {"id": "p2", "fitness": 0.4, "parents": ["p1"], "selected_ids": ["c1"]},
        ]
        got = BetaBinomialReputation().compute_injection_posteriors(
            programs, higher_is_better=False
        )
        want = compute_injection_posterior(programs, higher_is_better=False)
        assert got == want

    def test_noise_band_k_widens_dead_band(self):
        # cardX rides on parent pX; its child cx regresses within the noise band.
        programs = [
            {"id": "p0", "fitness": 0.5, "parents": [], "selected_ids": []},
            {"id": "pX", "fitness": 0.5, "parents": [], "selected_ids": ["cardX"]},
            {"id": "c1", "fitness": 0.5, "parents": ["p0"], "selected_ids": []},
            {"id": "c2", "fitness": 0.6, "parents": ["p0"], "selected_ids": []},
            {"id": "c3", "fitness": 0.4, "parents": ["p0"], "selected_ids": []},
            {"id": "c4", "fitness": 0.7, "parents": ["p0"], "selected_ids": []},
            {"id": "c5", "fitness": 0.3, "parents": ["p0"], "selected_ids": []},
            {"id": "cx", "fitness": 0.4, "parents": ["pX"], "selected_ids": []},
        ]
        default = BetaBinomialReputation().compute_injection_posteriors(programs)
        assert default["cardX"]["k_harm"] == 0
        assert default == compute_injection_posterior(programs)
        no_band = BetaBinomialReputation(noise_band_k=0.0).compute_injection_posteriors(
            programs
        )
        assert no_band["cardX"]["k_harm"] == 1

    def test_cold_prior_configurable(self):
        assert BetaBinomialReputation().cold_prior == (1.0, 1.0)
        assert BetaBinomialReputation(cold_prior=(2.0, 3.0)).cold_prior == (2.0, 3.0)


class TestThompsonAuctioneer:
    def test_seed_exact_match_with_legacy(self):
        candidates = [("a", 4.0, 1.0), ("b", 1.0, 1.0), ("c", 2.0, 5.0)] * 50
        new_winners, new_records = ThompsonAuctioneer().run(
            candidates, np.random.default_rng(7)
        )
        old_winners, old_records = run_card_auction(
            candidates, np.random.default_rng(7)
        )
        assert new_winners == old_winners
        assert new_records == old_records

    def test_baseline_prior_configurable(self):
        auct = ThompsonAuctioneer(baseline_prior=(1.0, 1000.0))
        winners, records = auct.run([("a", 2.0, 2.0)], np.random.default_rng(0))
        assert winners == ["a"]
        assert records[0]["baseline_a"] == 1.0
        assert records[0]["baseline_b"] == 1000.0

    def test_empty_candidates(self):
        winners, records = ThompsonAuctioneer().run([], np.random.default_rng(0))
        assert winners == []
        assert records == []


class TestTieredAdmitter:
    def test_tier_semantics(self):
        stats = make_idea_stats(
            [
                {"idea_id": "strong", "block": make_stats_block()},
                {
                    "idea_id": "strong",
                    "quartile": "Q4",
                    "block": make_stats_block(median=0.2),
                },
                {
                    "idea_id": "single-no-elite",
                    "block": make_stats_block(
                        intro_events=1, born_in_elite=0.0, median=0.147
                    ),
                },
                {
                    "idea_id": "neg-median",
                    "block": make_stats_block(median=-0.05, rel_median=-0.05),
                },
                {
                    "idea_id": "two-events-weak-sib",
                    "block": make_stats_block(
                        intro_events=2, sibling_win_allgens=0.5, p10=0.01
                    ),
                },
                {
                    "idea_id": "high-downside",
                    "block": make_stats_block(downside=0.6),
                },
            ]
        )
        got = TieredAdmitter().select(stats)
        assert [s.idea_id for s in got] == ["strong"]
        assert got[0].quartile == "ALL"

    def test_single_event_born_in_elite_kept(self):
        stats = make_idea_stats(
            [
                {
                    "idea_id": "newborn-elite",
                    "block": make_stats_block(intro_events=1, born_in_elite=1.0),
                }
            ]
        )
        assert [s.idea_id for s in TieredAdmitter().select(stats)] == ["newborn-elite"]

    def test_missing_metrics_never_admit(self):
        block = make_stats_block()
        block["IntroGain_best_rel_median"] = float("nan")
        stats = make_idea_stats([{"idea_id": "nan-rel", "block": block}])
        assert TieredAdmitter().select(stats) == []

        block = make_stats_block()
        block["SiblingWinRate_allgens"] = None
        stats = make_idea_stats([{"idea_id": "no-sib", "block": block}])
        assert TieredAdmitter().select(stats) == []

    def test_thresholds_configurable(self):
        stats = make_idea_stats(
            [{"idea_id": "mild", "block": make_stats_block(rel_median=0.005)}]
        )
        assert TieredAdmitter().select(stats) == []
        got = TieredAdmitter(min_rel_median=0.001).select(stats)
        assert [s.idea_id for s in got] == ["mild"]


class TestSignBasedAdmitter:
    def test_variant_c_semantics(self):
        stats = make_idea_stats(
            [
                {
                    "idea_id": "single-event-positive",
                    "block": make_stats_block(
                        intro_events=1,
                        posterior_a=2.0,
                        posterior_b=1.0,
                        median=0.147,
                        born_in_elite=0.0,
                    ),
                },
                {
                    "idea_id": "neg-median",
                    "block": make_stats_block(median=-0.01),
                },
                {
                    "idea_id": "confidently-harmful",
                    "block": make_stats_block(
                        intro_events=4, posterior_a=1.0, posterior_b=5.0, median=0.02
                    ),
                },
                {
                    "idea_id": "zero-events",
                    "block": make_stats_block(intro_events=0),
                },
                {
                    "idea_id": "zero-median",
                    "block": make_stats_block(median=0.0),
                },
            ]
        )
        got = SignBasedAdmitter().select(stats)
        assert [s.idea_id for s in got] == ["single-event-positive"]

    def test_quartile_rows_ignored_all_block_decides(self):
        stats = make_idea_stats(
            [
                {
                    "idea_id": "q4-only-strong",
                    "quartile": "Q4",
                    "block": make_stats_block(median=0.2),
                },
                {
                    "idea_id": "all-weak-q4-strong",
                    "block": make_stats_block(median=-0.01),
                },
                {
                    "idea_id": "all-weak-q4-strong",
                    "quartile": "Q4",
                    "block": make_stats_block(median=0.2),
                },
            ]
        )
        assert SignBasedAdmitter().select(stats) == []

    def test_one_row_per_idea_from_all_block(self):
        stats = make_idea_stats(
            [
                {"idea_id": "x", "block": make_stats_block(median=0.05)},
                {
                    "idea_id": "x",
                    "quartile": "Q4",
                    "block": make_stats_block(median=0.2),
                },
            ]
        )
        got = SignBasedAdmitter().select(stats)
        assert len(got) == 1
        assert got[0].quartile == "ALL"

    def test_nan_median_never_admits(self):
        block = make_stats_block()
        block["IntroGain_best_median"] = float("nan")
        stats = make_idea_stats([{"idea_id": "nan-median", "block": block}])
        assert SignBasedAdmitter().select(stats) == []

    def test_min_median_configurable(self):
        stats = make_idea_stats(
            [{"idea_id": "weak", "block": make_stats_block(median=0.005)}]
        )
        assert [s.idea_id for s in SignBasedAdmitter().select(stats)] == ["weak"]
        assert SignBasedAdmitter(min_median=0.01).select(stats) == []


class TestPermissiveAdmitter:
    def test_policy_b_semantics(self):
        stats = make_idea_stats(
            [
                {
                    "idea_id": "neg-but-not-harmful",
                    "block": make_stats_block(
                        intro_events=2, posterior_a=1.0, posterior_b=3.0, median=-0.02
                    ),
                },
                {
                    "idea_id": "confidently-harmful",
                    "block": make_stats_block(
                        intro_events=4, posterior_a=1.0, posterior_b=5.0
                    ),
                },
                {"idea_id": "zero-events", "block": make_stats_block(intro_events=0)},
            ]
        )
        got = PermissiveAdmitter().select(stats)
        assert [s.idea_id for s in got] == ["neg-but-not-harmful"]
