"""Unit + contract tests for the modular efficacy core.

BetaBinomialReputation must agree with the standalone posterior functions
(beta_binomial_posterior / compute_injection_posterior) under default config,
ThompsonAuctioneer's draw order (theta then baseline, per candidate) is pinned
seed-exact, and every threshold is exposed as a constructor parameter. The
admitters and the harm predicate have their semantics pinned directly here
over ``IdeaStats`` rows.
"""

from __future__ import annotations

import math

import numpy as np

from gigaevo.memory.core.admitter import PermissiveAdmitter, SignBasedAdmitter
from gigaevo.memory.core.auctioneer import AuctionCandidate, ThompsonAuctioneer
from gigaevo.memory.core.idea_stats import IdeaStats
from gigaevo.memory.core.reputation import BetaBinomialReputation
from gigaevo.memory.efficacy import beta_binomial_posterior
from gigaevo.memory.shared_memory.injection_posterior import (
    InjectionOutcome,
    compute_injection_posterior,
)
from gigaevo.memory.shared_memory.models import EvolutionStatistics


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
            (
                {"ALL": {"intro_events": 3, "posterior_a": 1.0, "posterior_b": 4.0}},
                True,
            ),
            (
                {"ALL": {"intro_events": 3, "posterior_a": 4.0, "posterior_b": 1.0}},
                False,
            ),
            (
                {"ALL": {"intro_events": 2, "posterior_a": 1.0, "posterior_b": 3.0}},
                False,
            ),
            (
                {"ALL": {"intro_events": 5, "posterior_a": 2.0, "posterior_b": 5.0}},
                True,
            ),
            (
                {
                    "best_ideas_snapshot": {
                        "intro_events": 5,
                        "posterior_a": 1.0,
                        "posterior_b": 9.0,
                    }
                },
                False,
            ),
            ({}, False),
            (None, False),
        ]
        for stats, expected in cases:
            typed = None if stats is None else EvolutionStatistics.model_validate(stats)
            assert rep.is_confidently_harmful(typed) is expected, stats

    def test_harm_min_events_configurable(self):
        harmful = EvolutionStatistics.model_validate(
            {"ALL": {"intro_events": 2, "posterior_a": 1.0, "posterior_b": 4.0}}
        )
        assert not BetaBinomialReputation().is_confidently_harmful(harmful)
        assert BetaBinomialReputation(harm_min_events=2).is_confidently_harmful(harmful)

    def test_injection_posteriors_match_legacy(self):
        programs = [
            InjectionOutcome(id="p1", fitness=0.5),
            InjectionOutcome(id="p2", fitness=0.6, parents=["p1"], selected_ids=["c1"]),
            InjectionOutcome(id="p3", fitness=0.4, parents=["p1"], selected_ids=["c2"]),
            InjectionOutcome(
                id="p4", fitness=0.7, parents=["p2"], selected_ids=["c1", "c3"]
            ),
            InjectionOutcome(id="p5", fitness=0.3, parents=["p3"], selected_ids=["c2"]),
        ]
        got = BetaBinomialReputation().compute_injection_posteriors(programs)
        want = compute_injection_posterior(programs)
        assert got == want

    def test_injection_posteriors_lower_is_better(self):
        programs = [
            InjectionOutcome(id="p1", fitness=0.5),
            InjectionOutcome(id="p2", fitness=0.4, parents=["p1"], selected_ids=["c1"]),
        ]
        got = BetaBinomialReputation().compute_injection_posteriors(
            programs, higher_is_better=False
        )
        want = compute_injection_posterior(programs, higher_is_better=False)
        assert got == want

    def test_noise_band_k_widens_dead_band(self):
        # cardX rides on parent pX; its child cx regresses within the noise band.
        programs = [
            InjectionOutcome(id="p0", fitness=0.5),
            InjectionOutcome(id="pX", fitness=0.5, selected_ids=["cardX"]),
            InjectionOutcome(id="c1", fitness=0.5, parents=["p0"]),
            InjectionOutcome(id="c2", fitness=0.6, parents=["p0"]),
            InjectionOutcome(id="c3", fitness=0.4, parents=["p0"]),
            InjectionOutcome(id="c4", fitness=0.7, parents=["p0"]),
            InjectionOutcome(id="c5", fitness=0.3, parents=["p0"]),
            InjectionOutcome(id="cx", fitness=0.4, parents=["pX"]),
        ]
        default = BetaBinomialReputation().compute_injection_posteriors(programs)
        assert default["cardX"].k_harm == 0
        assert default == compute_injection_posterior(programs)
        no_band = BetaBinomialReputation(noise_band_k=0.0).compute_injection_posteriors(
            programs
        )
        assert no_band["cardX"].k_harm == 1

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
