"""Thompson auctions: seed-exact draw order, EV gate + floor, budget caps."""

from __future__ import annotations

import numpy as np
from pydantic import ValidationError
import pytest
from scipy.stats import beta as beta_dist

from gigaevo.memory.events import MemoryAuctionRun, MemoryBudgetCap
from gigaevo.memory.read.auction import (
    AuctionBid,
    AuctionCandidate,
    BootstrapThompsonAuctioneer,
    EVThompsonAuctioneer,
    NoveltyDiscountedBootstrapAuctioneer,
    ThompsonAuctioneer,
    TopBidBudgeter,
    TopThetaBudgeter,
)
from gigaevo.programs.metrics.context import MetricsContext, MetricSpec


def _candidate(
    card_id: str, a: float = 2.0, b: float = 2.0, magnitude: float | None = None
) -> AuctionCandidate:
    return AuctionCandidate(
        card_id=card_id, posterior_a=a, posterior_b=b, magnitude=magnitude
    )


def _metrics(significant_change: float | None) -> MetricsContext:
    return MetricsContext(
        specs={
            "primary": MetricSpec(
                description="primary",
                higher_is_better=True,
                is_primary=True,
                significant_change=significant_change,
            )
        }
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
        auctioneer = EVThompsonAuctioneer(baseline_prior=(3.0, 3.0), ev_floor=0.0)
        candidates = [
            _candidate("c0", a=4.0, b=2.0, magnitude=0.5),
            _candidate("c1", a=2.0, b=4.0, magnitude=None),
        ]
        winners, slate = auctioneer.run(candidates, np.random.default_rng(42))

        replay = np.random.default_rng(42)
        warm_pool = [
            c.magnitude
            for c in candidates
            if c.magnitude is not None and c.magnitude > 0
        ]
        cold_mag = float(np.median(warm_pool)) if warm_pool else 1.0
        expected_bids = []
        for candidate in candidates:
            theta_bid = float(replay.beta(candidate.posterior_a, candidate.posterior_b))
            mag = candidate.magnitude if candidate.magnitude is not None else cold_mag
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

    def test_all_cold_round_uses_significant_change(self):
        # All-cold round with a metrics context: the cold bid takes the primary
        # metric's task-scaled significant_change, not a scale-blind constant.
        auctioneer = EVThompsonAuctioneer(metrics_context=_metrics(0.02))
        _, slate = auctioneer.run(
            [_candidate("cold", magnitude=None)], np.random.default_rng(1)
        )
        assert slate[0].magnitude == 0.02

    def test_all_cold_no_scale_uses_unscaled_placeholder(self):
        # Degenerate round: all cold AND the task declares no significant_change
        # (no metrics context). The cold bid rides the inert unit placeholder.
        auctioneer = EVThompsonAuctioneer()
        _, slate = auctioneer.run(
            [_candidate("cold", magnitude=None)], np.random.default_rng(1)
        )
        assert slate[0].magnitude == 1.0

    def test_unset_significant_change_uses_unscaled_placeholder(self):
        # significant_change is optional; when the primary metric leaves it unset
        # the cold bid falls back to the inert unit placeholder, not a fixed prior.
        auctioneer = EVThompsonAuctioneer(metrics_context=_metrics(None))
        _, slate = auctioneer.run(
            [_candidate("cold", magnitude=None)], np.random.default_rng(1)
        )
        assert slate[0].magnitude == 1.0

    def test_warm_pool_median_overrides_significant_change(self):
        # In-round realized helpful gains are more informative than the declared
        # threshold: the warm-pool median wins even when a metrics context is set.
        auctioneer = EVThompsonAuctioneer(metrics_context=_metrics(0.02))
        candidates = [
            _candidate("warm_hi", magnitude=0.6),
            _candidate("warm_lo", magnitude=0.2),
            _candidate("cold", magnitude=None),
        ]
        _, slate = auctioneer.run(candidates, np.random.default_rng(1))
        cold_bid = next(b for b in slate if b.card_id == "cold")
        assert cold_bid.magnitude == 0.4

    def test_cold_card_borrows_warm_pool_median(self):
        # Mixed round: cold card bids the median of the warm cards' magnitudes,
        # tracking the task's own realized gain scale rather than a fixed constant.
        auctioneer = EVThompsonAuctioneer()
        candidates = [
            _candidate("warm_hi", magnitude=0.6),
            _candidate("warm_lo", magnitude=0.2),
            _candidate("cold", magnitude=None),
        ]
        _, slate = auctioneer.run(candidates, np.random.default_rng(1))
        cold_bid = next(b for b in slate if b.card_id == "cold")
        assert cold_bid.magnitude == 0.4

    def test_cold_pool_ignores_nonpositive_warm_magnitudes(self):
        # Only positive (helpful) warm magnitudes set the cold scale; a harmful
        # or neutral card must not drag the exploration scale toward zero.
        auctioneer = EVThompsonAuctioneer()
        candidates = [
            _candidate("helpful", magnitude=0.4),
            _candidate("harmful", magnitude=-0.2),
            _candidate("neutral", magnitude=0.0),
            _candidate("cold", magnitude=None),
        ]
        _, slate = auctioneer.run(candidates, np.random.default_rng(1))
        cold_bid = next(b for b in slate if b.card_id == "cold")
        assert cold_bid.magnitude == 0.4

    def test_ev_floor_rejects_small_bids(self):
        auctioneer = EVThompsonAuctioneer(ev_floor=10.0)
        rng = np.random.default_rng(0)
        winners, slate = auctioneer.run(
            [_candidate("c0", a=50.0, b=1.0, magnitude=0.5)], rng
        )
        assert winners == []
        assert slate[0].selected is False

    def test_ev_floor_quantile_gates_on_baseline_ev_scale(self):
        # Floor = Beta(baseline_prior).ppf(q) x the round's own cold-magnitude
        # scale: a confident warm card clears it, a weak one bids under it.
        auctioneer = EVThompsonAuctioneer(ev_floor_quantile=0.75)
        winners, slate = auctioneer.run(
            [
                _candidate("strong", a=200.0, b=1.0, magnitude=0.5),
                _candidate("weak", a=1.0, b=200.0, magnitude=0.5),
            ],
            np.random.default_rng(0),
        )
        assert winners == ["strong"]
        assert next(b for b in slate if b.card_id == "weak").selected is False

    def test_ev_floor_quantile_rejects_cold_bid_below_baseline_tail(self):
        # An extreme quantile pushes the floor near the top of the cold scale
        # (Beta(3,3).ppf(0.9999) ~ 0.978 x cold_magnitude); a mid-confidence
        # cold bid (theta ~ 0.5) cannot clear it.
        from scipy.stats import beta as beta_dist

        q = 0.9999
        assert beta_dist.ppf(q, 3.0, 3.0) > 0.9
        auctioneer = EVThompsonAuctioneer(
            metrics_context=_metrics(0.01), ev_floor_quantile=q
        )
        winners, slate = auctioneer.run(
            [_candidate("cold", a=5.0, b=5.0, magnitude=None)],
            np.random.default_rng(0),
        )
        assert winners == []
        assert slate[0].selected is False

    def test_effective_floor_is_max_of_absolute_and_quantile(self, captured_events):
        # Beta(3,3).ppf(0.5) == 0.5 exactly, so q=0.5 with sig=0.3 sets the
        # quantile leg at 0.15.
        EVThompsonAuctioneer(
            metrics_context=_metrics(0.3), ev_floor=0.05, ev_floor_quantile=0.5
        ).run([_candidate("cold", magnitude=None)], np.random.default_rng(3))
        EVThompsonAuctioneer(
            metrics_context=_metrics(0.3), ev_floor=0.5, ev_floor_quantile=0.5
        ).run([_candidate("cold", magnitude=None)], np.random.default_rng(3))
        quantile_dominated, absolute_dominated = captured_events
        assert quantile_dominated.ev_floor == pytest.approx(0.15)
        assert absolute_dominated.ev_floor == pytest.approx(0.5)

    @pytest.mark.parametrize("q", [-0.1, 1.0, 1.5])
    def test_invalid_ev_floor_quantile_raises(self, q):
        with pytest.raises(ValueError):
            EVThompsonAuctioneer(ev_floor_quantile=q)

    def test_emits_ev_auction_event(self, captured_events):
        # All-cold round with a metrics context: the emitted cold_magnitude is
        # the borrowed significant_change scale, alongside the ev_floor.
        EVThompsonAuctioneer(metrics_context=_metrics(0.3), ev_floor=0.05).run(
            [_candidate("cold", magnitude=None)], np.random.default_rng(3)
        )
        (event,) = captured_events
        assert isinstance(event, MemoryAuctionRun)
        assert event.auction == "thompson_ev"
        assert event.cold_magnitude == 0.3
        assert event.ev_floor == 0.05


class TestBootstrapThompsonAuctioneer:
    def test_negative_only_candidate_does_not_borrow_positive_fallback(self):
        auctioneer = BootstrapThompsonAuctioneer()
        candidate = AuctionCandidate(
            card_id="harmful",
            posterior_a=50.0,
            posterior_b=1.0,
            magnitude=None,
            deltas=(-0.1,),
        )

        for seed in range(30):
            winners, slate = auctioneer.run([candidate], np.random.default_rng(seed))
            assert winners == []
            assert slate[0].bid <= 0.0
            assert slate[0].magnitude == 0.0

    def test_known_negative_delta_does_not_borrow_warm_scale(self):
        auctioneer = BootstrapThompsonAuctioneer(ev_floor_quantile=0.0)
        loser = AuctionCandidate(
            card_id="loser",
            posterior_a=50.0,
            posterior_b=1.0,
            magnitude=10.0,
            deltas=(-1.0,),
            staleness_weight=0.01,
        )
        warm = AuctionCandidate(
            card_id="warm",
            posterior_a=50.0,
            posterior_b=1.0,
            magnitude=10.0,
            deltas=(),
        )

        for seed in range(30):
            winners, slate = auctioneer.run([loser, warm], np.random.default_rng(seed))
            loser_bid = next(bid for bid in slate if bid.card_id == "loser")
            assert "loser" not in winners
            assert loser_bid.bid <= 0.0
            assert loser_bid.magnitude == pytest.approx(10.0)

    def test_cold_candidate_still_uses_sampled_positive_fallback(self):
        auctioneer = BootstrapThompsonAuctioneer(metrics_context=_metrics(0.02))

        _, slate = auctioneer.run(
            [
                AuctionCandidate(
                    card_id="cold",
                    posterior_a=50.0,
                    posterior_b=1.0,
                    magnitude=None,
                    deltas=(),
                )
            ],
            np.random.default_rng(0),
        )

        assert slate[0].magnitude == 0.02
        assert 0.0 < slate[0].bid < 0.02
        assert slate[0].support_kind == "cold_prior"

    def test_empty_delta_non_cold_candidate_does_not_borrow_fallback(self):
        auctioneer = BootstrapThompsonAuctioneer(metrics_context=_metrics(0.02))

        winners, slate = auctioneer.run(
            [
                AuctionCandidate(
                    card_id="unused-only",
                    posterior_a=50.0,
                    posterior_b=1.0,
                    magnitude=0.0,
                    deltas=(),
                )
            ],
            np.random.default_rng(0),
        )

        assert winners == []
        assert slate[0].magnitude == 0.0
        assert slate[0].bid == 0.0
        assert slate[0].support_kind == "zero_support"

    def test_support_n_is_staleness_scaled_like_eviction_support(self):
        # Probe eligibility and eviction adjudicability must partition on one
        # measure: decayed evidence re-earns probe eligibility exactly when it
        # drops below the eviction floor.
        auctioneer = BootstrapThompsonAuctioneer(metrics_context=_metrics(0.02))

        _, slate = auctioneer.run(
            [
                AuctionCandidate(
                    card_id="decayed",
                    posterior_a=3.0,
                    posterior_b=3.0,
                    magnitude=0.01,
                    deltas=(-0.1, -0.2),
                    delta_weights=(1.0, 1.0),
                    staleness_weight=0.5,
                )
            ],
            np.random.default_rng(0),
        )

        assert slate[0].support_n == pytest.approx(1.0)

    def test_bootstrap_uses_one_baseline_space_gate_for_slate(self):
        auctioneer = BootstrapThompsonAuctioneer(
            metrics_context=_metrics(0.02), ev_floor_quantile=0.0
        )
        candidates = [
            AuctionCandidate(
                card_id=f"cold-{i}",
                posterior_a=3.0,
                posterior_b=3.0,
                magnitude=None,
                deltas=(),
            )
            for i in range(10)
        ]

        _, slate = auctioneer.run(candidates, np.random.default_rng(123))

        assert len({bid.bid for bid in slate}) > 1
        baselines = {bid.baseline_theta for bid in slate}
        assert len(baselines) == 1
        (baseline_theta,) = baselines
        baseline_quantile = float(beta_dist.cdf(baseline_theta, 3.0, 3.0))
        gate_quantile = baseline_quantile ** (1.0 / len(candidates))
        for bid in slate:
            assert bid.baseline_quantile == pytest.approx(baseline_quantile)
            assert bid.gate_quantile == pytest.approx(gate_quantile)
            assert bid.theta_quantile == pytest.approx(
                beta_dist.cdf(bid.theta, bid.baseline_a, bid.baseline_b)
            )
            assert bid.selected is (bid.theta_quantile > gate_quantile)

        fires = 0
        trials = 200
        for seed in range(trials):
            winners, _ = auctioneer.run(candidates, np.random.default_rng(seed))
            fires += bool(winners)
        fire_rate = fires / trials
        assert 0.30 <= fire_rate <= 0.70

    def test_bootstrap_gate_preserves_card_posterior_strength(self):
        auctioneer = BootstrapThompsonAuctioneer(
            metrics_context=_metrics(0.02), ev_floor_quantile=0.0
        )
        candidates = [
            AuctionCandidate(
                card_id="strong",
                posterior_a=200.0,
                posterior_b=1.0,
                magnitude=None,
                deltas=(),
            ),
            AuctionCandidate(
                card_id="bad",
                posterior_a=1.0,
                posterior_b=200.0,
                magnitude=None,
                deltas=(),
            ),
        ]

        wins = {"strong": 0, "bad": 0}
        for seed in range(100):
            winners, _ = auctioneer.run(candidates, np.random.default_rng(seed))
            for winner in winners:
                wins[winner] += 1

        assert wins["strong"] >= 95
        assert wins["bad"] == 0


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

    def test_tied_bid_breaks_by_theta_then_card_id(self):
        slate = [
            _bid("b", theta=0.9, bid=0.1),
            _bid("a", theta=0.9, bid=0.1),
            _bid("c", theta=0.8, bid=0.1),
        ]
        assert TopBidBudgeter().cap(["c", "b", "a"], slate, max_cards=2) == [
            "a",
            "b",
        ]


class TestNoveltyDiscountedBootstrapAuctioneer:
    @staticmethod
    def _pair(hot_uses: int = 15) -> list[AuctionCandidate]:
        return [
            AuctionCandidate(
                card_id="hot",
                posterior_a=50.0,
                posterior_b=1.0,
                magnitude=0.9,
                deltas=(0.9,) * 8,
                use_count=hot_uses,
            ),
            AuctionCandidate(
                card_id="fresh",
                posterior_a=50.0,
                posterior_b=1.0,
                magnitude=0.5,
                deltas=(0.5,) * 8,
                use_count=0,
            ),
        ]

    def test_power_zero_is_bid_exact_with_base(self):
        base = BootstrapThompsonAuctioneer(ev_floor_quantile=0.765)
        novelty = NoveltyDiscountedBootstrapAuctioneer(
            ev_floor_quantile=0.765, novelty_power=0.0
        )
        for seed in range(10):
            base_winners, base_slate = base.run(
                self._pair(), np.random.default_rng(seed)
            )
            nov_winners, nov_slate = novelty.run(
                self._pair(), np.random.default_rng(seed)
            )
            assert nov_winners == base_winners
            assert [bid.bid for bid in nov_slate] == [bid.bid for bid in base_slate]
            assert [bid.theta for bid in nov_slate] == [bid.theta for bid in base_slate]

    def test_slate_stores_discounted_bid_and_use_count(self):
        candidates = self._pair(hot_uses=3)
        _, base_slate = BootstrapThompsonAuctioneer(ev_floor_quantile=0.5).run(
            candidates, np.random.default_rng(7)
        )
        _, nov_slate = NoveltyDiscountedBootstrapAuctioneer(
            ev_floor_quantile=0.5, novelty_power=0.5
        ).run(candidates, np.random.default_rng(7))
        for candidate, raw, discounted in zip(candidates, base_slate, nov_slate):
            assert discounted.use_count == candidate.use_count
            assert discounted.bid == pytest.approx(
                raw.bid * (1.0 + candidate.use_count) ** -0.5
            )

    def test_discount_redistributes_win_without_shrinking_volume(self):
        base = BootstrapThompsonAuctioneer(ev_floor_quantile=0.5)
        novelty = NoveltyDiscountedBootstrapAuctioneer(
            ev_floor_quantile=0.5, novelty_power=1.0
        )
        redistributed = 0
        for seed in range(30):
            base_winners, _ = base.run(self._pair(), np.random.default_rng(seed))
            nov_winners, _ = novelty.run(self._pair(), np.random.default_rng(seed))
            assert len(nov_winners) == len(base_winners)
            if base_winners == ["hot"] and nov_winners == ["fresh"]:
                redistributed += 1
        assert redistributed >= 25

    def test_zero_use_count_pays_no_tax(self):
        candidates = [
            AuctionCandidate(
                card_id="a",
                posterior_a=50.0,
                posterior_b=1.0,
                magnitude=0.5,
                deltas=(0.5,) * 4,
            ),
            AuctionCandidate(
                card_id="b",
                posterior_a=50.0,
                posterior_b=1.0,
                magnitude=0.3,
                deltas=(0.3,) * 4,
            ),
        ]
        base = BootstrapThompsonAuctioneer(ev_floor_quantile=0.5)
        novelty = NoveltyDiscountedBootstrapAuctioneer(
            ev_floor_quantile=0.5, novelty_power=1.0
        )
        for seed in range(10):
            base_winners, base_slate = base.run(candidates, np.random.default_rng(seed))
            nov_winners, nov_slate = novelty.run(
                candidates, np.random.default_rng(seed)
            )
            assert nov_winners == base_winners
            assert [bid.bid for bid in nov_slate] == [bid.bid for bid in base_slate]

    def test_negative_novelty_power_rejected(self):
        with pytest.raises(ValidationError):
            NoveltyDiscountedBootstrapAuctioneer(novelty_power=-0.1)


class TestBootstrapAuctioneerPricedNoise:
    def _run(self, se_a, se_b, seed):
        auctioneer = BootstrapThompsonAuctioneer()
        candidates = [
            AuctionCandidate(
                card_id="a",
                posterior_a=2.0,
                posterior_b=2.0,
                deltas=(0.3, -0.1),
                deltas_se=se_a,
            ),
            AuctionCandidate(
                card_id="b",
                posterior_a=2.0,
                posterior_b=2.0,
                deltas=(0.2, 0.1),
                deltas_se=se_b,
            ),
        ]
        return auctioneer.run(candidates, np.random.default_rng(seed))

    def test_zero_ses_replay_the_point_auction_exactly(self):
        for seed in range(10):
            absent_winners, absent_slate = self._run(None, None, seed)
            zero_winners, zero_slate = self._run((0.0, 0.0), (0.0, 0.0), seed)
            assert zero_winners == absent_winners
            assert [bid.bid for bid in zero_slate] == [bid.bid for bid in absent_slate]

    def test_confidently_harmful_stays_out_under_priced_noise(self):
        auctioneer = BootstrapThompsonAuctioneer()
        candidate = AuctionCandidate(
            card_id="harmful",
            posterior_a=50.0,
            posterior_b=1.0,
            deltas=(-1.0, -1.2, -0.9),
            deltas_se=(0.01, 0.01, 0.01),
        )
        for seed in range(30):
            winners, slate = auctioneer.run([candidate], np.random.default_rng(seed))
            assert winners == []
            assert slate[0].bid <= 0.0

    def test_marginally_harmful_keeps_a_bounded_positive_chance(self):
        auctioneer = BootstrapThompsonAuctioneer()
        candidate = AuctionCandidate(
            card_id="marginal",
            posterior_a=2.0,
            posterior_b=2.0,
            deltas=(-0.01,),
            deltas_se=(0.5,),
        )
        bids = [
            auctioneer.run([candidate], np.random.default_rng(seed))[1][0].bid
            for seed in range(100)
        ]
        assert any(bid > 0 for bid in bids)
        assert any(bid <= 0 for bid in bids)
