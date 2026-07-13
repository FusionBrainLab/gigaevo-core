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
    PendingDiscountedBootstrapAuctioneer,
    ThompsonAuctioneer,
    TopBidBudgeter,
    TopThetaBudgeter,
)
from gigaevo.memory.read.bootstrap import bootstrap_ev_samples, stable_rng
from gigaevo.programs.metrics.context import MetricsContext, MetricSpec


def _candidate(
    card_id: str, a: float = 2.0, b: float = 2.0, magnitude: float | None = None
) -> AuctionCandidate:
    return AuctionCandidate(
        card_id=card_id, posterior_a=a, posterior_b=b, magnitude=magnitude
    )


def test_candidate_pending_count_defaults_to_zero():
    assert _candidate("card").pending_count == 0


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


class _CoherenceRng:
    """Controlled RNG that exposes whether bid and gate consume the same u."""

    def __init__(self, u: float, *, bid_uses_beta: bool) -> None:
        self.u = u
        self.bid_uses_beta = bid_uses_beta
        self.uniform_calls = 0
        self.candidate_beta_calls = 0
        self.choice_indices = None

    def uniform(self) -> float:
        self.uniform_calls += 1
        return self.u

    def beta(self, a: float, b: float) -> float:
        if (a, b) == (7.0, 11.0):
            return 0.5
        self.candidate_beta_calls += 1
        if self.bid_uses_beta and self.candidate_beta_calls == 1:
            return 1.0 - self.u
        return self.u

    def choice(self, a, size, replace, p):
        n_samples, n_atoms = size
        rows = np.arange(n_samples) % int(a)
        self.choice_indices = np.repeat(rows[:, None], n_atoms, axis=1)
        return self.choice_indices


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
    @pytest.mark.parametrize("magnitude", [None, 0.4], ids=["cold", "warm"])
    def test_bid_and_gate_are_monotone_in_the_same_u(self, magnitude):
        auctioneer = EVThompsonAuctioneer(baseline_prior=(7.0, 11.0))
        candidate = _candidate("card", a=5.0, b=2.0, magnitude=magnitude)
        slates = []
        rngs = []
        for u in (0.1, 0.9):
            rng = _CoherenceRng(u, bid_uses_beta=True)
            _, slate = auctioneer.run([candidate], rng)
            rngs.append(rng)
            slates.append(slate[0])

        low, high = slates
        assert [rng.uniform_calls for rng in rngs] == [1, 1]
        assert low.theta < high.theta
        assert low.bid < high.bid
        assert low.bid / low.magnitude == pytest.approx(low.theta)
        assert high.bid / high.magnitude == pytest.approx(high.theta)
        assert high.selected is True

    def test_draw_order_shared_worlds_then_baselines(self):
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
            u = float(replay.uniform())
            theta = float(
                beta_dist.ppf(u, candidate.posterior_a, candidate.posterior_b)
            )
            mag = candidate.magnitude if candidate.magnitude is not None else cold_mag
            expected_bids.append((mag, theta, theta * mag))
        for candidate, (mag, theta, bid_value), bid in zip(
            candidates, expected_bids, slate
        ):
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
    def test_ev_reserve_defaults_to_byte_exact_legacy_quantile(self):
        candidates = [
            AuctionCandidate(
                card_id="hot",
                posterior_a=50.0,
                posterior_b=1.0,
                magnitude=0.9,
                deltas=(0.9,) * 8,
            ),
            AuctionCandidate(
                card_id="fresh",
                posterior_a=50.0,
                posterior_b=1.0,
                magnitude=0.5,
                deltas=(0.5,) * 8,
            ),
        ]
        legacy = BootstrapThompsonAuctioneer()
        explicit = BootstrapThompsonAuctioneer(ev_reserve_mode="quantile")

        legacy_winners, legacy_slate = legacy.run(candidates, np.random.default_rng(7))
        explicit_winners, explicit_slate = explicit.run(
            candidates, np.random.default_rng(7)
        )

        assert legacy.ev_reserve_mode == "quantile"
        assert legacy_winners == explicit_winners
        assert [bid.model_dump_json() for bid in legacy_slate] == [
            bid.model_dump_json() for bid in explicit_slate
        ]

    def test_risk_gate_consumes_no_shared_rng(self, captured_events):
        # The risk gate reads a CARD-LOCAL bootstrap vector (seeded only from the
        # card id), so switching quantile -> risk must not perturb the shared
        # round RNG: every bid, theta and baseline draw stays byte-identical.
        # Only the ev_* gate telemetry and the eligibility verdict may differ.
        candidates = [
            AuctionCandidate(
                card_id="a",
                posterior_a=50.0,
                posterior_b=1.0,
                magnitude=1.0,
                deltas=(1.0, 1.0, -0.5),
            ),
            AuctionCandidate(
                card_id="b",
                posterior_a=50.0,
                posterior_b=1.0,
                magnitude=1.0,
                deltas=(0.5, -1.0),
            ),
        ]
        quantile = BootstrapThompsonAuctioneer(n_bootstrap=32, ev_floor_quantile=0.0)
        risk = BootstrapThompsonAuctioneer(
            n_bootstrap=32, ev_reserve_mode="risk", ev_risk_alpha=0.2
        )

        _, quantile_slate = quantile.run(candidates, np.random.default_rng(7))
        _, risk_slate = risk.run(candidates, np.random.default_rng(7))

        for candidate, q_bid, r_bid in zip(candidates, quantile_slate, risk_slate):
            # Shared-stream draws are untouched by the card-local gate.
            assert r_bid.bid == pytest.approx(q_bid.bid)
            assert r_bid.theta == pytest.approx(q_bid.theta)
            assert r_bid.baseline_theta == pytest.approx(q_bid.baseline_theta)
            assert r_bid.magnitude == pytest.approx(q_bid.magnitude)
            # The reported probability is exactly the card-local bootstrap, off a
            # generator seeded from (card_id, len(deltas), n_bootstrap) alone.
            gate = bootstrap_ev_samples(
                candidate.deltas,
                0.0,
                1.0,
                32,
                stable_rng(candidate.card_id, len(candidate.deltas), 32),
            )
            assert r_bid.ev_positive_probability == pytest.approx(
                float(np.mean(gate > 0.0))
            )
            assert r_bid.ev_reserve_mode == "risk"
            assert r_bid.ev_risk_alpha == pytest.approx(0.2)
            assert q_bid.ev_positive_probability is None
        (event,) = captured_events[-1:]
        assert event.bids[0]["ev_reserve_mode"] == "risk"
        assert event.bids[0]["ev_risk_alpha"] == pytest.approx(0.2)

    def test_risk_alpha_one_rejected(self):
        # alpha=1.0 => risk_threshold = 1 - alpha = 0.0, so the sign gate
        # `P(EV>0) >= 0.0` admits even a card whose entire bootstrap-EV vector
        # is non-positive (P(EV>0)=0). Risk mode has no separate `bid > 0`
        # check, so that would inject a provably-losing card — the opposite of
        # an EV reserve. The alpha bound must be exclusive of 1.0, matching the
        # sibling ev_floor_quantile field.
        with pytest.raises(ValidationError):
            BootstrapThompsonAuctioneer(ev_reserve_mode="risk", ev_risk_alpha=1.0)

    def test_risk_reserve_admission_is_order_independent(self):
        # IIA: the risk gate seeds its bootstrap from the card id, so a card's
        # admission verdict cannot depend on which other cards share the round or
        # where it sits in the draw order. Under the old shared-stream gate,
        # trailing `target` behind two co-bidders shifted its EV vector and could
        # flip the verdict; the card-local vector makes it invariant.
        target = AuctionCandidate(
            card_id="target",
            posterior_a=50.0,
            posterior_b=1.0,
            magnitude=1.0,
            deltas=(1.0, 1.0, 1.0, 1.0, -0.5),
        )
        strong = [
            AuctionCandidate(
                card_id="strong-1",
                posterior_a=50.0,
                posterior_b=1.0,
                magnitude=2.0,
                deltas=(2.0, 2.0),
            ),
            AuctionCandidate(
                card_id="strong-2",
                posterior_a=50.0,
                posterior_b=1.0,
                magnitude=2.0,
                deltas=(2.0, 2.0),
            ),
        ]
        risk = BootstrapThompsonAuctioneer(
            n_bootstrap=32, ev_reserve_mode="risk", ev_risk_alpha=0.2
        )

        alone_winners, alone_slate = risk.run([target], np.random.default_rng(2))
        # target LAST, so two co-bidders consume the shared RNG before its draw.
        _, trailing_slate = risk.run([*strong, target], np.random.default_rng(2))
        alone_bid = alone_slate[0]
        trailing_bid = next(b for b in trailing_slate if b.card_id == "target")

        # Precondition: target clears the risk gate (P(EV>0) >= 1 - alpha = 0.8),
        # so the invariance below is exercised on an admitted card.
        assert alone_bid.rejected_by_ev_floor is False
        assert alone_bid.ev_positive_probability >= 0.8
        assert "target" in alone_winners
        # The risk-gate probability and verdict are identical across orderings.
        assert trailing_bid.ev_positive_probability == pytest.approx(
            alone_bid.ev_positive_probability
        )
        assert trailing_bid.rejected_by_ev_floor is False

    def test_risk_gate_admits_cold_card_unconditionally(self):
        # A genuinely cold card (no deltas, no magnitude) has no bootstrap-EV
        # history, so the risk reserve cannot estimate P(EV>0) from evidence. The
        # default admits it (probability pinned to 1.0) and defers exploration to
        # the Thompson draw, the no-card gate, and the probe lane -- a cold card
        # must NOT be benched the way a warm card with a weak track record is.
        # Reverting the branch to `None` makes every cold card ineligible (None
        # is filtered out of `eligible`), so cold cards would never win under
        # risk mode; the strictest admissible alpha proves the gate is bypassed.
        cold = AuctionCandidate(
            card_id="cold",
            posterior_a=3.0,
            posterior_b=3.0,
            magnitude=None,
            deltas=(),
        )
        risk = BootstrapThompsonAuctioneer(
            n_bootstrap=32, ev_reserve_mode="risk", ev_risk_alpha=0.01
        )
        _, slate = risk.run([cold], np.random.default_rng(0))
        assert slate[0].support_kind == "cold_prior"
        assert slate[0].ev_positive_probability == pytest.approx(1.0)
        assert slate[0].rejected_by_ev_floor is False

    def test_risk_gate_rejects_zero_support_known_card(self):
        # A known card with zero recorded deltas (only invalid/unused exposure,
        # never a direct causal gain) carries no positive-EV evidence: its
        # bootstrap support collapses to one zero atom, so P(EV>0)=0 and it is
        # rejected at every admissible alpha (<1). Under-evidenced cards stay
        # alive through the probe lane, not the auction, so this rejection is the
        # intended boundary -- pin it so a future `support_scale` tweak letting
        # zero-support cards borrow the cold scale cannot land silently.
        empty = AuctionCandidate(
            card_id="known-empty",
            posterior_a=50.0,
            posterior_b=1.0,
            magnitude=0.0,
            deltas=(),
        )
        risk = BootstrapThompsonAuctioneer(
            n_bootstrap=32, ev_reserve_mode="risk", ev_risk_alpha=0.99
        )
        _, slate = risk.run([empty], np.random.default_rng(0))
        assert slate[0].support_kind == "zero_support"
        assert slate[0].ev_positive_probability == pytest.approx(0.0)
        assert slate[0].rejected_by_ev_floor is True

    @pytest.mark.parametrize(
        ("magnitude", "deltas", "bid_uses_beta"),
        [(None, (), True), (0.6, (0.2, 1.0), False)],
        ids=["cold", "warm"],
    )
    def test_bid_and_gate_are_monotone_in_the_same_u(
        self, magnitude, deltas, bid_uses_beta
    ):
        auctioneer = BootstrapThompsonAuctioneer(
            baseline_prior=(7.0, 11.0), ev_floor_quantile=0.0
        )
        candidate = AuctionCandidate(
            card_id="card",
            posterior_a=5.0,
            posterior_b=2.0,
            magnitude=magnitude,
            deltas=deltas,
        )
        slates = []
        rngs = []
        for u in (0.1, 0.9):
            rng = _CoherenceRng(u, bid_uses_beta=bid_uses_beta)
            _, slate = auctioneer.run([candidate], rng)
            rngs.append(rng)
            slates.append(slate[0])

        low, high = slates
        assert [rng.uniform_calls for rng in rngs] == [1, 1]
        assert low.theta < high.theta
        assert low.bid < high.bid
        if deltas:
            atoms = np.asarray([*deltas, 0.0])
            for u, rng, bid in zip((0.1, 0.9), rngs, slates):
                samples = np.sort(atoms[rng.choice_indices].mean(axis=1))
                index = min(int(u * len(samples)), len(samples) - 1)
                assert bid.bid == pytest.approx(samples[index])
        else:
            assert low.bid / low.magnitude == pytest.approx(low.theta)
            assert high.bid / high.magnitude == pytest.approx(high.theta)
        assert high.selected is True

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

    def test_auction_leaves_probe_eligibility_to_probe_policy(self):
        _, slate = BootstrapThompsonAuctioneer().run(
            [
                AuctionCandidate(
                    card_id="cold",
                    posterior_a=3.0,
                    posterior_b=3.0,
                    magnitude=None,
                    deltas=(),
                )
            ],
            np.random.default_rng(0),
        )

        assert slate[0].support_kind == "cold_prior"
        assert slate[0].probe_eligible is False

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


class TestPendingDiscountedBootstrapAuctioneer:
    @staticmethod
    def _pair() -> list[AuctionCandidate]:
        return [
            AuctionCandidate(
                card_id="pending",
                posterior_a=50.0,
                posterior_b=1.0,
                magnitude=0.9,
                deltas=(0.9,) * 8,
                pending_count=3,
            ),
            AuctionCandidate(
                card_id="clear",
                posterior_a=50.0,
                posterior_b=1.0,
                magnitude=0.5,
                deltas=(0.5,) * 8,
                pending_count=0,
            ),
        ]

    def test_power_zero_is_bid_exact_with_base_and_consumes_no_rng(self):
        base = BootstrapThompsonAuctioneer(ev_floor_quantile=0.5)
        pending = PendingDiscountedBootstrapAuctioneer(
            ev_floor_quantile=0.5, pending_power=0.0
        )
        for seed in range(10):
            base_rng = np.random.default_rng(seed)
            pending_rng = np.random.default_rng(seed)

            base_winners, base_slate = base.run(self._pair(), base_rng)
            pending_winners, pending_slate = pending.run(self._pair(), pending_rng)

            assert pending_winners == base_winners
            assert [bid.bid for bid in pending_slate] == [bid.bid for bid in base_slate]
            assert [bid.theta for bid in pending_slate] == [
                bid.theta for bid in base_slate
            ]
            assert pending_rng.random() == base_rng.random()

    def test_positive_power_taxes_only_pending_bid_without_rng_draw(self):
        candidates = self._pair()
        base_rng = np.random.default_rng(7)
        pending_rng = np.random.default_rng(7)
        _, base_slate = BootstrapThompsonAuctioneer(ev_floor_quantile=0.5).run(
            candidates, base_rng
        )
        _, pending_slate = PendingDiscountedBootstrapAuctioneer(
            ev_floor_quantile=0.5, pending_power=0.5
        ).run(candidates, pending_rng)

        assert pending_slate[0].bid < base_slate[0].bid
        assert pending_slate[0].bid == pytest.approx(
            base_slate[0].bid * (1.0 + candidates[0].pending_count) ** -0.5
        )
        assert pending_slate[1].bid == base_slate[1].bid
        assert pending_rng.random() == base_rng.random()

    def test_negative_pending_power_rejected(self):
        with pytest.raises(ValidationError):
            PendingDiscountedBootstrapAuctioneer(pending_power=-0.1)


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


def test_bid_carries_candidate_staleness_audit_fields(captured_events):
    candidate = AuctionCandidate(
        card_id="audit",
        posterior_a=5.0,
        posterior_b=1.0,
        deltas=(0.2, 0.3),
        delta_weights=(0.25, 1.0),
        support_n_unstaled=2.0,
        gain_se=0.4,
    )
    _, slate = BootstrapThompsonAuctioneer().run([candidate], np.random.default_rng(0))
    bid = slate[0]

    # The auction copies the audit fields straight from the candidate...
    assert bid.support_n_unstaled == pytest.approx(2.0)
    assert bid.gain_se == pytest.approx(0.4)
    # ...support_n stays the staled reduction of delta_weights (0.25 + 1.0)...
    assert bid.support_n == pytest.approx(1.25)
    # ...so the per-event staleness bite is auditable off one slate row...
    assert bid.support_n / bid.support_n_unstaled == pytest.approx(0.625)
    # ...and both survive the JSON round-trip into MemoryReadSelection.slate.
    dumped = bid.model_dump(mode="json")
    assert dumped["support_n_unstaled"] == pytest.approx(2.0)
    assert dumped["gain_se"] == pytest.approx(0.4)
