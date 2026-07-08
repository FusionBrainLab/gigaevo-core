"""Reputation math: downside posterior, event blocks, BD partition."""

from __future__ import annotations

from math import inf, nan

import pytest
from scipy.stats import beta

from gigaevo.evolution.strategies.models import BehaviorSpace, LinearBinning
from gigaevo.memory.cards import (
    CardStatsBlock,
    CausalStrength,
    EvidenceAttribution,
    EvidenceSource,
)
from gigaevo.memory.read.reputation import (
    BDProximityReputation,
    BetaBinomialReputation,
    BootstrapReputation,
    beta_binomial_posterior,
    block_from_events,
)


def _bs(num_bins: int = 10, max_val: float = 1.0) -> BehaviorSpace:
    return BehaviorSpace(
        bins={"x": LinearBinning(min_val=0.0, max_val=max_val, num_bins=num_bins)}
    )


class TestBetaBinomialPosterior:
    def test_no_events_is_prior_with_nan_quantile(self):
        block = beta_binomial_posterior([])
        assert (block.posterior_a, block.posterior_b) == (1.0, 1.0)
        assert block.intro_events == 0
        assert block.p_help_lo20 != block.p_help_lo20
        assert block.efficacy_confident is False

    def test_counts_below_threshold_as_harm(self):
        block = beta_binomial_posterior([0.5, -0.5, 0.0, 0.2])
        assert block.intro_events == 4
        assert block.k_harm == 1
        assert (block.posterior_a, block.posterior_b) == (4.0, 2.0)
        assert block.p_help_mean == pytest.approx(4.0 / 6.0)

    def test_zero_gain_is_not_harm_at_default_threshold(self):
        block = beta_binomial_posterior([0.0, 0.0])
        assert block.k_harm == 0

    def test_negative_threshold_widens_the_band(self):
        block = beta_binomial_posterior([-0.05, -0.5], threshold=-0.1)
        assert block.k_harm == 1

    def test_invalid_events_are_forced_harm(self):
        block = beta_binomial_posterior([0.5], invalid_events=2)
        assert block.intro_events == 3
        assert block.k_harm == 2
        assert (block.posterior_a, block.posterior_b) == (2.0, 3.0)

    def test_unused_events_are_failure_trials(self):
        block = beta_binomial_posterior([0.5], unused_events=2)
        assert block.intro_events == 3
        assert block.k_harm == 2
        assert (block.posterior_a, block.posterior_b) == (2.0, 3.0)

    def test_non_finite_gains_dropped_from_counts(self):
        block = beta_binomial_posterior([nan, inf, 0.5])
        assert block.intro_events == 1
        assert block.k_harm == 0

    def test_lo20_matches_scipy_and_drives_confidence(self):
        block = beta_binomial_posterior([0.1] * 9 + [-0.2])
        assert block.p_help_lo20 == pytest.approx(float(beta.ppf(0.20, 10.0, 2.0)))
        assert block.efficacy_confident is True
        weak = beta_binomial_posterior([0.1, -0.2])
        assert weak.efficacy_confident is False


class TestBlockFromEvents:
    def test_no_events_no_block(self):
        assert block_from_events([]) is None

    def test_median_magnitude_stamped(self, make_event):
        block = block_from_events([make_event(0.1), make_event(0.3), make_event(0.2)])
        assert block is not None
        assert block.IntroGain_best_median == pytest.approx(0.2)

    def test_harm_counts_gains_below_zero(self, make_event):
        events = [make_event(g) for g in (0.10, 0.11, 0.09, -0.05)]
        block = block_from_events(events)
        assert block is not None
        assert block.k_harm == 1

    def test_wins_do_not_inflate_the_loss_count(self, make_event):
        # A card's own wins can never mask its losses: harm is a strict sign
        # test, so the -0.5 loss among {1,2,3} wins is still counted (monotone).
        events = [make_event(g) for g in (1.0, 2.0, 3.0, -0.5)]
        block = block_from_events(events)
        assert block is not None
        assert block.k_harm == 1

    def test_escalating_wins_never_hide_a_loss(self, make_event):
        # Same two losses, growing wins: the harm verdict on the losses is
        # invariant to how large the wins get (no win-inflation loophole).
        losses = (-0.3, -0.25)
        counts = [
            block_from_events([make_event(g) for g in losses]).k_harm,
            block_from_events([make_event(g) for g in (*losses, 0.05)]).k_harm,
            block_from_events([make_event(g) for g in (*losses, 10.0, 20.0)]).k_harm,
        ]
        assert counts == [2, 2, 2]

    def test_wins_never_reduce_harm_count(self, make_event):
        # A win must never buy immunity for a loss. A median-centred MAD band
        # shifts when a win adds a clipped zero, dropping a loss from the harm
        # count (non-monotone); counting every gain below zero keeps all four
        # losses as harm however many wins arrive.
        losses = (-1.0, -2.0, -3.0, -10.0)
        base = block_from_events([make_event(g) for g in (0.5, *losses)])
        plus_win = block_from_events([make_event(g) for g in (0.5, 0.5, *losses)])
        assert base is not None and plus_win is not None
        assert base.k_harm == plus_win.k_harm == 4

    def test_uniformly_harmful_card_counts_every_loss(self, make_event):
        # Every injection lost. A band derived from the card's own losses reads
        # roughly half of them as within-noise and the card as net-helpful; a
        # zero threshold counts all four as harm.
        block = block_from_events([make_event(g) for g in (-1.0, -2.0, -3.0, -10.0)])
        assert block is not None
        assert block.k_harm == 4
        assert block.p_help_mean < 0.5

    def test_harm_count_is_the_number_of_losses(self, make_event):
        block = block_from_events([make_event(g) for g in (2.0, -0.1, 0.0, -5.0, 3.0)])
        assert block is not None
        assert block.k_harm == 2

    def test_noop_card_is_not_efficacy_confident(self, make_event):
        # All-zero gains: none is below zero -> zero harm events -> a confident
        # downside posterior, but a zero central gain is a no-op, not a win.
        block = block_from_events([make_event(0.0) for _ in range(4)])
        assert block is not None
        assert block.IntroGain_best_median == 0.0
        assert block.efficacy_confident is False

    def test_positive_median_keeps_confidence(self, make_event):
        block = block_from_events(
            [make_event(0.1) for _ in range(9)] + [make_event(-0.2)]
        )
        assert block is not None
        assert block.IntroGain_best_median > 0
        assert block.efficacy_confident is True

    def test_invalid_events_forced_harm_without_magnitude(self, make_event):
        block = block_from_events([make_event(0.5), make_event(0.0, invalid=True)])
        assert block is not None
        assert block.intro_events == 2
        assert block.k_harm == 1
        assert block.IntroGain_best_median == pytest.approx(0.5)

    def test_unused_events_count_as_failures_without_magnitude(self, make_event):
        block = block_from_events([make_event(0.5), make_event(0.0, unused=True)])
        assert block is not None
        assert block.intro_events == 2
        assert block.k_harm == 1
        assert block.IntroGain_best_median == pytest.approx(0.5)

    def test_all_unused_has_zero_magnitude_and_no_confidence(self, make_event):
        block = block_from_events([make_event(0.0, unused=True) for _ in range(3)])
        assert block is not None
        assert block.intro_events == 3
        assert block.k_harm == 3
        assert block.IntroGain_best_median == 0.0
        assert block.efficacy_confident is False

    def test_all_invalid_has_zero_magnitude(self, make_event):
        block = block_from_events([make_event(0.0, invalid=True)])
        assert block is not None
        assert block.IntroGain_best_median == 0.0
        assert block.k_harm == 1

    def test_losing_founding_only_card_has_no_efficacy_block(self, make_event):
        # Founding evidence is origin/admission evidence only. It must not enter
        # the use-attributed posterior, confidence, or EV view.
        assert block_from_events([make_event(-0.4, founding=True)]) is None

    def test_winning_founding_only_card_also_has_no_efficacy_block(self, make_event):
        assert block_from_events([make_event(0.3, founding=True)]) is None

    def test_use_events_own_the_magnitude_over_founding(self, make_event):
        # Once real injection outcomes exist they are the gain scale; the
        # one-time birth delta no longer skews the median either way.
        block = block_from_events(
            [make_event(5.0, founding=True), make_event(0.1), make_event(0.2)]
        )
        assert block is not None
        assert block.IntroGain_best_median == pytest.approx(0.15)
        assert block.intro_events == 2

    def test_bundled_use_events_have_fractional_posterior_weight(self, make_event):
        bundled = make_event(0.2).model_copy(
            update={
                "attribution": EvidenceAttribution(
                    source=EvidenceSource.DIRECT,
                    causal_strength=CausalStrength.DIRECT_BUNDLED,
                    used_card_count=2,
                    credit_weight=0.5,
                )
            }
        )
        block = block_from_events([bundled])
        assert block is not None
        assert block.intro_events == pytest.approx(0.5)
        assert block.posterior_a == pytest.approx(1.5)
        assert block.posterior_b == pytest.approx(1.0)


class TestBetaBinomialReputation:
    def test_cold_card_gets_cold_prior(self, make_card):
        rep = BetaBinomialReputation()
        card = make_card()
        assert rep.card_stats(card) is None
        assert rep.card_posterior(card) == (3.0, 3.0)
        assert rep.card_magnitude(card) is None

    def test_card_with_events_resolves_through_block(self, make_card, make_event):
        rep = BetaBinomialReputation()
        card = make_card(
            gain_events=(make_event(0.2), make_event(0.4), make_event(-5.0))
        )
        assert rep.card_posterior(card) == (3.0, 2.0)
        assert rep.card_magnitude(card) == pytest.approx(0.2)

    def test_corrupt_posterior_falls_back_to_cold_prior(self, make_card):
        rep = BetaBinomialReputation(cold_prior=(2.0, 5.0))
        block = CardStatsBlock(posterior_a=0.0, posterior_b=-1.0)
        assert rep.is_confidently_harmful(block) is False

        class _Broken(BetaBinomialReputation):
            def card_stats(self, card, context=None):
                return block

        assert _Broken(cold_prior=(2.0, 5.0)).card_posterior(make_card()) == (2.0, 5.0)

    def test_is_confidently_harmful_needs_min_events(self):
        rep = BetaBinomialReputation()
        thin = CardStatsBlock(posterior_a=1.0, posterior_b=3.0, intro_events=2)
        assert rep.is_confidently_harmful(thin) is False
        dire = CardStatsBlock(posterior_a=1.0, posterior_b=11.0, intro_events=10)
        assert rep.is_confidently_harmful(dire) is True

    def test_is_confidently_harmful_spares_good_cards(self):
        rep = BetaBinomialReputation()
        good = CardStatsBlock(posterior_a=9.0, posterior_b=2.0, intro_events=9)
        assert rep.is_confidently_harmful(good) is False
        assert rep.is_confidently_harmful(None) is False

    def test_losing_founding_only_card_is_cold_for_the_auction(
        self, make_card, make_event
    ):
        rep = BetaBinomialReputation()
        card = make_card(gain_events=(make_event(-0.4, founding=True),))
        assert rep.card_magnitude(card) is None
        assert rep.card_posterior(card) == rep.cold_prior
        assert rep.event_deltas(card) == ()

    def test_unused_and_invalid_exposures_are_zero_ev_support(
        self, make_card, make_event
    ):
        rep = BetaBinomialReputation()
        card = make_card(
            gain_events=(make_event(0.0, unused=True), make_event(0.0, invalid=True))
        )
        assert rep.event_deltas(card) == (0.0, 0.0)
        assert rep.card_magnitude(card) == 0.0

    def test_projections_are_pure_over_a_resolved_block(self, make_card, make_event):
        # posterior_of/magnitude_of read an already-resolved block so the
        # reader can resolve card_stats ONCE per candidate and reuse it for
        # the auction and the render; the card-level accessors are the same
        # math composed with card_stats.
        rep = BetaBinomialReputation()
        card = make_card(
            gain_events=(make_event(0.2), make_event(0.4), make_event(-5.0))
        )
        block = rep.card_stats(card)
        assert rep.posterior_of(block) == rep.card_posterior(card)
        assert rep.magnitude_of(block) == rep.card_magnitude(card)
        assert rep.posterior_of(None) == (3.0, 3.0)
        assert rep.magnitude_of(None) is None


class TestBDProximityReputation:
    def test_no_context_delegates_to_fallback(self, make_card, make_event):
        rep = BDProximityReputation(behavior_space=_bs())
        card = make_card(gain_events=(make_event(0.5, metrics={"x": 0.15}),))
        assert rep.card_stats(card, None) == rep.fallback.card_stats(card, None)

    def test_in_cell_partition(self, make_card, make_event):
        from gigaevo.memory.cards import DecisionContext

        rep = BDProximityReputation(behavior_space=_bs())
        card = make_card(
            gain_events=(
                make_event(0.5, metrics={"x": 0.11}),
                make_event(0.6, metrics={"x": 0.19}),
                make_event(-0.9, metrics={"x": 0.91}),
            )
        )
        near = DecisionContext(parent_metrics={"x": 0.15})
        block = rep.card_stats(card, near)
        assert block is not None
        assert block.intro_events == 2
        assert block.k_harm == 0
        assert block.IntroGain_best_median == pytest.approx(0.55)
        far = DecisionContext(parent_metrics={"x": 0.95})
        far_block = rep.card_stats(card, far)
        assert far_block is not None
        assert far_block.intro_events == 1
        assert far_block.IntroGain_best_median == pytest.approx(-0.9)

    def test_cold_cell_delegates_to_fallback(self, make_card, make_event):
        from gigaevo.memory.cards import DecisionContext

        rep = BDProximityReputation(behavior_space=_bs())
        card = make_card(gain_events=(make_event(0.5, metrics={"x": 0.15}),))
        elsewhere = DecisionContext(parent_metrics={"x": 0.55})
        assert rep.card_stats(card, elsewhere) == rep.fallback.card_stats(card, None)

    def test_founding_only_cell_delegates_to_global_use_evidence(
        self, make_card, make_event
    ):
        from gigaevo.memory.cards import DecisionContext

        rep = BDProximityReputation(behavior_space=_bs())
        card = make_card(
            gain_events=(
                make_event(-0.5, founding=True, metrics={"x": 0.15}),
                make_event(-0.9, metrics={"x": 0.95}),
            )
        )
        founding_cell = DecisionContext(parent_metrics={"x": 0.15})

        assert rep.card_stats(card, founding_cell) == rep.fallback.card_stats(
            card, None
        )
        assert rep.event_deltas(card, founding_cell) == rep.fallback.event_deltas(
            card, None
        )

    def test_unused_in_cell_zero_support_does_not_delegate(self, make_card, make_event):
        from gigaevo.memory.cards import DecisionContext

        rep = BDProximityReputation(behavior_space=_bs())
        card = make_card(
            gain_events=(
                make_event(0.0, unused=True, metrics={"x": 0.15}),
                make_event(0.9, metrics={"x": 0.95}),
            )
        )
        near = DecisionContext(parent_metrics={"x": 0.15})

        block = rep.card_stats(card, near)
        assert block is not None
        assert block.IntroGain_best_median == 0.0
        assert rep.event_deltas(card, near) == (0.0,)

    @pytest.mark.parametrize("bad", [nan, inf])
    def test_non_finite_parent_coord_delegates(self, make_card, make_event, bad):
        from gigaevo.memory.cards import DecisionContext

        rep = BDProximityReputation(behavior_space=_bs())
        card = make_card(gain_events=(make_event(0.5, metrics={"x": 0.05}),))
        ctx = DecisionContext(parent_metrics={"x": bad})
        assert rep.card_stats(card, ctx) == rep.fallback.card_stats(card, None)

    def test_missing_parent_coord_delegates(self, make_card, make_event):
        from gigaevo.memory.cards import DecisionContext

        rep = BDProximityReputation(behavior_space=_bs())
        card = make_card(gain_events=(make_event(0.5, metrics={"x": 0.05}),))
        ctx = DecisionContext(parent_metrics={"y": 0.05})
        assert rep.card_stats(card, ctx) == rep.fallback.card_stats(card, None)

    def test_non_finite_event_coord_excluded_from_cell(self, make_card, make_event):
        from gigaevo.memory.cards import DecisionContext

        rep = BDProximityReputation(behavior_space=_bs())
        card = make_card(
            gain_events=(
                make_event(0.5, metrics={"x": 0.05}),
                make_event(-9.0, metrics={"x": nan}),
            )
        )
        ctx = DecisionContext(parent_metrics={"x": 0.05})
        block = rep.card_stats(card, ctx)
        assert block is not None
        assert block.intro_events == 1
        assert block.k_harm == 0

    def test_in_cell_snapshots_bounds_against_a_live_reindex(
        self, make_card, make_event, monkeypatch
    ):
        from gigaevo.evolution.strategies.models import (
            DynamicBehaviorSpace,
            LinearBinning,
        )
        from gigaevo.memory.cards import DecisionContext

        bs = DynamicBehaviorSpace(
            bins={"x": LinearBinning(min_val=0.0, max_val=1.0, num_bins=10)}
        )
        rep = BDProximityReputation(behavior_space=bs)
        card = make_card(
            gain_events=(
                make_event(0.5, metrics={"x": 0.15}),
                make_event(-0.9, metrics={"x": 0.85}),
            )
        )
        ctx = DecisionContext(parent_metrics={"x": 0.15})

        real_get_cell = DynamicBehaviorSpace.get_cell
        calls = {"n": 0}

        def reindex_after_parent(self, metrics):
            cell = real_get_cell(self, metrics)
            calls["n"] += 1
            if calls["n"] == 1:
                # A concurrent DynamicBehaviorSpace reindex moves the live bounds
                # after the parent cell is read; x=0.15 would fall in a new cell.
                bs.bins["x"].max_val = 0.3
            return cell

        monkeypatch.setattr(DynamicBehaviorSpace, "get_cell", reindex_after_parent)

        block = rep.card_stats(card, ctx)
        assert block is not None
        # The read pins one tessellation: parent (x=0.15) and its co-located event
        # share a cell despite the mid-read bound change, so exactly the near
        # event is in-cell (a live re-bin would delegate all events to fallback).
        assert block.intro_events == 1
        assert block.IntroGain_best_median == pytest.approx(0.5)


class _Store:
    def __init__(self, cards=()) -> None:
        self._cards = tuple(cards)

    def snapshot(self):
        return self._cards


class TestBootstrapReputation:
    def test_one_positive_use_event_is_priced_but_not_confident(
        self, make_card, make_event
    ):
        card = make_card(gain_events=(make_event(0.2),))
        rep = BootstrapReputation(
            BetaBinomialReputation(),
            _Store((card,)),
            n_bootstrap=128,
            confident_min_events=3,
        )
        block = rep.card_stats(card)
        assert block is not None
        assert block.IntroGain_best_median == pytest.approx(0.2)
        assert block.IntroGain_bootstrap_ev_mean is not None
        assert block.IntroGain_bootstrap_ev_mean > 0.0
        assert block.IntroGain_bootstrap_ev_lo20 is not None
        assert block.IntroGain_bootstrap_ev_lo20 >= 0.0
        assert block.p_help_lo20 is not None
        assert 0.0 <= block.p_help_lo20 <= 1.0
        assert block.p_help_lo20 != pytest.approx(block.IntroGain_bootstrap_ev_lo20)
        assert block.efficacy_confident is False

    def test_bootstrap_ev_does_not_overwrite_observed_median(
        self, make_card, make_event
    ):
        card = make_card(
            gain_events=(make_event(1.0), make_event(1.0), make_event(-10.0))
        )
        rep = BootstrapReputation(
            BetaBinomialReputation(),
            _Store((card,)),
            n_bootstrap=4096,
            confident_min_events=3,
        )
        block = rep.card_stats(card)
        assert block is not None
        assert block.IntroGain_best_median == pytest.approx(1.0)
        assert block.IntroGain_bootstrap_ev_mean is not None
        assert block.IntroGain_bootstrap_ev_mean < 0.0
        assert block.IntroGain_bootstrap_ev_lo20 is not None
        assert block.IntroGain_bootstrap_ev_lo20 < 0.0
        assert rep.magnitude_of(block) == pytest.approx(
            block.IntroGain_bootstrap_ev_mean
        )
        assert block.efficacy_confident is False

    def test_known_loser_does_not_borrow_positive_bank_scale(
        self, make_card, make_event
    ):
        loser = make_card(id="loser", gain_events=(make_event(-1.0),))
        winner = make_card(id="winner", gain_events=(make_event(10.0),))
        rep = BootstrapReputation(
            BetaBinomialReputation(),
            _Store((loser, winner)),
            n_bootstrap=4096,
            confident_min_events=1,
        )

        block = rep.card_stats(loser)

        assert block is not None
        assert block.IntroGain_bootstrap_ev_mean is not None
        assert block.IntroGain_bootstrap_ev_mean <= 0.0
        assert block.IntroGain_bootstrap_ev_lo20 is not None
        assert block.IntroGain_bootstrap_ev_lo20 <= 0.0
