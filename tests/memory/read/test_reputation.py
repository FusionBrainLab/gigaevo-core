"""Reputation math: noise band, downside posterior, event blocks, BD partition."""

from __future__ import annotations

from math import inf, nan

import pytest
from scipy.stats import beta

from gigaevo.evolution.strategies.models import BehaviorSpace, LinearBinning
from gigaevo.memory.cards import CardStatsBlock
from gigaevo.memory.read.reputation import (
    BDProximityReputation,
    BetaBinomialReputation,
    beta_binomial_posterior,
    block_from_events,
    robust_noise_band,
)


def _bs(num_bins: int = 10, max_val: float = 1.0) -> BehaviorSpace:
    return BehaviorSpace(
        bins={"x": LinearBinning(min_val=0.0, max_val=max_val, num_bins=num_bins)}
    )


class TestRobustNoiseBand:
    def test_empty_is_zero(self):
        assert robust_noise_band([]) == 0.0

    def test_flat_set_collapses_to_zero(self):
        assert robust_noise_band([0.5, 0.5, 0.5]) == 0.0

    def test_mad_to_sigma_scaling(self):
        assert robust_noise_band([1.0, 2.0, 3.0]) == pytest.approx(1.4826)

    def test_outlier_robust(self):
        assert robust_noise_band([1.0, 2.0, 3.0, 1000.0]) == pytest.approx(
            1.4826, rel=0.5
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

    def test_harm_uses_mad_noise_band(self, make_event):
        events = [make_event(g) for g in (0.10, 0.11, 0.09, -0.05)]
        block = block_from_events(events)
        assert block is not None
        assert block.k_harm == 1

    def test_within_band_dip_is_not_harm(self, make_event):
        events = [make_event(g) for g in (1.0, 2.0, 3.0, -0.5)]
        block = block_from_events(events)
        assert block is not None
        assert block.k_harm == 0

    def test_invalid_events_forced_harm_without_magnitude(self, make_event):
        block = block_from_events([make_event(0.5), make_event(0.0, invalid=True)])
        assert block is not None
        assert block.intro_events == 2
        assert block.k_harm == 1
        assert block.IntroGain_best_median == pytest.approx(0.5)

    def test_all_invalid_has_zero_magnitude(self, make_event):
        block = block_from_events([make_event(0.0, invalid=True)])
        assert block is not None
        assert block.IntroGain_best_median == 0.0
        assert block.k_harm == 1


class TestBetaBinomialReputation:
    def test_cold_card_gets_cold_prior(self, make_card):
        rep = BetaBinomialReputation()
        card = make_card()
        assert rep.card_stats(card) is None
        assert rep.card_posterior(card) == (1.0, 1.0)
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
