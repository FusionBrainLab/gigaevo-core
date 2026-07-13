"""Reputation math: downside posterior, event blocks, BD partition."""

from __future__ import annotations

from math import inf, nan

import pytest
from scipy.stats import beta, norm

from gigaevo.evolution.strategies.models import BehaviorSpace, LinearBinning
from gigaevo.memory.cards import (
    CardStatsBlock,
    CausalStrength,
    DecisionContext,
    EvidenceAttribution,
    EvidenceSource,
)
from gigaevo.memory.context.evidence import harm_mass
from gigaevo.memory.read.bootstrap import bootstrap_ev_samples, stable_rng
from gigaevo.memory.read.interfaces import ReputationModel
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


def _for_task(event, task_key: str):
    return event.model_copy(
        update={"context": event.context.model_copy(update={"task_key": task_key})}
    )


class _StubBetaPrior:
    source = "stub_cohort"

    def __init__(self, alpha: float, beta: float) -> None:
        self._parameters = (alpha, beta)

    def as_tuple(self) -> tuple[float, float]:
        return self._parameters


class _StubCohortPrior:
    def __init__(self, alpha: float, beta: float) -> None:
        self._prior = _StubBetaPrior(alpha, beta)

    def cold_card_prior(self, card, context=None):
        del card, context
        return self._prior


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

    def test_per_event_staleness_scales_valid_invalid_and_unused_mass(self, make_event):
        events = (
            make_event(0.5),
            make_event(0.0, invalid=True),
            make_event(0.0, unused=True),
        )

        block = block_from_events(events, staleness_weights=(0.25, 0.5, 0.125))

        assert block is not None
        assert block.intro_events == pytest.approx(0.875)
        assert block.k_harm == pytest.approx(0.625)
        assert block.posterior_a == pytest.approx(1.25)
        assert block.posterior_b == pytest.approx(1.625)
        assert block.IntroGain_best_median is None


class TestBetaBinomialReputation:
    def test_one_win_cannot_lower_help_below_its_cohort_prior(
        self, make_card, make_event
    ):
        prior_parameters = (4.0, 1.0)
        card = make_card(gain_events=(make_event(0.5),))

        coherent = BetaBinomialReputation(
            prior=_StubCohortPrior(*prior_parameters)
        ).card_stats(card)
        legacy = BetaBinomialReputation().card_stats(card)

        assert coherent is not None and legacy is not None
        assert coherent.p_help_mean >= prior_parameters[0] / sum(prior_parameters)
        assert (coherent.posterior_a, coherent.posterior_b) == (5.0, 1.0)
        assert (legacy.posterior_a, legacy.posterior_b) == (2.0, 1.0)
        assert legacy.p_help_mean == pytest.approx(2.0 / 3.0)

    def test_card_stats_keeps_native_magnitude_and_folds_foreign_sign_only(
        self, make_card, make_event
    ):
        native = tuple(_for_task(make_event(g), "task-a") for g in (0.2, 0.4))
        foreign = (
            _for_task(make_event(1000.0), "task-b"),
            _for_task(make_event(-1000.0), "task-b"),
            _for_task(make_event(0.0, invalid=True), "task-b"),
            _for_task(make_event(0.0, unused=True), "task-b"),
        )
        block = BetaBinomialReputation().card_stats(
            make_card(gain_events=(*native, *foreign)),
            DecisionContext(task_key="task-a"),
        )

        assert block is not None
        assert block.IntroGain_best_median == pytest.approx(0.3)
        assert block.intro_events == 2
        assert block.k_harm == 0
        assert (block.posterior_a, block.posterior_b) == (4.0, 3.0)
        assert block.foreign_help_events == 1
        assert block.foreign_total_events == 3

    def test_foreign_only_card_has_sign_posterior_but_no_native_support(
        self, make_card, make_event
    ):
        card = make_card(
            gain_events=(
                _for_task(make_event(99.0), "task-b"),
                _for_task(make_event(-99.0), "task-b"),
            )
        )

        block = BetaBinomialReputation().card_stats(
            card, DecisionContext(task_key="task-a")
        )

        assert block is not None
        assert block.intro_events == 0
        assert block.IntroGain_best_median is None
        assert (block.posterior_a, block.posterior_b) == (2.0, 2.0)
        assert block.foreign_help_events == 1
        assert block.foreign_total_events == 2

    def test_foreign_sign_fold_ignores_magnitude_and_uncertainty(
        self, make_card, make_event
    ):
        variants = (
            _for_task(make_event(1e-300, gain_se=1e300), "task-b"),
            _for_task(make_event(1e300, gain_se=0.0), "task-b"),
        )
        blocks = [
            BetaBinomialReputation().card_stats(
                make_card(gain_events=(foreign,)),
                DecisionContext(task_key="task-a"),
            )
            for foreign in variants
        ]

        assert all(block is not None for block in blocks)
        assert [
            (
                block.foreign_help_events,
                block.foreign_total_events,
                block.posterior_a,
                block.posterior_b,
            )
            for block in blocks
            if block is not None
        ] == [(1.0, 1.0, 2.0, 1.0)] * 2

    def test_task_partition_is_byte_identical_when_all_events_are_native(
        self, make_card, make_event
    ):
        events = (make_event(0.2), make_event(-0.1), make_event(0.0, unused=True))
        card = make_card(gain_events=events)

        expected = block_from_events(events)
        actual = BetaBinomialReputation().card_stats(card, DecisionContext(task_key=""))

        assert actual is not None and expected is not None
        assert actual.model_dump_json() == expected.model_dump_json()

    def test_bootstrap_support_excludes_foreign_magnitudes_weights_and_ses(
        self, make_card, make_event
    ):
        native = _for_task(make_event(0.2, gain_se=0.03), "task-a")
        foreign = _for_task(make_event(900.0, gain_se=80.0), "task-b")
        card = make_card(gain_events=(native, foreign))
        context = DecisionContext(task_key="task-a")
        rep = BetaBinomialReputation()

        assert rep.event_deltas(card, context) == (0.2,)
        assert rep.event_weights(card, context) == (1.0,)
        assert rep.evidence_events(card, context) == (native,)
        assert rep.event_ses(card, context) == (0.03,)

    def test_base_staleness_weights_align_with_ev_events_and_credit_stays_pure(
        self, make_card, make_event
    ):
        credited = make_event(0.2).model_copy(
            update={
                "attribution": EvidenceAttribution(
                    source=EvidenceSource.DIRECT,
                    causal_strength=CausalStrength.DIRECT_BUNDLED,
                    used_card_count=2,
                    credit_weight=0.5,
                )
            }
        )
        card = make_card(
            gain_events=(
                credited,
                make_event(-0.1),
                make_event(10.0, founding=True),
            )
        )
        rep = BetaBinomialReputation()

        assert isinstance(rep, ReputationModel)
        assert rep.staleness_weights(card) == (1.0, 1.0)
        assert rep.event_weights(card) == (0.5, 1.0)

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
    def test_in_cell_and_fallback_share_the_same_card_prior(
        self, make_card, make_event
    ):
        prior_parameters = (4.0, 1.0)
        rep = BDProximityReputation(
            behavior_space=_bs(),
            prior=_StubCohortPrior(*prior_parameters),
        )
        card = make_card(gain_events=(make_event(0.5, metrics={"x": 0.15}),))

        in_cell = rep.card_stats(card, DecisionContext(parent_metrics={"x": 0.15}))
        fallback = rep.card_stats(card, None)

        assert in_cell is not None and fallback is not None
        assert (in_cell.posterior_a, in_cell.posterior_b) == (5.0, 1.0)
        assert (fallback.posterior_a, fallback.posterior_b) == (5.0, 1.0)

    def test_foreign_events_are_not_bucketed_by_parent_metrics(
        self, make_card, make_event
    ):
        native = _for_task(make_event(0.2, metrics={"x": 0.15}), "task-a")
        foreign = _for_task(make_event(999.0, metrics={"x": 0.15}), "task-b")
        card = make_card(gain_events=(native, foreign))
        context = DecisionContext(task_key="task-a", parent_metrics={"x": 0.15})
        rep = BDProximityReputation(behavior_space=_bs())

        block = rep.card_stats(card, context)

        assert block is not None
        assert block.intro_events == 1
        assert block.IntroGain_best_median == pytest.approx(0.2)
        assert block.foreign_help_events == 1
        assert block.foreign_total_events == 1
        assert rep.event_deltas(card, context) == (0.2,)

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
    def test_all_unit_no_se_bootstrap_replays_the_legacy_draws_exactly(
        self, make_card, make_event, monkeypatch
    ):
        card = make_card(
            id="unit-weight-replay",
            gain_events=(make_event(0.5), make_event(-0.25), make_event(0.1)),
        )
        n_bootstrap = 256
        rep = BootstrapReputation(
            BetaBinomialReputation(),
            _Store((card,)),
            n_bootstrap=n_bootstrap,
        )

        expected = bootstrap_ev_samples(
            rep.event_deltas(card),
            0.0,
            1.0,
            n_bootstrap,
            stable_rng(card.id, len(rep.event_deltas(card)), n_bootstrap),
            delta_weights=rep.event_weights(card),
            ses=rep.event_ses(card),
        )
        captured = {}

        def recording_bootstrap(*args, **kwargs):
            samples = bootstrap_ev_samples(*args, **kwargs)
            captured["samples"] = samples
            return samples

        monkeypatch.setattr(
            "gigaevo.memory.read.reputation.bootstrap_ev_samples",
            recording_bootstrap,
        )
        block = rep.card_stats(card)

        assert block is not None
        assert (captured["samples"] == expected).all()
        assert block.IntroGain_bootstrap_ev_mean == float(expected.mean())

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


class TestSoftHarmMass:
    @pytest.mark.parametrize("gain", [-0.1, 0.1])
    def test_unknown_se_is_uninformative_on_either_side_of_threshold(self, gain):
        assert harm_mass(gain, None, 0.0) == pytest.approx(0.5)

    def test_exact_and_measured_se_regimes_are_preserved(self):
        assert harm_mass(-0.1, 0.0, 0.0) == 1.0
        assert harm_mass(0.1, 0.0, 0.0) == 0.0
        assert harm_mass(-0.1, 0.2, 0.0) == pytest.approx(float(norm.cdf(0.5)))

    def test_degraded_event_contributes_half_its_credit_to_harm(self, make_event):
        weight = 0.25
        degraded = make_event(-0.1, gain_se=None).model_copy(
            update={"attribution": EvidenceAttribution(credit_weight=weight)}
        )
        exact = degraded.model_copy(update={"gain_se": 0.0})

        degraded_block = block_from_events([degraded])
        exact_block = block_from_events([exact])

        assert degraded_block is not None and exact_block is not None
        assert degraded_block.k_harm == pytest.approx(0.5 * weight)
        assert exact_block.k_harm == pytest.approx(weight)

    def test_zero_ses_match_omitted_bit_exact(self):
        gains = [0.5, -0.2, 0.0, -0.1]
        assert beta_binomial_posterior(gains, event_ses=[0.0] * 4) == (
            beta_binomial_posterior(gains)
        )

    def test_noisy_events_accrue_fractional_harm(self):
        gains = [0.5, -0.2, 0.0, -0.1]
        strict = beta_binomial_posterior(gains)
        soft = beta_binomial_posterior(gains, event_ses=[0.05] * 4)
        assert strict.k_harm == 2
        assert soft.k_harm != int(soft.k_harm)
        assert 2.0 < soft.k_harm < 3.0

    def test_exact_event_at_threshold_is_not_harm(self):
        assert beta_binomial_posterior([0.0], event_ses=[0.0]).k_harm == 0

    def test_noisy_event_at_threshold_is_half_harm(self):
        block = beta_binomial_posterior([0.0], event_ses=[0.5])
        assert block.k_harm == pytest.approx(0.5)

    def test_non_finite_se_degrades_to_exact(self):
        noisy = beta_binomial_posterior([-0.1, 0.1], event_ses=[nan, inf])
        assert noisy == beta_binomial_posterior([-0.1, 0.1])

    def test_ses_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="event_ses length"):
            beta_binomial_posterior([0.1, 0.2], event_ses=[0.1])

    def test_block_from_events_reads_stamped_ses(self, make_event):
        strict = block_from_events([make_event(-0.1), make_event(0.5)])
        soft = block_from_events([make_event(-0.1, gain_se=0.05), make_event(0.5)])
        assert strict is not None and soft is not None
        assert strict.k_harm == 1
        assert 0.9 < soft.k_harm < 1.0


class TestEventSes:
    def test_aligned_with_event_deltas(self, make_card, make_event):
        rep = BetaBinomialReputation()
        card = make_card(
            gain_events=(
                make_event(0.3, gain_se=0.1),
                make_event(-0.2),
                make_event(0.0, invalid=True),
                make_event(0.0, unused=True),
                make_event(0.9, founding=True),
            )
        )
        deltas = rep.event_deltas(card)
        ses = rep.event_ses(card)
        assert len(ses) == len(deltas) == 4
        assert ses == (0.1, 0.0, 0.0, 0.0)

    def test_unknown_stored_se_is_preserved_and_bootstrap_safe(
        self, make_card, make_event
    ):
        card = make_card(gain_events=(make_event(-0.1, gain_se=None),))
        rep = BetaBinomialReputation()

        assert rep.event_ses(card) == (None,)
        assert (
            BootstrapReputation(
                rep, _Store((card,)), n_bootstrap=32, confident_min_events=1
            ).card_stats(card)
            is not None
        )

    def test_infinite_stored_se_degrades_to_exact(self, make_card, make_event):
        rep = BetaBinomialReputation()
        card = make_card(gain_events=(make_event(0.3, gain_se=float("inf")),))
        assert rep.event_ses(card) == (0.0,)

    def test_bd_proximity_partitions_ses_with_deltas(self, make_card, make_event):
        from gigaevo.memory.cards import DecisionContext

        rep = BDProximityReputation(behavior_space=_bs())
        card = make_card(
            gain_events=(
                make_event(0.5, gain_se=0.2, metrics={"x": 0.11}),
                make_event(-0.9, metrics={"x": 0.91}),
            )
        )
        near = DecisionContext(parent_metrics={"x": 0.15})
        assert rep.event_ses(card, near) == (0.2,)
        assert rep.event_ses(card, None) == rep.fallback.event_ses(card, None)
        assert rep.event_ses(card, None) == (0.2, 0.0)

    def test_bootstrap_reputation_delegates_to_inner(self, make_card, make_event):
        card = make_card(gain_events=(make_event(0.2, gain_se=0.1),))
        rep = BootstrapReputation(
            BetaBinomialReputation(),
            _Store((card,)),
            n_bootstrap=256,
            confident_min_events=3,
        )
        assert rep.event_ses(card) == (0.1,)

    def test_priced_noise_widens_the_ev_distribution(self, make_card, make_event):
        # Same card id -> same stable rng, so the noisy stack differs from the
        # exact one only by the jitter it prices in.
        def block(card):
            rep = BootstrapReputation(
                BetaBinomialReputation(),
                _Store((card,)),
                n_bootstrap=4096,
                confident_min_events=3,
            )
            return rep.card_stats(card)

        exact = block(
            make_card(id="fixed", gain_events=(make_event(0.2), make_event(0.1)))
        )
        noisy = block(
            make_card(
                id="fixed",
                gain_events=(
                    make_event(0.2, gain_se=0.5),
                    make_event(0.1, gain_se=0.5),
                ),
            )
        )
        assert exact is not None and noisy is not None
        assert noisy.IntroGain_bootstrap_ev_lo20 < exact.IntroGain_bootstrap_ev_lo20
        assert noisy.IntroGain_bootstrap_ev_hi80 > exact.IntroGain_bootstrap_ev_hi80
        assert noisy.IntroGain_bootstrap_ev_mean == pytest.approx(
            exact.IntroGain_bootstrap_ev_mean, abs=0.05
        )
