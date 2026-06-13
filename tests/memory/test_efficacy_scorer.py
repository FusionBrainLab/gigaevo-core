"""EfficacyScorer: the single owner of gain -> posterior math.

Pins the consolidation contract: one fitted cohort (child-deduplicated
baseline + noise band) serves both the card-side injection posterior and the
idea-side origin aggregation, parameterized from ``BetaBinomialReputation``.
"""

from __future__ import annotations

import pytest

from gigaevo.memory.core.reputation import BetaBinomialReputation
from gigaevo.memory.efficacy import (
    EfficacyScorer,
    GainObservation,
    beta_binomial_posterior,
)
from gigaevo.memory.shared_memory.injection_posterior import (
    InjectionOutcome,
    compute_injection_posterior,
)


def obs(child_id: str, parent_fitness: float, gain: float) -> GainObservation:
    return GainObservation(child_id=child_id, parent_fitness=parent_fitness, gain=gain)


class TestFittedCohort:
    def test_duplicate_child_observations_weigh_once(self):
        base = [
            obs("c1", 0.5, 0.10),
            obs("c2", 0.5, -0.30),
            obs("c3", 0.5, 0.02),
            obs("c4", 0.5, 0.04),
        ]
        duplicated = [*base, obs("c2", 0.5, -0.30), obs("c2", 0.5, -0.30)]

        fit_base = EfficacyScorer().fit(base)
        fit_dup = EfficacyScorer().fit(duplicated)

        assert fit_dup.epsilon == fit_base.epsilon
        events = [obs("c9", 0.5, -0.05)]
        assert fit_dup.posterior(events) == fit_base.posterior(events)

    def test_duplicate_weighting_would_have_shifted_the_band(self):
        base = [
            obs("c1", 0.5, 0.10),
            obs("c2", 0.5, -0.30),
            obs("c3", 0.5, 0.02),
            obs("c4", 0.5, 0.04),
        ]
        triple_outlier = [
            *base,
            obs("c2-bis", 0.5, -0.30),
            obs("c2-ter", 0.5, -0.30),
        ]
        assert (
            EfficacyScorer().fit(triple_outlier).epsilon
            != EfficacyScorer().fit(base).epsilon
        )

    def test_empty_cohort_scores_raw_gains_at_zero_threshold(self):
        fitted = EfficacyScorer().fit([])
        assert fitted.epsilon == 0.0
        block = fitted.posterior([obs("c1", 0.5, -0.01), obs("c2", 0.5, 0.02)])
        assert block.intro_events == 2
        assert block.k_harm == 1

    def test_posterior_adjusts_by_parent_local_baseline(self):
        cohort = [obs(f"lo{i}", 0.0, 0.50) for i in range(20)] + [
            obs(f"hi{i}", 1.0, 0.00) for i in range(20)
        ]
        fitted = EfficacyScorer(baseline_neighbors=5, noise_band_k=0.0).fit(cohort)
        helped_low = fitted.posterior([obs("x", 0.0, 0.60)])
        harmed_low = fitted.posterior([obs("y", 0.0, 0.40)])
        assert helped_low.k_harm == 0
        assert harmed_low.k_harm == 1
        helped_high = fitted.posterior([obs("z", 1.0, 0.10)])
        assert helped_high.k_harm == 0

    def test_noise_band_k_scales_the_dead_band(self):
        cohort = [
            obs("c1", 0.5, 0.10),
            obs("c2", 0.5, -0.10),
            obs("c3", 0.5, 0.05),
            obs("c4", 0.5, -0.05),
            obs("c5", 0.5, 0.0),
        ]
        narrow = EfficacyScorer(noise_band_k=0.0, noise_floor_rel=0.0).fit(cohort)
        wide = EfficacyScorer(noise_band_k=10.0).fit(cohort)
        assert narrow.epsilon == 0.0
        assert wide.epsilon > 0.0
        dip = [obs("d", 0.5, -0.04)]
        assert narrow.posterior(dip).k_harm == 1
        assert wide.posterior(dip).k_harm == 0

    def test_confidence_quantiles_come_from_the_scorer(self):
        gains = [obs(f"c{i}", 0.5, 0.1) for i in range(10)]
        strict = EfficacyScorer(confident_quantile=0.20, confident_threshold=0.99)
        lax = EfficacyScorer(confident_quantile=0.20, confident_threshold=0.5)
        assert strict.fit([]).posterior(gains).efficacy_confident is False
        assert lax.fit([]).posterior(gains).efficacy_confident is True


class TestNoiseFloorRel:
    def test_floor_scales_with_parent_fitness_magnitude(self):
        cohort = [obs("c1", 0.5, 0.0), obs("c2", 0.5, 0.0), obs("c3", 0.5, 0.0)]
        fitted = EfficacyScorer(noise_band_k=0.0, noise_floor_rel=1e-4).fit(cohort)
        assert fitted.epsilon == pytest.approx(1e-4 * 0.5)

    def test_floor_does_not_shrink_a_healthy_mad_band(self):
        cohort = [
            obs("c1", 0.5, 0.10),
            obs("c2", 0.5, -0.10),
            obs("c3", 0.5, 0.05),
            obs("c4", 0.5, -0.05),
            obs("c5", 0.5, 0.0),
        ]
        with_floor = EfficacyScorer(noise_band_k=1.0, noise_floor_rel=1e-4).fit(cohort)
        without_floor = EfficacyScorer(noise_band_k=1.0, noise_floor_rel=0.0).fit(
            cohort
        )
        assert with_floor.epsilon == without_floor.epsilon

    def test_empty_cohort_floor_is_zero(self):
        assert EfficacyScorer(noise_floor_rel=1e-4).fit([]).epsilon == 0.0


class TestInvalidObservations:
    def test_invalid_observations_excluded_from_cohort(self):
        base = [
            obs("c1", 0.5, 0.10),
            obs("c2", 0.5, -0.30),
            obs("c3", 0.5, 0.02),
            obs("c4", 0.5, 0.04),
        ]
        invalid = GainObservation(
            child_id="bad", parent_fitness=0.5, gain=0.0, invalid=True
        )
        assert EfficacyScorer().fit([*base, invalid]).epsilon == (
            EfficacyScorer().fit(base).epsilon
        )

    def test_invalid_event_is_forced_harm_in_posterior(self):
        fitted = EfficacyScorer().fit([])
        invalid = GainObservation(
            child_id="bad", parent_fitness=0.5, gain=0.0, invalid=True
        )
        block = fitted.posterior([obs("c1", 0.5, 0.10), invalid])
        assert block.intro_events == 2
        assert block.k_harm == 1

    def test_posterior_primitive_invalid_events_param(self):
        block = beta_binomial_posterior([0.2, 0.3], invalid_events=2)
        assert block.intro_events == 4
        assert block.k_harm == 2
        assert block.posterior_a == pytest.approx(3.0)
        assert block.posterior_b == pytest.approx(3.0)


class TestLeaveOneOutBaseline:
    def test_cohort_member_baseline_excludes_itself(self):
        cohort = [
            obs("c1", 0.5, 0.0),
            obs("c2", 0.5, 0.0),
            obs("c3", 0.5, -0.30),
        ]
        fitted = EfficacyScorer(noise_band_k=0.0, noise_floor_rel=0.0).fit(cohort)
        # Self-inclusion would pull c3's baseline toward its own -0.30.
        assert fitted.baseline(0.5, exclude_child_id="c3") == pytest.approx(0.0)
        assert fitted.adjusted_gain(cohort[2]) == pytest.approx(-0.30)

    def test_external_event_id_excludes_nothing(self):
        cohort = [
            obs("c1", 0.5, 0.10),
            obs("c2", 0.5, 0.20),
        ]
        fitted = EfficacyScorer(noise_band_k=0.0, noise_floor_rel=0.0).fit(cohort)
        assert fitted.baseline(0.5, exclude_child_id="elsewhere") == pytest.approx(0.15)


class TestReputationBinding:
    def test_scorer_carries_reputation_thresholds(self):
        rep = BetaBinomialReputation(
            baseline_neighbors=7,
            noise_band_k=2.5,
            noise_floor_rel=5e-4,
            confident_quantile=0.10,
            confident_threshold=0.6,
        )
        scorer = rep.scorer()
        assert scorer == EfficacyScorer(
            baseline_neighbors=7,
            noise_band_k=2.5,
            noise_floor_rel=5e-4,
            confident_quantile=0.10,
            confident_threshold=0.6,
        )

    def test_default_scorer_matches_default_reputation(self):
        assert BetaBinomialReputation().scorer() == EfficacyScorer()


class TestCardSideEquivalence:
    """The injection posterior is the scorer applied to per-child outcomes —
    consolidation must not move a single number on the card side."""

    PROGRAMS = [
        InjectionOutcome(id="p1", fitness=0.5),
        InjectionOutcome(id="p2", fitness=0.6, parents=["p1"], selected_ids=["c1"]),
        InjectionOutcome(id="p3", fitness=0.4, parents=["p1"], selected_ids=["c2"]),
        InjectionOutcome(id="p4", fitness=0.7, parents=["p2"], selected_ids=["c1"]),
        InjectionOutcome(id="p5", fitness=0.3, parents=["p3"], selected_ids=["c2"]),
    ]

    def test_scorer_reproduces_injection_posterior(self):
        via_module = compute_injection_posterior(self.PROGRAMS)
        via_reputation = BetaBinomialReputation().compute_injection_posteriors(
            self.PROGRAMS
        )
        assert via_module == via_reputation
        assert set(via_module) == {"c1", "c2"}

    def test_scorer_parameters_reach_the_card_side(self):
        default = compute_injection_posterior(self.PROGRAMS)
        no_band = compute_injection_posterior(
            self.PROGRAMS, scorer=EfficacyScorer(noise_band_k=0.0)
        )
        assert set(default) == set(no_band)


class TestPosteriorPrimitive:
    def test_counts_and_shape(self):
        block = beta_binomial_posterior([0.2, -0.1, 0.3], threshold=0.0)
        assert block.intro_events == 3
        assert block.k_harm == 1
        assert block.posterior_a == pytest.approx(3.0)
        assert block.posterior_b == pytest.approx(2.0)

    def test_non_finite_gains_are_dropped(self):
        block = beta_binomial_posterior([float("nan"), 0.1, float("inf")])
        assert block.intro_events == 1
