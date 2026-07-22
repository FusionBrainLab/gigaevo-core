"""Empirical-Bayes tuning of the reward prior scales (memory-v2 item 3)."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from pydantic import ValidationError
import pytest

from gigaevo.memory_v2.posterior import (
    BayesianResidualScaleGaussianRegressor,
    EmpiricalBayesRewardPriorEstimator,
    HierarchicalTerminalUtilityPosterior,
    TerminalUtilityPosteriorConfig,
)

from .conftest import synthetic_observations

_TUNED_PARAMETERS = (
    "baseline_prior_sd",
    "shared_effect_prior_sd",
    "card_effect_prior_sd",
    "reward_residual_sd_initial",
    "reward_residual_log_prior_sd",
)


@dataclass(frozen=True)
class _TwoBlockFeatureSpace:
    """Intercept + one card-effect column — exercises every tuned scale."""

    outcome_dim: int = 2

    @staticmethod
    def prior_variance(
        *, baseline_sd: float, shared_effect_sd: float, card_effect_sd: float
    ) -> np.ndarray:
        return np.asarray([baseline_sd**2, card_effect_sd**2], dtype=float)


def _reward_sample(
    rng: np.random.Generator,
    size: int,
    *,
    intercept: float,
    card_effect: float,
    noise_sd: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    card_indicator = rng.integers(0, 2, size=size).astype(float)
    design = np.column_stack([np.ones(size), card_indicator])
    values = (
        intercept + card_effect * card_indicator + rng.normal(0.0, noise_sd, size=size)
    )
    return design, values, np.zeros(size)


def _scales(config: TerminalUtilityPosteriorConfig) -> dict[str, float]:
    return {name: getattr(config, name) for name in _TUNED_PARAMETERS}


def _log_marginal(
    config: TerminalUtilityPosteriorConfig,
    design: np.ndarray,
    values: np.ndarray,
    measurement_sd: np.ndarray,
    space: _TwoBlockFeatureSpace,
) -> float:
    return (
        BayesianResidualScaleGaussianRegressor(config)
        .fit(design, values, measurement_sd, space)
        .log_marginal
    )


def test_hyperparameter_estimation_defaults_to_fixed() -> None:
    assert TerminalUtilityPosteriorConfig().hyperparameter_estimation == "fixed"


def test_empirical_bayes_is_a_noop_on_a_cold_bank() -> None:
    config = TerminalUtilityPosteriorConfig(hyperparameter_estimation="empirical_bayes")
    rng = np.random.default_rng(0)
    design, values, measurement_sd = _reward_sample(
        rng, 5, intercept=0.1, card_effect=0.2, noise_sd=0.2
    )

    tuned = EmpiricalBayesRewardPriorEstimator(config).estimate(
        design, values, measurement_sd, _TwoBlockFeatureSpace()
    )

    assert _scales(tuned) == _scales(config)


def test_empirical_bayes_never_decreases_the_log_marginal() -> None:
    config = TerminalUtilityPosteriorConfig(hyperparameter_estimation="empirical_bayes")
    rng = np.random.default_rng(1)
    design, values, measurement_sd = _reward_sample(
        rng, 120, intercept=0.3, card_effect=0.5, noise_sd=0.6
    )
    space = _TwoBlockFeatureSpace()

    tuned = EmpiricalBayesRewardPriorEstimator(config).estimate(
        design, values, measurement_sd, space
    )

    base_marginal = _log_marginal(config, design, values, measurement_sd, space)
    tuned_marginal = _log_marginal(tuned, design, values, measurement_sd, space)
    assert tuned_marginal >= base_marginal - 1e-6
    assert _scales(tuned) != _scales(config)


@pytest.mark.parametrize("bad_scale", [0.0, -1.0, math.inf, math.nan])
def test_with_scales_rejects_nonpositive_and_nonfinite_scales(bad_scale: float) -> None:
    config = TerminalUtilityPosteriorConfig(hyperparameter_estimation="empirical_bayes")
    estimator = EmpiricalBayesRewardPriorEstimator(config)
    scales = np.array([0.75, 0.35, 0.25, 0.20, 0.75])
    scales[3] = bad_scale  # reward_residual_sd_initial

    with pytest.raises(ValidationError):
        estimator._with_scales(scales)


def test_empirical_bayes_returns_only_finite_positive_in_bounds_scales() -> None:
    config = TerminalUtilityPosteriorConfig(hyperparameter_estimation="empirical_bayes")
    rng = np.random.default_rng(7)
    design, values, measurement_sd = _reward_sample(
        rng, 160, intercept=0.2, card_effect=0.1, noise_sd=0.9
    )

    tuned = EmpiricalBayesRewardPriorEstimator(config).estimate(
        design, values, measurement_sd, _TwoBlockFeatureSpace()
    )

    for name in _TUNED_PARAMETERS:
        scale = getattr(tuned, name)
        assert math.isfinite(scale) and scale > 0.0
    lower, upper = config.reward_residual_sd_bounds
    assert lower <= tuned.reward_residual_sd_initial <= upper


def test_empirical_bayes_lifts_the_residual_prior_for_noisy_tasks() -> None:
    config = TerminalUtilityPosteriorConfig(hyperparameter_estimation="empirical_bayes")
    rng = np.random.default_rng(2)
    design, values, measurement_sd = _reward_sample(
        rng, 160, intercept=0.2, card_effect=0.1, noise_sd=0.9
    )

    tuned = EmpiricalBayesRewardPriorEstimator(config).estimate(
        design, values, measurement_sd, _TwoBlockFeatureSpace()
    )

    # Realized residual noise (0.9) dwarfs the warm-start residual centre (0.20),
    # so the marginal-likelihood optimum lifts the residual prior centre.
    assert tuned.reward_residual_sd_initial > config.reward_residual_sd_initial


def test_fixed_estimation_leaves_the_live_reward_scale_untouched(
    posterior_model: HierarchicalTerminalUtilityPosterior,
    evolution_context,
    revisions,
) -> None:
    observations = synthetic_observations(evolution_context, revisions, per_arm=30)

    fitted = posterior_model.fit(observations, revisions)

    assert fitted.reward.card_effect_sd == posterior_model.config.card_effect_prior_sd


def test_empirical_bayes_reaches_the_live_reward_head(
    posterior_model: HierarchicalTerminalUtilityPosterior,
    evolution_context,
    revisions,
) -> None:
    eb_config = posterior_model.config.model_copy(
        update={"hyperparameter_estimation": "empirical_bayes"}
    )
    eb_model = HierarchicalTerminalUtilityPosterior(
        feature_map=posterior_model.feature_map, config=eb_config
    )
    observations = synthetic_observations(evolution_context, revisions, per_arm=30)

    fitted = eb_model.fit(observations, revisions)

    valid = sum(1 for row in observations if not row.invalid)
    assert fitted.reward.observations == valid
    # The tuned scale, not the frozen constant, reaches the fitted posterior.
    assert fitted.reward.card_effect_sd != eb_config.card_effect_prior_sd


def test_empirical_bayes_leaves_the_lineage_head_on_warm_start_scales(
    posterior_model: HierarchicalTerminalUtilityPosterior,
    evolution_context,
    revisions,
) -> None:
    eb_config = posterior_model.config.model_copy(
        update={"hyperparameter_estimation": "empirical_bayes"}
    )
    eb_model = HierarchicalTerminalUtilityPosterior(
        feature_map=posterior_model.feature_map, config=eb_config
    )
    observations = synthetic_observations(evolution_context, revisions, per_arm=30)

    fitted = eb_model.fit(observations, revisions)

    # EB tunes the primary reward head but the lineage head keeps warm-start
    # scales — its residuals are a different (unclipped) quantity.
    assert fitted.reward.card_effect_sd != eb_config.card_effect_prior_sd
    assert fitted.lineage_reward.card_effect_sd == eb_config.card_effect_prior_sd
