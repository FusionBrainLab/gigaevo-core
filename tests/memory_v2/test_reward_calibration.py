from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import norm

from gigaevo.memory.cards import Card
from gigaevo.memory_v2.calibration import (
    CalibrationDecision,
    CalibrationTrajectory,
    SafetyEnvironmentKey,
)
from gigaevo.memory_v2.features import EmbeddingPriorConfig
from gigaevo.memory_v2.models import (
    ApplicabilityRecord,
    ApplicabilitySpecification,
    CardSnapshot,
    CausalObservation,
    EvolutionContext,
    OutcomeMeasurement,
    PolicySpecification,
)
from gigaevo.memory_v2.posterior import (
    GaussianPosterior,
    GaussianPosteriorComponent,
    TerminalUtilityPosteriorConfig,
)
from gigaevo.memory_v2.reward_calibration import (
    REWARD_CALIBRATION_SCHEMA,
    _mixture_mean_sd,
    _pit_nll,
    _predict,
    reward_head_calibration,
)

from .conftest import synthetic_observations


def _calibration_decision(
    context: EvolutionContext,
    candidates: tuple[CardSnapshot, ...],
    *,
    decision_id: str = "decision-schema",
) -> CalibrationDecision:
    policy = PolicySpecification(
        safety_gate_mode="credible_joint_safe",
        max_treated_invalid_probability=0.25,
        max_incremental_invalid_probability=0.1,
        safety_alpha=0.1,
        offer_probability=0.5,
        proposal_exploration_probability=0.0,
        posterior_summary_samples=1024,
        proposal_worlds=512,
        abstain_effect=0.0,
        max_pending_per_card=2,
    )
    applicability = ApplicabilityRecord(
        specification=ApplicabilitySpecification(
            name="none",
            retrieval_applicability_contrast=False,
            policy_digest="a" * 64,
        ),
        status="disabled",
    )
    return CalibrationDecision(
        decision_id=decision_id,
        event_ordinal=0,
        context=context,
        lineage_registry=candidates,
        candidates=candidates,
        fitted_observation_ids=(),
        proposed_treatment_id=candidates[0].treatment_id,
        delivered=True,
        offer_probability=0.5,
        proposal_probability=0.5,
        joint_action_probability=0.25,
        reward_q_hat_control=0.0,
        reward_q_hat_treated=0.0,
        risk_q_hat_control=0.05,
        risk_q_hat_treated=0.05,
        policy=policy,
        applicability=applicability,
        card_kind_contrast=False,
        citation_contrast=False,
    )


def _trajectory(
    context: EvolutionContext,
    observations: tuple[CausalObservation, ...],
    *,
    trajectory_id: str = "traj-0",
) -> CalibrationTrajectory:
    cards = tuple({row.card.treatment_id: row.card for row in observations}.values())
    return CalibrationTrajectory(
        ledger_path=Path("synthetic.sqlite"),
        trajectory_id=trajectory_id,
        environment_key=SafetyEnvironmentKey.from_environment(context.environment),
        decisions=(
            _calibration_decision(
                context, cards, decision_id=f"{trajectory_id}-decision"
            ),
        ),
        observations=observations,
    )


def _loco_environment(
    context: EvolutionContext,
    *,
    aligned: bool,
    num_cards: int = 6,
    dimension: int = 3,
    per_card: int = 12,
    seed: int = 3,
    effect_slope: float = 0.2,
    noise_sd: float = 0.03,
) -> tuple[tuple[CalibrationTrajectory, ...], dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    cards = tuple(
        CardSnapshot.from_card(
            Card(
                id=f"card-{index}", task_key="task", description=f"lever number {index}"
            )
        )
        for index in range(num_cards)
    )
    axis = np.linspace(-1.0, 1.0, num_cards)
    effects = effect_slope * axis
    embeddings: dict[str, np.ndarray] = {}
    for index, card in enumerate(cards):
        if aligned:
            vector = rng.normal(0.0, 0.1, dimension)
            vector[0] = axis[index]
        else:
            vector = rng.normal(0.0, 1.0, dimension)
        embeddings[card.bank_card_id] = vector
    trajectories: list[CalibrationTrajectory] = []
    ordinal = 0
    for index, card in enumerate(cards):
        rows: list[CausalObservation] = []
        for treatment in (False, True):
            for _ in range(per_card):
                mean = effects[index] if treatment else 0.0
                value = float(rng.normal(mean, noise_sd))
                rows.append(
                    CausalObservation(
                        decision_id=f"decision-{ordinal}",
                        event_ordinal=ordinal,
                        card=card,
                        context=context,
                        treatment=treatment,
                        card_used=False,
                        offer_propensity=0.5,
                        proposal_propensity=0.5,
                        joint_action_propensity=0.25,
                        status="outcome",
                        measurement=OutcomeMeasurement(
                            value=value, se=None, kind="scalar"
                        ),
                        reward_q_hat_control=0.0,
                        reward_q_hat_treated=0.0,
                        risk_q_hat_control=0.05,
                        risk_q_hat_treated=0.05,
                    )
                )
                ordinal += 1
        trajectories.append(
            _trajectory(context, tuple(rows), trajectory_id=f"traj-{index}")
        )
    return tuple(trajectories), embeddings


def test_well_specified_reward_is_calibrated(evolution_context, revisions):
    observations = synthetic_observations(evolution_context, revisions, per_arm=25)
    report = reward_head_calibration(
        [_trajectory(evolution_context, observations)],
        config=TerminalUtilityPosteriorConfig(unknown_measurement_sd=0.02),
        min_observations=5,
    )
    assert report["schema"] == REWARD_CALIBRATION_SCHEMA
    assert report["embedding_prior"] is None
    (group,) = report["groups"]
    assert group["status"] == "development_estimate"
    predictive = group["predictive_calibration"]
    assert abs(predictive["predicted_mean_bias"]) < 0.05
    assert abs(predictive["pit_mean"] - 0.5) < 0.12
    assert predictive["pit_ks_statistic"] < 0.2
    coverage = {row["level"]: row["empirical"] for row in predictive["coverage"]}
    assert coverage[0.9] > 0.7


def test_overdispersion_widens_coverage_and_raises_nll(evolution_context, revisions):
    observations = synthetic_observations(evolution_context, revisions, per_arm=25)
    trajectories = [_trajectory(evolution_context, observations)]
    calibrated = reward_head_calibration(
        trajectories,
        config=TerminalUtilityPosteriorConfig(unknown_measurement_sd=0.02),
        min_observations=5,
    )["groups"][0]["predictive_calibration"]
    overdispersed = reward_head_calibration(
        trajectories,
        config=TerminalUtilityPosteriorConfig(unknown_measurement_sd=0.6),
        min_observations=5,
    )["groups"][0]["predictive_calibration"]
    calibrated_mid = {row["level"]: row["empirical"] for row in calibrated["coverage"]}
    over_mid = {row["level"]: row["empirical"] for row in overdispersed["coverage"]}
    assert over_mid[0.5] > calibrated_mid[0.5]
    assert overdispersed["mean_nll"] > calibrated["mean_nll"]
    assert overdispersed["sharpness"] > calibrated["sharpness"]


def test_no_embeddings_emits_core_block_only(evolution_context, revisions):
    observations = synthetic_observations(evolution_context, revisions, per_arm=20)
    report = reward_head_calibration(
        [_trajectory(evolution_context, observations)],
        min_observations=5,
    )
    (group,) = report["groups"]
    assert group["predictive_calibration"] is not None
    assert group["cold_start_loco"] is None
    assert group["embedding_neighborhood"] is None
    assert report["embedding_prior"] is None


def test_loco_credits_embedding_only_when_it_predicts_effect(evolution_context):
    prior = EmbeddingPriorConfig(raw_dimension=3, dimension=3, reward_prior_sd=0.5)
    config = TerminalUtilityPosteriorConfig(unknown_measurement_sd=0.02)
    aligned_trajectories, aligned_embeddings = _loco_environment(
        evolution_context, aligned=True
    )
    null_trajectories, null_embeddings = _loco_environment(
        evolution_context, aligned=False
    )
    aligned = reward_head_calibration(
        aligned_trajectories,
        config=config,
        card_embeddings=aligned_embeddings,
        embedding_prior=prior,
        min_observations=5,
    )["groups"][0]["cold_start_loco"]
    null = reward_head_calibration(
        null_trajectories,
        config=config,
        card_embeddings=null_embeddings,
        embedding_prior=prior,
        min_observations=5,
    )["groups"][0]["cold_start_loco"]
    assert aligned["mean_delta_nll"] < 0.0
    assert aligned["embedding_lowers_cold_start_nll"] is True
    assert aligned["mean_delta_nll"] < null["mean_delta_nll"]


def test_neighborhood_bins_partition_all_held_out_cards(evolution_context):
    prior = EmbeddingPriorConfig(raw_dimension=3, dimension=3, reward_prior_sd=0.5)
    trajectories, embeddings = _loco_environment(evolution_context, aligned=True)
    report = reward_head_calibration(
        trajectories,
        config=TerminalUtilityPosteriorConfig(unknown_measurement_sd=0.02),
        card_embeddings=embeddings,
        embedding_prior=prior,
        min_observations=5,
    )
    (group,) = report["groups"]
    loco = group["cold_start_loco"]
    neighborhood = group["embedding_neighborhood"]
    assert neighborhood["similarity_metric"] == "nearest_neighbour_cosine_quantile"
    binned = sum(entry["cards"] for entry in neighborhood["bins"])
    assert binned == loco["cards"]
    assert len(neighborhood["bins"]) == min(3, loco["cards"])


def test_min_observations_gate_reports_insufficient_evidence(
    evolution_context, revisions
):
    observations = synthetic_observations(evolution_context, revisions, per_arm=20)
    report = reward_head_calibration(
        [_trajectory(evolution_context, observations)],
        min_observations=10_000,
    )
    (group,) = report["groups"]
    assert group["status"] == "insufficient_evidence"
    assert group["predictive_calibration"] is None
    assert group["cold_start_loco"] is None


def _single_valid_row_trajectories(
    context: EvolutionContext, *, count: int = 6
) -> tuple[CalibrationTrajectory, ...]:
    trajectories: list[CalibrationTrajectory] = []
    for index in range(count):
        card = CardSnapshot.from_card(
            Card(id=f"solo-{index}", task_key="task", description=f"solo lever {index}")
        )
        row = CausalObservation(
            decision_id=f"solo-decision-{index}",
            event_ordinal=index,
            card=card,
            context=context,
            treatment=True,
            card_used=False,
            offer_propensity=0.5,
            proposal_propensity=0.5,
            joint_action_propensity=0.25,
            status="outcome",
            measurement=OutcomeMeasurement(value=0.1, se=None, kind="scalar"),
            reward_q_hat_control=0.0,
            reward_q_hat_treated=0.0,
            risk_q_hat_control=0.05,
            risk_q_hat_treated=0.05,
        )
        trajectories.append(_trajectory(context, (row,), trajectory_id=f"solo-{index}"))
    return tuple(trajectories)


def test_unscorable_group_reports_insufficient_without_crash(evolution_context):
    trajectories = _single_valid_row_trajectories(evolution_context, count=6)
    report = reward_head_calibration(trajectories, min_observations=5)
    (group,) = report["groups"]
    assert group["valid_observations"] == 6
    assert group["status"] == "insufficient_evidence"
    assert group["predictive_calibration"] is None
    assert group["cold_start_loco"] is None


def test_single_lineage_group_skips_loco_without_crash(evolution_context):
    prior = EmbeddingPriorConfig(raw_dimension=3, dimension=3, reward_prior_sd=0.5)
    trajectories, embeddings = _loco_environment(
        evolution_context, aligned=True, num_cards=1
    )
    report = reward_head_calibration(
        trajectories,
        config=TerminalUtilityPosteriorConfig(unknown_measurement_sd=0.02),
        card_embeddings=embeddings,
        embedding_prior=prior,
        min_observations=5,
    )
    (group,) = report["groups"]
    assert group["predictive_calibration"] is not None
    assert group["cold_start_loco"] is None
    assert group["embedding_neighborhood"] is None
    assert group["status"] == "development_estimate"


def test_contemporaneous_rows_excluded_from_prequential_history(evolution_context):
    cards = tuple(
        CardSnapshot.from_card(
            Card(id=f"tie-{index}", task_key="task", description=f"tie lever {index}")
        )
        for index in range(4)
    )
    ordinals = (0, 0, 1, 2)
    rows = tuple(
        CausalObservation(
            decision_id=f"tie-decision-{index}",
            event_ordinal=ordinal,
            card=cards[index],
            context=evolution_context,
            treatment=True,
            card_used=False,
            offer_propensity=0.5,
            proposal_propensity=0.5,
            joint_action_propensity=0.25,
            status="outcome",
            measurement=OutcomeMeasurement(
                value=0.1 * (index + 1), se=None, kind="scalar"
            ),
            reward_q_hat_control=0.0,
            reward_q_hat_treated=0.0,
            risk_q_hat_control=0.05,
            risk_q_hat_treated=0.05,
        )
        for index, ordinal in enumerate(ordinals)
    )
    report = reward_head_calibration(
        [_trajectory(evolution_context, rows)],
        config=TerminalUtilityPosteriorConfig(unknown_measurement_sd=0.02),
        min_observations=3,
    )
    (group,) = report["groups"]
    predictive = group["predictive_calibration"]
    # Prequential history must be STRICTLY earlier by event_ordinal, mirroring
    # the sibling head's ``event_ordinal >= decision.event_ordinal`` future
    # guard. The second ordinal-0 row has only a contemporaneous predecessor,
    # so it carries no scorable one-step-ahead history and must be skipped.
    assert predictive["observations"] == 2


def test_duplicate_trajectory_snapshots_raise(evolution_context, revisions):
    observations = synthetic_observations(evolution_context, revisions, per_arm=20)
    trajectory = _trajectory(evolution_context, observations)
    with pytest.raises(ValueError, match="duplicate"):
        reward_head_calibration([trajectory, trajectory], min_observations=5)


def test_mismatched_embedding_arguments_raise(evolution_context, revisions):
    observations = synthetic_observations(evolution_context, revisions, per_arm=20)
    with pytest.raises(ValueError, match="supplied together"):
        reward_head_calibration(
            [_trajectory(evolution_context, observations)],
            card_embeddings={"card-0": np.zeros(3)},
        )


def _mixture_posterior(
    mean: np.ndarray,
    covariance: np.ndarray,
    residual_sds: np.ndarray,
    weights: np.ndarray,
) -> GaussianPosterior:
    factor = np.linalg.cholesky(covariance)
    components = tuple(
        GaussianPosteriorComponent(
            mean=mean,
            covariance=covariance,
            factor=factor,
            residual_sd=float(sd),
            probability=float(weight),
        )
        for sd, weight in zip(residual_sds, weights)
    )
    variance_mean = float(np.average(residual_sds**2, weights=weights))
    return GaussianPosterior(
        mean=mean,
        covariance=covariance,
        residual_sd=math.sqrt(variance_mean),
        residual_variance_mean=variance_mean,
        log_marginal=0.0,
        observations=8,
        card_effect_sd=0.1,
        optimizer_method="fixture",
        optimizer_success=True,
        optimizer_iterations=1,
        hyperparameters_at_boundary=False,
        residual_scale_ess=float(len(weights)),
        residual_quadrature_error=0.0,
        residual_moment_error=0.0,
        coefficient_mean_error=0.0,
        residual_boundary_probability=0.0,
        residual_upper_boundary_probability=0.0,
        components=components,
    )


def test_predict_decomposes_exact_mixture_components():
    design = np.array([1.0, 0.5])
    mean = np.array([0.2, -0.1])
    covariance = np.diag([0.01, 0.04])
    residual_sds = np.array([0.1, 0.4])
    weights = np.array([0.3, 0.7])
    posterior = _mixture_posterior(mean, covariance, residual_sds, weights)
    measurement_variance = 0.02
    comp_means, comp_variances, comp_weights = _predict(
        posterior, design, measurement_variance
    )
    coefficient_variance = float(design @ covariance @ design)
    expected_mean = float(design @ mean)
    np.testing.assert_allclose(comp_means, [expected_mean, expected_mean])
    np.testing.assert_allclose(
        comp_variances,
        coefficient_variance + residual_sds**2 + measurement_variance,
    )
    np.testing.assert_allclose(comp_weights, weights)


def test_mixture_pit_nll_matches_closed_form_and_beats_moment_match():
    # A symmetric scale mixture: same mean, sharply different residual scales.
    # Its exact predictive is fat-tailed relative to the moment-matched Gaussian
    # that shares its mean and variance, so a tail draw is genuinely less
    # surprising under the mixture — the property the exact form must capture.
    means = np.array([0.0, 0.0])
    variances = np.array([0.01, 0.25])
    weights = np.array([0.5, 0.5])
    mixture_mean, mixture_sd = _mixture_mean_sd(means, variances, weights)
    assert mixture_mean == pytest.approx(0.0)
    assert mixture_sd == pytest.approx(math.sqrt(0.13))
    value = 3.0 * mixture_sd
    pit, nll = _pit_nll(means, variances, weights, value)
    sds = np.sqrt(variances)
    expected_pit = float(np.sum(weights * norm.cdf((value - means) / sds)))
    expected_density = float(np.sum(weights * norm.pdf(value, loc=means, scale=sds)))
    assert pit == pytest.approx(expected_pit)
    assert nll == pytest.approx(-math.log(expected_density))
    moment_matched_nll = -float(norm.logpdf(value, loc=mixture_mean, scale=mixture_sd))
    assert nll < moment_matched_nll - 0.2
