from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
from pydantic import ValidationError
import pytest
from scipy.stats import norm

from gigaevo.memory.cards import Card
from gigaevo.memory_v2.features import FeatureConfig, HierarchicalFeatureMap
from gigaevo.memory_v2.models import (
    CardSnapshot,
    CausalObservation,
    EvolutionContext,
    OutcomeMeasurement,
    PolicySpecification,
)
from gigaevo.memory_v2.policy import (
    ChanceConstrainedProbabilityMatchingPolicy,
    ProbabilityMatchingConfig,
    SafetyConstraint,
    _finite_probability_matching,
    _mix_finite_policy_with_exploration,
    safety_gate_admits,
)
from gigaevo.memory_v2.posterior import (
    BayesianResidualScaleGaussianRegressor,
    HierarchicalTerminalUtilityPosterior,
    PosteriorFitError,
    TerminalUtilityPosteriorConfig,
    _deterministic_safety_gate,
    _deterministic_safety_summary,
    _joint_gaussian_boundary_probability,
    _latent_to_gain,
    _normalized_gain_bounds,
)
from gigaevo.memory_v2.rng import EventRNG

from .conftest import synthetic_observations


class _InterceptOnlyFeatureSpace:
    outcome_dim = 1

    @staticmethod
    def prior_variance(**kwargs: float) -> np.ndarray:
        return np.asarray([kwargs["baseline_sd"] ** 2])


@pytest.mark.parametrize(
    ("probability_acceptable", "expected"),
    ((0.1000001, True), (0.10, False), (0.0999999, False)),
)
def test_incremental_harm_gate_uses_strict_rejection_boundary(
    probability_acceptable: float,
    expected: bool,
) -> None:
    assert (
        safety_gate_admits(
            gate_mode="exclude_confident_incremental_harm",
            probability_acceptable=probability_acceptable,
            alpha=0.10,
        )
        is expected
    )


def test_safety_constraint_rejects_ambiguous_absolute_limit_configuration() -> None:
    with pytest.raises(ValidationError, match="absolute invalidity limit"):
        SafetyConstraint(max_treated_invalid_probability=0.25)
    with pytest.raises(ValidationError, match="requires an absolute invalidity limit"):
        SafetyConstraint(gate_mode="credible_joint_safe")


def test_policy_specification_requires_explicit_gate_mode() -> None:
    policy = ChanceConstrainedProbabilityMatchingPolicy(
        safety=SafetyConstraint(),
        config=ProbabilityMatchingConfig(),
    )
    payload = policy.specification.model_dump(mode="python")

    assert payload["safety_gate_mode"] == "exclude_confident_incremental_harm"
    assert payload["max_treated_invalid_probability"] is None
    payload.pop("safety_gate_mode")
    with pytest.raises(ValidationError, match="safety_gate_mode"):
        PolicySpecification.model_validate(payload)


def test_incremental_gate_does_not_confuse_high_baseline_with_card_harm() -> None:
    mean = np.asarray([math.log(0.45 / 0.55), math.log(0.47 / 0.53)])
    covariance = np.asarray([[0.02, 0.01], [0.01, 0.02]])

    incremental = _deterministic_safety_summary(
        mean,
        covariance,
        max_treated_invalid_probability=None,
        max_incremental_invalid_probability=0.10,
        alpha=0.10,
        integration_tolerance=1e-8,
    )[2]
    joint = _deterministic_safety_summary(
        mean,
        covariance,
        max_treated_invalid_probability=0.25,
        max_incremental_invalid_probability=0.10,
        alpha=0.10,
        integration_tolerance=1e-8,
    )[2]

    assert incremental > 0.90
    assert joint < 0.10
    assert safety_gate_admits(
        gate_mode="exclude_confident_incremental_harm",
        probability_acceptable=incremental,
        alpha=0.10,
    )
    assert not safety_gate_admits(
        gate_mode="credible_joint_safe",
        probability_acceptable=joint,
        alpha=0.10,
    )


def test_incremental_gate_excludes_confident_excess_harm() -> None:
    mean = np.asarray([math.log(0.10 / 0.90), math.log(0.40 / 0.60)])
    covariance = np.asarray([[0.005, 0.0], [0.0, 0.005]])
    probability_acceptable = _deterministic_safety_summary(
        mean,
        covariance,
        max_treated_invalid_probability=None,
        max_incremental_invalid_probability=0.10,
        alpha=0.10,
        integration_tolerance=1e-8,
    )[2]

    assert probability_acceptable < 0.10
    assert not safety_gate_admits(
        gate_mode="exclude_confident_incremental_harm",
        probability_acceptable=probability_acceptable,
        alpha=0.10,
    )


@pytest.mark.parametrize(
    ("higher_is_better", "parent_fitness"),
    ((True, -5e-10), (False, 1.0 + 5e-10)),
)
def test_posterior_accepts_context_boundary_tolerance(
    evolution_context: EvolutionContext,
    *,
    higher_is_better: bool,
    parent_fitness: float,
) -> None:
    raw = evolution_context.model_dump(mode="python", exclude_computed_fields=True)
    raw["parent_metrics"]["fitness"] = parent_fitness
    raw["reward"]["higher_is_better"] = higher_is_better
    context = EvolutionContext.model_validate(raw)

    lower, upper = _normalized_gain_bounds(context)

    assert lower == pytest.approx(5e-10)
    assert upper == 1.0


def test_card_snapshot_audits_content_without_resetting_treatment() -> None:
    first = CardSnapshot.from_card(Card(id="card", description="first payload"))
    same = CardSnapshot.from_card(Card(id="card", description="first payload"))
    changed = CardSnapshot.from_card(Card(id="card", description="second payload"))

    assert first == same
    assert first.treatment_id == changed.treatment_id == "card"
    assert first.payload_sha256 != changed.payload_sha256
    assert first.bank_card_id == changed.bank_card_id
    assert first.delivered_text == "[card 1] id=card\nfirst payload"


def test_safety_prior_separates_control_rate_from_shared_treatment_mean(
    posterior_model: HierarchicalTerminalUtilityPosterior,
    revisions: tuple[CardSnapshot, CardSnapshot],
) -> None:
    shared_mean = -0.6
    calibrated = HierarchicalTerminalUtilityPosterior(
        feature_map=posterior_model.feature_map,
        config=TerminalUtilityPosteriorConfig(
            invalidity_prior_probability=0.2,
            safety_shared_effect_prior_mean=shared_mean,
        ),
    ).fit((), revisions)

    assert calibrated.safety.mean[0] == pytest.approx(math.log(0.2 / 0.8))
    assert calibrated.safety.mean[calibrated.space.baseline_dim] == pytest.approx(
        shared_mean
    )


def test_card_consolidation_aliases_pool_bank_effects_and_pending_budget(
    posterior_model: HierarchicalTerminalUtilityPosterior,
    evolution_context: EvolutionContext,
) -> None:
    historical = CardSnapshot.from_card(
        Card(id="historical", description="shared payload")
    )
    survivor = CardSnapshot.from_card(
        Card(
            id="survivor",
            description="shared payload",
            absorbed_ids=("historical",),
        )
    )
    space = posterior_model.feature_map.space((historical, survivor))

    historical_effect = space.effect(historical, evolution_context)
    survivor_effect = space.effect(survivor, evolution_context)
    assert space.bank_lineage_id(historical) == "survivor"
    assert space.bank_lineage_members(survivor) == {"historical", "survivor"}
    assert np.array_equal(
        historical_effect[space.card_effect_slice],
        survivor_effect[space.card_effect_slice],
    )

    policy = ChanceConstrainedProbabilityMatchingPolicy(
        safety=SafetyConstraint(),
        config=ProbabilityMatchingConfig(max_pending_per_card=2),
    )
    assert (
        policy.eligible_candidates((survivor,), pending_by_bank_card={"historical": 2})
        == ()
    )


def test_card_consolidation_rejects_ambiguous_survivors(
    posterior_model: HierarchicalTerminalUtilityPosterior,
) -> None:
    left = CardSnapshot.from_card(
        Card(id="left", description="left", absorbed_ids=("shared",))
    )
    right = CardSnapshot.from_card(
        Card(id="right", description="right", absorbed_ids=("shared",))
    )

    with pytest.raises(ValueError, match="exactly one survivor"):
        posterior_model.feature_map.space((left, right))


def test_card_kind_contrast_shares_a_clean_program_vs_insight_signal(
    posterior_model: HierarchicalTerminalUtilityPosterior,
    evolution_context: EvolutionContext,
) -> None:
    feature_map = HierarchicalFeatureMap(
        config=FeatureConfig(
            behavior_keys=posterior_model.feature_map.config.behavior_keys,
            card_kind_contrast=True,
        )
    )
    model = HierarchicalTerminalUtilityPosterior(
        feature_map=feature_map,
        config=posterior_model.config,
    )
    insight = CardSnapshot.from_card(
        Card(id="insight", kind="insight", description="bounded insight")
    )
    program = CardSnapshot.from_card(
        Card(
            id="program",
            kind="program",
            program_id="source-program",
            description="bounded program exemplar",
        )
    )
    space = feature_map.space((insight, program))

    assert space.effect(insight, evolution_context)[space.kind_effect_index] == -0.5
    assert space.effect(program, evolution_context)[space.kind_effect_index] == 0.5
    assert model.model_config_hash != posterior_model.model_config_hash


def test_retrieval_priority_probability_matches_within_each_source() -> None:
    cards = tuple(
        CardSnapshot.from_card(Card(id=card_id, description=card_id))
        for card_id in ("core-a", "core-b", "tail-a", "tail-b")
    )
    worlds = np.asarray(
        [
            [4.0, 1.0, 3.0, 2.0],
            [1.0, 4.0, 2.0, 3.0],
        ]
    )

    probabilities, abstain, variances, _ = _finite_probability_matching(
        cards,
        worlds,
        abstain_effect=0.0,
        preferred_ids=frozenset({"core-a", "core-b"}),
        preferred_probability=0.75,
    )

    assert probabilities == pytest.approx(
        {
            "core-a": 0.375,
            "core-b": 0.375,
            "tail-a": 0.125,
            "tail-b": 0.125,
        }
    )
    assert abstain == 0.0
    assert sum(variances.values()) > 0.0


def test_terminal_utility_posterior_detects_treatment_dependent_invalidity(
    posterior_model: HierarchicalTerminalUtilityPosterior,
    evolution_context: EvolutionContext,
    revisions: tuple[CardSnapshot, CardSnapshot],
) -> None:
    observations = synthetic_observations(evolution_context, revisions)
    fitted = posterior_model.fit(observations, revisions)
    good, bad = revisions
    good_prediction = fitted.prediction(
        good,
        evolution_context,
        np.random.default_rng(10),
        samples=4096,
        max_treated_invalid_probability=0.25,
        max_incremental_invalid_probability=0.10,
        safety_alpha=0.10,
    )
    bad_prediction = fitted.prediction(
        bad,
        evolution_context,
        np.random.default_rng(11),
        samples=4096,
        max_treated_invalid_probability=0.25,
        max_incremental_invalid_probability=0.10,
        safety_alpha=0.10,
    )

    assert fitted.safety.gradient_norm < 1e-5
    assert good_prediction.treated_invalid_probability < 0.20
    assert good_prediction.probability_safe > 0.85
    assert good_prediction.probability_helpful > 0.95
    assert bad_prediction.treated_invalid_probability > 0.25
    assert bad_prediction.probability_safe < 1e-6


def test_policy_abstains_when_reward_hyperparameter_fit_is_unhealthy(
    posterior_model: HierarchicalTerminalUtilityPosterior,
    evolution_context: EvolutionContext,
    revisions: tuple[CardSnapshot, CardSnapshot],
) -> None:
    fitted = posterior_model.fit(
        synthetic_observations(evolution_context, revisions, per_arm=4),
        revisions,
    )
    fitted.reward = replace(
        fitted.reward,
        optimizer_success=False,
        hyperparameters_at_boundary=True,
    )
    policy = ChanceConstrainedProbabilityMatchingPolicy(
        safety=SafetyConstraint(),
        config=ProbabilityMatchingConfig(
            posterior_summary_samples=128,
            proposal_worlds=64,
        ),
    )

    decision = policy.choose(
        posterior=fitted,
        candidates=revisions,
        context=evolution_context,
        rng=EventRNG("unhealthy-reward-fit"),
    )

    assert decision.proposed_card is None
    assert decision.abstain_probability == 1.0
    assert all(row.proposal_probability == 0.0 for row in decision.action_probabilities)


def test_policy_logs_the_exact_finite_probability_matching_distribution(
    posterior_model: HierarchicalTerminalUtilityPosterior,
    evolution_context: EvolutionContext,
    revisions: tuple[CardSnapshot, CardSnapshot],
) -> None:
    fitted = posterior_model.fit(
        synthetic_observations(evolution_context, revisions), revisions
    )
    policy = ChanceConstrainedProbabilityMatchingPolicy(
        safety=SafetyConstraint(
            gate_mode="credible_joint_safe",
            max_treated_invalid_probability=0.25,
            max_incremental_invalid_probability=0.10,
            alpha=0.10,
        ),
        config=ProbabilityMatchingConfig(
            posterior_summary_samples=512,
            proposal_worlds=256,
        ),
    )
    first = policy.choose(
        posterior=fitted,
        candidates=revisions,
        context=evolution_context,
        rng=EventRNG("a" * 64),
    )
    replay = policy.choose(
        posterior=fitted,
        candidates=revisions,
        context=evolution_context,
        rng=EventRNG("a" * 64),
    )

    assert first == replay
    assert (
        sum(row.proposal_probability for row in first.action_probabilities)
        + first.abstain_probability
        == 1.0
    )
    good = next(row for row in first.action_probabilities if row.bank_card_id == "good")
    assert good.safe
    assert good.proposal_probability > 0.0
    assert first.abstain_probability < 1.0
    bad = next(row for row in first.action_probabilities if row.bank_card_id == "bad")
    assert not bad.safe
    assert bad.proposal_probability == 0.0
    assert bad.joint_treated_probability == 0.0


def test_cold_safety_prior_is_calibrated_in_prediction_space(
    posterior_model: HierarchicalTerminalUtilityPosterior,
    evolution_context: EvolutionContext,
    revisions: tuple[CardSnapshot, CardSnapshot],
) -> None:
    fitted = posterior_model.fit((), revisions[:1])
    prediction = fitted.prediction(
        revisions[0],
        evolution_context,
        np.random.default_rng(12),
        samples=8192,
        max_treated_invalid_probability=0.25,
        max_incremental_invalid_probability=0.10,
        safety_alpha=0.10,
    )

    assert 0.03 < prediction.treated_invalid_probability < 0.10
    assert prediction.treated_invalid_upper < 0.20
    assert prediction.probability_safe > 0.85


def test_safety_feasibility_is_deterministic_under_laplace_posterior(
    posterior_model: HierarchicalTerminalUtilityPosterior,
    evolution_context: EvolutionContext,
    revisions: tuple[CardSnapshot, CardSnapshot],
) -> None:
    fitted = posterior_model.fit(
        synthetic_observations(evolution_context, revisions, per_arm=20), revisions
    )
    first = fitted.prediction(
        revisions[0],
        evolution_context,
        np.random.default_rng(100),
        samples=512,
        max_treated_invalid_probability=0.25,
        max_incremental_invalid_probability=0.10,
        safety_alpha=0.10,
    )
    second = fitted.prediction(
        revisions[0],
        evolution_context,
        np.random.default_rng(200),
        samples=512,
        max_treated_invalid_probability=0.25,
        max_incremental_invalid_probability=0.10,
        safety_alpha=0.10,
    )

    assert first.probability_safe == second.probability_safe
    assert first.treated_invalid_upper == second.treated_invalid_upper
    assert first.incremental_invalid_upper == second.incremental_invalid_upper


def test_feature_space_uses_absolute_fitness_and_card_map_interactions(
    posterior_model: HierarchicalTerminalUtilityPosterior,
    evolution_context: EvolutionContext,
    revisions: tuple[CardSnapshot, CardSnapshot],
) -> None:
    space = posterior_model.feature_map.space(revisions)
    high = evolution_context.model_copy(
        update={
            "parent_metrics": {
                **evolution_context.parent_metrics,
                "fitness": 0.9,
            }
        }
    )
    assert not np.allclose(
        space.context_features(evolution_context), space.context_features(high)
    )
    low_contrast = space.effect(revisions[0], evolution_context) - space.effect(
        revisions[1], evolution_context
    )
    high_contrast = space.effect(revisions[0], high) - space.effect(revisions[1], high)
    assert not np.allclose(low_contrast, high_contrast)


def test_feature_space_ignores_live_rebinning_coordinates(
    posterior_model: HierarchicalTerminalUtilityPosterior,
    evolution_context: EvolutionContext,
    revisions: tuple[CardSnapshot, CardSnapshot],
) -> None:
    space = posterior_model.feature_map.space(revisions)
    rebound_coordinates = tuple(
        row.model_copy(
            update={
                "dynamic_normalized": 1.0 - row.dynamic_normalized,
                "dynamic_lower_bound": row.dynamic_lower_bound - 10.0,
                "dynamic_upper_bound": row.dynamic_upper_bound + 10.0,
            }
        )
        for row in evolution_context.map_elites.coordinates
    )
    rebound = evolution_context.model_copy(
        update={
            "map_elites": evolution_context.map_elites.model_copy(
                update={"coordinates": rebound_coordinates}
            )
        }
    )

    assert np.array_equal(
        space.context_features(evolution_context),
        space.context_features(rebound),
    )


def test_outcome_model_rejects_mixed_offer_probabilities(
    posterior_model: HierarchicalTerminalUtilityPosterior,
    evolution_context: EvolutionContext,
    revisions: tuple[CardSnapshot, CardSnapshot],
) -> None:
    card = revisions[0]
    rows = tuple(
        CausalObservation(
            decision_id=f"adaptive-{ordinal}",
            event_ordinal=ordinal,
            card=card,
            context=evolution_context,
            treatment=bool(ordinal),
            offer_propensity=offer,
            proposal_propensity=1.0,
            joint_action_propensity=offer if ordinal else 1.0 - offer,
            status="outcome",
            measurement=OutcomeMeasurement(
                value=0.01,
                se=0.01,
                kind="paired",
                n_pairs=8,
                pairing_signature=f"cohort-{ordinal}",
            ),
            reward_q_hat_control=0.0,
            reward_q_hat_treated=0.0,
            risk_q_hat_control=0.05,
            risk_q_hat_treated=0.05,
        )
        for ordinal, offer in enumerate((0.2, 0.8))
    )

    with pytest.raises(ValueError, match="one fixed offer probability"):
        posterior_model.fit(rows, (card,))


def test_bounded_gain_link_respects_parent_specific_opportunity(
    posterior_model: HierarchicalTerminalUtilityPosterior,
    evolution_context: EvolutionContext,
    revisions: tuple[CardSnapshot, CardSnapshot],
) -> None:
    card = revisions[0]
    training_context = evolution_context.model_copy(
        update={
            "parent_metrics": {
                **evolution_context.parent_metrics,
                "fitness": 0.02,
            }
        }
    )
    rows: list[CausalObservation] = []
    for ordinal in range(80):
        treated = bool(ordinal % 2)
        value = 0.96 if treated else 0.0
        rows.append(
            CausalObservation(
                decision_id=f"bounded-{ordinal}",
                event_ordinal=ordinal,
                card=card,
                context=training_context,
                treatment=treated,
                offer_propensity=0.5,
                proposal_propensity=1.0,
                joint_action_propensity=0.5,
                status="outcome",
                measurement=OutcomeMeasurement(
                    value=value,
                    se=0.01,
                    kind="paired",
                    n_pairs=8,
                    pairing_signature=f"bounded-{ordinal}",
                ),
                reward_q_hat_control=0.0,
                reward_q_hat_treated=0.0,
                risk_q_hat_control=0.05,
                risk_q_hat_treated=0.05,
            )
        )
    fitted = posterior_model.fit(rows, (card,))
    constrained_context = evolution_context.model_copy(
        update={
            "parent_metrics": {
                **evolution_context.parent_metrics,
                "fitness": 0.99,
            },
            "parent_iteration": 250,
            "parent_generation": 250,
        }
    )
    prediction = fitted.prediction(
        card,
        constrained_context,
        np.random.default_rng(91),
        samples=4096,
        max_treated_invalid_probability=0.25,
        max_incremental_invalid_probability=0.10,
        safety_alpha=0.10,
    )

    assert -0.99 <= prediction.usable_gain_control_mean <= 0.01
    assert -0.99 <= prediction.usable_gain_treated_mean <= 0.01
    assert fitted.reward.card_effect_sd == posterior_model.config.card_effect_prior_sd


def test_reward_conditional_mean_is_not_double_clipped_by_residual_noise(
    evolution_context: EvolutionContext,
) -> None:
    latent = np.asarray([-3.0, -0.6, 0.0, 2.0])
    parent = evolution_context.parent_metrics[evolution_context.reward.primary_metric]
    reward = evolution_context.reward
    if reward.higher_is_better:
        lower = (reward.metric_lower_bound - parent) / reward.scale
        upper = (reward.metric_upper_bound - parent) / reward.scale
    else:
        lower = (parent - reward.metric_upper_bound) / reward.scale
        upper = (parent - reward.metric_lower_bound) / reward.scale

    assert _latent_to_gain(latent, evolution_context) == pytest.approx(
        np.clip(latent, lower, upper)
    )


def test_residual_quadrature_handles_lower_bound_mode_without_crashing() -> None:
    config = TerminalUtilityPosteriorConfig()
    fitted = BayesianResidualScaleGaussianRegressor(config).fit(
        np.ones((50, 1)),
        np.zeros(50),
        np.zeros(50),
        _InterceptOnlyFeatureSpace(),
    )

    assert fitted.optimizer_success
    assert not fitted.hyperparameters_at_boundary
    assert fitted.residual_boundary_probability > 0.5
    assert (
        fitted.residual_upper_boundary_probability
        < config.reward_residual_upper_boundary_mass_limit
    )
    assert (
        fitted.residual_quadrature_error <= config.reward_residual_quadrature_mass_rtol
    )
    assert fitted.residual_moment_error <= config.reward_residual_moment_rtol
    assert fitted.coefficient_mean_error <= config.reward_coefficient_mean_tolerance
    assert sum(
        component.probability for component in fitted.components
    ) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("prior_initial", "expected_upper_boundary"),
    ((0.2, False), (2.0, True)),
)
def test_residual_boundary_mass_matches_truncated_log_normal_prior(
    prior_initial: float,
    expected_upper_boundary: bool,
) -> None:
    config = TerminalUtilityPosteriorConfig(
        reward_residual_sd_initial=prior_initial,
        reward_residual_log_prior_sd=0.86,
    )
    fitted = BayesianResidualScaleGaussianRegressor(config).fit(
        np.empty((0, 1)),
        np.empty(0),
        np.empty(0),
        _InterceptOnlyFeatureSpace(),
    )
    lower, upper = map(math.log, config.reward_residual_sd_bounds)
    span = upper - lower
    prior_mean = math.log(config.reward_residual_sd_initial)
    prior_sd = config.reward_residual_log_prior_sd

    def prior_cdf(value: float) -> float:
        return float(norm.cdf((value - prior_mean) / prior_sd))

    normalization = prior_cdf(upper) - prior_cdf(lower)
    exact_lower_boundary_mass = (
        prior_cdf(lower + 0.01 * span) - prior_cdf(lower)
    ) / normalization
    exact_upper_boundary_mass = (
        prior_cdf(upper) - prior_cdf(upper - 0.01 * span)
    ) / normalization
    exact_boundary_mass = exact_lower_boundary_mass + exact_upper_boundary_mass

    assert fitted.residual_boundary_probability == pytest.approx(
        exact_boundary_mass, rel=1e-3
    )
    assert fitted.residual_upper_boundary_probability == pytest.approx(
        exact_upper_boundary_mass, rel=1e-3
    )
    assert fitted.hyperparameters_at_boundary is expected_upper_boundary


def test_residual_quadrature_is_healthy_for_concentrated_n250_posterior() -> None:
    config = TerminalUtilityPosteriorConfig()
    rng = np.random.default_rng(20260715)
    values = 0.05 + rng.normal(0.0, 0.10, size=250)
    fitted = BayesianResidualScaleGaussianRegressor(config).fit(
        np.ones((len(values), 1)),
        values,
        np.zeros_like(values),
        _InterceptOnlyFeatureSpace(),
    )

    assert fitted.optimizer_success
    assert not fitted.hyperparameters_at_boundary
    assert fitted.mean[0] == pytest.approx(float(np.mean(values)), abs=1e-3)
    assert fitted.residual_sd == pytest.approx(float(np.std(values)), abs=0.01)
    assert (
        fitted.residual_quadrature_error <= config.reward_residual_quadrature_mass_rtol
    )
    assert fitted.residual_moment_error <= config.reward_residual_moment_rtol
    assert fitted.coefficient_mean_error <= config.reward_coefficient_mean_tolerance
    assert fitted.optimizer_iterations < 1_000
    assert len(fitted.components) < 500


@pytest.mark.parametrize(
    "covariance",
    (
        np.asarray([[-1.0, 0.0], [0.0, 1.0]]),
        np.asarray([[0.0, 0.1], [0.1, 1.0]]),
    ),
)
def test_safety_quadrature_rejects_invalid_covariance(
    covariance: np.ndarray,
) -> None:
    with pytest.raises(PosteriorFitError, match="covariance"):
        _joint_gaussian_boundary_probability(
            np.zeros(2),
            covariance,
            lambda _eta0: 0.0,
            tolerance=1e-8,
        )


def test_safety_quadrature_handles_constant_plateau_boundary() -> None:
    mean = np.asarray([0.0, float(np.log(0.25 / 0.75))])
    covariance = np.eye(2)
    treated_upper, _, safe_probability, error = _deterministic_safety_summary(
        mean,
        covariance,
        max_treated_invalid_probability=0.25,
        max_incremental_invalid_probability=1.0,
        alpha=0.10,
        integration_tolerance=1e-8,
    )

    assert safe_probability == pytest.approx(0.5, abs=2e-8)
    assert error <= 1e-8
    assert _deterministic_safety_gate(
        mean,
        covariance,
        max_treated_invalid_probability=0.25,
        max_incremental_invalid_probability=1.0,
        alpha=0.10,
        integration_tolerance=1e-8,
    ) == pytest.approx((treated_upper, safe_probability, error))


def test_safety_quadrature_fails_closed_near_singular_covariance() -> None:
    covariance = np.asarray([[1.0, 0.999999999999], [0.999999999999, 1.0]])
    with pytest.raises(PosteriorFitError, match="too close to singular"):
        _joint_gaussian_boundary_probability(
            np.zeros(2),
            covariance,
            lambda _eta0: 0.0,
            tolerance=1e-8,
        )


def test_policy_excludes_card_when_safety_certification_fails(
    posterior_model: HierarchicalTerminalUtilityPosterior,
    evolution_context: EvolutionContext,
    revisions: tuple[CardSnapshot, CardSnapshot],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fitted = posterior_model.fit((), revisions[:1])

    def fail_certification(*_args: object, **_kwargs: object) -> object:
        raise PosteriorFitError("uncertified covariance")

    monkeypatch.setattr(
        "gigaevo.memory_v2.posterior._deterministic_safety_summary",
        fail_certification,
    )
    policy = ChanceConstrainedProbabilityMatchingPolicy(
        safety=SafetyConstraint(),
        config=ProbabilityMatchingConfig(
            posterior_summary_samples=128,
            proposal_worlds=64,
        ),
    )

    decision = policy.choose(
        posterior=fitted,
        candidates=revisions[:1],
        context=evolution_context,
        rng=EventRNG("failed-safety-certification"),
    )

    assert decision.abstain_probability == 1.0
    assert decision.proposed_card is None
    assert not decision.action_probabilities[0].safe
    assert decision.action_probabilities[0].prediction.probability_safe == 0.0
    assert decision.action_probabilities[0].prediction.safety_integration_error == 1.0


def test_default_policy_gives_a_cold_card_nonzero_exploration_support(
    posterior_model: HierarchicalTerminalUtilityPosterior,
    evolution_context: EvolutionContext,
    revisions: tuple[CardSnapshot, CardSnapshot],
) -> None:
    fitted = posterior_model.fit((), revisions[:1])
    policy = ChanceConstrainedProbabilityMatchingPolicy(
        safety=SafetyConstraint(),
        config=ProbabilityMatchingConfig(
            posterior_summary_samples=128,
            proposal_worlds=64,
            proposal_exploration_probability=0.05,
        ),
    )

    decision = policy.choose(
        posterior=fitted,
        candidates=revisions[:1],
        context=evolution_context,
        rng=EventRNG("cold-card-support"),
    )
    action = decision.action_probabilities[0]

    assert action.safe
    assert action.proposal_probability >= 0.05
    assert action.offer_probability == policy.config.offer_probability


def test_adaptive_safety_quadrature_is_conservative_near_singular_boundary() -> None:
    mean = np.asarray([-3.0, -2.2513606360])
    sd0, sd1, correlation = 0.2, 0.5, 0.9999
    covariance = np.asarray(
        [
            [sd0 * sd0, correlation * sd0 * sd1],
            [correlation * sd0 * sd1, sd1 * sd1],
        ]
    )

    treated_upper, difference_upper, safe_probability, integration_error = (
        _deterministic_safety_summary(
            mean,
            covariance,
            max_treated_invalid_probability=0.25,
            max_incremental_invalid_probability=0.10,
            alpha=0.10,
            integration_tolerance=1e-8,
        )
    )

    assert safe_probability == pytest.approx(0.88, abs=2e-7)
    assert safe_probability < 0.90
    assert difference_upper == pytest.approx(0.10606495, abs=2e-6)
    assert treated_upper < 0.25
    assert integration_error <= 1e-8


def test_safety_model_rejects_mixed_offer_probabilities(
    posterior_model: HierarchicalTerminalUtilityPosterior,
    evolution_context: EvolutionContext,
    revisions: tuple[CardSnapshot, CardSnapshot],
) -> None:
    card = revisions[0]
    rows = tuple(
        CausalObservation(
            decision_id=f"risk-adaptive-{ordinal}",
            event_ordinal=ordinal,
            card=card,
            context=evolution_context,
            treatment=bool(ordinal),
            offer_propensity=offer,
            proposal_propensity=1.0,
            joint_action_propensity=offer if ordinal else 1.0 - offer,
            status="outcome",
            measurement=OutcomeMeasurement(value=0.02, se=None, kind="scalar"),
            reward_q_hat_control=0.0,
            reward_q_hat_treated=0.0,
            risk_q_hat_control=0.05,
            risk_q_hat_treated=0.05,
        )
        for ordinal, offer in enumerate((0.2, 0.8))
    )

    with pytest.raises(ValueError, match="one fixed offer probability"):
        posterior_model.fit(rows, (card,))


def test_exploration_mixture_gives_every_safe_card_policy_support() -> None:
    treatment_ids = tuple(f"card-{index}" for index in range(80))
    finite = {
        treatment_id: (1.0 / 64.0 if index < 64 else 0.0)
        for index, treatment_id in enumerate(treatment_ids)
    }
    proposal, abstain = _mix_finite_policy_with_exploration(
        treatment_ids,
        frozenset(treatment_ids),
        finite,
        0.0,
        0.08,
    )

    assert all(probability >= 0.001 for probability in proposal.values())
    assert sum(proposal.values()) + abstain == pytest.approx(1.0)
