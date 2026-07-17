from __future__ import annotations

from gigaevo.memory_v2.models import (
    ApplicabilityRecord,
    ApplicabilitySpecification,
    CandidateActionProbability,
    CandidateUniverseRecord,
    CandidateUniverseSpecification,
    CardSnapshot,
    DecisionKey,
    DecisionRecord,
    EvolutionContext,
    PolicySpecification,
    PosteriorFitDiagnostics,
    PosteriorPrediction,
    SafetyGateMode,
    candidate_set_hash,
    canonical_digest,
)


def prediction(card: CardSnapshot) -> PosteriorPrediction:
    return PosteriorPrediction(
        treatment_id=card.treatment_id,
        usable_effect_mean=0.1,
        usable_effect_sd=0.05,
        probability_helpful=0.8,
        usable_gain_control_mean=0.0,
        usable_gain_treated_mean=0.095,
        control_invalid_probability=0.04,
        treated_invalid_probability=0.05,
        incremental_invalid_probability=0.01,
        treated_invalid_upper=0.12,
        incremental_invalid_upper=0.05,
        probability_safe=0.95,
        safety_integration_error=1e-10,
        probability_safe_and_helpful=0.78,
        usable_gain_predictive_sd=0.2,
    )


def decision_record(
    context: EvolutionContext,
    card: CardSnapshot,
    *,
    ordinal: int = 0,
    delivered: bool = True,
    run_seed: int = 17,
    attempt_id: str = "attempt-test",
    safety_gate_mode: SafetyGateMode = "credible_joint_safe",
    max_treated_invalid_probability: float | None = 0.25,
) -> DecisionRecord:
    offer = 0.6
    proposal = 1.0
    row = CandidateActionProbability(
        treatment_id=card.treatment_id,
        bank_card_id=card.bank_card_id,
        proposal_probability=proposal,
        proposal_mc_se=0.0,
        offer_probability=offer,
        joint_treated_probability=proposal * offer,
        joint_control_probability=proposal * (1.0 - offer),
        safe=True,
        prediction=prediction(card),
    )
    candidates = (card,)
    candidate_hash = candidate_set_hash(candidates)
    policy = PolicySpecification(
        safety_gate_mode=safety_gate_mode,
        max_treated_invalid_probability=max_treated_invalid_probability,
        max_incremental_invalid_probability=0.1,
        safety_alpha=0.1,
        offer_probability=0.6,
        proposal_exploration_probability=0.0,
        posterior_summary_samples=1024,
        proposal_worlds=512,
        abstain_effect=0.0,
        max_pending_per_card=2,
    )
    candidate_universe_specification = CandidateUniverseSpecification(
        policy_digest="u" * 64,
    )
    candidate_universe = CandidateUniverseRecord(
        specification=candidate_universe_specification,
        status="eligible_bank",
        eligible_bank_card_ids=(card.bank_card_id,),
    )
    applicability = ApplicabilityRecord(
        specification=ApplicabilitySpecification(
            name="none",
            retrieval_applicability_contrast=False,
            policy_digest="a" * 64,
        ),
        status="disabled",
    )
    posterior_config_hash = "c" * 64
    model_config_hash = canonical_digest(
        {
            "posterior": posterior_config_hash,
            "policy": policy.model_dump(mode="json", exclude={"digest"}),
            "candidate_universe": candidate_universe_specification.model_dump(
                mode="json", exclude={"digest"}
            ),
            "applicability": applicability.specification.model_dump(
                mode="json", exclude={"digest"}
            ),
        }
    )
    key = DecisionKey(
        run_id=context.run_id,
        run_seed=run_seed,
        task_key=context.environment.task_key,
        parent_id=context.parent_id,
        attempt_id=attempt_id,
        parent_iteration=context.parent_iteration,
        event_ordinal=ordinal,
        environment_hash=context.environment.digest,
        context_hash=canonical_digest(
            context.model_dump(mode="json", exclude_computed_fields=True)
        ),
        model_config_hash=model_config_hash,
        evidence_hash="d" * 64,
        model_evidence_hash="e" * 64,
        candidate_set_hash=candidate_hash,
        lineage_registry_hash=candidate_hash,
    )
    return DecisionRecord(
        decision_id=key.decision_id,
        run_seed=run_seed,
        attempt_id=key.attempt_id,
        event_ordinal=ordinal,
        rng_key=key.rng_key,
        evidence_hash=key.evidence_hash,
        model_evidence_hash=key.model_evidence_hash,
        candidate_set_hash=candidate_hash,
        lineage_registry_hash=key.lineage_registry_hash,
        context_hash=key.context_hash,
        model_config_hash=key.model_config_hash,
        posterior_config_hash=posterior_config_hash,
        policy=policy,
        candidate_universe=candidate_universe,
        applicability=applicability,
        fit_diagnostics=PosteriorFitDiagnostics(
            evidence_count=0,
            reward_observations=0,
            safety_observations=0,
            reward_residual_sd=0.2,
            reward_card_effect_sd=0.25,
            reward_optimizer_method="fixed_prior",
            reward_optimizer_success=True,
            reward_optimizer_iterations=0,
            reward_hyperparameters_at_boundary=False,
            reward_residual_scale_ess=8.0,
            reward_quadrature_error=0.0,
            reward_residual_moment_error=0.0,
            reward_coefficient_mean_error=0.0,
            reward_residual_boundary_probability=0.0,
            reward_residual_upper_boundary_probability=0.0,
            safety_optimizer_method="prior",
            safety_optimizer_iterations=0,
            safety_objective=0.0,
            safety_gradient_inf=0.0,
            safety_hessian_condition=1.0,
            offer_probability_hash=canonical_digest({}),
        ),
        context=context,
        lineage_registry=candidates,
        candidates=candidates,
        action_probabilities=(row,),
        pending_by_treatment={"older-treatment": 1},
        pending_by_bank_card={"older-bank": 1},
        censored_count=2,
        abstain_probability=0.0,
        proposed_treatment_id=card.treatment_id,
        delivered=delivered,
        offer_probability=offer,
        proposal_probability=proposal,
        joint_action_probability=(
            row.joint_treated_probability
            if delivered
            else row.joint_control_probability
        ),
        reward_q_hat_control=0.0,
        reward_q_hat_treated=0.095,
        risk_q_hat_control=0.04,
        risk_q_hat_treated=0.05,
    )
