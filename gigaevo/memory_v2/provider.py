"""Engine adapter for the durable, singleton memory-v2 policy."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
import math

from loguru import logger

from gigaevo.exceptions import MemoryStorageError
from gigaevo.memory.cards import AssignmentRecord, DecisionContext
from gigaevo.memory.events import (
    MemoryAssignment,
    MemoryReadSelection,
    emit_memory_event,
    memory_event_context,
)
from gigaevo.memory.provider import MemoryProvider
from gigaevo.memory.read.reader import MemorySelection
from gigaevo.memory.selection_leases import InFlightSelectionRegistry
from gigaevo.memory.storage.base import MemoryStore, ResearchResult
from gigaevo.memory_v2.candidates import CandidateSource
from gigaevo.memory_v2.context import DecisionContextSource
from gigaevo.memory_v2.events import MemoryV2Decision
from gigaevo.memory_v2.ledger import SqliteCausalLedger
from gigaevo.memory_v2.models import (
    ApplicabilityRecord,
    CandidateUniverseRecord,
    CardSnapshot,
    CausalObservation,
    DecisionKey,
    DecisionRecord,
    EvidenceSnapshot,
    EvolutionContext,
    PolicyDecision,
    PosteriorFitDiagnostics,
    candidate_set_hash,
    canonical_digest,
)
from gigaevo.memory_v2.policy import ChanceConstrainedProbabilityMatchingPolicy
from gigaevo.memory_v2.posterior import (
    FittedTerminalUtilityPosterior,
    HierarchicalTerminalUtilityPosterior,
)
from gigaevo.memory_v2.render import ImmutableCardRenderer
from gigaevo.memory_v2.rng import EventRNG
from gigaevo.programs.program import Program


class CausalBanditMemoryProvider(MemoryProvider):
    """Persist a two-stage card proposal/offer before returning its treatment."""

    def __init__(
        self,
        *,
        candidate_source: CandidateSource,
        context_source: DecisionContextSource,
        ledger: SqliteCausalLedger,
        posterior: HierarchicalTerminalUtilityPosterior,
        policy: ChanceConstrainedProbabilityMatchingPolicy,
        renderer: ImmutableCardRenderer,
        store: MemoryStore,
        selection_leases: InFlightSelectionRegistry,
        task_key: str,
        run_seed: int = 0,
    ) -> None:
        self.candidate_source = candidate_source
        self.context_source = context_source
        self.ledger = ledger
        self.posterior = posterior
        self.policy = policy
        if not math.isclose(
            posterior.config.reference_offer_probability,
            policy.config.offer_probability,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "memory-v2 posterior and policy offer probabilities must match"
            )
        self.renderer = renderer
        self.store = store
        self.selection_leases = selection_leases
        self.selection_leases.bind_store(store)
        self.task_key = task_key
        self.run_seed = run_seed
        self.decision_config_hash = canonical_digest(
            {
                "posterior": posterior.model_config_hash,
                "policy": policy.specification.model_dump(
                    mode="json", exclude={"digest"}
                ),
                "candidate_universe": candidate_source.specification.model_dump(
                    mode="json", exclude={"digest"}
                ),
                "applicability": candidate_source.applicability_specification.model_dump(
                    mode="json", exclude={"digest"}
                ),
            }
        )
        self._lock = asyncio.Lock()
        self._posterior_cache_key: tuple[str, tuple[str, ...]] | None = None
        self._posterior_cache: FittedTerminalUtilityPosterior | None = None

    async def select_cards(
        self,
        program: Program,
        *,
        task_description: str,
        metrics_description: str,
        parent_context: str | None = None,
        pending_counts: Mapping[str, int] | None = None,
    ) -> MemorySelection:
        del pending_counts
        attempt_id = self.selection_leases.active_attempt_for_parent(program.id)
        if attempt_id is None:
            # MemoryContextStage also runs while evaluating newly created children.
            # Those DAGs are not treatment-assignment points and must not consume a
            # behavior-policy draw or create an indefinitely pending ledger row.
            return MemorySelection()
        try:
            context = await self.context_source.snapshot(program)
            preparation_evidence = self.ledger.snapshot()
            with memory_event_context(
                program_id=program.id,
                parent_ids=(program.id,),
            ):
                research = await self.candidate_source.prepare(
                    program,
                    task_key=self.task_key,
                    task_description=task_description,
                    metrics_description=metrics_description,
                    parent_context=parent_context,
                    pending_by_bank_card=preparation_evidence.pending_by_bank_card,
                    max_pending_per_card=self.policy.config.max_pending_per_card,
                )
            async with self._lock:
                return await self._select_locked(
                    program,
                    attempt_id=attempt_id,
                    context=context,
                    research=research,
                    task_description=task_description,
                    metrics_description=metrics_description,
                    parent_context=parent_context,
                )
        except Exception as exc:
            logger.opt(exception=True).error(
                "[MemoryV2][Provider] decision failed before treatment: {}", exc
            )
            raise

    async def _select_locked(
        self,
        program: Program,
        *,
        attempt_id: str,
        context: EvolutionContext,
        research: ResearchResult,
        task_description: str,
        metrics_description: str,
        parent_context: str | None,
    ) -> MemorySelection:
        if self.selection_leases.active_attempt_for_parent(program.id) != attempt_id:
            raise MemoryStorageError(
                f"selection attempt {attempt_id!r} is no longer active for "
                f"parent {program.id!r}"
            )
        ordinal = self.ledger.next_event_ordinal(self.task_key)
        context_hash = canonical_digest(
            context.model_dump(mode="json", exclude_computed_fields=True)
        )
        for _ in range(3):
            lease_snapshot = self.selection_leases.selection_snapshot()
            evidence = self.ledger.snapshot()
            slate = await self.candidate_source.candidate_snapshot(
                program,
                task_key=self.task_key,
                task_description=task_description,
                metrics_description=metrics_description,
                parent_context=parent_context,
                pending_by_bank_card=evidence.pending_by_bank_card,
                max_pending_per_card=self.policy.config.max_pending_per_card,
                research=research,
            )
            lineage_registry = tuple(
                CardSnapshot.from_card(card) for card in slate.lineage_registry
            )
            revisions = tuple(CardSnapshot.from_card(card) for card in slate.candidates)
            eligible = self.policy.eligible_candidates(
                revisions,
                pending_by_bank_card=evidence.pending_by_bank_card,
            )
            if {row.bank_card_id for row in eligible} != set(
                slate.candidate_universe.eligible_bank_card_ids
            ):
                raise MemoryStorageError(
                    "candidate universe and posterior pending filters produced "
                    "different slates"
                )
            candidate_hash = candidate_set_hash(eligible)
            lineage_registry_hash = candidate_set_hash(lineage_registry)
            key = DecisionKey(
                run_id=context.run_id,
                run_seed=self.run_seed,
                task_key=self.task_key,
                parent_id=program.id,
                attempt_id=attempt_id,
                parent_iteration=program.iteration,
                event_ordinal=ordinal,
                environment_hash=context.environment.digest,
                context_hash=context_hash,
                model_config_hash=self.decision_config_hash,
                evidence_hash=evidence.version,
                model_evidence_hash=evidence.model_version,
                candidate_set_hash=candidate_hash,
                lineage_registry_hash=lineage_registry_hash,
            )
            fitted = self._fit(
                evidence.model_version,
                evidence.observations,
                evidence.posterior_reward_observations,
                lineage_registry,
            )
            decision = self.policy.choose(
                posterior=fitted,
                candidates=eligible,
                context=context,
                rng=EventRNG(key.rng_key),
                applicable_bank_card_ids=frozenset(
                    slate.applicability.applicable_bank_card_ids
                ),
            )
            if self._reserve_candidate_slate(
                program,
                attempt_id=attempt_id,
                expected_lease_version=lease_snapshot.version,
                candidates=eligible,
            ):
                retained_ids = (
                    (decision.proposed_card.bank_card_id,)
                    if decision.proposed_card is not None
                    else ()
                )
                try:
                    return self._commit(
                        key=key,
                        attempt_id=attempt_id,
                        context=context,
                        lineage_registry=lineage_registry,
                        candidates=eligible,
                        candidate_universe=slate.candidate_universe,
                        applicability=slate.applicability,
                        decision=decision,
                        evidence=evidence,
                        fitted=fitted,
                    )
                finally:
                    self._retain_selected_lease(attempt_id, retained_ids)
        raise MemoryStorageError("candidate slate changed during three lease retries")

    def _reserve_candidate_slate(
        self,
        program: Program,
        *,
        attempt_id: str,
        expected_lease_version: str,
        candidates: Sequence[CardSnapshot],
    ) -> bool:
        active_attempt = self.selection_leases.active_attempt_for_parent(program.id)
        attempts = self.selection_leases.attempts_for_parent(program.id)
        if active_attempt != attempt_id or attempt_id not in attempts:
            raise MemoryStorageError(
                f"selection attempt {attempt_id!r} is not active for parent "
                f"{program.id!r}"
            )
        expected = {candidate.bank_card_id: candidate for candidate in candidates}

        def exact_card(card_id: str):
            current = self.store.get(card_id)
            if current is None or CardSnapshot.from_card(current) != expected[card_id]:
                return None
            return current

        card_ids = tuple(expected)
        reservation = self.selection_leases.reserve_selection(
            attempt_id,
            card_ids,
            expected_version=expected_lease_version,
            card_lookup=exact_card,
        )
        if reservation.committed and reservation.card_ids == card_ids:
            return True
        self.selection_leases.retain_attempt_cards(attempt_id, ())
        return False

    def _retain_selected_lease(
        self, attempt_id: str, retained_ids: Sequence[str]
    ) -> None:
        try:
            self.selection_leases.retain_attempt_cards(attempt_id, retained_ids)
        except Exception:
            # Keeping the temporary slate leased is conservative. The attempt
            # cleanup still releases it, while the causal row remains usable.
            logger.opt(exception=True).warning(
                "[MemoryV2][Provider] failed to release temporary slate leases"
            )

    def _fit(
        self,
        evidence_version: str,
        observations: Sequence[CausalObservation],
        reward_observations: Sequence[CausalObservation],
        candidates: tuple[CardSnapshot, ...],
    ) -> FittedTerminalUtilityPosterior:
        descriptor_key = tuple(
            card.model_dump_json()
            for card in sorted(candidates, key=lambda row: row.treatment_id)
        )
        cache_key = (evidence_version, descriptor_key)
        if self._posterior_cache_key != cache_key or self._posterior_cache is None:
            self._posterior_cache = self.posterior.fit(
                observations,
                candidates,
                reward_observations=reward_observations,
            )
            self._posterior_cache_key = cache_key
        return self._posterior_cache

    def _commit(
        self,
        *,
        key: DecisionKey,
        attempt_id: str,
        context: EvolutionContext,
        lineage_registry: tuple[CardSnapshot, ...],
        candidates: tuple[CardSnapshot, ...],
        candidate_universe: CandidateUniverseRecord,
        applicability: ApplicabilityRecord,
        decision: PolicyDecision,
        evidence: EvidenceSnapshot,
        fitted: FittedTerminalUtilityPosterior,
    ) -> MemorySelection:
        proposed = decision.proposed_card
        prediction = None
        if proposed is not None:
            prediction = next(
                row.prediction
                for row in decision.action_probabilities
                if row.treatment_id == proposed.treatment_id
            )
        scale = context.reward.scale
        record = DecisionRecord(
            decision_id=key.decision_id,
            run_seed=key.run_seed,
            attempt_id=attempt_id,
            event_ordinal=key.event_ordinal,
            rng_key=key.rng_key,
            evidence_hash=key.evidence_hash,
            model_evidence_hash=key.model_evidence_hash,
            candidate_set_hash=key.candidate_set_hash,
            lineage_registry_hash=key.lineage_registry_hash,
            context_hash=key.context_hash,
            model_config_hash=key.model_config_hash,
            posterior_config_hash=self.posterior.model_config_hash,
            card_kind_contrast=self.posterior.feature_map.config.card_kind_contrast,
            policy=self.policy.specification,
            candidate_universe=candidate_universe,
            applicability=applicability,
            fit_diagnostics=PosteriorFitDiagnostics(
                evidence_count=fitted.evidence_count,
                reward_observations=fitted.reward.observations,
                safety_observations=fitted.safety.observations,
                reward_residual_sd=fitted.reward.residual_sd,
                reward_card_effect_sd=fitted.reward.card_effect_sd,
                reward_optimizer_method=fitted.reward.optimizer_method,
                reward_optimizer_success=fitted.reward.optimizer_success,
                reward_optimizer_iterations=fitted.reward.optimizer_iterations,
                reward_hyperparameters_at_boundary=(
                    fitted.reward.hyperparameters_at_boundary
                ),
                reward_residual_scale_ess=fitted.reward.residual_scale_ess,
                reward_quadrature_error=fitted.reward.residual_quadrature_error,
                reward_residual_moment_error=fitted.reward.residual_moment_error,
                reward_coefficient_mean_error=fitted.reward.coefficient_mean_error,
                reward_residual_boundary_probability=(
                    fitted.reward.residual_boundary_probability
                ),
                reward_residual_upper_boundary_probability=(
                    fitted.reward.residual_upper_boundary_probability
                ),
                safety_optimizer_method=fitted.safety.optimizer_method,
                safety_optimizer_iterations=fitted.safety.optimizer_iterations,
                safety_objective=fitted.safety.objective,
                safety_gradient_inf=fitted.safety.gradient_norm,
                safety_hessian_condition=fitted.safety.hessian_condition,
                offer_probability_hash=canonical_digest(
                    fitted.offer_probability_by_treatment
                ),
            ),
            context=context,
            lineage_registry=lineage_registry,
            fitted_observation_ids=tuple(
                row.decision_id for row in evidence.observations
            ),
            candidates=candidates,
            action_probabilities=decision.action_probabilities,
            pending_by_treatment=dict(evidence.pending_by_treatment),
            pending_by_bank_card=dict(evidence.pending_by_bank_card),
            censored_count=evidence.censored_count,
            ineligible_count=evidence.ineligible_count,
            abstain_probability=decision.abstain_probability,
            proposed_treatment_id=(
                proposed.treatment_id if proposed is not None else None
            ),
            delivered=decision.delivered,
            offer_probability=decision.offer_probability,
            proposal_probability=decision.proposal_probability,
            joint_action_probability=decision.joint_action_probability,
            reward_q_hat_control=(
                max(-scale, min(scale, prediction.usable_gain_control_mean * scale))
                if prediction
                else None
            ),
            reward_q_hat_treated=(
                max(-scale, min(scale, prediction.usable_gain_treated_mean * scale))
                if prediction
                else None
            ),
            risk_q_hat_control=(
                prediction.control_invalid_probability if prediction else None
            ),
            risk_q_hat_treated=(
                prediction.treated_invalid_probability if prediction else None
            ),
        )
        selected_ids = (
            (proposed.bank_card_id,)
            if proposed is not None and decision.delivered
            else ()
        )
        rendered = self.renderer.render(proposed) if selected_ids and proposed else ""
        legacy_context = DecisionContext(
            task_key=self.task_key,
            parent_metrics=dict(context.parent_metrics),
            parent_id=context.parent_id,
            search_phase=f"iteration:{context.parent_iteration}",
            parent_quality_quantile=context.map_elites.parent_quality_quantile,
        )
        assignment = AssignmentRecord(
            decision_id=key.decision_id,
            policy_version=f"MemoryV2:{key.model_config_hash[:16]}",
            task_key=self.task_key,
            ordered_eligible_ids=tuple(
                row.bank_card_id
                for row in sorted(
                    decision.action_probabilities,
                    key=lambda row: (-row.proposal_probability, row.treatment_id),
                )
            ),
            assigned_ids=selected_ids,
            delivered_ids=selected_ids,
            arm="injected" if selected_ids else "none",
            probe_arm=("treated" if decision.delivered else "control")
            if proposed is not None
            else "none",
            randomized=proposed is not None,
            propensity_kind=(
                "probe_bernoulli" if proposed is not None else "observational"
            ),
            propensities=(
                {proposed.bank_card_id: decision.offer_probability}
                if proposed is not None and decision.offer_probability is not None
                else {}
            ),
            ope_eligible=proposed is not None,
            q_hat_control=record.reward_q_hat_control,
            q_hat_treated=record.reward_q_hat_treated,
            predicted_help=(
                {proposed.bank_card_id: prediction.probability_helpful}
                if proposed is not None and prediction is not None
                else {}
            ),
            predicted_gain=(
                {proposed.bank_card_id: prediction.usable_effect_mean * scale}
                if proposed is not None and prediction is not None
                else {}
            ),
            predicted_no_card_gain=(
                {proposed.bank_card_id: prediction.usable_gain_control_mean * scale}
                if proposed is not None and prediction is not None
                else {}
            ),
            pending_by_card={},
            context=legacy_context,
            bd_cell=context.map_elites.parent_cell,
            timestamp=datetime.now(UTC),
        )
        read_event = MemoryReadSelection(
            decision_id=key.decision_id,
            mutation_mode=applicability.specification.mutation_mode,
            max_cards=1,
            research_iterations=applicability.research_iterations,
            candidate_ids=tuple(card.bank_card_id for card in candidates),
            auction_winner_ids=(
                (proposed.bank_card_id,) if proposed is not None else ()
            ),
            budgeted_ids=selected_ids,
            selected_ids=selected_ids,
            empty_reason="" if selected_ids else "policy_control_or_abstain",
        )
        v2_event = MemoryV2Decision(record=record)
        assignment_event = MemoryAssignment(assignment=assignment)
        selection = MemorySelection(
            cards=(rendered,) if rendered else (),
            card_ids=selected_ids,
            decision_id=key.decision_id,
            assignment=assignment,
            preformatted=True,
        )

        # Every fallible exposure object is complete. Publish the immutable
        # causal row last, immediately before handing the assignment to the
        # prompt stage; telemetry below is deliberately best-effort.
        self.ledger.record_decision(record)
        with memory_event_context(
            decision_id=key.decision_id,
            program_id=context.parent_id,
            parent_ids=(context.parent_id,),
        ):
            self._emit(read_event)
            self._emit(v2_event)
            self._emit(assignment_event)
        return selection

    @staticmethod
    def _emit(event: object) -> None:
        try:
            emit_memory_event(event)  # type: ignore[arg-type]
        except Exception:
            logger.opt(exception=True).warning(
                "[MemoryV2][Provider] analytics event emission failed"
            )
