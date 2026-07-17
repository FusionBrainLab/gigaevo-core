"""Chance-constrained probability-matching policy for memory v2."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from gigaevo.memory_v2.models import (
    CandidateActionProbability,
    CardSnapshot,
    EvolutionContext,
    PolicyDecision,
    PolicySpecification,
    SafetyGateMode,
)
from gigaevo.memory_v2.posterior import FittedTerminalUtilityPosterior
from gigaevo.memory_v2.rng import EventRNG


def _mix_finite_policy_with_exploration(
    treatment_ids: tuple[str, ...],
    safe_ids: frozenset[str],
    finite_probability: Mapping[str, float],
    finite_abstain_probability: float,
    exploration_probability: float,
) -> tuple[dict[str, float], float]:
    """Give every feasible action logged support without changing unsafe mass."""

    if not safe_ids:
        return {treatment_id: 0.0 for treatment_id in treatment_ids}, 1.0
    floor = exploration_probability / len(safe_ids)
    proposal = {
        treatment_id: (
            (1.0 - exploration_probability) * finite_probability[treatment_id] + floor
            if treatment_id in safe_ids
            else 0.0
        )
        for treatment_id in treatment_ids
    }
    abstain = (1.0 - exploration_probability) * finite_abstain_probability
    return proposal, abstain


def _finite_probability_matching(
    cards: tuple[CardSnapshot, ...],
    effect_worlds: Sequence[Sequence[float]] | np.ndarray,
    *,
    abstain_effect: float,
) -> tuple[dict[str, float], float, dict[str, float], dict[str, float]]:
    """Return one-pool winner frequencies, abstention, MC SEs, and last world."""

    treatment_ids = tuple(card.treatment_id for card in cards)
    finite_probability = {treatment_id: 0.0 for treatment_id in treatment_ids}
    finite_mc_variance = {treatment_id: 0.0 for treatment_id in treatment_ids}
    if not cards:
        return finite_probability, 1.0, finite_mc_variance, {}
    worlds = tuple(effect_worlds)
    if not worlds:
        raise ValueError("probability matching requires posterior worlds")
    denominator = float(len(worlds))
    winners: dict[str | None, int] = {None: 0}
    for card in cards:
        winners[card.treatment_id] = 0
    for effects in worlds:
        winner_index = max(
            range(len(cards)),
            key=lambda index: (
                effects[index],
                cards[index].treatment_id,
            ),
        )
        treatment_id = (
            None
            if effects[winner_index] <= abstain_effect
            else cards[winner_index].treatment_id
        )
        winners[treatment_id] += 1
    abstain_probability = winners[None] / denominator
    for card in cards:
        treatment_id = card.treatment_id
        probability = winners[treatment_id] / denominator
        finite_probability[treatment_id] = probability
        finite_mc_variance[treatment_id] = (
            probability * (1.0 - probability) / denominator
        )

    last_effects = {
        card.treatment_id: float(worlds[-1][index]) for index, card in enumerate(cards)
    }
    return (
        finite_probability,
        abstain_probability,
        finite_mc_variance,
        last_effects,
    )


class SafetyConstraint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    gate_mode: SafetyGateMode = "exclude_confident_incremental_harm"
    max_treated_invalid_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    max_incremental_invalid_probability: float = Field(default=0.10, ge=-1.0, le=1.0)
    alpha: float = Field(default=0.10, gt=0.0, lt=0.5)

    @model_validator(mode="after")
    def _validate_gate(self) -> SafetyConstraint:
        if self.gate_mode == "exclude_confident_incremental_harm":
            if self.max_treated_invalid_probability is not None:
                raise ValueError(
                    "incremental-harm mode requires the absolute invalidity limit "
                    "to be disabled"
                )
        elif self.max_treated_invalid_probability is None:
            raise ValueError(
                "credible-joint-safe mode requires an absolute invalidity limit"
            )
        return self


def safety_gate_admits(
    *,
    gate_mode: SafetyGateMode,
    probability_acceptable: float,
    alpha: float,
) -> bool:
    """Apply the logged posterior admission boundary."""

    if gate_mode == "exclude_confident_incremental_harm":
        return probability_acceptable > alpha
    return probability_acceptable >= 1.0 - alpha


class ProbabilityMatchingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    offer_probability: float = Field(default=0.50, gt=0.0, lt=1.0)
    proposal_exploration_probability: float = Field(default=0.05, ge=0.0, lt=1.0)
    posterior_summary_samples: int = Field(default=1024, ge=128)
    proposal_worlds: int = Field(default=512, ge=64)
    abstain_effect: float = 0.0
    max_pending_per_card: int = Field(default=2, ge=1)


class ChanceConstrainedProbabilityMatchingPolicy:
    """Finite probability matching inside the configured admissible set.

    The ``proposal_worlds`` posterior draws define the actual categorical policy,
    not merely a diagnostic approximation. Sampling from their empirical winner
    distribution therefore gives an exact, logged proposal probability for this
    configured policy. A second Bernoulli draw creates the card-vs-control arm.
    """

    def __init__(
        self,
        *,
        safety: SafetyConstraint,
        config: ProbabilityMatchingConfig,
    ) -> None:
        self.safety = safety
        self.config = config
        self.specification = PolicySpecification(
            safety_gate_mode=safety.gate_mode,
            max_treated_invalid_probability=(safety.max_treated_invalid_probability),
            max_incremental_invalid_probability=(
                safety.max_incremental_invalid_probability
            ),
            safety_alpha=safety.alpha,
            offer_probability=config.offer_probability,
            proposal_exploration_probability=(config.proposal_exploration_probability),
            posterior_summary_samples=config.posterior_summary_samples,
            proposal_worlds=config.proposal_worlds,
            abstain_effect=config.abstain_effect,
            max_pending_per_card=config.max_pending_per_card,
        )

    def eligible_candidates(
        self,
        candidates: Sequence[CardSnapshot],
        *,
        pending_by_bank_card: Mapping[str, int],
    ) -> tuple[CardSnapshot, ...]:
        return tuple(
            sorted(
                (
                    card
                    for card in candidates
                    if sum(
                        pending_by_bank_card.get(bank_id, 0)
                        for bank_id in card.bank_lineage_ids
                    )
                    < self.config.max_pending_per_card
                ),
                key=lambda row: row.treatment_id,
            )
        )

    def choose(
        self,
        *,
        posterior: FittedTerminalUtilityPosterior,
        candidates: Sequence[CardSnapshot],
        context: EvolutionContext,
        rng: EventRNG,
        applicable_bank_card_ids: frozenset[str] = frozenset(),
    ) -> PolicyDecision:
        cards = tuple(sorted(candidates, key=lambda row: row.treatment_id))
        if not cards:
            return PolicyDecision(abstain_probability=1.0)
        candidate_bank_ids = {card.bank_card_id for card in cards}
        if not applicable_bank_card_ids <= candidate_bank_ids:
            raise ValueError("RAG applicability contains a non-candidate card")
        if not math.isclose(
            posterior.reference_offer_probability,
            self.config.offer_probability,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("posterior and policy offer-probability references differ")
        predictions = posterior.predictions(
            cards,
            context,
            rng.generator("posterior-summary"),
            samples=self.config.posterior_summary_samples,
            max_treated_invalid_probability=(
                self.safety.max_treated_invalid_probability
            ),
            max_incremental_invalid_probability=(
                self.safety.max_incremental_invalid_probability
            ),
            safety_alpha=self.safety.alpha,
            applicable_bank_card_ids=applicable_bank_card_ids,
        )
        if (
            not posterior.reward.optimizer_success
            or posterior.reward.hyperparameters_at_boundary
        ):
            return PolicyDecision(
                action_probabilities=tuple(
                    CandidateActionProbability(
                        treatment_id=card.treatment_id,
                        bank_card_id=card.bank_card_id,
                        proposal_probability=0.0,
                        proposal_mc_se=0.0,
                        offer_probability=None,
                        joint_treated_probability=0.0,
                        joint_control_probability=0.0,
                        safe=False,
                        prediction=predictions[card.treatment_id],
                    )
                    for card in cards
                ),
                abstain_probability=1.0,
            )
        safe_cards = tuple(
            card
            for card in cards
            if safety_gate_admits(
                gate_mode=self.safety.gate_mode,
                probability_acceptable=(
                    predictions[card.treatment_id].probability_safe
                ),
                alpha=self.safety.alpha,
            )
        )

        proposal_rng = rng.generator("proposal-worlds")
        effect_worlds = posterior.sample_usable_effects(
            safe_cards,
            context,
            proposal_rng,
            samples=self.config.proposal_worlds,
            applicable_bank_card_ids=applicable_bank_card_ids,
        )
        (
            finite_safe_probability,
            finite_abstain_probability,
            finite_mc_variance,
            last_effects,
        ) = _finite_probability_matching(
            safe_cards,
            effect_worlds,
            abstain_effect=self.config.abstain_effect,
        )
        finite_probability = {
            card.treatment_id: finite_safe_probability.get(card.treatment_id, 0.0)
            for card in cards
        }
        exploration = self.config.proposal_exploration_probability
        safe_ids = frozenset(card.treatment_id for card in safe_cards)
        proposal_probability, abstain_probability = _mix_finite_policy_with_exploration(
            tuple(card.treatment_id for card in cards),
            safe_ids,
            finite_probability,
            finite_abstain_probability,
            exploration,
        )
        actions: list[CandidateActionProbability] = []
        offer_probability: dict[str, float | None] = {}
        for card in cards:
            treatment_id = card.treatment_id
            rho = proposal_probability[treatment_id]
            prediction = predictions[treatment_id]
            offer = None
            if treatment_id in safe_ids:
                offer = self.config.offer_probability
            offer_probability[treatment_id] = offer
            actions.append(
                CandidateActionProbability(
                    treatment_id=treatment_id,
                    bank_card_id=card.bank_card_id,
                    proposal_probability=rho,
                    proposal_mc_se=(1.0 - exploration)
                    * math.sqrt(max(finite_mc_variance.get(treatment_id, 0.0), 0.0)),
                    offer_probability=offer,
                    joint_treated_probability=(0.0 if offer is None else rho * offer),
                    joint_control_probability=(
                        0.0 if offer is None else rho * (1.0 - offer)
                    ),
                    safe=treatment_id in safe_ids,
                    prediction=prediction,
                )
            )

        proposed_id = self._sample_proposal(
            cards,
            proposal_probability,
            abstain_probability,
            rng.uniform("proposal-categorical"),
        )
        if proposed_id is None:
            return PolicyDecision(
                action_probabilities=tuple(actions),
                abstain_probability=abstain_probability,
                sampled_effects=last_effects,
            )
        proposed = next(card for card in cards if card.treatment_id == proposed_id)
        offer = offer_probability[proposed_id]
        if offer is None:
            raise RuntimeError("unsafe card acquired non-zero proposal probability")
        delivered = rng.uniform("offer-bernoulli") < offer
        rho = proposal_probability[proposed_id]
        joint = rho * (offer if delivered else 1.0 - offer)
        return PolicyDecision(
            proposed_card=proposed,
            delivered=delivered,
            offer_probability=offer,
            proposal_probability=rho,
            joint_action_probability=joint,
            action_probabilities=tuple(actions),
            abstain_probability=abstain_probability,
            sampled_effects=last_effects,
        )

    @staticmethod
    def _sample_proposal(
        cards: tuple[CardSnapshot, ...],
        probabilities: Mapping[str, float],
        abstain_probability: float,
        draw: float,
    ) -> str | None:
        cumulative = abstain_probability
        if draw < cumulative:
            return None
        for card in cards:
            cumulative += probabilities[card.treatment_id]
            if draw < cumulative:
                return card.treatment_id
        # Floating point closure: the validated probabilities sum to one.
        return cards[-1].treatment_id if cards else None
