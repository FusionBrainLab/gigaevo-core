"""Conservative causal retirement for the live memory-v2 card bank."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

from loguru import logger

from gigaevo.memory.cards import Card
from gigaevo.memory_v2.features import FeatureSpace
from gigaevo.memory_v2.ledger import SqliteCausalLedger
from gigaevo.memory_v2.models import (
    CardSnapshot,
    CausalObservation,
    EvidenceSnapshot,
    EvolutionContext,
    RagApplicability,
)
from gigaevo.memory_v2.policy import SafetyConstraint
from gigaevo.memory_v2.posterior import (
    HierarchicalTerminalUtilityPosterior,
    PosteriorFitError,
)
from gigaevo.memory_v2.rng import EventRNG


@dataclass(frozen=True)
class _RetirementVerdict:
    evidence_version: str
    revision: CardSnapshot


class CausalRetirementEvictor:
    """Retire cards with supported, globally non-viable causal posteriors.

    A card is removable only when it has randomized treatment support, no
    immediate or lineage outcome pending, and its optimistic probability of
    being both safe and helpful is below the configured boundary in every
    modeled context in which its lineage was proposed.

    Verdicts are deliberately one-shot. ``CardAdmissionGate.sweep`` computes
    them and immediately revalidates the exact card revision before deletion.
    They cannot become an admission-time filter or survive a changed evidence
    snapshot.
    """

    def __init__(
        self,
        *,
        ledger: SqliteCausalLedger,
        posterior: HierarchicalTerminalUtilityPosterior,
        safety: SafetyConstraint,
        min_treated: int = 2,
        min_global_control: int = 2,
        min_distinct_contexts: int = 2,
        posterior_samples: int = 4096,
        max_viability_probability: float = 0.10,
        mc_confidence_z: float = 1.96,
    ) -> None:
        if min_treated < 1 or min_global_control < 1:
            raise ValueError("treated and pooled-control support must be positive")
        if min_distinct_contexts < 1:
            raise ValueError("min_distinct_contexts must be positive")
        if posterior_samples < 256:
            raise ValueError("posterior_samples must be at least 256")
        if not 0.0 < max_viability_probability < 0.5:
            raise ValueError("max_viability_probability must be in (0, 0.5)")
        if mc_confidence_z <= 0.0:
            raise ValueError("mc_confidence_z must be positive")
        self.ledger = ledger
        self.posterior = posterior
        self.safety = safety
        self.min_treated = min_treated
        self.min_global_control = min_global_control
        self.min_distinct_contexts = min_distinct_contexts
        self.posterior_samples = posterior_samples
        self.max_viability_probability = max_viability_probability
        self.mc_confidence_z = mc_confidence_z
        self._verdicts: dict[str, _RetirementVerdict] = {}

    def should_evict(self, card: Card) -> bool:
        """Consume and revalidate one verdict produced by the current sweep."""

        verdict = self._verdicts.pop(card.id, None)
        if verdict is None:
            return False
        if self.ledger.snapshot().version != verdict.evidence_version:
            return False
        try:
            revision = CardSnapshot.from_card(card)
        except ValueError:
            return False
        return revision == verdict.revision

    def eviction_reason(self, card: Card) -> str:
        del card
        return (
            "supported causal posterior assigns little probability to safe, "
            "helpful utility in every observed context"
        )

    def sweep(self, cards: Sequence[Card]) -> list[str]:
        evidence = self.ledger.snapshot()
        revisions: list[CardSnapshot] = []
        cards_by_id: dict[str, Card] = {}
        for card in cards:
            try:
                revision = CardSnapshot.from_card(card)
            except ValueError:
                continue
            revisions.append(revision)
            cards_by_id[revision.bank_card_id] = card

        self._verdicts.clear()
        if not revisions or not evidence.observations:
            return []
        try:
            fitted = self.posterior.fit(
                evidence.observations,
                tuple(revisions),
                lineage_observations=evidence.lineage_observations,
            )
        except (PosteriorFitError, ValueError) as exc:
            logger.warning(
                "[MemoryV2][Retirement] posterior fit failed closed: {}", exc
            )
            return []
        if (
            not fitted.reward.optimizer_success
            or fitted.reward.hyperparameters_at_boundary
        ):
            return []

        global_controls = sum(not row.treatment for row in evidence.observations)
        rng = EventRNG(evidence.model_version)
        for revision in revisions:
            rows = self._lineage_rows(revision, evidence.observations)
            if self._has_pending_lineage(revision, evidence) or not self._supported(
                rows, global_controls
            ):
                continue
            contexts = self._distinct_contexts(rows, fitted.space)
            if len(contexts) < self.min_distinct_contexts:
                continue
            if not self._non_viable_everywhere(
                fitted=fitted,
                revision=revision,
                contexts=contexts,
                rng=rng,
            ):
                continue
            self._verdicts[revision.bank_card_id] = _RetirementVerdict(
                evidence_version=evidence.version,
                revision=revision,
            )

        retired_ids = sorted(self._verdicts)
        if retired_ids:
            logger.info(
                "[MemoryV2][Retirement] proposed {}/{} cards for retirement: {}",
                len(retired_ids),
                len(cards_by_id),
                retired_ids,
            )
        return retired_ids

    def _non_viable_everywhere(
        self,
        *,
        fitted,
        revision: CardSnapshot,
        contexts: Sequence[EvolutionContext],
        rng: EventRNG,
    ) -> bool:
        # Keep a card if it could be useful either without a RAG judgment or
        # under the most favorable semantic judgment. NOT_APPLICABLE is not a
        # reason to preserve an otherwise-useless action.
        applicability_states = (
            RagApplicability.UNASSESSED,
            RagApplicability.APPLICABLE,
        )
        for context_index, context in enumerate(contexts):
            for applicability_index, applicability in enumerate(applicability_states):
                try:
                    prediction = fitted.prediction(
                        revision,
                        context,
                        rng.generator(
                            revision.treatment_id,
                            context_index * len(applicability_states)
                            + applicability_index,
                        ),
                        samples=self.posterior_samples,
                        max_treated_invalid_probability=(
                            self.safety.max_treated_invalid_probability
                        ),
                        max_incremental_invalid_probability=(
                            self.safety.max_incremental_invalid_probability
                        ),
                        safety_alpha=self.safety.alpha,
                        rag_applicability=applicability,
                    )
                except (PosteriorFitError, ValueError) as exc:
                    logger.warning(
                        "[MemoryV2][Retirement] prediction failed closed for {}: {}",
                        revision.bank_card_id,
                        exc,
                    )
                    return False
                if (
                    prediction.safety_integration_error
                    > fitted.safety_integration_tolerance
                ):
                    return False
                if (
                    self._wilson_upper(prediction.probability_safe_and_helpful)
                    > self.max_viability_probability
                ):
                    return False
        return True

    def _supported(
        self,
        rows: Sequence[CausalObservation],
        global_controls: int,
    ) -> bool:
        return (
            sum(row.treatment for row in rows) >= self.min_treated
            and global_controls >= self.min_global_control
        )

    @staticmethod
    def _lineage_rows(
        revision: CardSnapshot,
        rows: Sequence[CausalObservation],
    ) -> tuple[CausalObservation, ...]:
        lineage_ids = set(revision.bank_lineage_ids)
        return tuple(
            row
            for row in rows
            if row.card.bank_card_id in lineage_ids
            or row.card.treatment_id in lineage_ids
        )

    @staticmethod
    def _has_pending_lineage(
        revision: CardSnapshot,
        evidence: EvidenceSnapshot,
    ) -> bool:
        for bank_id in revision.bank_lineage_ids:
            if (
                evidence.pending_by_bank_card.get(bank_id, 0)
                or evidence.pending_by_treatment.get(bank_id, 0)
                or evidence.lineage_pending_by_bank_card.get(bank_id, 0)
                or evidence.lineage_pending_by_treatment.get(bank_id, 0)
            ):
                return True
        return False

    @staticmethod
    def _distinct_contexts(
        rows: Sequence[CausalObservation],
        feature_space: FeatureSpace,
    ) -> tuple[EvolutionContext, ...]:
        contexts: dict[tuple[float, ...], EvolutionContext] = {}
        for row in rows:
            key = tuple(
                float(value) for value in feature_space.context_features(row.context)
            )
            contexts.setdefault(key, row.context)
        return tuple(contexts[key] for key in sorted(contexts))

    def _wilson_upper(self, probability: float) -> float:
        n = float(self.posterior_samples)
        z = self.mc_confidence_z
        denominator = 1.0 + z * z / n
        center = (probability + z * z / (2.0 * n)) / denominator
        margin = (
            z
            / denominator
            * math.sqrt(probability * (1.0 - probability) / n + z * z / (4.0 * n * n))
        )
        return min(1.0, center + margin)
