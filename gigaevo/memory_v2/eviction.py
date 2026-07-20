"""Conservative causal retirement for the live memory-v2 card bank."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from typing import Protocol

from loguru import logger
import numpy as np

from gigaevo.memory.cards import Card
from gigaevo.memory_v2.models import (
    CardSnapshot,
    CausalObservation,
    EvidenceSnapshot,
    EvolutionContext,
    RagApplicability,
)
from gigaevo.memory_v2.policy import SafetyConstraint
from gigaevo.memory_v2.posterior import (
    FittedTerminalUtilityPosterior,
    PosteriorFitError,
    normalized_gain_bounds,
)
from gigaevo.memory_v2.rng import EventRNG


@dataclass(frozen=True)
class _RetirementVerdict:
    model_evidence_version: str
    revision: CardSnapshot


class CausalEvidenceSource(Protocol):
    def snapshot(self) -> EvidenceSnapshot: ...


class RetirementPosterior(Protocol):
    def fit(
        self,
        observations: Sequence[CausalObservation],
        cards: Sequence[CardSnapshot],
        *,
        lineage_observations: Sequence[CausalObservation] = (),
    ) -> FittedTerminalUtilityPosterior: ...


class CausalRetirementEvictor:
    """Retire cards with supported, globally non-viable causal posteriors.

    A card is removable only when it has randomized treatment support, no
    immediate or lineage outcome pending, and its optimistic probability of
    being both safe and helpful is below the configured boundary in every
    modeled MAP-Elites context in which its lineage was proposed. For retirement,
    "helpful" means clearing a low quantile of the task ledger's own non-zero
    randomized-control gain magnitudes; the read policy still evaluates ordinary
    helpfulness relative to zero.

    Verdicts are deliberately one-shot. ``CardAdmissionGate.sweep`` computes
    them and immediately revalidates the exact card revision before deletion.
    They cannot become an admission-time filter or survive a changed evidence
    snapshot.
    """

    def __init__(
        self,
        *,
        ledger: CausalEvidenceSource,
        posterior: RetirementPosterior,
        safety: SafetyConstraint,
        min_treated: int = 2,
        min_global_control: int = 2,
        min_distinct_contexts: int = 2,
        posterior_samples: int = 4096,
        practical_effect_quantile: float = 0.10,
        max_viability_probability: float = 0.10,
        mc_confidence_z: float = 1.96,
    ) -> None:
        if min_treated < 1 or min_global_control < 1:
            raise ValueError("treated and pooled-control support must be positive")
        if min_distinct_contexts < 1:
            raise ValueError("min_distinct_contexts must be positive")
        if posterior_samples < 256:
            raise ValueError("posterior_samples must be at least 256")
        if not 0.0 < practical_effect_quantile <= 0.5:
            raise ValueError("practical_effect_quantile must be in (0, 0.5]")
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
        self.practical_effect_quantile = practical_effect_quantile
        self.max_viability_probability = max_viability_probability
        self.mc_confidence_z = mc_confidence_z
        self._verdicts: dict[str, _RetirementVerdict] = {}

    def should_evict(self, card: Card) -> bool:
        """Consume and revalidate one verdict produced by the current sweep."""

        verdict = self._verdicts.pop(card.id, None)
        if verdict is None:
            return False
        evidence = self.ledger.snapshot()
        if evidence.model_version != verdict.model_evidence_version:
            return False
        if self._has_pending_lineage(verdict.revision, evidence):
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
            "practically useful utility in every supported MAP-Elites context"
        )

    def sweep(self, cards: Sequence[Card]) -> list[str]:
        evidence = self.ledger.snapshot()
        revisions: list[CardSnapshot] = []
        for card in cards:
            try:
                revision = CardSnapshot.from_card(card)
            except ValueError:
                continue
            revisions.append(revision)

        self._verdicts.clear()
        usefulness_rows = evidence.observations
        if not revisions or not usefulness_rows:
            return []
        minimum_useful_effect = self._practical_effect_threshold(usefulness_rows)
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
        for name, head in (
            ("immediate", fitted.reward),
            ("lineage", fitted.lineage_reward),
        ):
            if not head.optimizer_success:
                reason = "optimizer failed"
                residual_hint = ""
            elif head.hyperparameters_at_boundary:
                reason = "residual scale reached the configured upper bound"
                residual_hint = (
                    "; raise the upper value in "
                    "memory.posterior_config.reward_residual_sd_bounds "
                    "for high-noise tasks"
                )
            elif head.residual_boundary_probability > self.safety.alpha:
                reason = (
                    "residual-scale boundary mass "
                    f"{head.residual_boundary_probability:.6g} exceeds "
                    f"alpha={self.safety.alpha:.6g}"
                )
                residual_hint = (
                    "; adjust memory.posterior_config.reward_residual_sd_bounds "
                    "(including a lower floor for low-noise or variance-floor tasks)"
                )
            else:
                continue
            logger.warning(
                "[MemoryV2][Retirement] disabled for evidence {}: {} reward head {}{}",
                evidence.version,
                name,
                reason,
                residual_hint,
            )
            return []

        # Invalid controls still inform the risk head, but without a gain
        # measurement they cannot support reward-based retirement.
        global_controls = sum(
            not row.treatment and row.measurement is not None for row in usefulness_rows
        )
        rng = EventRNG(evidence.model_version)
        for revision in revisions:
            rows = self._lineage_rows(revision, usefulness_rows)
            if self._has_pending_lineage(revision, evidence) or not self._supported(
                rows, global_controls
            ):
                continue
            contexts = self._distinct_contexts(rows)
            # Cheap coarse gate before the stricter assessable-context check in
            # _non_viable_everywhere.
            if len(contexts) < self.min_distinct_contexts:
                continue
            if not self._non_viable_everywhere(
                fitted=fitted,
                revision=revision,
                contexts=contexts,
                rng=rng,
                minimum_useful_effect=minimum_useful_effect,
            ):
                continue
            self._verdicts[revision.bank_card_id] = _RetirementVerdict(
                model_evidence_version=evidence.model_version,
                revision=revision,
            )

        retired_ids = sorted(self._verdicts)
        if retired_ids:
            logger.info(
                "[MemoryV2][Retirement] proposed {}/{} cards for retirement: {}",
                len(retired_ids),
                len(revisions),
                retired_ids,
            )
        return retired_ids

    def _non_viable_everywhere(
        self,
        *,
        fitted: FittedTerminalUtilityPosterior,
        revision: CardSnapshot,
        contexts: Sequence[EvolutionContext],
        rng: EventRNG,
        minimum_useful_effect: float,
    ) -> bool:
        # Keep a card if it could be useful either without a RAG judgment or
        # under the most favorable semantic judgment. NOT_APPLICABLE is not a
        # reason to preserve an otherwise-useless action.
        applicability_states = (
            RagApplicability.UNASSESSED,
            RagApplicability.APPLICABLE,
        )
        assessable_contexts = 0
        for context in contexts:
            try:
                _, feasible_headroom = normalized_gain_bounds(context)
            except ValueError as exc:
                logger.warning(
                    "[MemoryV2][Retirement] invalid gain bounds for {}: {}",
                    revision.bank_card_id,
                    exc,
                )
                return False
            if feasible_headroom > minimum_useful_effect:
                assessable_contexts += 1
        if assessable_contexts < self.min_distinct_contexts:
            return False

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
                        minimum_helpful_effect=minimum_useful_effect,
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
                    logger.warning(
                        "[MemoryV2][Retirement] safety integration failed closed "
                        "for {}: error {:.6g} exceeds tolerance {:.6g}",
                        revision.bank_card_id,
                        prediction.safety_integration_error,
                        fitted.safety_integration_tolerance,
                    )
                    return False
                if (
                    self._wilson_upper(prediction.probability_safe_and_helpful)
                    > self.max_viability_probability
                ):
                    return False
        return True

    def _practical_effect_threshold(
        self,
        rows: Sequence[CausalObservation],
    ) -> float:
        magnitudes = [
            abs(row.measurement.value) / row.context.reward.scale
            for row in rows
            if not row.treatment
            and row.measurement is not None
            and row.measurement.value != 0.0
        ]
        # Exact zeros are deliberately excluded: the quantile estimates the
        # scale of a realized non-trivial step instead of collapsing toward a
        # harm-only boundary on plateau-heavy tasks. Require the configured
        # pooled-control support in measured non-zero magnitudes before using
        # that conditional scale. The immediate-control denomination is also
        # applied to lineage-inclusive utility; option value enters both arms,
        # so this mismatch is fail-keep.
        if len(magnitudes) < self.min_global_control:
            logger.debug(
                "[MemoryV2][Retirement] only {} non-zero measured randomized-control "
                "gains (need {}); using zero (harm-only) practical threshold",
                len(magnitudes),
                self.min_global_control,
            )
            return 0.0
        threshold = float(
            np.quantile(magnitudes, self.practical_effect_quantile, method="linear")
        )
        if not math.isfinite(threshold) or threshold <= 0.0:
            return 0.0
        logger.debug(
            "[MemoryV2][Retirement] practical effect threshold q={:.3g} "
            "is {:.6g} from {} non-zero randomized-control gains",
            self.practical_effect_quantile,
            threshold,
            len(magnitudes),
        )
        return threshold

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
        return tuple(row for row in rows if row.card.bank_card_id in lineage_ids)

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
    ) -> tuple[EvolutionContext, ...]:
        contexts: dict[tuple[str, tuple[int, ...]], EvolutionContext] = {}
        for row in rows:
            if not row.treatment:
                continue
            key = (
                row.context.map_elites.island_id,
                row.context.map_elites.parent_cell,
            )
            contexts.setdefault(key, row.context)
        return tuple(contexts[key] for key in sorted(contexts))

    def _wilson_upper(self, probability: float) -> float:
        """Conservative binomial bound for plain i.i.d. posterior MC draws."""

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
