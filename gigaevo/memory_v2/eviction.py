"""Bayesian harm eviction over the current memory-v2 posterior."""

from __future__ import annotations

from collections.abc import Sequence
import math

from loguru import logger

from gigaevo.memory.cards import Card
from gigaevo.memory_v2.ledger import SqliteCausalLedger
from gigaevo.memory_v2.models import (
    CardSnapshot,
    CausalObservation,
    EvolutionContext,
)
from gigaevo.memory_v2.policy import SafetyConstraint
from gigaevo.memory_v2.posterior import (
    HierarchicalTerminalUtilityPosterior,
    PosteriorFitError,
)
from gigaevo.memory_v2.rng import EventRNG


class CausalPosteriorEvictor:
    """Retire exact revisions that are non-viable across observed model states."""

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
        max_viability_probability: float = 0.05,
        mc_confidence_z: float = 2.576,
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
        self._verdict_version = ""
        self._harmful_treatments: set[str] = set()

    def should_evict(self, card: Card) -> bool:
        if self.ledger.snapshot().version != self._verdict_version:
            return False
        try:
            treatment_id = CardSnapshot.from_card(card).treatment_id
        except ValueError:
            return False
        return treatment_id in self._harmful_treatments

    def eviction_reason(self, card: Card) -> str:
        del card
        return (
            "current causal posterior indicates non-viable card utility across "
            "every observed model context"
        )

    def sweep(self, cards: Sequence[Card]) -> list[str]:
        snapshot = self.ledger.snapshot()
        revisions: list[CardSnapshot] = []
        cards_by_treatment: dict[str, Card] = {}
        for card in cards:
            try:
                revision = CardSnapshot.from_card(card)
            except ValueError:
                continue
            revisions.append(revision)
            cards_by_treatment[revision.treatment_id] = card

        self._verdict_version = snapshot.version
        self._harmful_treatments = set()
        if not revisions or not snapshot.observations:
            return []
        try:
            fitted = self.posterior.fit(
                snapshot.observations,
                tuple(revisions),
                lineage_observations=snapshot.lineage_observations,
            )
        except (PosteriorFitError, ValueError) as exc:
            logger.warning("[MemoryV2][Evictor] posterior fit failed closed: {}", exc)
            return []
        if (
            not fitted.reward.optimizer_success
            or fitted.reward.hyperparameters_at_boundary
        ):
            return []

        by_treatment: dict[str, list[CausalObservation]] = {}
        for observation in snapshot.observations:
            by_treatment.setdefault(observation.card.treatment_id, []).append(
                observation
            )
        global_controls = sum(not row.treatment for row in snapshot.observations)
        rng = EventRNG(snapshot.model_version)
        for revision in revisions:
            if any(
                snapshot.pending_by_bank_card.get(bank_id, 0)
                for bank_id in revision.bank_lineage_ids
            ) or snapshot.pending_by_treatment.get(revision.treatment_id, 0):
                continue
            treatment_rows = by_treatment.get(revision.treatment_id, [])
            if not self._supported(treatment_rows, global_controls):
                continue
            contexts = self._distinct_contexts(treatment_rows, fitted.space)
            if len(contexts) < self.min_distinct_contexts:
                continue
            non_viable_everywhere = True
            for index, context in enumerate(contexts):
                prediction = fitted.prediction(
                    revision,
                    context,
                    rng.generator(revision.treatment_id, index),
                    samples=self.posterior_samples,
                    max_treated_invalid_probability=(
                        self.safety.max_treated_invalid_probability
                    ),
                    max_incremental_invalid_probability=(
                        self.safety.max_incremental_invalid_probability
                    ),
                    safety_alpha=self.safety.alpha,
                )
                if (
                    prediction.safety_integration_error
                    > fitted.safety_integration_tolerance
                ):
                    non_viable_everywhere = False
                    break
                viability_upper = self._wilson_upper(
                    prediction.probability_safe_and_helpful
                )
                if viability_upper > self.max_viability_probability:
                    non_viable_everywhere = False
                    break
            if non_viable_everywhere:
                self._harmful_treatments.add(revision.treatment_id)

        evicted_ids = [
            cards_by_treatment[treatment_id].id
            for treatment_id in sorted(self._harmful_treatments)
        ]
        if evicted_ids:
            logger.info(
                "[MemoryV2][Evictor] retiring {}/{} cards: {}",
                len(evicted_ids),
                len(cards),
                evicted_ids,
            )
        return evicted_ids

    def _supported(
        self,
        rows: Sequence[CausalObservation],
        global_controls: int,
    ) -> bool:
        treated = sum(row.treatment for row in rows)
        return (
            treated >= self.min_treated and global_controls >= self.min_global_control
        )

    @staticmethod
    def _distinct_contexts(
        rows: Sequence[CausalObservation],
        feature_space,
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
