"""Explicit mixed-effect features for the memory-v2 terminal-utility model."""

from __future__ import annotations

import math

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from gigaevo.memory_v2.models import CardSnapshot, EvolutionContext


class FeatureConfig(BaseModel):
    """Stable feature schema; changing it changes the model-config hash."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    behavior_keys: tuple[str, ...]
    progress_log_scale: float = Field(default=100.0, gt=1.0)
    card_kind_contrast: bool = False
    retrieval_applicability_contrast: bool = False
    citation_contrast: bool = False


class HierarchicalFeatureMap:
    def __init__(self, *, config: FeatureConfig) -> None:
        if len(set(config.behavior_keys)) != len(config.behavior_keys):
            raise ValueError("behavior_keys must be unique")
        self.config = config

    def space(self, cards: tuple[CardSnapshot, ...]) -> FeatureSpace:
        return FeatureSpace(self.config, cards)


class FeatureSpace:
    """Finite design space with shared and contextual card-lineage effects."""

    def __init__(self, config: FeatureConfig, cards: tuple[CardSnapshot, ...]) -> None:
        all_cards = tuple(cards)
        by_treatment: dict[str, CardSnapshot] = {}
        for card in all_cards:
            previous = by_treatment.get(card.treatment_id)
            if previous is not None and previous.bank_card_id != card.bank_card_id:
                raise ValueError(f"conflicting card lineage {card.treatment_id!r}")
            # Historical decisions retain their exact payload snapshots. The
            # latest snapshot is only a descriptor; model features depend on the
            # stable treatment lineage, not prose revisions.
            by_treatment[card.treatment_id] = card
        self.config = config
        self.cards = tuple(
            sorted(by_treatment.values(), key=lambda row: row.treatment_id)
        )
        self._canonical_bank_id = self._resolve_bank_lineages(all_cards)
        bank_ids = sorted(set(self._canonical_bank_id.values()))
        self._bank_index = {card_id: index for index, card_id in enumerate(bank_ids)}

    @staticmethod
    def _resolve_bank_lineages(
        cards: tuple[CardSnapshot, ...],
    ) -> dict[str, str]:
        """Resolve historical aliases to one unambiguous live lineage."""

        adjacency: dict[str, set[str]] = {}
        absorbed: set[str] = set()
        actual: set[str] = set()
        for card in cards:
            actual.add(card.bank_card_id)
            adjacency.setdefault(card.bank_card_id, set())
            for alias in card.absorbed_bank_card_ids:
                absorbed.add(alias)
                adjacency.setdefault(alias, set()).add(card.bank_card_id)
                adjacency[card.bank_card_id].add(alias)

        canonical: dict[str, str] = {}
        unseen = set(adjacency)
        while unseen:
            seed = min(unseen)
            component: set[str] = set()
            frontier = [seed]
            while frontier:
                node = frontier.pop()
                if node in component:
                    continue
                component.add(node)
                frontier.extend(adjacency[node] - component)
            unseen.difference_update(component)
            roots = sorted((component & actual) - absorbed)
            if len(roots) != 1:
                raise ValueError(
                    "bank alias lineage must have exactly one survivor; "
                    f"component={sorted(component)!r}, survivors={roots!r}"
                )
            survivor = roots[0]
            canonical.update({member: survivor for member in component})
        return canonical

    def bank_lineage_id(self, card: CardSnapshot) -> str:
        try:
            return self._canonical_bank_id[card.bank_card_id]
        except KeyError as exc:
            raise ValueError(
                f"card bank id {card.bank_card_id!r} is outside this feature space"
            ) from exc

    @property
    def context_dim(self) -> int:
        # Intercept, oriented fitness, progress, and stable behavior coordinates.
        return 3 + len(self.config.behavior_keys)

    @property
    def baseline_dim(self) -> int:
        return self.context_dim

    @property
    def shared_effect_dim(self) -> int:
        return self.context_dim

    @property
    def kind_effect_dim(self) -> int:
        return int(self.config.card_kind_contrast)

    @property
    def kind_effect_index(self) -> int:
        if not self.config.card_kind_contrast:
            raise ValueError("card-kind contrast is disabled")
        return self.shared_effect_dim

    @property
    def retrieval_effect_dim(self) -> int:
        return int(self.config.retrieval_applicability_contrast)

    @property
    def retrieval_effect_index(self) -> int:
        if not self.config.retrieval_applicability_contrast:
            raise ValueError("retrieval-applicability contrast is disabled")
        return self.shared_effect_dim + self.kind_effect_dim

    @property
    def citation_effect_dim(self) -> int:
        return int(self.config.citation_contrast)

    @property
    def citation_effect_index(self) -> int:
        if not self.config.citation_contrast:
            raise ValueError("citation contrast is disabled")
        return self.shared_effect_dim + self.kind_effect_dim + self.retrieval_effect_dim

    @property
    def card_context_dim(self) -> int:
        # Card rankings may change with fitness, search progress, and MAP position.
        return self.context_dim

    @property
    def card_effect_slice(self) -> slice:
        start = (
            self.shared_effect_dim
            + self.kind_effect_dim
            + self.retrieval_effect_dim
            + self.citation_effect_dim
        )
        return slice(start, start + len(self._bank_index) * self.card_context_dim)

    @property
    def effect_dim(self) -> int:
        return self.card_effect_slice.stop

    @property
    def outcome_dim(self) -> int:
        return self.baseline_dim + self.effect_dim

    def context_features(self, context: EvolutionContext) -> np.ndarray:
        snapshot = context.map_elites
        coordinates = {row.key: row for row in snapshot.coordinates}
        if set(coordinates) != set(self.config.behavior_keys):
            raise ValueError(
                "context behavior axes differ from the configured feature schema"
            )
        primary_metric = context.reward.primary_metric
        if primary_metric not in context.parent_metrics:
            raise ValueError(f"parent metrics omit primary metric {primary_metric!r}")
        primary_raw = context.parent_metrics[primary_metric]
        primary_normalized = (
            primary_raw - context.reward.metric_lower_bound
        ) / context.reward.scale
        primary_normalized = min(max(primary_normalized, 0.0), 1.0)
        oriented_primary = (
            primary_normalized
            if context.reward.higher_is_better
            else 1.0 - primary_normalized
        )
        # Smoothly approaches one without erasing all temporal information after
        # an arbitrary iteration boundary.
        progress = math.log1p(context.parent_iteration) / math.log1p(
            context.parent_iteration + self.config.progress_log_scale
        )
        result = [
            1.0,
            2.0 * oriented_primary - 1.0,
            progress,
        ]
        semantic = [
            2.0 * coordinates[key].semantic_normalized - 1.0
            for key in self.config.behavior_keys
        ]
        result.extend(semantic)
        return np.asarray(result, dtype=float)

    def baseline(self, card: CardSnapshot, context: EvolutionContext) -> np.ndarray:
        self.bank_lineage_id(card)
        return self.context_features(context)

    def _card_deviation(
        self, card: CardSnapshot, context: EvolutionContext
    ) -> np.ndarray:
        bank_id = self.bank_lineage_id(card)
        shared = self.context_features(context)
        result = np.zeros(len(self._bank_index) * self.card_context_dim)
        start = self._bank_index[bank_id] * self.card_context_dim
        result[start : start + self.card_context_dim] = shared
        return result / math.sqrt(self.card_context_dim)

    def effect(
        self,
        card: CardSnapshot,
        context: EvolutionContext,
        *,
        rag_contrast: float = 0.0,
        use_contrast: float = 0.0,
    ) -> np.ndarray:
        result = np.zeros(self.effect_dim, dtype=float)
        context_features = self.context_features(context)
        result[: self.context_dim] = context_features
        if self.config.card_kind_contrast:
            # Effect coding makes this coefficient the program-minus-insight
            # contrast while the shared intercept remains their midpoint.
            result[self.kind_effect_index] = {
                "insight": -0.5,
                "program": 0.5,
            }.get(card.kind, 0.0)
        if self.config.retrieval_applicability_contrast:
            if rag_contrast not in (-0.5, 0.0, 0.5):
                raise ValueError(
                    "RAG contrast must be centered tri-state effect coding"
                )
            result[self.retrieval_effect_index] = rag_contrast
        if self.config.citation_contrast:
            # Citation is post-treatment uptake, never eligibility: cited and
            # delivered-but-uncited rows both stay in every head, split by one
            # effect-coded coefficient. Predictions use the 0.0 midpoint.
            if use_contrast not in (-0.5, 0.0, 0.5):
                raise ValueError(
                    "citation contrast must be centered tri-state effect coding"
                )
            result[self.citation_effect_index] = use_contrast
        result[self.card_effect_slice] = self._card_deviation(card, context)
        return result

    def design(
        self,
        card: CardSnapshot,
        context: EvolutionContext,
        treatment: bool | float,
        *,
        rag_contrast: float = 0.0,
        use_contrast: float = 0.0,
    ) -> np.ndarray:
        action_weight = float(treatment)
        effect = action_weight * self.effect(
            card,
            context,
            rag_contrast=rag_contrast,
            use_contrast=use_contrast,
        )
        return np.concatenate((self.baseline(card, context), effect))

    def baseline_design(
        self, card: CardSnapshot, context: EvolutionContext
    ) -> np.ndarray:
        return self.design(card, context, False)

    def effect_design(
        self,
        card: CardSnapshot,
        context: EvolutionContext,
        *,
        rag_contrast: float = 0.0,
        use_contrast: float = 0.0,
    ) -> np.ndarray:
        return self.effect(
            card,
            context,
            rag_contrast=rag_contrast,
            use_contrast=use_contrast,
        )

    def prior_variance(
        self,
        *,
        baseline_sd: float,
        shared_effect_sd: float,
        card_effect_sd: float,
    ) -> np.ndarray:
        values = np.full(self.outcome_dim, shared_effect_sd**2, dtype=float)
        values[: self.baseline_dim] = baseline_sd**2
        effect_start = self.baseline_dim
        values[
            effect_start + self.card_effect_slice.start : effect_start
            + self.card_effect_slice.stop
        ] = card_effect_sd**2
        return values

    def effect_coefficients(self, coefficients: np.ndarray) -> np.ndarray:
        return coefficients[self.baseline_dim :]
