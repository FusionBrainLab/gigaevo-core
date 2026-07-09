"""Projection from card evidence to auction candidates."""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from gigaevo.memory.cards import Card, CardStatsBlock, DecisionContext
from gigaevo.memory.context import GlobalMemoryContext
from gigaevo.memory.read.auction import AuctionCandidate
from gigaevo.memory.read.interfaces import ReputationModel


class AuctionCandidateProjector(BaseModel):
    """Single policy seam that turns reputation evidence into auction input."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    prior: Any | None = Field(
        default=None,
        description="Optional cold-card prior policy; absent preserves reputation cold_prior.",
    )
    context_model: Any = Field(default_factory=GlobalMemoryContext)
    no_card_evidence: Any | None = Field(
        default=None,
        description="Optional dynamic no-card evidence used by the abstention gate.",
    )

    def project(
        self,
        *,
        card: Card,
        block: CardStatsBlock | None,
        reputation: ReputationModel,
        context: DecisionContext | None,
    ) -> AuctionCandidate:
        posterior_a, posterior_b = reputation.posterior_of(block)
        prior_source = "reputation"
        if self.prior is not None and self._is_cold_or_corrupt(
            block, posterior_a, posterior_b
        ):
            prior = self.prior.cold_card_prior(card, context)
            posterior_a, posterior_b = prior.as_tuple()
            prior_source = prior.source
        baseline = (
            self.no_card_evidence.summary_for(context)
            if self.no_card_evidence is not None
            else None
        )
        context_key = self.context_model.key_for(context).label()
        return AuctionCandidate(
            card_id=card.id,
            posterior_a=posterior_a,
            posterior_b=posterior_b,
            magnitude=reputation.magnitude_of(block),
            deltas=reputation.event_deltas(card, context),
            delta_weights=reputation.event_weights(card, context),
            staleness_weight=reputation.staleness_weight(card, context),
            prior_source=prior_source,
            context_key=context_key,
            baseline_a=baseline.prior.alpha if baseline is not None else None,
            baseline_b=baseline.prior.beta if baseline is not None else None,
            baseline_source=baseline.source if baseline is not None else "",
            no_card_baseline=baseline.baseline if baseline is not None else None,
            no_card_n=baseline.evidence_n if baseline is not None else 0.0,
        )

    @staticmethod
    def _is_cold_or_corrupt(
        block: CardStatsBlock | None, posterior_a: float, posterior_b: float
    ) -> bool:
        if block is None or block.posterior_a is None or block.posterior_b is None:
            return True
        return not (
            math.isfinite(float(posterior_a))
            and math.isfinite(float(posterior_b))
            and posterior_a > 0.0
            and posterior_b > 0.0
        )
