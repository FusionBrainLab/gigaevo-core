"""Projection from card evidence to auction candidates."""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from gigaevo.memory.cards import Card, CardStatsBlock, DecisionContext
from gigaevo.memory.context import GlobalMemoryContext
from gigaevo.memory.context.no_card import NoCardGateSummary
from gigaevo.memory.read.auction import AuctionCandidate
from gigaevo.memory.read.interfaces import ReputationModel


def _use_count(card: Card) -> int:
    """Deterministic injection count: non-founding gain events (one per child
    the card was actually injected for). Event-based, no wall clock — identical
    decisions give identical counts regardless of run pacing."""
    return sum(1 for event in card.gain_events if not event.founding)


class AuctionCandidateProjector(BaseModel):
    """Single policy seam that turns reputation evidence into auction input."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

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
        context_key = self.context_model.key_for(context).label()
        return AuctionCandidate(
            card_id=card.id,
            posterior_a=posterior_a,
            posterior_b=posterior_b,
            magnitude=reputation.magnitude_of(block),
            deltas=reputation.event_deltas(card, context),
            delta_weights=reputation.event_weights(card, context),
            deltas_se=reputation.event_ses(card, context),
            staleness_weight=reputation.staleness_weight(card, context),
            prior_source=prior_source,
            context_key=context_key,
            use_count=_use_count(card),
        )

    def decision_baseline(
        self, context: DecisionContext | None
    ) -> NoCardGateSummary | None:
        if self.no_card_evidence is None:
            return None
        return self.no_card_evidence.summary_for(context)

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
