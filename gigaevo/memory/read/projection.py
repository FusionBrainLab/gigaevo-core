"""Projection from card evidence to auction candidates."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from gigaevo.memory.cards import (
    Card,
    CardStatsBlock,
    ContextualGain,
    DecisionContext,
)
from gigaevo.memory.context import GlobalMemoryContext
from gigaevo.memory.context.evidence import split_events_by_task
from gigaevo.memory.context.no_card import NoCardGateSummary
from gigaevo.memory.read.auction import AuctionCandidate
from gigaevo.memory.read.interfaces import ReputationModel


def _use_count(card: Card, context: DecisionContext | None) -> int:
    """Deterministic injection count: non-founding gain events (one per child
    the card was actually injected for). Event-based, no wall clock — identical
    decisions give identical counts regardless of run pacing."""
    task_key = context.task_key if context is not None else ""
    native, _ = split_events_by_task(card.gain_events, task_key)
    return sum(1 for event in native if not event.founding)


def _latest_native_se(
    events: tuple[ContextualGain, ...], ses: tuple[float | None, ...]
) -> float | None:
    """Paired se of the temporally-latest native gain event, None-safe.

    Audit-only: mirrors the recency ordering the cold prior uses so the field
    reflects the freshest evidence the card carries. Events with no timestamp
    sort oldest (unknown recency never outranks a stamped event); ties resolve
    to the earliest listed. Returns None when the rows do not align or are empty."""
    if not events or len(events) != len(ses):
        return None
    latest = max(
        range(len(events)),
        key=lambda i: (
            events[i].context.timestamp is not None,
            events[i].context.timestamp or datetime.min,
        ),
    )
    return ses[latest]


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
        pending_counts: Mapping[str, int] | None = None,
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
        deltas = reputation.event_deltas(card, context)
        credit = reputation.event_weights(card, context)
        staleness = reputation.staleness_weights(card, context)
        if len(credit) != len(deltas) or len(staleness) != len(deltas):
            raise ValueError(
                "event_weights and staleness_weights must align with event_deltas"
            )
        combined_weights = tuple(
            float(event_credit) * float(event_age)
            for event_credit, event_age in zip(credit, staleness)
        )
        support_n_unstaled = sum(
            max(0.0, float(event_credit))
            for event_credit in credit
            if math.isfinite(float(event_credit))
        )
        ses = reputation.event_ses(card, context)
        gain_se = _latest_native_se(reputation.evidence_events(card, context), ses)
        return AuctionCandidate(
            card_id=card.id,
            posterior_a=posterior_a,
            posterior_b=posterior_b,
            magnitude=reputation.magnitude_of(block),
            deltas=deltas,
            delta_weights=combined_weights,
            deltas_se=ses,
            staleness_weight=1.0,
            support_n_unstaled=support_n_unstaled,
            gain_se=gain_se,
            prior_source=prior_source,
            context_key=context_key,
            use_count=_use_count(card, context),
            pending_count=(
                0 if pending_counts is None else pending_counts.get(card.id, 0)
            ),
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
