"""Shared read-side component protocols."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from gigaevo.memory.cards import Card, CardStatsBlock, ContextualGain, DecisionContext
from gigaevo.memory.context.no_card import NoCardGateSummary
from gigaevo.memory.read.auction import AuctionBid, AuctionCandidate
from gigaevo.memory.storage.base import ResearchResult


@runtime_checkable
class Shortlister(Protocol):
    """Turns the mutation context into researched candidate cards."""

    async def shortlist(
        self,
        *,
        parents: list[Any],
        mutation_mode: str,
        task_description: str,
        metrics_description: str,
        exclude_ids: frozenset[str] = frozenset(),
        parent_contexts: list[str] | None = None,
    ) -> ResearchResult: ...


@runtime_checkable
class ReputationModel(Protocol):
    """Owns all per-card efficacy statistics derived from injection outcomes."""

    @property
    def requires_decision_context(self) -> bool: ...

    @property
    def policy_min_effective_events(self) -> float: ...

    def card_stats(
        self, card: Card, context: DecisionContext | None = None
    ) -> CardStatsBlock | None: ...

    def prior_base(
        self, card: Card, context: DecisionContext | None = None
    ) -> tuple[float, float]: ...

    def posterior_of(self, block: CardStatsBlock | None) -> tuple[float, float]: ...

    def magnitude_of(self, block: CardStatsBlock | None) -> float | None: ...

    def event_deltas(
        self, card: Card, context: DecisionContext | None = None
    ) -> tuple[float, ...]: ...

    def event_weights(
        self, card: Card, context: DecisionContext | None = None
    ) -> tuple[float, ...]: ...

    def evidence_events(
        self, card: Card, context: DecisionContext | None = None
    ) -> tuple[ContextualGain, ...]: ...

    def event_ses(
        self, card: Card, context: DecisionContext | None = None
    ) -> tuple[float | None, ...]: ...

    def staleness_weights(
        self, card: Card, context: DecisionContext | None = None
    ) -> tuple[float, ...]: ...


class EvictionFacingReputation(ReputationModel, Protocol):
    """Reputation that also serves the write-side eviction sweep; the read-side
    decorators delegate this surface to their inner model, so their inners must
    declare it even though no read path calls it."""

    def is_confidently_harmful(self, block: CardStatsBlock | None) -> bool: ...

    def eviction_contexts(self, card: Card) -> tuple[DecisionContext | None, ...]: ...


class DecayCompatibleReputation(EvictionFacingReputation, Protocol):
    """Reputation whose threshold fields can drive posterior-count decay."""

    harm_min_events: int
    harm_quantile: float
    harm_threshold: float
    confident_quantile: float
    confident_threshold: float

    def card_stats_with_staleness(
        self,
        card: Card,
        context: DecisionContext | None = None,
        *,
        staleness_weights: Sequence[float],
    ) -> CardStatsBlock | None: ...


@runtime_checkable
class Auctioneer(Protocol):
    """Decides which candidate cards are injected into a mutation prompt."""

    def run(
        self,
        candidates: list[AuctionCandidate],
        rng: Any,
        *,
        baseline: NoCardGateSummary | None = None,
    ) -> tuple[list[str], list[AuctionBid]]: ...


@runtime_checkable
class CandidateProjector(Protocol):
    """Projects one resolved card/reputation view into auction input and
    resolves the decision-level no-card baseline the auctioneer gates on."""

    def project(
        self,
        *,
        card: Card,
        block: CardStatsBlock | None,
        reputation: ReputationModel,
        context: DecisionContext | None,
        pending_counts: Mapping[str, int] | None = None,
    ) -> AuctionCandidate: ...

    def decision_baseline(
        self, context: DecisionContext | None
    ) -> NoCardGateSummary | None: ...


@runtime_checkable
class ProbePolicy(Protocol):
    """Optional post-auction exploration policy."""

    def apply(
        self,
        *,
        budgeted_ids: list[str],
        slate: list[AuctionBid],
        max_cards: int,
        rng: Any,
    ) -> tuple[list[str], list[AuctionBid]]: ...


@runtime_checkable
class Budgeter(Protocol):
    """Caps the auction's emergent winner set to the mutator-facing budget."""

    def cap(
        self, card_ids: list[str], slate: list[AuctionBid], max_cards: int
    ) -> list[str]: ...


@runtime_checkable
class CardRenderer(Protocol):
    """Renders one card into its mutator-facing text block."""

    def render(self, card: Card | None, block: CardStatsBlock | None = None) -> str: ...
