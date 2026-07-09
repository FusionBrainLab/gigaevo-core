"""Shared read-side component protocols."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from gigaevo.memory.cards import Card, CardStatsBlock, ContextualGain, DecisionContext
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

    def card_stats(
        self, card: Card, context: DecisionContext | None = None
    ) -> CardStatsBlock | None: ...

    def posterior_of(self, block: CardStatsBlock | None) -> tuple[float, float]: ...

    def magnitude_of(self, block: CardStatsBlock | None) -> float | None: ...

    def is_confidently_harmful(self, block: CardStatsBlock | None) -> bool: ...

    def event_deltas(
        self, card: Card, context: DecisionContext | None = None
    ) -> tuple[float, ...]: ...

    def event_weights(
        self, card: Card, context: DecisionContext | None = None
    ) -> tuple[float, ...]: ...

    def evidence_events(
        self, card: Card, context: DecisionContext | None = None
    ) -> tuple[ContextualGain, ...]: ...

    def staleness_weight(
        self, card: Card, context: DecisionContext | None = None
    ) -> float: ...


class DecayCompatibleReputation(ReputationModel, Protocol):
    """Reputation whose threshold fields can drive posterior-count decay."""

    harm_min_events: int
    harm_quantile: float
    harm_threshold: float
    confident_quantile: float
    confident_threshold: float


class NoCardBaseline(Protocol):
    """Fitted no-card progress baseline used by gain stamping."""

    has_evidence: bool

    def baseline_for(self, outcome: Any) -> float: ...


@runtime_checkable
class Auctioneer(Protocol):
    """Decides which candidate cards are injected into a mutation prompt."""

    def run(
        self, candidates: list[AuctionCandidate], rng: Any
    ) -> tuple[list[str], list[AuctionBid]]: ...


@runtime_checkable
class AuctionCandidateProjector(Protocol):
    """Projects one resolved card/reputation view into auction input."""

    def project(
        self,
        *,
        card: Card,
        block: CardStatsBlock | None,
        reputation: ReputationModel,
        context: DecisionContext | None,
    ) -> AuctionCandidate: ...


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
