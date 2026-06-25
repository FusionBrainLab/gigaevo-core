from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from gigaevo.memory.context import DecisionContext
from gigaevo.memory.core.auctioneer import AuctionBid, AuctionCandidate
from gigaevo.memory.shared_memory.card_dedup import DedupDecision
from gigaevo.memory.shared_memory.models import (
    AnyCard,
    CardStatsBlock,
)


@runtime_checkable
class ReputationModel(Protocol):
    """Owns all per-card efficacy statistics derived from injection outcomes."""

    def posterior(
        self, gains: Sequence[float], *, threshold: float = 0.0
    ) -> CardStatsBlock: ...

    def card_stats(
        self, card: AnyCard, context: DecisionContext | None = None
    ) -> CardStatsBlock | None: ...

    def card_posterior(
        self, card: AnyCard, context: DecisionContext | None = None
    ) -> tuple[float, float]: ...

    def card_magnitude(
        self, card: AnyCard, context: DecisionContext | None = None
    ) -> float | None: ...

    def is_confidently_harmful(self, block: CardStatsBlock | None) -> bool: ...


@runtime_checkable
class Auctioneer(Protocol):
    """Decides which candidate cards are injected into a mutation prompt."""

    def run(
        self, candidates: list[AuctionCandidate], rng: Any
    ) -> tuple[list[str], list[AuctionBid]]: ...


@runtime_checkable
class CardRetriever(Protocol):
    """Runs the backend research pass and resolves shortlisted card ids."""

    def research(
        self,
        query: str,
        *,
        planning_request: str | None = None,
        exclude_ids: frozenset[str] = frozenset(),
        random_drop_dose: int = 0,
    ) -> Any: ...

    def get_card(self, card_id: str) -> AnyCard | None: ...


@runtime_checkable
class CardExcluder(Protocol):
    """Decides which card ids must be pruned from the candidate pool BEFORE
    retrieval ranks them (filter-first lineage gate)."""

    def exclude_for(self, program: Any) -> frozenset[str]: ...

    def dose_for(self, program: Any) -> int: ...


@runtime_checkable
class CardShortlister(Protocol):
    """Builds the selector-LLM query and parses its structured decision back
    into an ordered candidate-id shortlist."""

    def build_query(
        self,
        *,
        parents: list[Any],
        mutation_mode: str,
        task_description: str,
        metrics_description: str,
        max_cards: int,
        parent_contexts: list[str] | None = None,
    ) -> str: ...

    def build_core_request(
        self,
        *,
        parents: list[Any],
        mutation_mode: str,
        task_description: str,
        metrics_description: str,
        max_cards: int,
        parent_contexts: list[str] | None = None,
    ) -> str: ...

    def shortlist(self, raw_memory: Any) -> list[str]: ...


@runtime_checkable
class Budgeter(Protocol):
    """Caps the auction's emergent winner set to the mutator-facing budget."""

    def cap(
        self, card_ids: list[str], slate: list[AuctionBid], max_cards: int
    ) -> list[str]: ...


@runtime_checkable
class CardRenderer(Protocol):
    """Renders one card into its mutator-facing text block from its resolved
    ``card_stats`` block (the same authority the auction bid on)."""

    def render(
        self, card: AnyCard | None, block: CardStatsBlock | None = None
    ) -> str: ...


@runtime_checkable
class Deduplicator(Protocol):
    """Reconciles an incoming card against the existing bank into a
    DedupDecision (add / discard / update-with-merges)."""

    def reconcile(
        self, card: AnyCard, bank: Mapping[str, AnyCard]
    ) -> DedupDecision: ...


@runtime_checkable
class Evictor(Protocol):
    """Decides which cards must leave the bank based on their reputation."""

    def should_evict(self, card: AnyCard) -> bool: ...

    def sweep(self, bank: Mapping[str, AnyCard]) -> Sequence[str]: ...
