from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from gigaevo.memory.core.auctioneer import AuctionBid, AuctionCandidate
from gigaevo.memory.core.idea_stats import IdeaStats
from gigaevo.memory.efficacy import EfficacyScorer
from gigaevo.memory.shared_memory.card_dedup import DedupDecision
from gigaevo.memory.shared_memory.injection_posterior import InjectionOutcome
from gigaevo.memory.shared_memory.models import (
    AnyCard,
    CardStatsBlock,
    EvolutionStatistics,
)


@runtime_checkable
class ReputationModel(Protocol):
    """Owns all per-card efficacy statistics derived from injection outcomes."""

    def scorer(self) -> EfficacyScorer: ...

    def posterior(
        self, gains: Sequence[float], *, threshold: float = 0.0
    ) -> CardStatsBlock: ...

    def card_posterior(self, card: AnyCard) -> tuple[float, float]: ...

    def is_confidently_harmful(
        self, evolution_statistics: EvolutionStatistics | None
    ) -> bool: ...

    def compute_injection_posteriors(
        self,
        programs: Sequence[InjectionOutcome],
        *,
        higher_is_better: bool = True,
    ) -> dict[str, CardStatsBlock]: ...


@runtime_checkable
class MemoryAdmitter(Protocol):
    """Decides which tracked ideas enter the shared bank from the per-idea
    origin-analysis rows (one ``IdeaStats`` per idea x quartile block)."""

    def select(self, stats: Sequence[IdeaStats]) -> list[IdeaStats]: ...


@runtime_checkable
class Auctioneer(Protocol):
    """Decides which candidate cards are injected into a mutation prompt."""

    def run(
        self, candidates: list[AuctionCandidate], rng: Any
    ) -> tuple[list[str], list[AuctionBid]]: ...


@runtime_checkable
class CardRetriever(Protocol):
    """Runs the backend research pass and resolves shortlisted card ids."""

    def research(self, query: str, *, planning_request: str | None = None) -> Any: ...

    def get_card(self, card_id: str) -> AnyCard | None: ...


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
    ) -> str: ...

    def build_core_request(
        self,
        *,
        parents: list[Any],
        mutation_mode: str,
        task_description: str,
        metrics_description: str,
        max_cards: int,
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
    """Renders one card into its mutator-facing text block."""

    def render(self, card: AnyCard | None) -> str: ...


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
