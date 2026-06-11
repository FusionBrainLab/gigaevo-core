from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from gigaevo.memory.core.idea_stats import IdeaStats


@runtime_checkable
class ReputationModel(Protocol):
    """Owns all per-card efficacy statistics derived from injection outcomes."""

    def posterior(
        self, gains: Sequence[float], *, threshold: float = 0.0
    ) -> dict[str, Any]: ...

    def card_posterior(self, card: Any) -> tuple[float, float]: ...

    def is_confidently_harmful(
        self, evolution_statistics: Mapping[str, Any] | None
    ) -> bool: ...

    def compute_injection_posteriors(
        self,
        programs: Sequence[Mapping[str, Any]],
        *,
        higher_is_better: bool = True,
    ) -> dict[str, dict[str, Any]]: ...


@runtime_checkable
class MemoryAdmitter(Protocol):
    """Decides which tracked ideas enter the shared bank from the per-idea
    origin-analysis rows (one ``IdeaStats`` per idea x quartile block)."""

    def select(self, stats: Sequence[IdeaStats]) -> list[IdeaStats]: ...


@runtime_checkable
class Auctioneer(Protocol):
    """Decides which candidate cards are injected into a mutation prompt."""

    def run(
        self, candidates: list[tuple[str, float, float]], rng: Any
    ) -> tuple[list[str], list[dict]]: ...


@runtime_checkable
class CardRetriever(Protocol):
    """Runs the backend research pass and resolves shortlisted card ids."""

    def research(self, query: str, *, planning_request: str | None = None) -> Any: ...

    def get_card(self, card_id: str) -> Any: ...


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
        self, card_ids: list[str], slate: list[dict], max_cards: int
    ) -> list[str]: ...


@runtime_checkable
class CardRenderer(Protocol):
    """Renders one card into its mutator-facing text block."""

    def render(self, card: Any) -> str: ...


@runtime_checkable
class Deduplicator(Protocol):
    """Reconciles an incoming card against the existing bank into a
    DedupDecision (add / discard / update-with-merges)."""

    def reconcile(self, card: Any, bank: Mapping[str, Any]) -> Any: ...


@runtime_checkable
class Evictor(Protocol):
    """Decides which cards must leave the bank based on their reputation."""

    def should_evict(self, card: Any) -> bool: ...

    def sweep(self, bank: Mapping[str, Any]) -> Sequence[str]: ...
