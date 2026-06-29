"""Abstract base class for memory backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from gigaevo.memory.shared_memory.card_conversion import AnyCard
    from gigaevo.memory.shared_memory.models import CardT


class GigaEvoMemoryBase(ABC):
    """Abstract base for memory backends.

    Subclasses must implement all abstract methods.
    """

    @property
    @abstractmethod
    def is_ready(self) -> bool:
        """True if memory is fully initialized and ready for operations."""
        ...

    @property
    @abstractmethod
    def checkpoint_path(self) -> Path:
        """Directory the backend persists to.

        The write-path orchestration (write ledger) anchors its sidecar files
        here instead of reaching into a concrete backend's config object.
        """
        ...

    @abstractmethod
    def save_card(self, card: dict[str, Any] | AnyCard) -> str:
        """Persist a memory card directly into the bank."""
        ...

    @abstractmethod
    def save_card_direct(self, card: AnyCard) -> str:
        """Persist one already-normalized card and return its id.

        The write path (gate admit, stats restamp) authors typed cards, so it
        bypasses the dict-normalizing ``save_card``. Part of the formal write
        contract so a new backend satisfies the librarian without mimicking
        ``AmemGamMemory`` internals.
        """
        ...

    @abstractmethod
    def apply_merges(self, merges: list[tuple[str, AnyCard]]) -> list[str]:
        """Overwrite each target id with its pre-computed merged card.

        Returns the ids that were updated, in input order (a failed target is
        logged and skipped). Called by ``CardAdmissionGate.merge`` when the
        librarian folds a new idea into an existing card.
        """
        ...

    @abstractmethod
    def save(self, data: str, category: str = "general") -> str:
        """Save a text description as a new memory card."""
        ...

    @abstractmethod
    def search(self, query: str, memory_state: str | None = None) -> str:
        """Search memory cards."""
        ...

    @abstractmethod
    def nearest(
        self, note: str, k: int, card_type: type[CardT]
    ) -> list[tuple[CardT, float]]:
        """Return the ``k`` cards of ``card_type`` nearest ``note``, as
        (card, distance) ascending.

        The writer's nearest-card primitive: the online pre-gate near-duplicate
        check, reconcile grounding, the consolidation sweep, and exemplar twin
        dedup all rank cards through this one method, parametrized by the kind
        each wants. Part of the formal write contract (it satisfies the
        librarian's ``NeighborSource``) so a new backend works with the writer
        without exposing a concrete vector store to the write path.
        """
        ...

    @abstractmethod
    def get_card(self, card_id: str) -> AnyCard | None:
        """Return a card by ID, or None if not found."""
        ...

    @abstractmethod
    def all_cards_snapshot(self) -> dict[str, AnyCard]:
        """Read-only snapshot of the whole bank, keyed by id.

        The single accessor the write-path orchestration (gate sweep,
        consolidation, stats restamp) uses instead of reaching into the
        backend's private card store. Implementations must return a shallow
        copy so a pass stays stable while the live index mutates.
        """
        ...

    @abstractmethod
    def rebuild(self) -> None:
        """Persist cards, re-export JSONL, and rebuild the GAM index."""
        ...

    @abstractmethod
    def delete(self, memory_id: str) -> bool:
        """Delete a memory card by ID. Return True if removed."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Clean up resources."""
        ...
