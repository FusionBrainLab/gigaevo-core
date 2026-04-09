"""Abstract base class for memory backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from gigaevo.memory.shared_memory.card_conversion import AnyCard


class GigaEvoMemoryBase(ABC):
    """Abstract base for memory backends.

    Subclasses must implement all abstract methods.
    """

    @property
    @abstractmethod
    def is_ready(self) -> bool:
        """True if memory is fully initialized and ready for operations."""
        ...

    @abstractmethod
    def save_card(self, card: dict[str, Any] | AnyCard) -> str:
        """Save a memory card, with optional dedup against existing cards."""
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
    def get_card(self, card_id: str) -> AnyCard | None:
        """Return a card by ID, or None if not found."""
        ...

    @abstractmethod
    def get_card_write_stats(self) -> dict[str, int]:
        """Return write statistics for cards."""
        ...

    @abstractmethod
    def rebuild(self) -> None:
        """Persist cards, re-export JSONL, rebuild GAM index and dedup retrievers."""
        ...

    @abstractmethod
    def delete(self, memory_id: str) -> bool:
        """Delete a memory card by ID. Return True if removed."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Clean up resources."""
        ...
