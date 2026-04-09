"""Abstract base class for memory backends."""

from __future__ import annotations

from abc import ABC, abstractmethod


class GigaEvoMemoryBase(ABC):
    """Abstract base for memory backends.

    Subclasses must implement ``save``, ``search``, and ``delete``.
    """

    @abstractmethod
    def save(self, data: str, category: str = "general") -> str:
        """Save a text description as a new memory card."""
        ...

    @abstractmethod
    def search(self, query: str, memory_state: str | None = None) -> str:
        """Search memory cards."""
        ...

    @abstractmethod
    def delete(self, memory_id: str) -> bool:
        """Delete a memory card by ID. Return True if removed."""
        ...
