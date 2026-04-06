"""Abstract base class for memory backends."""

from __future__ import annotations

from abc import ABC, abstractmethod


class GigaEvoMemoryBase(ABC):
    """Abstract base for memory backends.

    Subclasses must implement ``save``, ``search``, and ``delete``.
    """

    @abstractmethod
    def save(self, data: str) -> str:
        """Persist a text observation and return a card identifier."""
        ...

    @abstractmethod
    def search(self, query: str) -> str:
        """Search memory and return a formatted result string."""
        ...

    @abstractmethod
    def delete(self, memory_id: str) -> bool:
        """Delete a memory card by ID. Return True if removed."""
        ...
