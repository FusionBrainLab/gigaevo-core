"""Abstract base class for memory backends."""

from __future__ import annotations


class GigaEvoMemoryBase:
    """Abstract base for memory backends.

    Subclasses must implement save, search, and delete.
    """

    def save(self, data: str) -> str:
        raise NotImplementedError

    def search(self, query: str) -> str:
        raise NotImplementedError

    def delete(self, memory_id: str) -> bool:
        raise NotImplementedError
