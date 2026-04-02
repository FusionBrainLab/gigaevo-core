"""Platform-backed memory implementation built on top of gigaevo-memory."""

from .shared_memory.memory import (
    AmemGamMemory,
    GigaEvoMemoryBase,
    normalize_memory_card,
)

__all__ = ["AmemGamMemory", "GigaEvoMemoryBase", "normalize_memory_card"]
