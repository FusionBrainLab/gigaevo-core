"""Core memory orchestrator: card storage, search, sync, and dedup.

The concrete backend ``AmemGamMemory`` is intentionally NOT re-exported here:
its Redis/Chroma/GAM closure is heavy. Import it from
``gigaevo.memory.shared_memory.memory`` directly so the package root stays
import-light for leaf tools.
"""

from __future__ import annotations

from gigaevo.memory.shared_memory.base import GigaEvoMemoryBase
from gigaevo.memory.shared_memory.card_conversion import normalize_memory_card
from gigaevo.memory.shared_memory.models import (
    AnyCard,
    MemoryCard,
    ProgramCard,
    Strategy,
)

__all__ = [
    "AnyCard",
    "GigaEvoMemoryBase",
    "MemoryCard",
    "ProgramCard",
    "Strategy",
    "normalize_memory_card",
]
