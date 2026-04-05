"""GigaEvo Memory — card-based memory system for evolution-guided mutation.

Public API:
    AmemGamMemory      — main memory backend (local or API-backed)
    MemoryConfig       — configuration for AmemGamMemory (Pydantic)
    ApiConfig          — API connection settings
    GamConfig          — GAM retriever settings
    MemoryCard         — general idea/insight card
    ProgramCard        — top-performing program card
    AnyCard            — union type (MemoryCard | ProgramCard)
    ConnectedIdea      — idea reference linked to a program
    normalize_memory_card — normalize raw dict into typed card model
    GigaEvoMemoryBase  — abstract base for memory backends
"""

from __future__ import annotations

from gigaevo.memory.shared_memory.card_conversion import (
    GigaEvoMemoryBase,
    normalize_memory_card,
)
from gigaevo.memory.shared_memory.memory import AmemGamMemory
from gigaevo.memory.shared_memory.memory_config import (
    ApiConfig,
    GamConfig,
    MemoryConfig,
)
from gigaevo.memory.shared_memory.models import (
    AnyCard,
    ConnectedIdea,
    LocalMemorySnapshot,
    MemoryCard,
    MemoryCardExplanation,
    ProgramCard,
    Strategy,
)

__all__ = [
    "AmemGamMemory",
    "ApiConfig",
    "AnyCard",
    "ConnectedIdea",
    "GamConfig",
    "GigaEvoMemoryBase",
    "LocalMemorySnapshot",
    "MemoryCard",
    "MemoryCardExplanation",
    "MemoryConfig",
    "ProgramCard",
    "Strategy",
    "normalize_memory_card",
]
