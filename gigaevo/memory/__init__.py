"""GigaEvo Memory — card-based memory system for evolution-guided mutation.

Subpackages:
    shared_memory/  — Core orchestrator: card storage, search, sync, dedup
    _vendor/        — Vendored MIT libs: A_mem (vector store), GAM_root (research agent)
    ideas_tracker/  — Writer subsystem: the librarian authors cards from completed runs

Public API:
    AmemGamMemory      — main memory backend (local or API-backed)
    MemoryConfig       — configuration for AmemGamMemory (Pydantic)
    ApiConfig          — API connection settings
    GamConfig          — GAM retriever settings
    MemoryCard         — general idea/insight card
    ProgramCard        — top-performing program card
    AnyCard            — union type (MemoryCard | ProgramCard)
    normalize_memory_card — normalize raw dict into typed card model
    GigaEvoMemoryBase  — abstract base for memory backends
"""

from __future__ import annotations

from gigaevo.memory.shared_memory.base import GigaEvoMemoryBase
from gigaevo.memory.shared_memory.card_conversion import normalize_memory_card
from gigaevo.memory.shared_memory.memory import AmemGamMemory
from gigaevo.memory.shared_memory.memory_config import (
    ApiConfig,
    GamConfig,
    MemoryConfig,
)
from gigaevo.memory.shared_memory.models import (
    AnyCard,
    MemoryCard,
    ProgramCard,
    Strategy,
)
from gigaevo.memory.system import MemorySystem

__all__ = [
    "AmemGamMemory",
    "ApiConfig",
    "AnyCard",
    "GamConfig",
    "GigaEvoMemoryBase",
    "MemoryCard",
    "MemoryConfig",
    "MemorySystem",
    "ProgramCard",
    "Strategy",
    "normalize_memory_card",
]
