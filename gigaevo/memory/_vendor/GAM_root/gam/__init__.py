from __future__ import annotations

from gigaevo.memory._vendor.GAM_root.gam.agents import ResearchAgent
from gigaevo.memory._vendor.GAM_root.gam.generator import AbsGenerator, AMemGenerator
from gigaevo.memory._vendor.GAM_root.gam.retriever import AbsRetriever, ChromaRetriever, IndexRetriever
from gigaevo.memory._vendor.GAM_root.gam.schemas import (
    EnoughDecision,
    Hit,
    InMemoryMemoryStore,
    InMemoryPageStore,
    MemoryState,
    MemoryUpdate,
    Page,
    ReflectionDecision,
    ResearchOutput,
    Result,
    SearchPlan,
)

__version__ = "0.1.0"
__all__ = [
    "ResearchAgent",
    "AbsGenerator",
    "AMemGenerator",
    "AbsRetriever",
    "IndexRetriever",
    "ChromaRetriever",
    "MemoryState",
    "Page",
    "MemoryUpdate",
    "SearchPlan",
    "Hit",
    "Result",
    "EnoughDecision",
    "ReflectionDecision",
    "ResearchOutput",
    "InMemoryMemoryStore",
    "InMemoryPageStore",
]
