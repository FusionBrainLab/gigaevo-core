"""
Schemas Module

This module exposes all core data models and protocol definitions for the GAM (General-Agentic-Memory) framework.
It organizes memory, page, search, tool, and result schemas for unified import and type safety across the system.
"""

from __future__ import annotations

from .memory import InMemoryMemoryStore, MemoryState, MemoryStore, MemoryUpdate
from .page import InMemoryPageStore, Page, PageStore
from .result import (
    Decision,
    ResearchOutput,
    Result,
    TopIdea,
)
from .search import Hit, Retriever, SearchPlan
from .tools import Tool, ToolRegistry, ToolResult

# =============================
# Model rebuilding for forward references
# =============================
# Rebuild models so forward references such as Page resolve consistently in
# concurrent environments.
MemoryUpdate.model_rebuild()
ResearchOutput.model_rebuild()

# JSON Schema constants for LLM and system validation
PLANNING_SCHEMA = SearchPlan.model_json_schema()
DECISION_SCHEMA = Decision.model_json_schema()

__all__ = [
    "MemoryState",
    "MemoryUpdate",
    "MemoryStore",
    "InMemoryMemoryStore",
    "Page",
    "PageStore",
    "InMemoryPageStore",
    "SearchPlan",
    "Retriever",
    "Hit",
    "ToolResult",
    "Tool",
    "ToolRegistry",
    "Result",
    "ResearchOutput",
    "TopIdea",
    "Decision",
    "PLANNING_SCHEMA",
    "DECISION_SCHEMA",
]
