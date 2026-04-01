"""
Schemas Module

This module exposes all core data models and protocol definitions for the GAM (General-Agentic-Memory) framework.
It organizes memory, page, search, tool, and result schemas for unified import and type safety across the system.
"""
from .memory import InMemoryMemoryStore, MemoryState, MemoryStore, MemoryUpdate
from .page import InMemoryPageStore, Page, PageStore
from .result import (
    EnoughDecision,
    ExperimentalDecision,
    GenerateRequests,
    ReflectionDecision,
    ResearchOutput,
    Result,
    TopIdea,
)
from .search import Hit, Retriever, SearchPlan
from .tools import Tool, ToolRegistry, ToolResult

# =============================
# Model rebuilding for forward references
# =============================
# 显式重建模型以确保在并发环境下所有前向引用（如 'Page'）都正确解析
# 这对于多线程环境尤为重要
MemoryUpdate.model_rebuild()
ResearchOutput.model_rebuild()

# JSON Schema constants for LLM and system validation
PLANNING_SCHEMA = SearchPlan.model_json_schema()
INTEGRATE_SCHEMA = Result.model_json_schema()
INFO_CHECK_SCHEMA = EnoughDecision.model_json_schema()
GENERATE_REQUESTS_SCHEMA = GenerateRequests.model_json_schema()
EXPERIMENTAL_DECISION_SCHEMA = ExperimentalDecision.model_json_schema()

__all__ = [
    "MemoryState", "MemoryUpdate", "MemoryStore", "InMemoryMemoryStore",
    "Page", "PageStore", "InMemoryPageStore",
    "SearchPlan", "Retriever", "Hit",
    "ToolResult", "Tool", "ToolRegistry",
    "Result", "EnoughDecision", "ReflectionDecision", "ResearchOutput", "GenerateRequests",
    "TopIdea", "ExperimentalDecision",
    "PLANNING_SCHEMA", "INTEGRATE_SCHEMA", "INFO_CHECK_SCHEMA", "GENERATE_REQUESTS_SCHEMA",
    "EXPERIMENTAL_DECISION_SCHEMA",
]
