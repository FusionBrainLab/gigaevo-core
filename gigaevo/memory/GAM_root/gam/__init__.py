# -*- coding: utf-8 -*-
"""
General Agentic Memory (GAM) Framework

A dual-agent architecture for building long-term memory with deep research capabilities.

Key Components:
- MemoryAgent: Builds structured memory from raw messages
- ResearchAgent: Performs multi-iteration research with reflection
"""

from __future__ import annotations

# Core agents
from GAM_root.gam.agents import MemoryAgent, ResearchAgent

# Generators
from GAM_root.gam.generator import AbsGenerator, OpenAIGenerator, VLLMGenerator

# Retrievers
from GAM_root.gam.retriever import AbsRetriever, IndexRetriever

# 尝试导入可选检索器
try:
    from GAM_root.gam.retriever import BM25Retriever
except ImportError:
    BM25Retriever = None  # type: ignore

try:
    from GAM_root.gam.retriever import DenseRetriever
except ImportError:
    DenseRetriever = None  # type: ignore
try:
    from GAM_root.gam.retriever import ChromaRetriever
except ImportError:
    ChromaRetriever = None  # type: ignore

# Configurations
from GAM_root.gam.config import (
    OpenAIGeneratorConfig,
    VLLMGeneratorConfig,
    DenseRetrieverConfig,
    BM25RetrieverConfig,
    IndexRetrieverConfig
)

# Schemas
from GAM_root.gam.schemas import (
    MemoryState,
    Page,
    MemoryUpdate,
    SearchPlan,
    Hit,
    Result,
    EnoughDecision,
    ReflectionDecision,
    ResearchOutput,
    InMemoryMemoryStore,
    InMemoryPageStore
)

__version__ = "0.1.0"
__all__ = [
    # Core agents
    "MemoryAgent",
    "ResearchAgent",
    
    # Generators
    "AbsGenerator",
    "OpenAIGenerator",
    "VLLMGenerator",
    
    # Retrievers
    "AbsRetriever",
    "IndexRetriever",
    "BM25Retriever",
    "DenseRetriever",
    "ChromaRetriever",
    
    # Configurations
    "OpenAIGeneratorConfig",
    "VLLMGeneratorConfig",
    "DenseRetrieverConfig",
    "BM25RetrieverConfig",
    "IndexRetrieverConfig",
    
    # Schemas
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
