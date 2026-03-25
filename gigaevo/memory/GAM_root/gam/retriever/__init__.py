# -*- coding: utf-8 -*-
"""
Retriever Module

This module contains all retrieval implementations for the GAM framework.
Retrievers provide different search strategies for finding relevant information.

Available Retrievers:
- DenseRetriever: Semantic search using dense vector embeddings
- BM25Retriever: Keyword-based search using BM25 algorithm
- IndexRetriever: Direct page access by index
"""

from __future__ import annotations

from .base import AbsRetriever
from .index_retriever import IndexRetriever

# Lazy imports to avoid dependency issues
try:
    from .bm25 import BM25Retriever
except Exception:
    BM25Retriever = None  # type: ignore
    import warnings
    warnings.warn("BM25Retriever not available (pyserini dependencies may be missing)")

try:
    from .dense_retriever import DenseRetriever
except Exception:
    DenseRetriever = None  # type: ignore
    import warnings
    warnings.warn("DenseRetriever not available (FlagEmbedding dependencies may be missing)")

try:
    from .chroma_retriever import ChromaRetriever
except Exception:
    ChromaRetriever = None  # type: ignore
    import warnings
    warnings.warn("ChromaRetriever not available (chromadb dependencies may be missing)")

__all__ = [
    "AbsRetriever",
    "IndexRetriever",
]

# Only add retrievers if they were successfully imported
if BM25Retriever is not None:
    __all__.append("BM25Retriever")
if DenseRetriever is not None:
    __all__.append("DenseRetriever")
if ChromaRetriever is not None:
    __all__.append("ChromaRetriever")
