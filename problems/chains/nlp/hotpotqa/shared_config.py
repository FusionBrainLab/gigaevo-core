"""Shared configuration for nlp/hotpotqa chain evolution.

Delegates to the canonical hotpotqa package for all dataset and retriever logic.
LLM endpoint is env-driven via HOTPOTQA_CHAIN_URL.
"""

from problems.chains.hotpotqa.shared_config import (
    BM25S_INDEX_DIR,
    CORPUS_PATH,
    DATASET_CONFIG,
    LLM_CONFIG,
    build_retriever,
    load_context,
    load_jsonl,
    outer_context_builder,
    preprocess_sample,
)

__all__ = [
    "LLM_CONFIG",
    "build_retriever",
    "load_context",
    "outer_context_builder",
    "preprocess_sample",
    "load_jsonl",
    "DATASET_CONFIG",
    "BM25S_INDEX_DIR",
    "CORPUS_PATH",
]
