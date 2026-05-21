"""Shared configuration for nlp/aime chain evolution.

Delegates to the canonical aime package for all dataset and LLM logic.
Math olympiad problem solving with up to 3 LLM reasoning steps. No tools.
"""

from problems.chains.aime.shared_config import (
    DATASET_CONFIG,
    LLM_CONFIG,
    load_context,
    outer_context_builder,
)

__all__ = [
    "DATASET_CONFIG",
    "LLM_CONFIG",
    "load_context",
    "outer_context_builder",
]
