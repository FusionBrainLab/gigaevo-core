"""Prompt co-evolution infrastructure for GigaEvo.

Provides stats tracking, stages, and pipeline builder for running a prompt
evolution problem alongside the main HotpotQA evolution run.
"""

from gigaevo.prompts.coevolution.stats import (
    PromptMutationStats,
    PromptStatsProvider,
    RedisPromptStatsProvider,
)

__all__ = [
    "PromptMutationStats",
    "PromptStatsProvider",
    "RedisPromptStatsProvider",
]
