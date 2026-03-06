"""Types for prompt evolution infrastructure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, TypedDict


@dataclass
class CallLog:
    """Log of a single LLM call with cost tracking."""

    prompt_tokens: int
    completion_tokens: int
    cost: float  # Actual cost in dollars
    cost_utilization: float  # Fraction of max_cost budget

    @property
    def total_tokens(self) -> int:
        """Total tokens used in this call."""
        return self.prompt_tokens + self.completion_tokens


class OutputDict(TypedDict):
    """Output from run_prompts."""

    predictions: List[Any]  # Raw response strings
    call_logs: List[CallLog]
