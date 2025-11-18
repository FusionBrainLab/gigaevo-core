"""LangGraph-based agents for LLM workflows.

This package contains LangGraph state machine agents that encapsulate
LLM-based workflows with clean separation of concerns and extensibility
for future multi-step reasoning, tool calling, and reflection loops.

Each agent defines its own domain-specific state schema using TypedDict.
"""

from gigaevo.llm.agents.base import LangGraphAgent
from gigaevo.llm.agents.factories import (
    create_insights_agent,
    create_lineage_agent,
    create_scoring_agent,
)

__all__ = [
    "LangGraphAgent",
    "create_insights_agent",
    "create_lineage_agent",
    "create_scoring_agent",
]
