"""Insights stage - clean wrapper using LangGraphStage.

Usage:
    >>> from src.programs.stages.insights import create_insights_stage
    >>> stage = create_insights_stage(llm, task_description, metrics_context)
"""

from typing import Any

from langchain_openai import ChatOpenAI

from src.llm.agents.factories import create_insights_agent
from src.llm.models import MultiModelRouter
from src.programs.metrics.context import MetricsContext
from src.programs.stages.langgraph_stage import LangGraphStage


def create_insights_stage(
    llm: ChatOpenAI | MultiModelRouter,
    task_description: str,
    metrics_context: MetricsContext,
    max_insights: int = 7,
    **kwargs: Any
) -> LangGraphStage:
    """Create a ready-to-use insights stage.
    
    Args:
        llm: LangChain chat model or router
        task_description: Description of the evolutionary task
        metrics_context: Metrics context for formatting
        max_insights: Maximum number of insights to generate
        **kwargs: Additional arguments passed to LangGraphStage (e.g., timeout, stage_name)
        
    Returns:
        Configured LangGraphStage wrapping InsightsAgent
        
    Example:
        >>> stage = create_insights_stage(
        ...     llm=my_llm,
        ...     task_description="Maximize triangle areas",
        ...     metrics_context=ctx,
        ...     timeout=1200.0,
        ...     stage_name="GenerateInsights"
        ... )
    """
    # Create agent via factory
    agent = create_insights_agent(
        llm=llm,
        task_description=task_description,
        metrics_context=metrics_context,
        max_insights=max_insights,
    )
    
    # Wrap in generic stage
    return LangGraphStage(
        agent=agent,
        prepare_inputs=lambda program: {"program": program},
        **kwargs
    )
