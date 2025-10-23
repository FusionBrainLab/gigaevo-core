"""Lineage insights stage - clean wrapper using LangGraphStage.

Usage:
    >>> from src.programs.stages.insights_lineage import create_lineage_stage
    >>> stage = create_lineage_stage(
    ...     llm, task_description, metrics_context, storage,
    ...     ancestry_selector=AncestrySelector(...)
    ... )
"""

from datetime import datetime
from typing import Any

from langchain_openai import ChatOpenAI

from src.database.program_storage import ProgramStorage
from src.llm.agents.factories import create_lineage_agent
from src.llm.models import MultiModelRouter
from src.programs.metrics.context import MetricsContext
from src.programs.program import Program
from src.programs.stages.ancestry_selector import AncestrySelector
from src.programs.stages.langgraph_stage import LangGraphStage


class LineageStage(LangGraphStage):
    """Lineage analysis stage wrapper.
    
    This stage performs expensive LLM-based lineage analysis (parent→child)
    and is cacheable. The individual parent→child analysis doesn't change.
    
    Dynamic ancestor/descendant context is handled by MutationContextStage
    which fetches current lineage data (including fresh descendants) directly
    from program.lineage.children and stage_results.
    """
    pass


def create_lineage_stage(
    llm: ChatOpenAI | MultiModelRouter,
    task_description: str,
    metrics_context: MetricsContext,
    storage: ProgramStorage,
    ancestry_selector: AncestrySelector,
    **kwargs: Any
) -> LineageStage:
    """Create a ready-to-use lineage insights stage.
    
    Args:
        llm: LangChain chat model or router
        task_description: Description of the optimization task
        metrics_context: Metrics context for formatting
        storage: Program storage for fetching parent programs
        ancestry_selector: Selector for choosing which parent(s) to analyze.
                          If None, uses first parent with best_fitness strategy.
        **kwargs: Additional arguments passed to LineageStage (e.g., timeout, stage_name)
        
    Returns:
        Configured LineageStage (cacheable) wrapping LineageAgent
        
    Example:
        >>> stage = create_lineage_stage(
        ...     llm=my_llm,
        ...     task_description="Maximize triangle areas",
        ...     metrics_context=ctx,
        ...     storage=storage,
        ...     ancestry_selector=AncestrySelector(
        ...         storage=storage,
        ...         metrics_context=ctx,
        ...         strategy="best_fitness",
        ...         max_parents=1
        ...     ),
        ...     timeout=1200.0,
        ... )
    """

    
    # Create agent via factory
    agent = create_lineage_agent(
        llm=llm,
        task_description=task_description,
        metrics_context=metrics_context,
    )
    
    # Define async input preparation that fetches parents from storage
    async def prepare_lineage_inputs(program: Program) -> dict[str, list[Program] | Program]:
        """Select and fetch parent programs for lineage analysis."""
        # Use ancestry selector to get parent ID(s)
        parent_ids = await ancestry_selector.select(program)
        
        if not parent_ids:
            raise ValueError("No parents selected for lineage analysis")
        
        # Fetch all selected parents
        parents = []
        for parent_id in parent_ids:
            parent = await storage.get(parent_id)
            if parent:
                parents.append(parent)
        
        if not parents:
            raise ValueError(f"No parent programs found in storage")
        
        return {
            "parents": parents,
            "child": program
        }
    
    return LineageStage(
        agent=agent,
        prepare_inputs=prepare_lineage_inputs,
        should_skip=lambda program: (
            program.is_root,
            "No parents in lineage (likely a root program)"
        ),
        **kwargs
    )
