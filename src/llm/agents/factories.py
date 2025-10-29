"""Factory functions for creating pre-configured agents.

These factories encapsulate all the complexity of agent creation:
- Loading prompts from files
- Formatting system prompts with task-specific info
- Creating metrics formatters
- Wiring everything together

Stages just call these factories and use the agents - no LLM logic in stages!
"""

from langchain_openai import ChatOpenAI

from src.llm.agents.insights import InsightsAgent
from src.llm.agents.lineage import LineageAgent
from src.llm.agents.scoring import ScoringAgent
from src.llm.models import MultiModelRouter
from src.programs.metrics.context import MetricsContext
from src.programs.metrics.formatter import MetricsFormatter
from src.prompts import InsightsPrompts, LineagePrompts, ScoringPrompts


def create_insights_agent(
    llm: ChatOpenAI | MultiModelRouter,
    task_description: str,
    metrics_context: MetricsContext,
    max_insights: int = 7,
) -> InsightsAgent:
    """Create a fully configured insights agent.

    This factory does ALL the setup:
    - Loads prompts from files
    - Formats system prompt with task description
    - Creates metrics formatter
    - Returns ready-to-use agent

    Args:
        llm: LangChain chat model or multi-model router
        task_description: Description of the evolutionary task
        metrics_context: Metrics context for formatting
        max_insights: Maximum number of insights to generate

    Returns:
        Ready-to-use InsightsAgent

    Example:
        >>> agent = create_insights_agent(
        ...     llm=llm,
        ...     task_description="Maximize triangle areas",
        ...     metrics_context=metrics_context
        ... )
        >>> insights = await agent.arun(program)
    """
    # Load prompts from files
    system_template = InsightsPrompts.system()
    user_template = InsightsPrompts.user()
    metrics_formatter = MetricsFormatter(metrics_context)

    # Format system prompt with task description
    system_prompt = system_template.format(
        task_description=task_description,
        max_insights=max_insights,
        metrics_description=metrics_formatter.format_metrics_description(),
    )

    # Return configured agent
    return InsightsAgent(
        llm=llm,
        system_prompt_template=system_prompt,
        user_prompt_template=user_template,
        max_insights=max_insights,
        metrics_formatter=metrics_formatter,
    )


def create_lineage_agent(
    llm: ChatOpenAI | MultiModelRouter,
    task_description: str,
    metrics_context: MetricsContext,
) -> LineageAgent:
    """Create a fully configured lineage agent.

    This factory does ALL the setup:
    - Loads prompts from files
    - Creates metrics formatter
    - Returns ready-to-use agent

    Args:
        llm: LangChain chat model or multi-model router
        task_description: Description of the optimization task
        metrics_context: Metrics context for formatting

    Returns:
        Ready-to-use LineageAgent

    Example:
        >>> agent = create_lineage_agent(
        ...     llm=llm,
        ...     task_description="Maximize triangle areas",
        ...     metrics_context=metrics_context
        ... )
        >>> insights = await agent.arun(parent, child)
    """
    # Load prompts from files
    system_prompt = LineagePrompts.system()
    user_template = LineagePrompts.user()

    # Create metrics formatter
    metrics_formatter = MetricsFormatter(metrics_context)

    # Return configured agent
    return LineageAgent(
        llm=llm,
        system_prompt=system_prompt,
        user_prompt_template=user_template,
        task_description=task_description,
        metrics_formatter=metrics_formatter,
    )


def create_scoring_agent(
    llm: ChatOpenAI | MultiModelRouter,
    trait_description: str,
    max_score: float,
) -> ScoringAgent:
    """Create a fully configured scoring agent.

    This factory does ALL the setup:
    - Loads prompts from files
    - Returns ready-to-use agent

    Args:
        llm: LangChain chat model or multi-model router

    Returns:
        Ready-to-use ScoringAgent

    Example:
        >>> agent = create_scoring_agent(
        ...     llm=llm,
        ...     trait_description="code novelty",
        ...     max_score=1.0
        ... )
        >>> score = await agent.arun(program)
    """
    # Load prompts from files
    system_prompt = ScoringPrompts.system()
    user_template = ScoringPrompts.user()

    # Return configured agent
    return ScoringAgent(
        llm=llm,
        system_prompt=system_prompt,
        user_prompt_template=user_template,
        trait_description=trait_description,
        max_score=max_score,
    )
