"""Factory functions for creating pre-configured agents.

These factories encapsulate all the complexity of agent creation:
- Loading prompts from files
- Formatting system prompts with task-specific info
- Creating metrics formatters
- Wiring everything together

Stages just call these factories and use the agents - no LLM logic in stages!
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from langchain_openai import ChatOpenAI

from gigaevo.llm.agents.insights import InsightsAgent
from gigaevo.llm.agents.lineage import LineageAgent
from gigaevo.llm.agents.memory_selector import MemorySelectorAgent
from gigaevo.llm.agents.mutation import MutationAgent
from gigaevo.llm.agents.scoring import ScoringAgent
from gigaevo.llm.models import MultiModelRouter
from gigaevo.programs.metrics.context import MetricsContext
from gigaevo.programs.metrics.formatter import MetricsFormatter
from gigaevo.prompts import (
    InsightsPrompts,
    LineagePrompts,
    ScoringPrompts,
)

if TYPE_CHECKING:
    from gigaevo.prompts.fetcher import PromptFetcher


def create_mutation_agent(
    llm: ChatOpenAI | MultiModelRouter,
    task_description: str,
    metrics_context: MetricsContext,
    mutation_mode: str = "rewrite",
    prompts_dir: str | Path | None = None,
    prompt_fetcher: PromptFetcher | None = None,
) -> MutationAgent:
    """Create a fully configured mutation agent.

    This factory does ALL the setup:
    - Loads prompts from files (or initial fetch from prompt_fetcher)
    - Formats system prompt with task description and metrics
    - Returns ready-to-use agent

    Args:
        llm: LangChain chat model or multi-model router
        task_description: Description of the optimization task
        metrics_context: Metrics context for formatting
        mutation_mode: "rewrite" or "diff"
        prompts_dir: Optional prompts directory (e.g. config.prompts.dir). If None,
            package defaults are used. Ignored when prompt_fetcher is provided.
        prompt_fetcher: Optional PromptFetcher for dynamic prompt co-evolution.
            When provided, the fetcher is used both for the initial system prompt
            and for refreshing it on every build_prompt() call (if is_dynamic=True).
            When None, a FixedDirPromptFetcher(prompts_dir) is created.

    Returns:
        Ready-to-use MutationAgent

    Example:
        >>> agent = create_mutation_agent(
        ...     llm=llm,
        ...     task_description="Maximize triangle areas",
        ...     metrics_context=metrics_context,
        ...     mutation_mode="rewrite"
        ... )
        >>> result = await agent.arun(parents, mutation_mode)
    """
    from gigaevo.prompts.fetcher import FixedDirPromptFetcher

    # Use provided fetcher or create a default fixed-dir one
    fetcher = (
        prompt_fetcher
        if prompt_fetcher is not None
        else FixedDirPromptFetcher(prompts_dir)
    )

    # Load initial system prompt template via the fetcher
    initial_fetch = fetcher.fetch("mutation", "system")
    system_template = initial_fetch.text

    # Load user template via fetcher (co-evolved if champion has user prompt, else from files)
    user_fetch = fetcher.fetch("mutation", "user")
    user_template = user_fetch.text

    # Create metrics formatter
    metrics_formatter = MetricsFormatter(metrics_context)

    # Format system prompt with task description and metrics
    system_prompt = system_template.format(
        task_description=task_description,
        metrics_description=metrics_formatter.format_metrics_description(),
    )

    # Return configured agent with fetcher wired in for dynamic updates
    return MutationAgent(
        llm=llm,
        system_prompt=system_prompt,
        user_prompt_template=user_template,
        mutation_mode=mutation_mode,
        prompt_fetcher=fetcher,
        task_description=task_description,
        metrics_context=metrics_context,
    )


def create_insights_agent(
    llm: ChatOpenAI | MultiModelRouter,
    task_description: str,
    metrics_context: MetricsContext,
    max_insights: int = 7,
    prompts_dir: str | Path | None = None,
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
        prompts_dir: Optional prompts directory (e.g. config.prompts.dir). If None, package defaults are used.

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
    system_template = InsightsPrompts.system(prompts_dir=prompts_dir)
    user_template = InsightsPrompts.user(prompts_dir=prompts_dir)
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
    prompts_dir: str | Path | None = None,
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
        prompts_dir: Optional prompts directory (e.g. config.prompts.dir). If None, package defaults are used.

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
    system_prompt = LineagePrompts.system(prompts_dir=prompts_dir)
    user_template = LineagePrompts.user(prompts_dir=prompts_dir)

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
    prompts_dir: str | Path | None = None,
) -> ScoringAgent:
    """Create a fully configured scoring agent.

    This factory does ALL the setup:
    - Loads prompts from files
    - Returns ready-to-use agent

    Args:
        llm: LangChain chat model or multi-model router
        prompts_dir: Optional prompts directory (e.g. config.prompts.dir). If None, package defaults are used.

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
    system_prompt = ScoringPrompts.system(prompts_dir=prompts_dir)
    user_template = ScoringPrompts.user(prompts_dir=prompts_dir)

    # Return configured agent
    return ScoringAgent(
        llm=llm,
        system_prompt=system_prompt,
        user_prompt_template=user_template,
        trait_description=trait_description,
        max_score=max_score,
    )


def create_memory_selector_agent(
    llm: ChatOpenAI | MultiModelRouter,
) -> MemorySelectorAgent:
    """Create a red-agent-backed memory selector for filtering memory ideas."""
    _ = llm  # kept for API compatibility with existing call sites
    return MemorySelectorAgent()
