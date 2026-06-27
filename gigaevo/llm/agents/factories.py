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

from gigaevo.llm.agents.consolidate_cards import ConsolidateAgent
from gigaevo.llm.agents.insights import InsightsAgent
from gigaevo.llm.agents.lineage import LineageAgent
from gigaevo.llm.agents.mutation import MutationAgent
from gigaevo.llm.agents.mutation_suggestions import MutationSuggestionAgent
from gigaevo.llm.agents.program_author import ProgramAuthorAgent
from gigaevo.llm.agents.reconcile import ReconcileAgent
from gigaevo.llm.models import MultiModelRouter
from gigaevo.programs.metrics.context import MetricsContext
from gigaevo.programs.metrics.formatter import MetricsFormatter
from gigaevo.prompts import (
    ConsolidatePrompts,
    InsightsPrompts,
    LineagePrompts,
    MutationSuggestionsPrompts,
    ProgramAuthorPrompts,
    ReconcilePrompts,
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
    max_insights: int = 5,
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


def create_mutation_suggestion_agent(
    llm: ChatOpenAI | MultiModelRouter,
    task_description: str,
    metrics_context: MetricsContext,
    max_insights: int = 5,
    prompts_dir: str | Path | None = None,
) -> MutationSuggestionAgent:
    """Create a fully configured mutation-suggestion agent.

    The suggestion agent consumes the descriptive intra/extra memory cards
    plus a backward ancestral trail and produces actionable suggestions in
    the ``ProgramInsights`` schema (wire-compatible with the legacy insights
    stage).

    Args:
        llm: LangChain chat model or multi-model router.
        task_description: Description of the optimization task.
        metrics_context: Metrics context for formatting.
        max_insights: Maximum suggestions to generate.
        prompts_dir: Optional prompts directory (e.g. ``config.prompts.dir``).
            If None, package defaults are used.

    Returns:
        Ready-to-use ``MutationSuggestionAgent``.
    """
    system_template = MutationSuggestionsPrompts.system(prompts_dir=prompts_dir)
    user_template = MutationSuggestionsPrompts.user(prompts_dir=prompts_dir)
    metrics_formatter = MetricsFormatter(metrics_context)

    system_prompt = system_template.format(
        task_description=task_description,
        metrics_description=metrics_formatter.format_metrics_description(),
        max_insights=max_insights,
    )

    return MutationSuggestionAgent(
        llm=llm,
        system_prompt=system_prompt,
        user_prompt_template=user_template,
        max_insights=max_insights,
        metrics_formatter=metrics_formatter,
        metrics_context=metrics_context,
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


def create_reconcile_agent(
    llm: ChatOpenAI | MultiModelRouter,
    task_description: str,
    prompts_dir: str | Path | None = None,
) -> ReconcileAgent:
    """Create the librarian's reconcile agent (memory write path).

    Bakes the task into the system prompt's CONTEXT once at construction — the
    task is fixed for a run — and injects the user template, mirroring the
    insights/lineage factories. No metrics: the librarian reasons over the diff
    and existing cards, not over per-program metric blocks.

    Args:
        llm: LangChain chat model or multi-model router.
        task_description: Description of the optimization task.
        prompts_dir: Optional prompts directory (e.g. ``config.prompts.dir``).
            If None, package defaults are used.

    Returns:
        Ready-to-use ``ReconcileAgent``.
    """
    system_template = ReconcilePrompts.system(prompts_dir=prompts_dir)
    user_template = ReconcilePrompts.user(prompts_dir=prompts_dir)
    system_prompt = system_template.format(task_description=task_description)
    return ReconcileAgent(
        llm=llm,
        system_prompt=system_prompt,
        user_prompt_template=user_template,
    )


def create_consolidate_agent(
    llm: ChatOpenAI | MultiModelRouter,
    task_description: str,
    prompts_dir: str | Path | None = None,
) -> ConsolidateAgent:
    """Create the librarian's consolidate agent (periodic dedup pass).

    Args:
        llm: LangChain chat model or multi-model router.
        task_description: Description of the optimization task.
        prompts_dir: Optional prompts directory (e.g. ``config.prompts.dir``).
            If None, package defaults are used.

    Returns:
        Ready-to-use ``ConsolidateAgent``.
    """
    system_template = ConsolidatePrompts.system(prompts_dir=prompts_dir)
    user_template = ConsolidatePrompts.user(prompts_dir=prompts_dir)
    system_prompt = system_template.format(task_description=task_description)
    return ConsolidateAgent(
        llm=llm,
        system_prompt=system_prompt,
        user_prompt_template=user_template,
    )


def create_program_author_agent(
    llm: ChatOpenAI | MultiModelRouter,
    task_description: str,
    prompts_dir: str | Path | None = None,
) -> ProgramAuthorAgent:
    """Create the librarian's exemplar program-author agent.

    Args:
        llm: LangChain chat model or multi-model router.
        task_description: Description of the optimization task.
        prompts_dir: Optional prompts directory (e.g. ``config.prompts.dir``).
            If None, package defaults are used.

    Returns:
        Ready-to-use ``ProgramAuthorAgent``.
    """
    system_template = ProgramAuthorPrompts.system(prompts_dir=prompts_dir)
    user_template = ProgramAuthorPrompts.user(prompts_dir=prompts_dir)
    system_prompt = system_template.format(task_description=task_description)
    return ProgramAuthorAgent(
        llm=llm,
        system_prompt=system_prompt,
        user_prompt_template=user_template,
    )
