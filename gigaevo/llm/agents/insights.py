"""Insights agent for program analysis using LangGraph.

This agent analyzes programs to generate actionable insights for evolution.
ALL LLM-related logic lives here - stages are just thin wrappers.
"""

from typing import Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from gigaevo.llm.agents.base import LangGraphAgent
from gigaevo.llm.models import MultiModelRouter
from gigaevo.programs.metrics.formatter import MetricsFormatter
from gigaevo.programs.program import OPTIMIZATION_STAGES, Program


class ProgramInsight(BaseModel):
    """Single structured insight about a program.

    Schema v2 (2026-05-23): split the legacy free-string ``insight`` into
    structured grounding fields (``anchor_quote``, ``evidence_source``,
    ``mechanism``, ``substitute``, ``evidence_refs``, ``relation_to_lineage``)
    so the suggester cannot hide a missing anchor or missing substitute
    inside one ≤35-word sentence. ``insight`` is kept as an OPTIONAL legacy
    fallback for the off-path :class:`InsightsAgent` (legacy ``InsightsStage``
    prompt still emits the free string); the renderer prefers the structured
    fields when present.
    """

    type: str = Field(
        description=(
            "1-2 word snake_case label for the *specific* code issue "
            "(e.g. early_termination, threshold_tuning, policy_bias). "
            "Generic labels such as optimization or refactor are rejected."
        )
    )
    anchor_quote: str = Field(
        default="",
        description=(
            "Literal string copied verbatim from one of the evidence inputs "
            "(numeric constant, identifier, cluster label, fitness number). "
            "Must appear in the source verbatim."
        ),
    )
    evidence_source: str = Field(
        default="",
        description=(
            "Which input the anchor came from: program | metrics | intra_memory "
            "| memory_cards | ancestral_trail | evolutionary_statistics."
        ),
    )
    mechanism: str = Field(
        default="",
        description=(
            "One-clause explanation of why the anchor moves the primary metric. "
            "Must be concrete — vague claims like 'improves performance' are rejected."
        ),
    )
    mechanism_source: Literal[
        "",
        "program",
        "metrics",
        "intra_memory",
        "memory_cards",
        "ancestral_trail",
        "evolutionary_statistics",
    ] = Field(
        default="",
        description=(
            "Which input the LEVER IDEA (mechanism) came from — distinct from "
            "evidence_source, which records where the anchor_quote came from. "
            "A suggestion that transposes a card mechanism onto a code anchor "
            "therefore has evidence_source=program but mechanism_source=memory_cards. "
            "Empty when the mechanism is the analyst's own synthesis."
        ),
    )
    card_id: str = Field(
        default="",
        description=(
            "When mechanism_source is memory_cards: the id of the card the "
            "mechanism was transposed from (shown as `[card N] id=<id>`). "
            "Empty otherwise."
        ),
    )
    substitute: str = Field(
        default="",
        description=(
            "A concrete target value, replacement pattern, or specific guard — "
            "never a bare direction (a direction such as 'reduce X' is rejected; "
            "give an explicit replacement value or pattern)."
        ),
    )
    evidence_refs: list[str] = Field(
        default_factory=list,
        description=(
            "Optional list of identifiers of the exact evidence items cited "
            "(memory card IDs, cluster labels, trail depth_back values). "
            "Empty when not citing specific items."
        ),
    )
    relation_to_lineage: str = Field(
        default="",
        description=(
            "Optional one-clause note describing how this suggestion DIFFERS "
            "from prior tried strategies (required when refining a "
            "negative/regressed cluster)."
        ),
    )
    insight: str = Field(
        default="",
        description=(
            "DEPRECATED legacy free-string insight (≤35 words). Retained for "
            "the legacy InsightsAgent prompt; new prompts populate the "
            "structured fields above instead."
        ),
    )
    tag: Literal["beneficial", "harmful", "fragile", "rigid", "neutral"] = Field(
        description=(
            "How to act on the insight: beneficial (preserve/extend) | "
            "harmful (remove/replace) | fragile (robustify) | "
            "rigid (parameterize/tune) | neutral (low priority)."
        )
    )
    severity: Literal["high", "medium", "low"] = Field(
        description="Severity / priority of the insight: high | medium | low."
    )


class ProgramInsights(BaseModel):
    """Collection of program insights."""

    insights: list[ProgramInsight] = Field(
        description="List of actionable insights",
    )


class InsightsState(TypedDict):
    """Complete state for insights analysis.

    This is the LangGraph state - it flows through all nodes.
    """

    # Input
    program: Program

    # LLM interaction
    messages: list[BaseMessage]
    llm_response: AIMessage | ProgramInsights | None

    # Output
    insights: ProgramInsights | None

    # Metadata
    metadata: dict


class InsightsAgent(LangGraphAgent):
    """Agent for generating program insights.

    This agent does ALL the heavy lifting:
    - Formats metrics and errors
    - Builds prompts
    - Calls LLM
    - Parses structured output

    Stages just call agent.arun(program) and store results.
    """

    StateSchema = InsightsState

    def __init__(
        self,
        llm: ChatOpenAI | MultiModelRouter,
        system_prompt_template: str,
        user_prompt_template: str,
        max_insights: int,
        metrics_formatter: MetricsFormatter,
    ):
        """Initialize insights agent.

        Args:
            llm: LangChain chat model or router
            system_prompt_template: System prompt template (with {task_description} etc)
            user_prompt_template: User prompt template (with {code}, {metrics}, etc)
            max_insights: Maximum insights to generate
            metrics_formatter: Formatter for program metrics
        """
        self.system_prompt_template = system_prompt_template
        self.user_prompt_template = user_prompt_template
        self.max_insights = max_insights
        self.metrics_formatter = metrics_formatter
        structured_llm = llm.with_structured_output(ProgramInsights)

        super().__init__(structured_llm)

    def build_prompt(self, state: InsightsState) -> InsightsState:
        """Build insights prompt - ALL formatting logic here.

        This method does:
        - Format metrics using metrics_formatter
        - Build error section
        - Format prompts with all variables
        - Create LangChain messages
        """
        program = state["program"]

        # Format metrics (agent responsibility!)
        metrics_text = (
            self.metrics_formatter.format_metrics_block(program.metrics)
            if program.metrics
            else "No metrics available"
        )

        errors = program.format_errors(
            include_traceback=True, exclude_stages=set(OPTIMIZATION_STAGES)
        )
        error_section = (
            f"**Error Analysis**: Focus on fixing or avoiding failure modes from stages:\n{errors}"
            if errors
            else ""
        )

        user_prompt = self.user_prompt_template.format(
            code=program.code,
            metrics=metrics_text,
            error_section=error_section,
            max_insights=self.max_insights,
        )

        state["messages"] = [
            SystemMessage(content=self.system_prompt_template),
            HumanMessage(content=user_prompt),
        ]

        return state

    def parse_response(self, state: InsightsState) -> InsightsState:
        """Parse LLM response (already validated by LangChain structured output)."""
        llm_response = state["llm_response"]
        if not isinstance(llm_response, ProgramInsights):
            raise ValueError(f"Expected ProgramInsights, got {type(llm_response)}")
        state["insights"] = llm_response
        return state

    async def arun(
        self,
        program: Program,
    ) -> ProgramInsights:
        """Run insights analysis on a program.

        Args:
            program: Program to analyze

        Returns:
            List of insight dicts with "type" and "insight" keys
            lineage_data: is not used for now; in the future we can use it to add more context to the insights
        """
        initial_state: InsightsState = {
            "program": program,
            "messages": [],
            "llm_response": None,
            "insights": None,
            "metadata": {"program_id": program.id},
        }

        final_state = await self.graph.ainvoke(initial_state)
        return final_state["insights"]
