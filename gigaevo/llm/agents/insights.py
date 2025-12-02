"""Insights agent for program analysis using LangGraph.

This agent analyzes programs to generate actionable insights for evolution.
ALL LLM-related logic lives here - stages are just thin wrappers.
"""

from typing import TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from gigaevo.llm.agents.base import LangGraphAgent
from gigaevo.llm.models import MultiModelRouter
from gigaevo.programs.metrics.formatter import MetricsFormatter
from gigaevo.programs.program import Program


class ProgramInsight(BaseModel):
    """Single structured insight about a program."""

    type: str = Field(description="Insight category")
    insight: str = Field(description="Actionable insight with evidence (≤35 words)")
    tag: str = Field(description="Tag for the insight")
    severity: str = Field(description="Severity of the insight")


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
    llm_response: AIMessage | ProgramInsights

    # Output
    insights: ProgramInsights

    # Metadata
    metadata: dict


class InsightsAgent(LangGraphAgent):
    StateSchema = InsightsState
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
        llm = llm.with_structured_output(ProgramInsights)

        super().__init__(llm)

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

        errors = program.format_errors(include_traceback=True)
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
        llm_response: ProgramInsights = state["llm_response"]
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
            "llm_response": None,  # type: ignore
            "insights": None,  # type: ignore
            "metadata": {"program_id": program.id},
        }

        final_state = await self.graph.ainvoke(initial_state)
        return final_state["insights"]
