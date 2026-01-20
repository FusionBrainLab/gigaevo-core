"""Lineage agent for analyzing program evolution using LangGraph.

This agent analyzes parent→child transitions to identify successful strategies.
ALL LLM-related logic lives here - stages are just thin wrappers.
"""

import difflib
from typing import TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from gigaevo.llm.agents.base import LangGraphAgent
from gigaevo.llm.models import MultiModelRouter
from gigaevo.programs.metrics.formatter import MetricsFormatter
from gigaevo.programs.program import Program


class TransitionInsight(BaseModel):
    """Single insight about a parent→child transition."""

    strategy: str = Field(
        description="Strategy type: imitation, avoidance, generalization, exploration"
    )
    description: str = Field(
        description="Specific explanation with evidence (≤30 words)"
    )


class TransitionInsights(BaseModel):
    """Collection of transition insights."""

    insights: list[TransitionInsight] = Field(
        description="List of 3-5 strategy insights", min_length=3, max_length=5
    )


class TransitionAnalysis(BaseModel):
    """Complete transition analysis output."""

    from_id: str = Field(alias="from")
    to_id: str = Field(alias="to")
    parent_metrics: dict[str, float]
    child_metrics: dict[str, float]
    diff_blocks: list[str]
    insights: TransitionInsights

    class Config:
        populate_by_name = True


class LineageState(TypedDict):
    """Complete state for lineage analysis."""

    parent: Program
    child: Program
    messages: list[BaseMessage]
    llm_response: AIMessage | TransitionInsights
    delta: float
    diff_blocks: list[str]
    insights: list[dict]
    full_analysis: TransitionAnalysis
    metadata: dict


class LineageAgent(LangGraphAgent):
    StateSchema = LineageState
    """Agent for analyzing program lineage.

    This agent does ALL the heavy lifting:
    - Computes deltas
    - Formats diffs
    - Formats metrics
    - Builds prompts
    - Calls LLM
    - Parses structured output

    Stages just call agent.arun(parent, child) and store results.
    """

    StateSchema = LineageState

    def __init__(
        self,
        llm: ChatOpenAI | MultiModelRouter,
        system_prompt: str,
        user_prompt_template: str,
        task_description: str,
        metrics_formatter: MetricsFormatter,
    ):
        """Initialize lineage agent.

        Args:
            llm: LangChain chat model or router
            system_prompt: System prompt (no template vars)
            user_prompt_template: User prompt template (with {delta}, {diff_blocks}, etc)
            task_description: Description of optimization task
            metrics_formatter: Formatter for program metrics
        """
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template
        self.task_description = task_description
        self.metrics_formatter = metrics_formatter

        llm = llm.with_structured_output(TransitionInsights)

        super().__init__(llm)

    def _compute_diff_blocks(self, parent_code: str, child_code: str) -> list[str]:
        """Compute unified diff blocks between parent and child code.

        Returns list of diff hunks (excluding file headers).
        Returns empty list if codes are identical.
        """
        if parent_code.strip() == child_code.strip():
            return []

        diff_lines = list(
            difflib.unified_diff(
                parent_code.strip().splitlines(),
                child_code.strip().splitlines(),
                lineterm="",
                n=3,  # Context lines
            )
        )

        if not diff_lines:
            return []

        # Skip file headers (--- and +++)
        content = [line for line in diff_lines if not line.startswith(("---", "+++"))]
        if not content:
            return []

        # Group into hunks
        hunks: list[str] = []
        current_hunk: list[str] = []

        for line in content:
            if line.startswith("@@"):
                # Start of new hunk
                if current_hunk:
                    hunks.append("\n".join(current_hunk))
                current_hunk = [line]
            else:
                current_hunk.append(line)

        # Add last hunk
        if current_hunk:
            hunks.append("\n".join(current_hunk))

        return hunks

    def build_prompt(self, state: LineageState) -> LineageState:
        """Build lineage analysis prompt - ALL formatting logic here.

        This method does:
        - Compute metric deltas
        - Format diff blocks
        - Format additional metrics
        - Get error summaries
        - Build complete prompt
        """
        parent = state["parent"]
        child = state["child"]

        # Compute delta using primary metric from context
        primary_key = self.metrics_formatter.context.get_primary_key()
        parent_fitness = parent.metrics[primary_key]
        child_fitness = child.metrics[primary_key]
        delta = child_fitness - parent_fitness

        # Store delta in state for prompt formatting
        state["delta"] = delta

        # Compute diff blocks (agent responsibility!)
        diff_blocks = self._compute_diff_blocks(parent.code, child.code)

        # Store diff blocks in state for later use
        state["diff_blocks"] = diff_blocks

        if diff_blocks:
            rendered_blocks = "\n\n".join(
                [
                    f"--- Block {i + 1} ---\n```diff\n{block}\n```"
                    for i, block in enumerate(diff_blocks)
                ]
            )
        else:
            rendered_blocks = "(No code differences detected)"

        # Format additional metrics (agent responsibility!)
        additional_metrics_str = (
            self.metrics_formatter.format_delta_block(
                parent=parent.metrics, child=child.metrics, include_primary=False
            )
            if self.metrics_formatter
            else ""
        )

        parent_errors = parent.format_errors(include_traceback=True)
        child_errors = child.format_errors(include_traceback=True)

        metric_name = self.metrics_formatter.context.get_primary_key()
        metric_description = self.metrics_formatter.context.get_description(metric_name)
        higher_is_better = self.metrics_formatter.context.is_higher_better(metric_name)

        # Compute interpretation based on direction
        higher_is_better_text = (
            "↑ higher is better" if higher_is_better else "↓ lower is better"
        )
        is_improvement = (delta > 0) == higher_is_better
        delta_interpretation = "IMPROVEMENT ✓" if is_improvement else "REGRESSION ✗"

        user_prompt = self.user_prompt_template.format(
            task_description=self.task_description,
            metric_name=metric_name,
            metric_description=metric_description,
            delta=delta,
            higher_is_better_text=higher_is_better_text,
            delta_interpretation=delta_interpretation,
            parent_errors=parent_errors,
            child_errors=child_errors,
            additional_metrics=additional_metrics_str,
            diff_blocks=rendered_blocks,
            parent_code=parent.code,
        )

        # Create messages
        state["messages"] = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_prompt),
        ]

        return state

    def parse_response(self, state: LineageState) -> LineageState:
        """Parse LLM response and build complete analysis output."""
        llm_response = state["llm_response"]

        if not isinstance(llm_response, TransitionInsights):
            raise ValueError(f"Expected TransitionInsights, got {type(llm_response)}")

        state["insights"] = llm_response

        parent = state["parent"]
        child = state["child"]

        state["full_analysis"] = TransitionAnalysis(
            from_id=parent.id,
            to_id=child.id,
            parent_metrics=parent.metrics,
            child_metrics=child.metrics,
            diff_blocks=state["diff_blocks"],
            insights=llm_response,
        )

        return state

    async def arun(
        self, *, parents: list[Program], program: Program
    ) -> list[TransitionAnalysis]:
        """Run lineage analysis on parent→child transitions for multiple parents.

        Args:
            program: Child program
            parents: List of parent programs to analyze

        Returns:
            List of LineageAnalysis for each parent→child transition
        """
        analyses: list[TransitionAnalysis] = []

        for parent in parents:
            initial_state: LineageState = {
                "parent": parent,
                "child": program,
                "messages": [],
                "llm_response": None,  # type: ignore
                "delta": 0.0,
                "diff_blocks": [],
                "insights": [],
                "full_analysis": {},
                "metadata": {"parent_id": parent.id, "child_id": program.id},
            }

            final_state = await self.graph.ainvoke(initial_state)
            analyses.append(final_state["full_analysis"])  # type: ignore

        return analyses
