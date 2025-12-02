# gigaevo/programs/stages/insights.py
from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI

from gigaevo.llm.agents.factories import create_insights_agent
from gigaevo.llm.agents.insights import ProgramInsights
from gigaevo.llm.models import MultiModelRouter
from gigaevo.programs.core_types import StageIO, VoidInput
from gigaevo.programs.metrics.context import MetricsContext
from gigaevo.programs.program import Program
from gigaevo.programs.stages.langgraph_stage import LangGraphStage
from gigaevo.programs.stages.stage_registry import StageRegistry


class InsightsOutput(StageIO):
    """Single-field wrapper so downstream stages get a strict schema."""

    insights: ProgramInsights


@StageRegistry.register(description="LLM insights for a single program")
class InsightsStage(LangGraphStage):
    """
    Runs the Insights agent on the current Program.

    - InputsModel: none (DAG provides no inputs here)
    - OutputModel: InsightsOutput (wraps ProgramInsights)
    - Injects the live Program into the agent call as `program`
    """

    InputsModel = VoidInput
    OutputModel = InsightsOutput

    def __init__(
        self,
        *,
        llm: ChatOpenAI | MultiModelRouter,
        task_description: str,
        metrics_context: MetricsContext,
        max_insights: int = 7,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            agent=create_insights_agent(
                llm, task_description, metrics_context, max_insights
            ),
            program_kwarg="program",
            **kwargs,
        )

    async def compute(self, program: Program) -> InsightsOutput:
        return InsightsOutput(insights=await self.agent.arun(program))
