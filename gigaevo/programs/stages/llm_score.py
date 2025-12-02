from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI

from gigaevo.llm.agents.factories import create_scoring_agent
from gigaevo.llm.models import MultiModelRouter
from gigaevo.programs.core_types import VoidInput
from gigaevo.programs.program import Program
from gigaevo.programs.stages.common import Box
from gigaevo.programs.stages.langgraph_stage import LangGraphStage
from gigaevo.programs.stages.stage_registry import StageRegistry


@StageRegistry.register(description="LLM score for a single program")
class LLMScoreStage(LangGraphStage):
    """
    Runs the Scoring agent on the current Program.
    - InputsModel: VoidInput (DAG provides no inputs here)
    - OutputModel: Box[dict[str, float]] (contains the score for the trait)
    - Injects the live Program into the agent call as `program`
    """

    InputsModel = VoidInput
    OutputModel = Box[dict[str, float]]

    def __init__(
        self,
        *,
        llm: ChatOpenAI | MultiModelRouter,
        trait_description: str,
        max_score: float,
        score_key: str,
        **kwargs: Any,
    ) -> None:
        agent = create_scoring_agent(
            llm=llm,
            trait_description=trait_description,
            max_score=max_score,
        )
        super().__init__(agent=agent, program_kwarg="program", **kwargs)
        self.score_key = score_key

    async def postprocess(
        self, program: Program, agent_result: float
    ) -> Box[dict[str, float]]:
        return Box[dict[str, float]](data={self.score_key: agent_result})
