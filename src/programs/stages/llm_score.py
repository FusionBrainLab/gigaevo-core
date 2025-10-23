"""Scoring stage - wrapper with metric name transformation.

Usage:
    >>> from src.programs.stages.llm_score import ScoringStage
    >>> stage = ScoringStage(
    ...     llm=my_llm,
    ...     trait_description="code novelty",
    ...     score_metric_name="novelty"
    ... )
"""

from datetime import datetime
from typing import Any

from langchain_openai import ChatOpenAI
from loguru import logger

from src.llm.agents.factories import create_scoring_agent
from src.llm.models import MultiModelRouter
from src.programs.core_types import ProgramStageResult, StageState
from src.programs.program import Program
from src.programs.stages.base import Stage
from src.programs.utils import build_stage_result


class ScoringStage(Stage):
    """Stage that scores programs and returns {metric_name: score}.
    
    The scoring agent returns {"score": value}, but we transform it to
    {score_metric_name: value} so each scoring stage has a unique metric name.
    """
    
    def __init__(
        self,
        llm: ChatOpenAI | MultiModelRouter,
        trait_description: str,
        score_metric_name: str,
        max_expected_score: float = 1.0,
        **kwargs: Any
    ):
        """Initialize scoring stage.
        
        Args:
            llm: LangChain chat model or router
            trait_description: Description of trait to score (e.g., "code novelty")
            score_metric_name: Key for storing score in output dict
            max_expected_score: Maximum allowed score
            **kwargs: Additional arguments passed to Stage base class
        """
        super().__init__(**kwargs)
        self.agent = create_scoring_agent(llm=llm)
        self.trait_description = trait_description
        self.score_metric_name = score_metric_name
        self.max_expected_score = max_expected_score
        
        logger.info(
            f"[{self.stage_name}] Initialized ScoringStage "
            f"(trait={trait_description}, metric={score_metric_name})"
        )
    
    async def _execute_stage(
        self, program: Program, started_at: datetime
    ) -> ProgramStageResult:
        """Execute scoring agent and rename output."""
        
        # Call agent
        score = await self.agent.arun(
            program=program,
            trait_description=self.trait_description,
            max_score=self.max_expected_score
        )
        
        # Agent returns float, we return {metric_name: score}
        output = {self.score_metric_name: score}
        
        return build_stage_result(
            status=StageState.COMPLETED,
            started_at=started_at,
            output=output,
            stage_name=self.stage_name,
        )
