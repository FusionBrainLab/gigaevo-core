from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from gigaevo.database.program_storage import ProgramStorage
from gigaevo.llm.models import MultiModelRouter
from gigaevo.problems.context import ProblemContext


class EvolutionContext(BaseModel):
    """Context for the evolution process. All stages should be able to access this context. Can be extended with more context if needed."""

    problem_ctx: ProblemContext = Field(
        ..., description="Problem description with all related files"
    )
    llm_wrapper: MultiModelRouter = Field(
        ..., description="LLM wrapper to use for LLM calls"
    )
    storage: ProgramStorage = Field(
        ..., description="Storage containing all programs and their metadata"
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)
