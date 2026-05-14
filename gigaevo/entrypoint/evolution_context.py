from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from gigaevo.database.program_storage import ProgramStorage
from gigaevo.llm.models import MultiModelRouter
from gigaevo.memory.provider import MemoryProvider, NullMemoryProvider
from gigaevo.problems.context import ProblemContext
from gigaevo.programs.stages.archive_gate import ArchiveGateProvider

if TYPE_CHECKING:
    pass


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
    prompts_dir: str | Path | None = Field(
        default=None,
        description="Optional directory for prompt templates (e.g. from config.prompts.dir). Same layout as gigaevo/prompts.",
    )
    prompt_fetcher: Any | None = Field(
        default=None,
        description="Optional PromptFetcher for dynamic prompt co-evolution. "
        "When None, agents use FixedDirPromptFetcher(prompts_dir) by default.",
    )
    memory_provider: MemoryProvider = Field(
        default_factory=NullMemoryProvider,
        description="Memory provider for DAG pipeline. NullMemoryProvider is a no-op.",
    )
    archive_gate_provider: ArchiveGateProvider | None = Field(
        default=None,
        description=(
            "Provider for archive-insertion gating before InsightsStage. "
            "None disables gating (ArchivePotentialGateStage fails open). "
            "Built at config-instantiation time from evolution_strategy."
        ),
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)
