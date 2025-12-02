from __future__ import annotations

from typing import Any

from loguru import logger

from gigaevo.database.program_storage import ProgramStorage
from gigaevo.llm.agents.factories import create_lineage_agent
from gigaevo.llm.agents.lineage import TransitionAnalysis
from gigaevo.llm.models import ChatOpenAI, MultiModelRouter
from gigaevo.programs.core_types import (
    ProgramStageResult,
    StageError,
    StageIO,
    StageState,
    VoidInput,
)
from gigaevo.programs.metrics.context import MetricsContext
from gigaevo.programs.program import Program
from gigaevo.programs.stages.base import Stage
from gigaevo.programs.stages.cache_handler import NO_CACHE
from gigaevo.programs.stages.common import ListOf
from gigaevo.programs.stages.langgraph_stage import LangGraphStage
from gigaevo.programs.stages.stage_registry import StageRegistry


class LineageAnalysesOutput(StageIO):
    """List of LineageAnalysis for each parent→child transition."""

    analyses: list[TransitionAnalysis]


class TransitionAnalysisList(StageIO):
    items: list[TransitionAnalysis]


@StageRegistry.register(
    description="Compute LLM lineage analysis (parent → child) using parent IDs"
)
class LineageStage(LangGraphStage):
    """
    Uses DAG input `parents` to fetch parent Programs, injects the current Program
    as `program`, calls the lineage agent, and returns analyses (same order as parents).
    """

    InputsModel = VoidInput
    OutputModel = LineageAnalysesOutput  # Output: list of TransitionAnalysis from parent<i> to child

    def __init__(
        self,
        *,
        llm: ChatOpenAI | MultiModelRouter,
        task_description: str,
        metrics_context: MetricsContext,
        storage: ProgramStorage,
        **kwargs: Any,
    ):
        # Inject live Program instance as `program` kwarg for the agent
        super().__init__(
            agent=create_lineage_agent(llm, task_description, metrics_context),
            program_kwarg="program",
            **kwargs,
        )
        self.storage = storage

    async def preprocess(
        self, program: Program, params: VoidInput
    ) -> dict[str, Program] | ProgramStageResult:
        ids: list[str] = list(program.lineage.parents)
        return {"parents": await self.storage.mget(ids)}


class LineagesToDescendantsInputs(StageIO):
    descendant_ids: ListOf[str]


@StageRegistry.register(
    description="From a list of descendant IDs, return analyses for current→child transitions."
)
class LineagesToDescendants(Stage):
    """
    Input:  ListOf[str](items=[child_id, ...])
    Output: ListOf[TransitionAnalysis] for transitions (this_program -> each selected child)
    """

    InputsModel = LineagesToDescendantsInputs
    OutputModel = TransitionAnalysisList
    cache_handler = NO_CACHE  # descendants and their lineage may evolve over time

    def __init__(
        self, *, storage: ProgramStorage, source_stage_name: str, **kwargs: Any
    ):
        super().__init__(**kwargs)
        self.storage = storage
        self.source_stage_name = source_stage_name

    async def compute(
        self, program: Program
    ) -> TransitionAnalysisList | ProgramStageResult:
        child_ids = list(self.params.descendant_ids.items)
        if not child_ids:
            return ProgramStageResult(
                status=StageState.SKIPPED,
                error=StageError(
                    type="Skip",
                    message="No descendant IDs provided for lineage analysis",
                    stage=self.stage_name,
                ),
            )

        children: list[Program] = await self.storage.mget(child_ids)
        want_parent = program.id
        out: list[TransitionAnalysis] = []

        for child in children:
            res = child.stage_results.get(self.source_stage_name)
            if not res or res.output is None:
                continue

            analyses: LineageAnalysesOutput = res.output
            # from all parents of this child, pick the one where (from == current program) and (to == this child)
            for a in analyses.analyses:
                if a.from_id == want_parent:
                    out.append(a)
                    logger.info(
                        f"[LineagesToDescendants] Added transition analysis for {a.from_id} -> {a.to_id}"
                    )
                    break

        return TransitionAnalysisList(items=out)


class LineagesFromAncestorsInputs(StageIO):
    ancestor_ids: ListOf[str]


@StageRegistry.register(
    description="From a list of ancestor IDs, return analyses for parent→current transitions."
)
class LineagesFromAncestors(Stage):
    """
    Input:  ListOf[str](items=[parent_id, ...])
    Output: ListOf[TransitionAnalysis] for transitions (parent -> this_program)
    """

    InputsModel = LineagesFromAncestorsInputs
    OutputModel = TransitionAnalysisList
    cache_handler = NO_CACHE

    def __init__(
        self, *, storage: ProgramStorage, source_stage_name: str, **kwargs: Any
    ):
        super().__init__(**kwargs)
        self.source_stage_name = source_stage_name
        self.storage = storage

    async def compute(
        self, program: Program
    ) -> TransitionAnalysisList | ProgramStageResult:
        parent_ids: list[str] = list(self.params.ancestor_ids.items)
        if not parent_ids:
            return ProgramStageResult(
                status=StageState.SKIPPED,
                error=StageError(
                    type="Skip",
                    message="No ancestor IDs provided for lineage analysis",
                    stage=self.stage_name,
                ),
            )
        res: ProgramStageResult = program.stage_results.get(self.source_stage_name)
        if not res or res.output is None:
            return ProgramStageResult(
                status=StageState.SKIPPED,
                error=StageError(
                    type="Skip",
                    message="No transitions computed for this program",
                    stage=self.stage_name,
                ),
            )
        analyses: list[TransitionAnalysis] = res.output.analyses
        want_child = program.id
        out = [a for a in analyses if a.to_id == want_child and a.from_id in parent_ids]
        for a in out:
            logger.info(
                f"[LineagesFromAncestors] Added transition analysis for {a.from_id} -> {a.to_id}"
            )

        return TransitionAnalysisList(items=out)
