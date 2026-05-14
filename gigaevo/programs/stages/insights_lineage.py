from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from loguru import logger

from gigaevo.database.program_storage import ProgramStorage
from gigaevo.llm.agents.factories import create_lineage_agent
from gigaevo.llm.agents.lineage import TransitionAnalysis
from gigaevo.llm.models import ChatOpenAI, MultiModelRouter
from gigaevo.programs.core_types import (
    ProgramStageResult,
    StageIO,
)
from gigaevo.programs.metrics.context import MetricsContext
from gigaevo.programs.program import Program
from gigaevo.programs.stages.ancestry_selector import AncestrySelector
from gigaevo.programs.stages.base import Stage
from gigaevo.programs.stages.cache_handler import NO_CACHE
from gigaevo.programs.stages.common import CacheOnlyInput, ListOf
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

    # CacheOnlyInput.cache_on lets an upstream stage (e.g. FetchOpponentIdsStage)
    # invalidate cached lineage analyses when the opponent set rotates without
    # changing what `preprocess()`/`compute()` actually read.
    InputsModel: type[StageIO] = CacheOnlyInput
    OutputModel: type[StageIO] = LineageAnalysesOutput

    def __init__(
        self,
        *,
        llm: ChatOpenAI | MultiModelRouter,
        task_description: str,
        metrics_context: MetricsContext,
        storage: ProgramStorage,
        prompts_dir: str | Path | None = None,
        descendant_selector: AncestrySelector | None = None,
        **kwargs: Any,
    ):
        # Inject live Program instance as `program` kwarg for the agent
        super().__init__(
            agent=create_lineage_agent(
                llm,
                task_description,
                metrics_context,
                prompts_dir=prompts_dir,
            ),
            program_kwarg="program",
            **kwargs,
        )
        self.storage = storage
        # When set, used to short-circuit analyses whose (parent→failed-child)
        # transition would never be read by the parent's future
        # LineagesToDescendants stage. Should match the selector wired into
        # DescendantProgramIds. Pass ``None`` to disable the optimisation.
        self.descendant_selector = descendant_selector

    async def preprocess(
        self, program: Program, params: StageIO
    ) -> dict[str, Any] | ProgramStageResult:
        ids: list[str] = list(program.lineage.parents)
        logger.info(
            "[LineageStage] program={} n_parents={}",
            program.id[:8],
            len(ids),
        )
        parents = await self.storage.mget(ids)
        if self.descendant_selector is not None and program.is_failed and parents:
            parents = await self._filter_parents_for_failed_child(program, parents)
            logger.info(
                "[LineageStage] program={} failed=True kept {}/{} parents after"
                " descendant-selector simulation",
                program.id[:8],
                len(parents),
                len(ids),
            )
        return {"parents": parents}

    async def _filter_parents_for_failed_child(
        self, program: Program, parents: list[Program]
    ) -> list[Program]:
        """Drop parents whose DescendantProgramIds would not pick this failed child.

        For each parent Q, simulate ``descendant_selector.select`` on
        ``Q.children`` (plus the in-memory ``program`` for fresh metrics).
        Keep the parent only when the selector would pick ``program`` —
        otherwise the future ``(Q → program)`` analysis is dead weight.
        """
        assert self.descendant_selector is not None
        kept: list[Program] = []
        for parent in parents:
            sibling_ids = [c for c in parent.lineage.children if c != program.id]
            siblings = await self.storage.mget(sibling_ids) if sibling_ids else []
            picked = await self.descendant_selector.select(siblings + [program])
            if any(p.id == program.id for p in picked):
                kept.append(parent)
            else:
                logger.info(
                    "[LineageStage] skip {} -> {} (failed; outside top-{})",
                    parent.id[:8],
                    program.id[:8],
                    self.descendant_selector.max_selected,
                )
        return kept


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
        child_ids = list(
            cast(LineagesToDescendantsInputs, self.params).descendant_ids.items
        )
        if not child_ids:
            return ProgramStageResult.skipped(
                message="No descendant IDs provided for lineage analysis",
                stage=self.stage_name,
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
                        "[LineagesToDescendants] Added transition for {} -> {}",
                        a.from_id,
                        a.to_id,
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
        parent_ids: list[str] = list(
            cast(LineagesFromAncestorsInputs, self.params).ancestor_ids.items
        )
        if not parent_ids:
            return ProgramStageResult.skipped(
                message="No ancestor IDs provided for lineage analysis",
                stage=self.stage_name,
            )
        res: ProgramStageResult | None = program.stage_results.get(
            self.source_stage_name
        )
        if not res or res.output is None:
            return ProgramStageResult.skipped(
                message="No transitions computed for this program",
                stage=self.stage_name,
            )
        analyses: list[TransitionAnalysis] = res.output.analyses
        want_child = program.id
        out = [a for a in analyses if a.to_id == want_child and a.from_id in parent_ids]
        for a in out:
            logger.info(
                "[LineagesFromAncestors] Added transition for {} -> {}",
                a.from_id,
                a.to_id,
            )

        return TransitionAnalysisList(items=out)
