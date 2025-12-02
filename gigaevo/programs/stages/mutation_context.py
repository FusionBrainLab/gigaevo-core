# gigaevo/programs/stages/mutation_context.py
from __future__ import annotations

from typing import Optional

from loguru import logger

from gigaevo.evolution.mutation.context import (
    MUTATION_CONTEXT_METADATA_KEY,
    CompositeMutationContext,
    EvolutionaryStatisticsMutationContext,
    FamilyTreeMutationContext,
    InsightsMutationContext,
    MetricsMutationContext,
)
from gigaevo.llm.agents.lineage import TransitionAnalysis
from gigaevo.programs.metrics.context import MetricsContext
from gigaevo.programs.metrics.formatter import MetricsFormatter
from gigaevo.programs.program import Program
from gigaevo.programs.stages.base import Stage
from gigaevo.programs.stages.cache_handler import NO_CACHE
from gigaevo.programs.stages.collector import EvolutionaryStatistics
from gigaevo.programs.stages.common import FloatDictContainer, StageIO, StringContainer
from gigaevo.programs.stages.insights import InsightsOutput
from gigaevo.programs.stages.insights_lineage import TransitionAnalysisList
from gigaevo.programs.stages.stage_registry import StageRegistry


class MutationContextInputs(StageIO):
    """
    Optional upstream signals the stage can consume.
      - metrics: validated floats, e.g. from EnsureMetricsStage (FloatDictContainer)
      - insights: ProgramInsights wrapped by the Insights stage output
      - lineage_ancestors: TransitionAnalysisList (from collector+lineage stages on ancestors)
      - lineage_descendants: TransitionAnalysisList (from collector+lineage stages on descendants)
      - evolutionary_statistics: EvolutionaryStatistics (from EvolutionaryStatisticsCollector)
    """

    metrics: Optional[FloatDictContainer]
    insights: Optional[InsightsOutput]
    lineage_ancestors: Optional[TransitionAnalysisList]
    lineage_descendants: Optional[TransitionAnalysisList]
    evolutionary_statistics: Optional[EvolutionaryStatistics]


@StageRegistry.register(
    description="Assemble mutation context from metrics/insights/lineage"
)
class MutationContextStage(Stage):
    """
    Builds a CompositeMutationContext from whatever inputs are available.

    Notes:
      - Non-cacheable: lineage/descendant data evolves over time.
      - Writes context into Program.metadata[MUTATION_CONTEXT_METADATA_KEY].
      - Returns the context wrapped in AnyContainer so downstream stages can consume it.
    """

    InputsModel = MutationContextInputs
    OutputModel = StringContainer
    cache_handler = NO_CACHE

    def __init__(self, *, metrics_context: MetricsContext, **kwargs):
        super().__init__(**kwargs)
        self.metrics_context = metrics_context
        self.metadata_key = MUTATION_CONTEXT_METADATA_KEY

    async def compute(self, program: Program) -> StageIO:
        contexts: list = []
        params: MutationContextInputs = self.params

        if params.metrics is not None:
            metrics_map = params.metrics.data
            formatter = MetricsFormatter(self.metrics_context)
            contexts.append(
                MetricsMutationContext(metrics=metrics_map, metrics_formatter=formatter)
            )

        if params.insights is not None:
            insights = params.insights.insights
            contexts.append(InsightsMutationContext(insights=insights))

        ancestor_lineages: list[TransitionAnalysis] = []
        if params.lineage_ancestors is not None:
            ancestor_lineages = params.lineage_ancestors.items

        descendant_lineages: list[TransitionAnalysis] = []
        if params.lineage_descendants is not None:
            descendant_lineages = params.lineage_descendants.items

        if ancestor_lineages or descendant_lineages:
            formatter = MetricsFormatter(self.metrics_context)
            contexts.append(
                FamilyTreeMutationContext(
                    ancestors=ancestor_lineages,
                    descendants=descendant_lineages,
                    metrics_formatter=formatter,
                )
            )

        if params.evolutionary_statistics is not None:
            contexts.append(
                EvolutionaryStatisticsMutationContext(
                    evolutionary_statistics=params.evolutionary_statistics,
                    metrics_context=self.metrics_context,
                )
            )

        if not contexts:
            logger.info(
                "[{}] No upstream context available for {}",
                type(self).__name__,
                program.id[:8],
            )

        context = CompositeMutationContext(contexts=contexts).format()
        program.set_metadata(self.metadata_key, context)
        return StringContainer(data=context)
