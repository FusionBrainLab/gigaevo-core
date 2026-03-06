from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence, Tuple, cast

from loguru import logger

from gigaevo.exceptions import ValidationError
from gigaevo.programs.program import Program
from gigaevo.programs.stages.base import Stage
from gigaevo.programs.stages.common import AnyContainer, Box

from gigaevo.programs.stages.stage_registry import StageRegistry
from gigaevo.evolution.mutation.context import MutationContext

from gigaevo.programs.core_types import StageIO, VoidInput
from gigaevo.programs.stages import Stage
from gigaevo.programs.stages.python_executors import PythonCodeExecutor, ValidatorInput, CallValidatorFunction, ValidatorOutput
from gigaevo.programs.stages.mutation_context import *

from custom.metrics_formatter import BroaderMetricsFormatter


class DictInput(StageIO):
    data: Box[dict[str, float | str]]


class StrDictInput(StageIO):
    data: Box[dict[str, str]]


class FloatDictInput(StageIO):
    data: Box[dict[str, str]]


@StageRegistry.register(description="LLM insights for a single program")
class ComputeTimeStage(Stage):
    InputsModel = VoidInput
    OutputModel = Box[dict[str, float]]
    cacheable: bool = True

    async def compute(self, program: Program) -> StageIO:

        time = {"runtime": program.stage_results['CallProgramFunction'].duration_seconds()} 

        logger.debug(
            "[TimeStage] Time of {}: {:.2} s",
            program.id,
            time["runtime"],
        )

        return Box[dict[str, float]](data=time)


@StageRegistry.register(
    description="Call a validator function from a Python file on program output (+ optional context)."
)
class BroaderCallValidatorFunction(PythonCodeExecutor):
    """Loads validator file and calls function `validate(context?, program_output)`."""
    InputsModel = ValidatorInput
    OutputModel = Box[dict[str, float | str]]

    def __init__(self, *, path: Path, function_name: str = "validate", **kwargs: Any):
        super().__init__(
            function_name=function_name, python_path=[Path(path).parent], **kwargs
        )
        p = Path(path)
        if not p.exists():
            raise ValidationError(f"Validator file not found: {p}")
        try:
            self._validator_code = p.read_text(encoding="utf-8")
        except OSError as e:
            raise ValidationError(f"Failed to read validator file: {e}") from e

    def _code_str(self, program: Program) -> str:
        return self._validator_code

    # def parse_output(self, x: Any) -> Tuple[dict[str, float], Any]:
    #     return x if isinstance(x, tuple) else (x, None)

    def _build_call(self, program: Program) -> tuple[Sequence[Any], dict[str, Any]]:
        params = cast(ValidatorInput, self.params)
        payload = params.payload.data
        if params.context is not None:
            context = params.context.data
        else:
            context = None
        return ([context, payload] if context is not None else [payload]), {}

@StageRegistry.register(
    description="Distill numeric metrics from CallValidator"
)
class DistillMetrics(Stage):
    InputsModel = DictInput
    OutputModel = Box[dict[str, float]]
    cacheable: bool = True

    async def compute(self, program: Program) -> StageIO:
        metrics = {}
        for key, val in self.params.data.data.items():
            if isinstance(val, float):
                metrics[key] = val
        logger.debug(f"[DistillMetrics] stage completed with {len(metrics)} /n{self.params.data=}")
        return Box[dict[str, float]](data=metrics)
    
    
@StageRegistry.register(
    description="Distill aux info from CallValidator"
)
class DistillNonMetrics(Stage):
    InputsModel = DictInput
    OutputModel = Box[dict[str, str]]
    cacheable: bool = True

    async def compute(self, program: Program) -> StageIO:
        non_metrics = {}
        params: DictInput = self.params
        for key, val in params.data.data.items():
            if isinstance(val, str):
                non_metrics[key] = val
        logger.debug(f"[DistillNonMetrics] stage completed with {len(non_metrics)}/n{self.params.data=}")
        return Box[dict[str, str]](data=non_metrics)
    

@StageRegistry.register(
    description="Ensure non metrics and set formatter"
)
class EnsureNonMetricsStage(Stage):
    InputsModel = StrDictInput
    OutputModel = Box[dict[str, str]]
    async def compute(self, program: Program) -> StageIO:
        # logger.debug(f"[EnsureNonMetricsStage] {self.params.data.data["aux info"]=}")
        program.set_metadata("aux_info", self.params.data.data["aux info"])
        return self.params.data


class MutationContextInputs(StageIO):
    """
    Optional upstream signals the stage can consume.
      - metrics: validated floats, e.g. from EnsureMetricsStage (FloatDictContainer)
      - insights: ProgramInsights wrapped by the Insights stage output
      - lineage_ancestors: TransitionAnalysisList (from collector+lineage stages on ancestors)
      - lineage_descendants: TransitionAnalysisList (from collector+lineage stages on descendants)
    """

    metrics: Optional[FloatDictContainer]
    non_metrics: Optional[Box[dict[str, str]]]
    insights: Optional[InsightsOutput]
    lineage_ancestors: Optional[TransitionAnalysisList]
    lineage_descendants: Optional[TransitionAnalysisList]
    evolutionary_statistics: Optional[EvolutionaryStatistics]


class NonMetricsMutationContext(MutationContext):
    """Context with program metrics."""

    non_metrics: dict[str, str]

    class Config:
        arbitrary_types_allowed = True

    def format(self) -> str:
        lines = ["## Program execution aux info", ""]
        for _, val in self.non_metrics.items():
            lines.append(val)
        logger.debug(f"[NonMetricsMutationContext] {lines=}")
        return "\n".join(lines)


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
    cacheable: bool = False

    def __init__(self, *, metrics_context: MetricsContext, **kwargs):
        super().__init__(**kwargs)
        self.metrics_context = metrics_context
        self.metadata_key = MUTATION_CONTEXT_METADATA_KEY

    async def compute(self, program: Program) -> StageIO:
        contexts: list = []
        params: MutationContextInputs = self.params

        if params.metrics is not None:
            metrics_map = params.metrics.data
            formatter = BroaderMetricsFormatter(self.metrics_context)
            contexts.append(
                MetricsMutationContext(metrics=metrics_map, metrics_formatter=formatter)
            )
        if params.non_metrics is not None:
            context = NonMetricsMutationContext(non_metrics=params.non_metrics.data)
            aux_info = context.format()
            program.set_metadata("aux_info", aux_info)
            # logger.debug(f"{context.format()=}")
            contexts.append(context)
            
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
            formatter = BroaderMetricsFormatter(self.metrics_context)
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
            logger.info(
                "[{}] HELLO",
                EvolutionaryStatisticsMutationContext(
                    evolutionary_statistics=params.evolutionary_statistics,
                    metrics_context=self.metrics_context,
                ).format()
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
    
