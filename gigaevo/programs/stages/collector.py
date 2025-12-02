from __future__ import annotations

from abc import abstractmethod
from typing import Any, List, TypeVar

from loguru import logger
from pydantic import Field

from gigaevo.database.program_storage import ProgramStorage
from gigaevo.programs.core_types import (
    ProgramStageResult,
    StageIO,
    VoidInput,
    VoidOutput,
)
from gigaevo.programs.metrics.context import VALIDITY_KEY, MetricsContext
from gigaevo.programs.program import Program
from gigaevo.programs.stages.ancestry_selector import AncestrySelector
from gigaevo.programs.stages.base import Stage
from gigaevo.programs.stages.cache_handler import NO_CACHE
from gigaevo.programs.stages.common import StringList
from gigaevo.programs.stages.stage_registry import StageRegistry

T = TypeVar("T")


class RelatedCollectorBase(Stage):
    """
    Two-phase collector:
      1) _collect_programs(program)  -> list[Program]
      2) _process(program, programs) -> StageIO | ProgramStageResult

    Subclasses set a concrete OutputModel and override the two abstract methods.
    """

    InputsModel = VoidInput
    OutputModel = VoidOutput
    cache_handler = NO_CACHE  # lineage-derived sets usually change over time

    def __init__(self, *, storage: ProgramStorage, **kwargs: Any):
        super().__init__(**kwargs)
        self.storage = storage

    @abstractmethod
    async def _collect_programs(self, program: Program) -> list[Program]: ...

    @abstractmethod
    async def _process(
        self, program: Program, programs: list[Program]
    ) -> StageIO | ProgramStageResult: ...

    async def compute(self, program: Program) -> StageIO | ProgramStageResult:
        related = await self._collect_programs(program)
        return await self._process(program, related)


@StageRegistry.register(description="Collect related Program IDs (List[str])")
class ProgramIdsCollector(RelatedCollectorBase):
    OutputModel = StringList

    async def _process(self, program: Program, programs: List[Program]) -> StringList:
        return StringList(items=[p.id for p in programs])


@StageRegistry.register(description="Collect ids of descendant Programs")
class DescendantProgramIds(ProgramIdsCollector):
    cache_handler = NO_CACHE

    def __init__(self, *, selector: AncestrySelector, **kwargs: Any):
        super().__init__(**kwargs)
        self.selector = selector

    async def _collect_programs(self, program: Program) -> list[Program]:
        selected = await self.selector.select(
            await self.storage.mget(program.lineage.children)
        )
        logger.info(
            f"[DescendantProgramIds] Selected {len(selected)} programs for {program.id} with children {program.lineage.children}"
        )
        return selected


@StageRegistry.register(description="Collect ids of ancestor Programs")
class AncestorProgramIds(ProgramIdsCollector):
    cache_handler = NO_CACHE

    def __init__(self, *, selector: AncestrySelector, **kwargs: Any):
        super().__init__(**kwargs)
        self.selector = selector

    async def _collect_programs(self, program: Program) -> list[Program]:
        selected = await self.selector.select(
            await self.storage.mget(program.lineage.parents)
        )
        logger.info(
            f"[AncestorProgramIds] Selected {len(selected)} programs for {program.id} with parents {program.lineage.parents}"
        )
        return selected


class GenerationMetrics(StageIO):
    """Metrics for a single generation (main metric only)."""

    best: float | None = Field(
        None,
        description="Best fitness in generation (valid only), None if no valid programs have the metric",
    )
    worst: float | None = Field(
        None,
        description="Worst fitness in generation (valid only), None if no valid programs have the metric",
    )
    average: float | None = Field(
        None,
        description="Average fitness in generation (valid only), None if no valid programs have the metric",
    )
    valid_rate: float = Field(description="Valid rate in generation")
    # num_children statistics
    avg_num_children: float = Field(
        description="Average number of children per program"
    )
    max_num_children: int = Field(
        description="Maximum number of children any program has"
    )
    program_count: int = Field(description="Number of programs in generation")


class EvolutionaryStatistics(StageIO):
    # program statistics
    generation: int = Field(description="Generation")
    iteration: int | None = Field(None, description="Evolution loop iteration number")
    current_program_metrics: dict[str, float] = Field(
        description="Metrics of the current program"
    )
    # global statistics (all programs) - keyed by metric name
    best_fitness: dict[str, float] = Field(
        description="Best fitness per metric (valid only)"
    )
    worst_fitness: dict[str, float] = Field(
        description="Worst fitness per metric (valid only)"
    )
    average_fitness: dict[str, float] = Field(
        description="Average fitness per metric (valid only)"
    )
    valid_rate: float = Field(description="Valid rate")
    # global num_children statistics
    total_program_count: int = Field(description="Total number of programs")
    avg_num_children: float = Field(
        description="Average number of children per program"
    )
    max_num_children: int = Field(
        description="Maximum number of children any program has"
    )
    # generation statistics - keyed by metric name
    best_fitness_in_generation: dict[str, float] = Field(
        description="Best fitness per metric in generation (valid only)"
    )
    worst_fitness_in_generation: dict[str, float] = Field(
        description="Worst fitness per metric in generation (valid only)"
    )
    average_fitness_in_generation: dict[str, float] = Field(
        description="Average fitness per metric in generation (valid only)"
    )
    valid_rate_in_generation: float = Field(description="Valid rate in generation")
    # iteration statistics - keyed by metric name
    best_fitness_in_iteration: dict[str, float] | None = Field(
        None, description="Best fitness per metric in iteration (valid only)"
    )
    worst_fitness_in_iteration: dict[str, float] | None = Field(
        None, description="Worst fitness per metric in iteration (valid only)"
    )
    average_fitness_in_iteration: dict[str, float] | None = Field(
        None, description="Average fitness per metric in iteration (valid only)"
    )
    valid_rate_in_iteration: float | None = Field(
        None, description="Valid rate in iteration"
    )
    # ancestor statistics - keyed by metric name
    ancestor_count: int = Field(description="Number of ancestors (immediate parents)")
    best_fitness_in_ancestors: dict[str, float] = Field(
        description="Best fitness per metric in ancestors (valid only)"
    )
    worst_fitness_in_ancestors: dict[str, float] = Field(
        description="Worst fitness per metric in ancestors (valid only)"
    )
    average_fitness_in_ancestors: dict[str, float] = Field(
        description="Average fitness per metric in ancestors (valid only)"
    )
    valid_rate_in_ancestors: float = Field(description="Valid rate in ancestors")
    # descendant statistics - keyed by metric name
    descendant_count: int = Field(
        description="Number of descendants (immediate children)"
    )
    best_fitness_in_descendants: dict[str, float] = Field(
        description="Best fitness per metric in descendants (valid only)"
    )
    worst_fitness_in_descendants: dict[str, float] = Field(
        description="Worst fitness per metric in descendants (valid only)"
    )
    average_fitness_in_descendants: dict[str, float] = Field(
        description="Average fitness per metric in descendants (valid only)"
    )
    valid_rate_in_descendants: float = Field(description="Valid rate in descendants")
    # history of main metric across all generations - keyed by generation number
    generation_history: dict[int, GenerationMetrics] = Field(
        default_factory=dict,
        description="History of main metric stats per generation (best/worst/avg/valid_rate)",
    )


def _compute_fitness_stats_all_metrics(
    programs: list[Program],
    metrics_context: MetricsContext,
) -> tuple[dict[str, float], dict[str, float], dict[str, float], float]:
    """Compute best, worst, average fitness for all metrics and valid rate for a group of programs.

    Metrics that are absent from all valid programs are skipped (not included in result dicts).

    Returns:
        Tuple of (best_dict, worst_dict, average_dict, valid_rate) where dicts are keyed by metric name
    """
    metric_keys = list(metrics_context.specs.keys())

    if not programs:
        return ({}, {}, {}, 0.0)

    valid_programs = [p for p in programs if p.metrics.get(VALIDITY_KEY, 0) > 0]
    valid_rate = len(valid_programs) / len(programs)

    if not valid_programs:
        return ({}, {}, {}, valid_rate)

    best_dict: dict[str, float] = {}
    worst_dict: dict[str, float] = {}
    average_dict: dict[str, float] = {}

    for metric_key in metric_keys:
        higher_is_better = metrics_context.is_higher_better(metric_key)
        # Only include programs that actually have this metric
        fitness_values = [
            p.metrics[metric_key] for p in valid_programs if metric_key in p.metrics
        ]

        # Skip metrics that no valid program has
        if not fitness_values:
            continue

        if higher_is_better:
            best_dict[metric_key] = max(fitness_values)
            worst_dict[metric_key] = min(fitness_values)
        else:
            best_dict[metric_key] = min(fitness_values)
            worst_dict[metric_key] = max(fitness_values)

        average_dict[metric_key] = sum(fitness_values) / len(fitness_values)

    return (best_dict, worst_dict, average_dict, valid_rate)


def _compute_num_children_stats(programs: list[Program]) -> tuple[float, int, int]:
    """Compute num_children statistics for a group of programs.

    Returns:
        Tuple of (avg_num_children, max_num_children, program_count)
    """
    if not programs:
        return (0.0, 0, 0)

    children_counts = [p.lineage.child_count for p in programs]
    avg_num_children = sum(children_counts) / len(children_counts)
    max_num_children = max(children_counts)

    return (avg_num_children, max_num_children, len(programs))


def _compute_main_metric_stats(
    programs: list[Program],
    metric_key: str,
    higher_is_better: bool,
) -> GenerationMetrics:
    """Compute best, worst, average, valid rate, and num_children stats for the main metric.

    Returns None for best/worst/average if no valid programs have the metric.

    Returns:
        GenerationMetrics with all statistics
    """
    avg_children, max_children, program_count = _compute_num_children_stats(programs)

    if not programs:
        return GenerationMetrics(
            best=None,
            worst=None,
            average=None,
            valid_rate=0.0,
            avg_num_children=0.0,
            max_num_children=0,
            program_count=0,
        )

    valid_programs = [p for p in programs if p.metrics.get(VALIDITY_KEY, 0) > 0]
    valid_rate = len(valid_programs) / len(programs)

    # Only include valid programs that actually have the metric
    fitness_values = [
        p.metrics[metric_key] for p in valid_programs if metric_key in p.metrics
    ]

    if not fitness_values:
        return GenerationMetrics(
            best=None,
            worst=None,
            average=None,
            valid_rate=valid_rate,
            avg_num_children=avg_children,
            max_num_children=max_children,
            program_count=program_count,
        )

    if higher_is_better:
        best = max(fitness_values)
        worst = min(fitness_values)
    else:
        best = min(fitness_values)
        worst = max(fitness_values)

    average = sum(fitness_values) / len(fitness_values)

    return GenerationMetrics(
        best=best,
        worst=worst,
        average=average,
        valid_rate=valid_rate,
        avg_num_children=avg_children,
        max_num_children=max_children,
        program_count=program_count,
    )


async def _get_ancestors(storage: ProgramStorage, program: Program) -> list[Program]:
    """Get immediate parent programs (depth 1)."""
    return await storage.mget(program.lineage.parents)


async def _get_descendants(storage: ProgramStorage, program: Program) -> list[Program]:
    """Get immediate child programs (depth 1)."""
    return await storage.mget(program.lineage.children)


@StageRegistry.register(description="Evolutionary statistics collector")
class EvolutionaryStatisticsCollector(RelatedCollectorBase):
    OutputModel = EvolutionaryStatistics

    def __init__(self, *, metrics_context: MetricsContext, **kwargs: Any):
        super().__init__(**kwargs)
        self.metrics_context = metrics_context

    async def _collect_programs(self, program: Program) -> list[Program]:
        return await self.storage.get_all()

    async def _process(
        self, program: Program, programs: list[Program]
    ) -> EvolutionaryStatistics:
        # Global statistics (all programs)
        best, worst, avg, valid_rate = _compute_fitness_stats_all_metrics(
            programs, self.metrics_context
        )
        global_avg_children, global_max_children, total_count = (
            _compute_num_children_stats(programs)
        )

        # Program's generation
        generation = program.generation
        iteration = program.get_metadata("iteration")

        # Generation statistics (programs in same generation)
        gen_programs = [p for p in programs if p.generation == generation]
        gen_best, gen_worst, gen_avg, gen_valid_rate = (
            _compute_fitness_stats_all_metrics(gen_programs, self.metrics_context)
        )

        # Iteration statistics (programs in same iteration)
        iter_best, iter_worst, iter_avg, iter_valid_rate = None, None, None, None
        if iteration is not None:
            iter_programs = [
                p for p in programs if p.get_metadata("iteration") == iteration
            ]
            iter_best, iter_worst, iter_avg, iter_valid_rate = (
                _compute_fitness_stats_all_metrics(iter_programs, self.metrics_context)
            )

        # Ancestor statistics (depth 1 - immediate parents)
        ancestors = await _get_ancestors(self.storage, program)
        anc_best, anc_worst, anc_avg, anc_valid_rate = (
            _compute_fitness_stats_all_metrics(ancestors, self.metrics_context)
        )

        # Descendant statistics (depth 1 - immediate children)
        descendants = await _get_descendants(self.storage, program)
        desc_best, desc_worst, desc_avg, desc_valid_rate = (
            _compute_fitness_stats_all_metrics(descendants, self.metrics_context)
        )

        # Generation history (main metric only, across all generations)
        main_metric = self.metrics_context.get_primary_key()
        higher_is_better = self.metrics_context.is_higher_better(main_metric)

        # Group programs by generation
        programs_by_gen: dict[int, list[Program]] = {}
        for p in programs:
            gen = p.generation
            if gen not in programs_by_gen:
                programs_by_gen[gen] = []
            programs_by_gen[gen].append(p)

        # Compute main metric stats for each generation
        generation_history: dict[int, GenerationMetrics] = {}
        for gen_num, gen_progs in sorted(programs_by_gen.items()):
            generation_history[gen_num] = _compute_main_metric_stats(
                gen_progs, main_metric, higher_is_better
            )

        return EvolutionaryStatistics(
            # Program statistics
            generation=generation,
            iteration=iteration,
            current_program_metrics=program.metrics,
            # Global statistics
            best_fitness=best,
            worst_fitness=worst,
            average_fitness=avg,
            valid_rate=valid_rate,
            total_program_count=total_count,
            avg_num_children=global_avg_children,
            max_num_children=global_max_children,
            # Generation statistics
            best_fitness_in_generation=gen_best,
            worst_fitness_in_generation=gen_worst,
            average_fitness_in_generation=gen_avg,
            valid_rate_in_generation=gen_valid_rate,
            # Iteration statistics
            best_fitness_in_iteration=iter_best,
            worst_fitness_in_iteration=iter_worst,
            average_fitness_in_iteration=iter_avg,
            valid_rate_in_iteration=iter_valid_rate,
            # Ancestor statistics
            ancestor_count=len(ancestors),
            best_fitness_in_ancestors=anc_best,
            worst_fitness_in_ancestors=anc_worst,
            average_fitness_in_ancestors=anc_avg,
            valid_rate_in_ancestors=anc_valid_rate,
            # Descendant statistics
            descendant_count=len(descendants),
            best_fitness_in_descendants=desc_best,
            worst_fitness_in_descendants=desc_worst,
            average_fitness_in_descendants=desc_avg,
            valid_rate_in_descendants=desc_valid_rate,
            # Generation history
            generation_history=generation_history,
        )
