from __future__ import annotations

from abc import abstractmethod
import bisect
from typing import Any, TypeVar

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
from gigaevo.programs.program import EXCLUDE_FOR_ANALYTICS, Program
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

    InputsModel: type[StageIO] = VoidInput
    OutputModel: type[StageIO] = VoidOutput
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

    async def _process(self, program: Program, programs: list[Program]) -> StringList:
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
            "[DescendantProgramIds] Selected {} programs for {} with children {}",
            len(selected),
            program.id,
            program.lineage.children,
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
            "[AncestorProgramIds] Selected {} programs for {} with parents {}",
            len(selected),
            program.id,
            program.lineage.parents,
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
    # iteration-window statistics - keyed by metric name. The window is a
    # trailing range of iterations [program.iteration - N, program.iteration]
    # sized by ``EvolutionaryStatisticsCollector.iteration_window_size``;
    # fields are None when the feature is disabled (N=0).
    best_fitness_in_iteration: dict[str, float] | None = Field(
        None, description="Best fitness per metric in the iteration window (valid only)"
    )
    worst_fitness_in_iteration: dict[str, float] | None = Field(
        None,
        description="Worst fitness per metric in the iteration window (valid only)",
    )
    average_fitness_in_iteration: dict[str, float] | None = Field(
        None,
        description="Average fitness per metric in the iteration window (valid only)",
    )
    valid_rate_in_iteration: float | None = Field(
        None, description="Valid rate in the iteration window"
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
    return await storage.mget(program.lineage.parents, exclude=EXCLUDE_FOR_ANALYTICS)


async def _get_descendants(storage: ProgramStorage, program: Program) -> list[Program]:
    """Get immediate child programs (depth 1)."""
    return await storage.mget(program.lineage.children, exclude=EXCLUDE_FOR_ANALYTICS)


@StageRegistry.register(description="Evolutionary statistics collector")
class EvolutionaryStatisticsCollector(RelatedCollectorBase):
    OutputModel = EvolutionaryStatistics

    def __init__(
        self,
        *,
        metrics_context: MetricsContext,
        iteration_window_size: int = 8,
        **kwargs: Any,
    ):
        if iteration_window_size < 0:
            raise ValueError(
                f"iteration_window_size must be >= 0, got {iteration_window_size}"
            )
        super().__init__(**kwargs)
        self.metrics_context = metrics_context
        #: Trailing-window size for iteration cohort aggregation. The window
        #: for a program at iteration ``i`` is the set of programs with
        #: iteration ``∈ [i - N, i]`` (the program itself is always included).
        #: Set to 0 to disable iteration-cohort stats; fields are then None.
        #: Default 8 gives prompts ~8 mutants of recent cohort context — wide
        #: enough to smooth single-program noise, narrow enough to track
        #: short-horizon trend changes.
        self.iteration_window_size = iteration_window_size
        # Population-level stats cache (keyed on list identity from snapshot).
        # Within a single snapshot epoch, all programs see the same population,
        # so global stats, per-generation stats, and generation history are
        # identical.  Computing them once instead of N times reduces O(N²) to O(N).
        self._cached_related_pop_id: int = -1
        self._related_cache: dict[str, Program] = {}
        self._cached_pop_id: int = -1
        self._cached_global: (
            tuple[
                dict[str, float],
                dict[str, float],
                dict[str, float],
                float,
                float,
                int,
                int,
            ]
            | None
        ) = None
        self._cached_gen_stats: (
            dict[
                int, tuple[dict[str, float], dict[str, float], dict[str, float], float]
            ]
            | None
        ) = None
        self._cached_gen_history: dict[int, GenerationMetrics] | None = None
        # Iteration cohort index: programs sorted by iteration, plus a
        # parallel list of iteration keys for bisect lookup. Rebuilt once
        # per snapshot epoch alongside the other caches.
        self._cached_iter_sorted: list[Program] | None = None
        self._cached_iter_keys: list[int] | None = None

    #: Skip metadata (89% of payload) and stage_results (10%) during
    #: deserialization.  The collector only reads metrics, lineage, and
    #: generation — never metadata or stage output.  Iteration-level stats
    #: degrade gracefully (iteration = None → block skipped).
    _EXCLUDE = EXCLUDE_FOR_ANALYTICS

    async def _collect_programs(self, program: Program) -> list[Program]:
        return await self.storage.snapshot.get_all(self.storage, exclude=self._EXCLUDE)

    async def _ensure_related_cache(self, programs: list[Program]) -> None:
        """Build ancestor/descendant lookup from the population, fetching only misses.

        Most parents/children are already in ``programs`` (the population
        snapshot).  This method indexes the population by ID and only hits
        Redis for IDs not found in the population (e.g. programs evicted from
        the MAP-Elites archive).  Eliminates ~N individual mget calls.

        Uses ``EXCLUDE_FOR_ANALYTICS`` for any fallback fetches — must NOT be
        shared with stages that need full program objects.
        """
        pop_id = id(programs)
        if pop_id == self._cached_related_pop_id:
            return

        # Index population by ID (O(N), ~0.2ms for N=5000)
        pop_by_id: dict[str, Program] = {p.id: p for p in programs}

        # Collect all needed parent/child IDs
        all_ids: set[str] = set()
        for p in programs:
            all_ids.update(p.lineage.parents)
            all_ids.update(p.lineage.children)

        # Only fetch IDs not already in the population
        missing_ids = all_ids - pop_by_id.keys()
        if missing_ids:
            fetched = await self.storage.mget(
                list(missing_ids), exclude=EXCLUDE_FOR_ANALYTICS
            )
            for p in fetched:
                pop_by_id[p.id] = p

        self._related_cache = pop_by_id
        self._cached_related_pop_id = pop_id

    def _ensure_population_cache(self, programs: list[Program]) -> None:
        """Compute and cache population-level stats if not already cached."""
        pop_id = id(programs)
        if pop_id == self._cached_pop_id:
            return

        best, worst, avg, valid_rate = _compute_fitness_stats_all_metrics(
            programs, self.metrics_context
        )
        global_avg_children, global_max_children, total_count = (
            _compute_num_children_stats(programs)
        )
        self._cached_global = (
            best,
            worst,
            avg,
            valid_rate,
            global_avg_children,
            global_max_children,
            total_count,
        )

        # Group by generation
        programs_by_gen: dict[int, list[Program]] = {}
        for p in programs:
            gen = p.generation
            if gen not in programs_by_gen:
                programs_by_gen[gen] = []
            programs_by_gen[gen].append(p)

        # Per-generation fitness stats
        gen_stats: dict[
            int, tuple[dict[str, float], dict[str, float], dict[str, float], float]
        ] = {}
        for gen_num, gen_progs in programs_by_gen.items():
            gen_stats[gen_num] = _compute_fitness_stats_all_metrics(
                gen_progs, self.metrics_context
            )
        self._cached_gen_stats = gen_stats

        # Generation history (main metric)
        main_metric = self.metrics_context.get_primary_key()
        higher_is_better = self.metrics_context.is_higher_better(main_metric)
        generation_history: dict[int, GenerationMetrics] = {}
        for gen_num, gen_progs in sorted(programs_by_gen.items()):
            generation_history[gen_num] = _compute_main_metric_stats(
                gen_progs, main_metric, higher_is_better
            )
        self._cached_gen_history = generation_history

        # Iteration cohort index (skipped when the feature is disabled).
        if self.iteration_window_size > 0:
            sorted_by_iter = sorted(programs, key=lambda p: p.iteration)
            self._cached_iter_sorted = sorted_by_iter
            self._cached_iter_keys = [p.iteration for p in sorted_by_iter]
        else:
            self._cached_iter_sorted = None
            self._cached_iter_keys = None

        self._cached_pop_id = pop_id

    def _compute_iteration_window_stats(
        self, program: Program
    ) -> tuple[
        dict[str, float] | None,
        dict[str, float] | None,
        dict[str, float] | None,
        float | None,
    ]:
        """Aggregate stats over the trailing iteration window of ``program``.

        Window = programs whose iteration ``∈ [program.iteration - N, program.iteration]``
        where ``N == self.iteration_window_size``. Returns ``(None, None, None, None)``
        when the feature is disabled (``N == 0``). The program itself is always
        in its own window so the cohort is non-empty for any valid input.
        """
        if (
            self.iteration_window_size == 0
            or self._cached_iter_sorted is None
            or self._cached_iter_keys is None
        ):
            return (None, None, None, None)

        lo = program.iteration - self.iteration_window_size
        hi = program.iteration
        left = bisect.bisect_left(self._cached_iter_keys, lo)
        right = bisect.bisect_right(self._cached_iter_keys, hi)
        window = self._cached_iter_sorted[left:right]
        return _compute_fitness_stats_all_metrics(window, self.metrics_context)

    async def _process(
        self, program: Program, programs: list[Program]
    ) -> EvolutionaryStatistics:
        # Pre-fetch all ancestors/descendants in one batch (cached per epoch)
        await self._ensure_related_cache(programs)
        # Population-level stats (cached per snapshot epoch)
        self._ensure_population_cache(programs)
        assert self._cached_global is not None
        assert self._cached_gen_stats is not None
        assert self._cached_gen_history is not None
        (
            best,
            worst,
            avg,
            valid_rate,
            global_avg_children,
            global_max_children,
            total_count,
        ) = self._cached_global

        # Program's generation
        generation = program.generation
        iteration = program.iteration

        # Generation statistics (cached)
        gen_best, gen_worst, gen_avg, gen_valid_rate = self._cached_gen_stats.get(
            generation, ({}, {}, {}, 0.0)
        )

        # Iteration cohort aggregates: trailing window over programs with
        # iteration ∈ [program.iteration - N, program.iteration]. When the
        # feature is disabled (N=0) the fields stay None for schema stability.
        iter_best, iter_worst, iter_avg, iter_valid_rate = (
            self._compute_iteration_window_stats(program)
        )

        # Ancestor statistics (depth 1 - immediate parents, from batch cache)
        ancestors = [
            self._related_cache[pid]
            for pid in program.lineage.parents
            if pid in self._related_cache
        ]
        anc_best, anc_worst, anc_avg, anc_valid_rate = (
            _compute_fitness_stats_all_metrics(ancestors, self.metrics_context)
        )

        # Descendant statistics (depth 1 - immediate children, from batch cache)
        descendants = [
            self._related_cache[cid]
            for cid in program.lineage.children
            if cid in self._related_cache
        ]
        desc_best, desc_worst, desc_avg, desc_valid_rate = (
            _compute_fitness_stats_all_metrics(descendants, self.metrics_context)
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
            generation_history=self._cached_gen_history,
        )
