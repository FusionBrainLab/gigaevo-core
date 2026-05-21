from __future__ import annotations

from abc import abstractmethod
import bisect
from collections.abc import Sequence
import statistics
from typing import Any, Literal, TypeVar

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

DEFAULT_ITER_WINDOW_RADIUS = 15
MEDIAN_HORIZON = 10
# Bootstrap-only fallback when MAD cannot be estimated (window <4 valid).
# Kept identical to the pre-bundle constant so behaviour at start-of-run is
# unchanged.
TREND_DELTA_RATIO = 0.05
# Minimum sample size to compute MAD meaningfully — below this, fall back
# to TREND_DELTA_RATIO. Not a regime threshold; a data-availability gate.
N_MIN_FOR_MAD = 4
# Minimum sample size for archive quartile estimation. Same justification:
# below 4 samples a quartile is degenerate.
N_MIN_ARCHIVE = 4
# Numerical guard against true-zero MAD — keeps the inequality strict
# without acting as a regime cutoff.
_TREND_EPSILON = 1e-12


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


TrendLabel = Literal["rising", "flat", "falling"]


class EvolutionaryStatistics(StageIO):
    """Snapshot of population state seen by a focal program in the mutation prompt."""

    generation: int = Field(description="Program lineage generation depth")
    iteration: int | None = Field(None, description="Evolution loop iteration number")
    current_program_metrics: dict[str, float] = Field(
        description="Metrics of the current program"
    )
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
    total_program_count: int = Field(description="Total number of programs")
    avg_num_children: float = Field(
        description="Average number of children per program"
    )
    max_num_children: int = Field(
        description="Maximum number of children any program has"
    )
    iter_window_lo: int | None = Field(None, description="Window lower bound")
    iter_window_hi: int | None = Field(None, description="Window upper bound")
    iter_window_programs: int = Field(0, description="Programs in window")
    iter_window_valid: int = Field(0, description="Valid programs in window")
    iter_window_best_fitness: float | None = Field(
        None, description="Best primary fitness in window"
    )
    iter_window_best_iter: int | None = Field(
        None, description="Iter of window's best fitness"
    )
    iter_window_rank: int | None = Field(
        None, description="Focal's rank among window valid programs (1=best)"
    )
    iter_window_median_before: float | None = Field(
        None, description="Median of last 10 valid fitnesses before focal"
    )
    iter_window_median_after: float | None = Field(
        None, description="Median of first 10 valid fitnesses after focal"
    )
    iter_window_trend: TrendLabel = Field("flat", description="Window trend")
    iter_window_trend_thirds: tuple[float | None, float | None, float | None] = Field(
        (None, None, None), description="Medians of window thirds, in iter order"
    )
    iter_window_invalid_streak_max: int = Field(
        0, description="Max consecutive invalid in window"
    )
    iter_window_invalid_count: int = Field(0, description="Invalid count in window")
    iters_since_last_new_best: int = Field(
        0, description="Iters since global running-best last advanced (at focal)"
    )
    archive_valid_fitnesses: tuple[float, ...] = Field(
        default=(),
        description=(
            "Sorted-ascending valid primary fitnesses from the population (whole-run "
            "archive snapshot). Render-time input for quartile/regime classification."
        ),
    )
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


def _max_invalid_streak(window: list[Program]) -> int:
    streak = max_streak = 0
    for p in window:
        if p.metrics.get(VALIDITY_KEY, 0) > 0:
            streak = 0
        else:
            streak += 1
            if streak > max_streak:
                max_streak = streak
    return max_streak


def _mad(values: Sequence[float]) -> float:
    """Median absolute deviation. Returns 0 for degenerate input."""
    if len(values) < 2:
        return 0.0
    s = sorted(values)
    m = s[len(s) // 2]
    abs_devs = sorted(abs(v - m) for v in values)
    return abs_devs[len(abs_devs) // 2]


def _trend_from_thirds(
    fits: list[float], higher_is_better: bool
) -> tuple[TrendLabel, tuple[float | None, float | None, float | None]]:
    # 'rising' means improving, 'falling' means worsening — direction-aware so
    # the LLM sees the same word regardless of higher_is_better.
    n = len(fits)
    if n < 6:
        return ("flat", (None, None, None))

    t1 = statistics.median(fits[: n // 3])
    t2 = statistics.median(fits[n // 3 : 2 * n // 3])
    t3 = statistics.median(fits[2 * n // 3 :])

    # Noise floor: MAD of recent valid fits when sample size permits, else the
    # legacy ratio of |t1|. MAD is a textbook nonparametric noise estimator —
    # no chosen number, just the run's own dispersion.
    if n >= N_MIN_FOR_MAD:
        noise_floor = _mad(fits)
    else:
        noise_floor = abs(t1) * TREND_DELTA_RATIO
    noise_floor = max(noise_floor, _TREND_EPSILON)

    if higher_is_better:
        going_up = (t3 - t1) > noise_floor and t3 >= t2
        going_down = (t1 - t3) > noise_floor and t3 <= t2
    else:
        going_up = (t1 - t3) > noise_floor and t3 <= t2
        going_down = (t3 - t1) > noise_floor and t3 >= t2

    if going_up:
        return ("rising", (t1, t2, t3))
    if going_down:
        return ("falling", (t1, t2, t3))
    return ("flat", (t1, t2, t3))


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
        iteration_window_radius: int = DEFAULT_ITER_WINDOW_RADIUS,
        **kwargs: Any,
    ):
        if iteration_window_radius < 0:
            raise ValueError(
                f"iteration_window_radius must be >= 0, got {iteration_window_radius}"
            )
        super().__init__(**kwargs)
        self.metrics_context = metrics_context
        self.iteration_window_radius = iteration_window_radius
        # Population-level stats cache (keyed on list identity from snapshot).
        # Within a single snapshot epoch, all programs see the same population,
        # so global stats are identical.  Computing them once instead of N
        # times reduces O(N²) to O(N).
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

        # Iteration cohort index. Only programs with a real iteration value
        # (>= 0) are indexed; seed programs occasionally lack one and would
        # corrupt the bisect order.
        if self.iteration_window_radius > 0:
            with_iter = [p for p in programs if p.iteration is not None]
            sorted_by_iter = sorted(with_iter, key=lambda p: p.iteration)
            self._cached_iter_sorted = sorted_by_iter
            self._cached_iter_keys = [p.iteration for p in sorted_by_iter]
        else:
            self._cached_iter_sorted = None
            self._cached_iter_keys = None

        self._cached_pop_id = pop_id

    def _compute_iter_window_fields(
        self,
        program: Program,
        primary_key: str,
        higher_is_better: bool,
    ) -> dict[str, Any]:
        empty = {
            "iter_window_lo": None,
            "iter_window_hi": None,
            "iter_window_programs": 0,
            "iter_window_valid": 0,
            "iter_window_best_fitness": None,
            "iter_window_best_iter": None,
            "iter_window_rank": None,
            "iter_window_median_before": None,
            "iter_window_median_after": None,
            "iter_window_trend": "flat",
            "iter_window_trend_thirds": (None, None, None),
            "iter_window_invalid_streak_max": 0,
            "iter_window_invalid_count": 0,
            "iters_since_last_new_best": 0,
        }

        if (
            self.iteration_window_radius == 0
            or self._cached_iter_sorted is None
            or self._cached_iter_keys is None
            or program.iteration is None
        ):
            return empty

        focal_iter = program.iteration
        lo = focal_iter - self.iteration_window_radius
        hi = focal_iter + self.iteration_window_radius
        left = bisect.bisect_left(self._cached_iter_keys, lo)
        right = bisect.bisect_right(self._cached_iter_keys, hi)
        window = self._cached_iter_sorted[left:right]
        window_size = len(window)

        valid_with_fit = [
            p
            for p in window
            if p.metrics.get(VALIDITY_KEY, 0) > 0 and primary_key in p.metrics
        ]
        # Snapshot-lag safeguard: when the collector runs for `program`, the
        # population snapshot may not yet contain it (or may contain a stale
        # view with VALIDITY_KEY=0). The focal program is logically a member
        # of its own window, so include it here using the up-to-date metrics
        # passed in by the pipeline. Without this, top-of-window programs hit
        # the silent `sorted.index(focal_fit) → ValueError → rank=None` path
        # and the rank line disappears from rendered stats.
        focal_valid = program.metrics.get(VALIDITY_KEY, 0) > 0
        focal_fit = program.metrics.get(primary_key)
        focal_already_counted = any(p.id == program.id for p in valid_with_fit)
        if focal_valid and focal_fit is not None and not focal_already_counted:
            valid_with_fit = valid_with_fit + [program]
        invalid_count = window_size - sum(
            1 for p in window if p.metrics.get(VALIDITY_KEY, 0) > 0
        )
        invalid_streak_max = _max_invalid_streak(window)

        out: dict[str, Any] = {
            "iter_window_lo": lo,
            "iter_window_hi": hi,
            "iter_window_programs": window_size,
            "iter_window_valid": len(valid_with_fit),
            "iter_window_invalid_streak_max": invalid_streak_max,
            "iter_window_invalid_count": invalid_count,
            "iter_window_best_fitness": None,
            "iter_window_best_iter": None,
            "iter_window_rank": None,
            "iter_window_median_before": None,
            "iter_window_median_after": None,
            "iter_window_trend": "flat",
            "iter_window_trend_thirds": (None, None, None),
            "iters_since_last_new_best": self._compute_global_plateau(
                focal_iter, primary_key, higher_is_better
            ),
        }

        if not valid_with_fit:
            return out

        fits = [p.metrics[primary_key] for p in valid_with_fit]
        best_fit = max(fits) if higher_is_better else min(fits)
        best_iter = next(
            p.iteration for p in valid_with_fit if p.metrics[primary_key] == best_fit
        )
        out["iter_window_best_fitness"] = best_fit
        out["iter_window_best_iter"] = best_iter

        if focal_valid and focal_fit is not None:
            # Count-based rank (1 = best). Robust to duplicate fitness values
            # — `sorted.index` previously returned the first match for ties,
            # which under-counted equal-fit competitors.
            if higher_is_better:
                better = sum(1 for f in fits if f > focal_fit)
            else:
                better = sum(1 for f in fits if f < focal_fit)
            out["iter_window_rank"] = better + 1

        before = [
            p.metrics[primary_key] for p in valid_with_fit if p.iteration < focal_iter
        ][-MEDIAN_HORIZON:]
        after = [
            p.metrics[primary_key] for p in valid_with_fit if p.iteration > focal_iter
        ][:MEDIAN_HORIZON]
        if before:
            out["iter_window_median_before"] = statistics.median(before)
        if after:
            out["iter_window_median_after"] = statistics.median(after)

        trend_label, thirds = _trend_from_thirds(fits, higher_is_better)
        out["iter_window_trend"] = trend_label
        out["iter_window_trend_thirds"] = thirds
        return out

    def _compute_global_plateau(
        self,
        focal_iter: int,
        primary_key: str,
        higher_is_better: bool,
    ) -> int:
        if self._cached_iter_sorted is None or self._cached_iter_keys is None:
            return 0

        hi_idx = bisect.bisect_right(self._cached_iter_keys, focal_iter)
        candidates = self._cached_iter_sorted[:hi_idx]

        running_best: float | None = None
        last_new_best_iter: int | None = None
        for p in candidates:
            if p.metrics.get(VALIDITY_KEY, 0) <= 0:
                continue
            fit = p.metrics.get(primary_key)
            if fit is None:
                continue
            if running_best is None or (
                fit > running_best if higher_is_better else fit < running_best
            ):
                running_best = fit
                last_new_best_iter = p.iteration

        if last_new_best_iter is None:
            return focal_iter
        return focal_iter - last_new_best_iter

    async def _process(
        self, program: Program, programs: list[Program]
    ) -> EvolutionaryStatistics:
        # Pre-fetch all ancestors/descendants in one batch (cached per epoch)
        await self._ensure_related_cache(programs)
        # Population-level stats (cached per snapshot epoch)
        self._ensure_population_cache(programs)
        assert self._cached_global is not None
        (
            best,
            worst,
            avg,
            valid_rate,
            global_avg_children,
            global_max_children,
            total_count,
        ) = self._cached_global

        primary_key = self.metrics_context.get_primary_key()
        higher_is_better = self.metrics_context.is_higher_better(primary_key)

        iter_window_fields = self._compute_iter_window_fields(
            program, primary_key, higher_is_better
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

        # Archive snapshot of valid primary fitnesses — sorted ascending so the
        # renderer can compute quartile boundaries in O(1) without resorting.
        archive_valid_fitnesses: tuple[float, ...] = tuple(
            sorted(
                p.metrics[primary_key]
                for p in programs
                if p.metrics.get(VALIDITY_KEY, 0) > 0 and primary_key in p.metrics
            )
        )

        return EvolutionaryStatistics(
            generation=program.generation,
            iteration=program.iteration,
            current_program_metrics=program.metrics,
            best_fitness=best,
            worst_fitness=worst,
            average_fitness=avg,
            valid_rate=valid_rate,
            total_program_count=total_count,
            avg_num_children=global_avg_children,
            max_num_children=global_max_children,
            ancestor_count=len(ancestors),
            best_fitness_in_ancestors=anc_best,
            worst_fitness_in_ancestors=anc_worst,
            average_fitness_in_ancestors=anc_avg,
            valid_rate_in_ancestors=anc_valid_rate,
            descendant_count=len(descendants),
            best_fitness_in_descendants=desc_best,
            worst_fitness_in_descendants=desc_worst,
            average_fitness_in_descendants=desc_avg,
            valid_rate_in_descendants=desc_valid_rate,
            archive_valid_fitnesses=archive_valid_fitnesses,
            **iter_window_fields,
        )
