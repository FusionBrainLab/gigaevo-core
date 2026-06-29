"""Pure fitness/attribution helpers shared across the write-path components.

These read only a ``Program``'s metrics and frozen mutation metadata — no store,
no LLM, no event loop — so the record extractor, the card-stats updater, and the
exemplar selection in the orchestrator all draw the same validity/sentinel
semantics from one place.
"""

from __future__ import annotations

import math

from gigaevo.evolution.mutation.constants import (
    MUTATION_MEMORY_BASE_METRICS_METADATA_KEY,
    MUTATION_MEMORY_BASE_SELECTED_IDS_METADATA_KEY,
    MUTATION_OUTPUT_METADATA_KEY,
)
from gigaevo.programs.metrics.context import VALIDITY_KEY, MetricsContext
from gigaevo.programs.program import Program


def valid_fitness(
    prog: Program,
    fitness_key: str,
    metrics_context: MetricsContext | None = None,
) -> float | None:
    """Fitness under ``fitness_key``, or ``None`` for invalid programs.

    ``is_valid`` is a contract: every evaluated program carries it, so a missing
    flag is treated as invalid — the same strict semantics record eligibility
    uses. Invalid programs carry a sentinel floor fitness; treating it as a real
    value would manufacture catastrophic harm (invalid child) or phantom
    improvement (invalid parent baseline). When ``metrics_context`` is wired, a
    fitness equal to the metric's sentinel is rejected even if the program
    claims validity. Contrast: :meth:`MetricsContext.is_valid` defaults a
    missing flag to valid — correct for metric aggregation, not for stat
    tracking.
    """
    is_valid = prog.metrics.get(VALIDITY_KEY)
    if is_valid is None or is_valid <= 0:
        return None
    fit = prog.metrics.get(fitness_key)
    if fit is None or not math.isfinite(fit):
        return None
    if metrics_context is not None and metrics_context.is_sentinel(fitness_key, fit):
        return None
    return float(fit)


def evaluated_invalid(
    prog: Program,
    fitness_key: str,
    metrics_context: MetricsContext | None = None,
) -> bool:
    """True iff the program was evaluated and judged invalid — a real negative
    outcome the posterior must count as harm, unlike a program that simply never
    produced a fitness (missing ``is_valid``: not evaluated, no signal)."""
    is_valid = prog.metrics.get(VALIDITY_KEY)
    if is_valid is None:
        return False
    if is_valid <= 0:
        return True
    fit = prog.metrics.get(fitness_key)
    if fit is None or not math.isfinite(fit):
        return False
    return metrics_context is not None and metrics_context.is_sentinel(fitness_key, fit)


def base_fitness(
    base_metrics: dict[str, float],
    fitness_key: str,
    metrics_context: MetricsContext | None,
) -> float | None:
    """Base parent's reward baseline from its frozen metrics, mirroring the
    validity/sentinel semantics of :func:`valid_fitness`."""
    is_valid = base_metrics.get(VALIDITY_KEY)
    if is_valid is None or is_valid <= 0:
        return None
    fit = base_metrics.get(fitness_key)
    if fit is None or not math.isfinite(fit):
        return None
    if metrics_context is not None and metrics_context.is_sentinel(fitness_key, fit):
        return None
    return float(fit)


def card_ids_used(prog: Program) -> list[str]:
    """Card ids the mutator declared it applied, from the stamped structured output."""
    out = prog.get_metadata(MUTATION_OUTPUT_METADATA_KEY)
    if isinstance(out, dict):
        return list(out.get("card_ids_used", []) or [])
    return []


def base_selected_ids(prog: Program) -> list[str]:
    """Cards selected for the mutator's named base parent, frozen at birth."""
    ids = prog.get_metadata(MUTATION_MEMORY_BASE_SELECTED_IDS_METADATA_KEY)
    return list(ids) if isinstance(ids, list) else []


def base_metrics(prog: Program) -> dict[str, float]:
    """The base parent's metric dict, frozen at birth."""
    metrics = prog.get_metadata(MUTATION_MEMORY_BASE_METRICS_METADATA_KEY)
    return dict(metrics) if isinstance(metrics, dict) else {}


def select_top_programs(
    programs: list[Program],
    *,
    best_programs_percent: float,
    fitness_key: str,
    higher_is_better: bool,
    metrics_context: MetricsContext | None,
) -> list[tuple[Program, float]]:
    """The top-fitness slice of valid programs, as (program, fitness) pairs."""
    if best_programs_percent <= 0:
        return []
    scored = [
        (prog, fit)
        for prog in programs
        for fit in (valid_fitness(prog, fitness_key, metrics_context),)
        if fit is not None
    ]
    if not scored:
        return []
    scored.sort(key=lambda pair: (pair[1], pair[0].id), reverse=higher_is_better)
    count = max(1, math.ceil(len(scored) * best_programs_percent / 100.0))
    return scored[:count]
