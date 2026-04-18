"""Thin adversarial-emit helpers that route through the canonical registry.

Each helper constructs a Pydantic event from `gigaevo.adversarial.events` (or
`gigaevo.monitoring.events` for shared ones) and emits it via
`gigaevo.monitoring.emit.emit`. Callers no longer assemble dicts or call
`logger.info(...)` directly — that is the single emission seam now.

Dead helpers (`emit_cache_hit`, `emit_cache_miss`, `emit_gradient_inject`,
`emit_archive_move`) were removed per the plan: CACHE_* is subsumed by
STAGE_EXEC, GRADIENT_INJECT and ARCHIVE_MOVE were overfit adversarial-specific
failure-mode instrumentation that nobody consumed.
"""

from __future__ import annotations

from typing import Any

from gigaevo.adversarial.events import CellPick, HofFetch, HofRotate, TrackerWrite
from gigaevo.monitoring.emit import emit
from gigaevo.monitoring.events import LineageTrend, MetricEmit


def emit_tracker_write(
    *,
    pairs_count: int,
    positive_count: int,
    d_wins_added: int,
    g_resisted_added: int,
    d_faced_added: int,
    gen: int | None = None,
) -> None:
    emit(
        TrackerWrite(
            pairs_count=pairs_count,
            positive_count=positive_count,
            d_wins_added=d_wins_added,
            g_resisted_added=g_resisted_added,
            d_faced_added=d_faced_added,
            gen=gen,
        )
    )


def emit_hof_fetch(
    *,
    label: str,
    n_elites: int,
    fitness_key: str,
    gen: int | None = None,
    k_requested: int | None = None,
    cells_populated: int | None = None,
) -> None:
    emit(
        HofFetch(
            label=label,
            n_elites=n_elites,
            fitness_key=fitness_key,
            gen=gen,
            k_requested=k_requested,
            cells_populated=cells_populated,
        )
    )


def emit_hof_rotate(
    *,
    label: str,
    old_hof_size: int,
    new_hof_size: int,
    gen: int | None = None,
    fitness_key: str | None = None,
) -> None:
    emit(
        HofRotate(
            label=label,
            old_hof_size=old_hof_size,
            new_hof_size=new_hof_size,
            gen=gen,
            fitness_key=fitness_key,
        )
    )


def emit_cell_pick(
    *,
    label: str,
    cell_id: str,
    program_id: str,
    fitness_key: str,
    fitness_value: float,
    gen: int | None = None,
) -> None:
    emit(
        CellPick(
            label=label,
            cell_id=cell_id,
            program_id=program_id,
            fitness_key=fitness_key,
            fitness_value=fitness_value,
            gen=gen,
        )
    )


def emit_metric_emit(
    *,
    program_id: str,
    metric: str,
    value: Any,
) -> None:
    emit(MetricEmit(program_id=program_id, metric=metric, value=value))


def emit_lineage_trend(
    *,
    program_id: str,
    trend: float | None,
    gen: int | None = None,
) -> None:
    emit(LineageTrend(program_id=program_id, trend=trend, gen=gen))
