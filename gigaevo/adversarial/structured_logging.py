"""Structured JSON logging for v3 adversarial evolution (log-based verification).

Canonical events emitted during pipeline execution. Each event is a JSON object
logged via loguru with [EVENT_TYPE] prefix, parseable for post-hoc audit.
"""

from __future__ import annotations

import json
from typing import Any


def emit_tracker_write(
    pairs_count: int,
    positive_count: int,
    d_wins_added: int,
    g_resisted_added: int,
    d_faced_added: int,
    gen: int | None = None,
) -> dict[str, Any]:
    """TRACKER_WRITE: inverted-index dual-write completion.

    Emitted by DGImprovementTracker.record_batch after all five key families
    are written (per-G sorted sets, best-pairs, d_wins SET, g_resisted SET,
    d_delta HASH) in a single Redis pipeline.
    """
    return {
        "event": "TRACKER_WRITE",
        "gen": gen,
        "pairs_count": pairs_count,
        "positive_count": positive_count,
        "d_wins_added": d_wins_added,
        "g_resisted_added": g_resisted_added,
        "d_faced_added": d_faced_added,
    }


def emit_hof_fetch(
    label: str,
    n_elites: int,
    fitness_key: str,
    gen: int | None = None,
) -> dict[str, Any]:
    """HOF_FETCH: current HoF fetched from archive for opponent sampling.

    Emitted by FetchOpponentIdsStage when it loads the opponent HoF from Redis.
    """
    return {
        "event": "HOF_FETCH",
        "gen": gen,
        "label": label,
        "n_elites": n_elites,
        "fitness_key": fitness_key,
    }


def emit_hof_rotate(
    label: str,
    old_hof_size: int,
    new_hof_size: int,
    gen: int | None = None,
) -> dict[str, Any]:
    """HOF_ROTATE: HoF changed (new elites added or removed).

    Emitted by cache invalidation logic when archive state changes between
    DAG steps, signaling opponent HoF may have rotated.
    """
    return {
        "event": "HOF_ROTATE",
        "gen": gen,
        "label": label,
        "old_hof_size": old_hof_size,
        "new_hof_size": new_hof_size,
    }


def emit_cell_pick(
    label: str,
    cell_id: str,
    program_id: str,
    fitness_key: str,
    fitness_value: float,
    gen: int | None = None,
) -> dict[str, Any]:
    """CELL_PICK: CellStratifiedRedisOpponentArchiveProvider picked one elite from cell.

    Emitted per-cell by get_top_k when selecting distinct-cell opponents.
    """
    return {
        "event": "CELL_PICK",
        "gen": gen,
        "label": label,
        "cell_id": cell_id,
        "program_id": program_id,
        "fitness_key": fitness_key,
        "fitness_value": fitness_value,
    }


def emit_cache_hit(
    stage_name: str,
    cache_key: str,
    gen: int | None = None,
) -> dict[str, Any]:
    """CACHE_HIT: in-memory cache hit (no Redis round-trip)."""
    return {
        "event": "CACHE_HIT",
        "gen": gen,
        "stage_name": stage_name,
        "cache_key": cache_key,
    }


def emit_cache_miss(
    stage_name: str,
    cache_key: str,
    gen: int | None = None,
) -> dict[str, Any]:
    """CACHE_MISS: cache miss, refreshing from Redis."""
    return {
        "event": "CACHE_MISS",
        "gen": gen,
        "stage_name": stage_name,
        "cache_key": cache_key,
    }


def emit_lineage_trend(
    program_id: str,
    d_id: str,
    parent_d_id: str,
    trend: float | None,
    n_shared: int,
    gen: int | None = None,
) -> dict[str, Any]:
    """LINEAGE_TREND: SharedBenchmarkLineageStage computed child vs parent trend.

    Emitted after computing shared-benchmark trend for child-parent pair.
    """
    return {
        "event": "LINEAGE_TREND",
        "gen": gen,
        "program_id": program_id,
        "d_id": d_id,
        "parent_d_id": parent_d_id,
        "trend": trend,
        "n_shared": n_shared,
    }


def emit_metric_emit(
    program_id: str,
    metric_name: str,
    metric_value: Any,
    gen: int | None = None,
) -> dict[str, Any]:
    """METRIC_EMIT: metric written to program.metrics dict for BD binning.

    Emitted by stages (tracker coverage, lineage) when computed metrics are
    stored in program.metrics.
    """
    return {
        "event": "METRIC_EMIT",
        "gen": gen,
        "program_id": program_id,
        "metric_name": metric_name,
        "metric_value": metric_value,
    }


def emit_gradient_inject(
    program_id: str,
    label: str,
    opponent_count: int,
    gen: int | None = None,
) -> dict[str, Any]:
    """GRADIENT_INJECT: gradient prompt injected with opponent feedback."""
    return {
        "event": "GRADIENT_INJECT",
        "gen": gen,
        "program_id": program_id,
        "label": label,
        "opponent_count": opponent_count,
    }


def emit_archive_move(
    program_id: str,
    old_cell: str | None,
    new_cell: str,
    fitness_key: str,
    fitness_value: float,
    gen: int | None = None,
) -> dict[str, Any]:
    """ARCHIVE_MOVE: program placed or moved in archive cell.

    Emitted after each program is added to archive (or cell changed due to
    metric re-evaluation).
    """
    return {
        "event": "ARCHIVE_MOVE",
        "gen": gen,
        "program_id": program_id,
        "old_cell": old_cell,
        "new_cell": new_cell,
        "fitness_key": fitness_key,
        "fitness_value": fitness_value,
    }


def format_json_log(event_dict: dict[str, Any]) -> str:
    """Format event dict as JSON log line for loguru."""
    return json.dumps(event_dict, separators=(",", ":"), default=str)
