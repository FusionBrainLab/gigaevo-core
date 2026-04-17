"""Adversarial evaluate.py for Pop A (Constructor) — v3 clean.

v3 drops the ALPHA·quality + (1-ALPHA)·resistance scalarization. The scalar
`fitness` is now just normalized quality: min(min_area / Q_MAX, 1.0). Selection
and BD x-axis read `fitness`; the BD y-axis is `wins` (# D improvers this G
has resisted, written by ComputeGResistedCountStage from the career tracker).

Receives:
    opponent_results: list of callables improve(points) -> improved_points  (from Pop B)
    program_output:   (11, 2) np.ndarray  (this Constructor's point configuration)

Returns: (metrics_dict, artifact_dict)
    metrics_dict — float-only metrics for MAP-Elites + paper.
        fitness        = min(min_area / Q_MAX, 1.0)  ∈ [0, 1]
        actual_fitness = raw min_area (paper reporting, comparable to baseline 0.03449).
    artifact_dict — per-opponent fitness deltas for DGTrackerStage, aligned
        with opponent_results order. Suppressed in the LLM prompt by
        NullArtifactStage in the adversarial pipeline.
"""

from __future__ import annotations

from helper import get_smallest_triangle_area, get_unit_triangle, is_inside_triangle
import numpy as np

Q_MAX = 0.0365

INVALID_METRICS = {
    "fitness": -1000.0,
    "is_valid": 0.0,
    "actual_fitness": -1000.0,
    "mean_improvement": -1000.0,
    "best_post_improvement": -1000.0,
    "n_opponents": 0.0,
}


def _invalid_artifact(n: int) -> dict:
    return {
        "role": "constructor",
        "n_opponents": n,
        "per_opp_pre": [float("nan")] * n,
        "per_opp_post": [float("nan")] * n,
        "per_opp_delta": [float("nan")] * n,
    }


def _validate_config(points: object) -> np.ndarray | None:
    try:
        pts = np.asarray(points, dtype=float)
    except (ValueError, TypeError):
        return None
    if pts.ndim != 2 or pts.shape != (11, 2):
        return None
    if not np.all(np.isfinite(pts)):
        return None
    A, B, C = get_unit_triangle()
    if not is_inside_triangle(pts, A, B, C):
        return None
    return pts


def evaluate(opponent_results: list, program_output: object) -> tuple[dict, dict]:
    """Cross-play: constructor config vs opponent improvers.

    Returns (metrics, artifact). The artifact carries per-opponent fitness
    deltas aligned with opponent_results order, consumed by DGTrackerStage
    to populate the (D, G, fitness_delta) career tracker.
    """
    n_in = len(opponent_results) if opponent_results else 0
    points = _validate_config(program_output)
    if points is None:
        return INVALID_METRICS, _invalid_artifact(n_in)

    raw_quality = float(get_smallest_triangle_area(points))
    if raw_quality <= 0:
        return INVALID_METRICS, _invalid_artifact(n_in)

    fitness = min(raw_quality / Q_MAX, 1.0)

    if not opponent_results:
        metrics = {
            "fitness": float(fitness),
            "is_valid": 1.0,
            "actual_fitness": raw_quality,
            "mean_improvement": 0.0,
            "best_post_improvement": raw_quality,
            "n_opponents": 0.0,
        }
        return metrics, _invalid_artifact(0)

    n = len(opponent_results)
    per_opp_pre: list[float] = [float("nan")] * n
    per_opp_post: list[float] = [float("nan")] * n
    per_opp_delta: list[float] = [float("nan")] * n

    deltas: list[float] = []
    post_qualities: list[float] = [raw_quality]  # self baseline anchors the max

    for i, improve_fn in enumerate(opponent_results):
        if not callable(improve_fn):
            continue
        try:
            improved = improve_fn(points.copy())
            improved = _validate_config(improved)
            if improved is None:
                continue
            post_q = float(get_smallest_triangle_area(improved))
            delta = post_q - raw_quality
            deltas.append(delta)
            post_qualities.append(post_q)
            per_opp_pre[i] = raw_quality
            per_opp_post[i] = post_q
            per_opp_delta[i] = delta
        except Exception:
            continue

    artifact = {
        "role": "constructor",
        "n_opponents": n,
        "per_opp_pre": per_opp_pre,
        "per_opp_post": per_opp_post,
        "per_opp_delta": per_opp_delta,
    }

    mean_delta = sum(deltas) / len(deltas) if deltas else 0.0

    metrics = {
        "fitness": float(fitness),
        "is_valid": 1.0,
        "actual_fitness": raw_quality,
        "mean_improvement": float(mean_delta),
        "best_post_improvement": float(max(post_qualities)),
        "n_opponents": float(len(deltas)),
    }
    return metrics, artifact
