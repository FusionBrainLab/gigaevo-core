"""Adversarial evaluate.py for Pop B (Improver).

Receives:
    opponent_results: list of (11, 2) np.ndarray  (point configs from Pop A)
    program_output:   callable improve(points) -> improved_points

Fitness = mean normalized improvement across opponent configurations
    actual_fitness = best post-improvement min_area achieved (for paper reporting)
"""

from __future__ import annotations

from helper import get_smallest_triangle_area, get_unit_triangle, is_inside_triangle
import numpy as np

Q_MAX = 0.0365

INVALID = {
    "fitness": -1.0,
    "is_valid": 0.0,
    "actual_fitness": -1.0,
    "mean_improvement_raw": -1.0,
    "mean_pre_quality": -1.0,
    "mean_post_quality": -1.0,
    "max_post_quality": -1.0,
    "n_opponents": 0.0,
}


def _validate_config(points: object) -> np.ndarray | None:
    """Validate a point configuration. Returns array or None if invalid."""
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


def evaluate(opponent_results: list, program_output: object) -> dict[str, float]:
    """Cross-play: improver vs opponent constructor configs."""
    improve_fn = program_output
    if not callable(improve_fn):
        return INVALID

    if not opponent_results:
        return INVALID

    deltas_norm = []
    pre_qualities = []
    post_qualities = []

    for config in opponent_results:
        config = _validate_config(config)
        if config is None:
            continue
        pre_q = float(get_smallest_triangle_area(config))
        if pre_q <= 0:
            continue
        try:
            improved = improve_fn(config.copy())
            improved = _validate_config(improved)
            if improved is None:
                deltas_norm.append(0.0)
                pre_qualities.append(pre_q)
                post_qualities.append(pre_q)
                continue
            post_q = float(get_smallest_triangle_area(improved))
            delta = max(post_q - pre_q, 0.0)
            deltas_norm.append(min(delta / Q_MAX, 1.0))
            pre_qualities.append(pre_q)
            post_qualities.append(post_q)
        except Exception:
            deltas_norm.append(0.0)
            pre_qualities.append(pre_q)
            post_qualities.append(pre_q)

    if not deltas_norm:
        return INVALID

    fitness = sum(deltas_norm) / len(deltas_norm)
    mean_improvement_raw = sum(d * Q_MAX for d in deltas_norm) / len(deltas_norm)

    return {
        "fitness": float(fitness),
        "is_valid": 1.0,
        "actual_fitness": float(max(post_qualities)),
        "mean_improvement_raw": float(mean_improvement_raw),
        "mean_pre_quality": float(sum(pre_qualities) / len(pre_qualities)),
        "mean_post_quality": float(sum(post_qualities) / len(post_qualities)),
        "max_post_quality": float(max(post_qualities)),
        "n_opponents": float(len(deltas_norm)),
    }
