"""Adversarial evaluate.py for Pop A (Constructor).

Receives:
    opponent_results: list of callables improve(points) -> improved_points  (from Pop B)
    program_output:   (11, 2) np.ndarray  (this Constructor's point configuration)

Fitness = ALPHA * quality + (1 - ALPHA) * resistance
    quality    = min(min_area / Q_MAX, 1.0)
    resistance = 1 - mean(normalized improvements by opponents)
    actual_fitness = raw min_area (tracked separately for paper reporting)
"""

from __future__ import annotations

from helper import get_smallest_triangle_area, get_unit_triangle, is_inside_triangle
import numpy as np

Q_MAX = 0.0365
ALPHA = 0.5

INVALID = {
    "fitness": -1.0,
    "is_valid": 0.0,
    "actual_fitness": -1.0,
    "quality": -1.0,
    "resistance": -1.0,
    "mean_improvement": -1.0,
    "best_post_improvement": -1.0,
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
    """Cross-play: constructor config vs opponent improvers."""
    points = _validate_config(program_output)
    if points is None:
        return INVALID

    raw_quality = float(get_smallest_triangle_area(points))
    if raw_quality <= 0:
        return INVALID

    quality = min(raw_quality / Q_MAX, 1.0)

    # Cold start: no opponents yet
    if not opponent_results:
        return {
            "fitness": quality,
            "is_valid": 1.0,
            "actual_fitness": raw_quality,
            "quality": quality,
            "resistance": 1.0,
            "mean_improvement": 0.0,
            "best_post_improvement": raw_quality,
            "n_opponents": 0.0,
        }

    deltas = []
    post_qualities = []

    for improve_fn in opponent_results:
        if not callable(improve_fn):
            deltas.append(0.0)
            post_qualities.append(raw_quality)
            continue
        try:
            improved = improve_fn(points.copy())
            improved = _validate_config(improved)
            if improved is None:
                deltas.append(0.0)
                post_qualities.append(raw_quality)
                continue
            post_q = float(get_smallest_triangle_area(improved))
            delta = max(post_q - raw_quality, 0.0)
            deltas.append(delta)
            post_qualities.append(post_q)
        except Exception:
            deltas.append(0.0)
            post_qualities.append(raw_quality)

    mean_delta = sum(deltas) / len(deltas) if deltas else 0.0
    mean_delta_norm = min(mean_delta / Q_MAX, 1.0)
    resistance = 1.0 - mean_delta_norm

    fitness = ALPHA * quality + (1.0 - ALPHA) * resistance

    return {
        "fitness": float(fitness),
        "is_valid": 1.0,
        "actual_fitness": raw_quality,
        "quality": float(quality),
        "resistance": float(resistance),
        "mean_improvement": float(mean_delta),
        "best_post_improvement": float(max(post_qualities))
        if post_qualities
        else raw_quality,
        "n_opponents": float(len(deltas)),
    }
