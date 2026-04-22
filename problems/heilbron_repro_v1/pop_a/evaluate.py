"""Adversarial evaluate.py for Pop A (Constructor) — binary resistance scoring.

Receives:
    opponent_results: list of callables improve(points) -> improved_points  (from Pop B)
    program_output:   (11, 2) np.ndarray  (this Constructor's point configuration)

Fitness = ALPHA * quality + (1 - ALPHA) * resistance
    quality    = min(min_area / Q_MAX, 1.0)
    resistance = mean(float(delta_i <= 0))  — 1 if opponent failed to improve, 0 if succeeded
    actual_fitness = raw min_area (tracked separately for paper reporting)

For sigmoid resistance scoring use pop_a_soft (IV2 soft-fitness variant).
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


def evaluate(opponent_results: list, program_output: object):
    """Cross-play: constructor config vs opponent improvers.

    Returns (metrics, artifact). The artifact carries per_opp_metrics aligned
    index-wise with opponent_results so SBF-LineageStage can replay the
    ConfigurableAggregator downstream. per_opp_delta is kept as a redundant
    back-compat alias.
    """
    points = _validate_config(program_output)
    if points is None:
        return INVALID, {
            "role": "constructor",
            "per_opp_metrics": [],
            "per_opp_delta": [],
        }

    raw_quality = float(get_smallest_triangle_area(points))
    if raw_quality <= 0:
        return INVALID, {
            "role": "constructor",
            "per_opp_metrics": [],
            "per_opp_delta": [],
        }

    quality = min(raw_quality / Q_MAX, 1.0)

    if not opponent_results:
        return (
            {
                "fitness": quality,
                "is_valid": 1.0,
                "actual_fitness": raw_quality,
                "quality": quality,
                "resistance": 1.0,
                "mean_improvement": 0.0,
                "best_post_improvement": raw_quality,
                "n_opponents": 0.0,
            },
            {"role": "constructor", "per_opp_metrics": [], "per_opp_delta": []},
        )

    resistance_scores = []
    deltas = []
    post_qualities = []
    per_opp_metrics: list[dict[str, float]] = []

    for improve_fn in opponent_results:
        if not callable(improve_fn):
            resistance_scores.append(1.0)
            deltas.append(0.0)
            post_qualities.append(raw_quality)
            per_opp_metrics.append(
                {
                    "post_q": float(raw_quality),
                    "delta": 0.0,
                    "resistance_score": 1.0,
                    "is_valid": 1.0,
                }
            )
            continue
        try:
            improved = improve_fn(points.copy())
            improved = _validate_config(improved)
            if improved is None:
                post_q = raw_quality
                delta = 0.0
            else:
                post_q = float(get_smallest_triangle_area(improved))
                delta = max(post_q - raw_quality, 0.0)
            resistance_score = float(delta <= 0)
            resistance_scores.append(resistance_score)
            deltas.append(delta)
            post_qualities.append(post_q)
            per_opp_metrics.append(
                {
                    "post_q": float(post_q),
                    "delta": float(delta),
                    "resistance_score": float(resistance_score),
                    "is_valid": 1.0,
                }
            )
        except Exception:
            resistance_scores.append(1.0)
            deltas.append(0.0)
            post_qualities.append(raw_quality)
            per_opp_metrics.append(
                {
                    "post_q": float(raw_quality),
                    "delta": 0.0,
                    "resistance_score": 1.0,
                    "is_valid": 1.0,
                }
            )

    mean_delta = sum(deltas) / len(deltas) if deltas else 0.0
    resistance = (
        sum(resistance_scores) / len(resistance_scores) if resistance_scores else 1.0
    )
    fitness = ALPHA * quality + (1.0 - ALPHA) * resistance

    metrics = {
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
    artifact = {
        "role": "constructor",
        "per_opp_metrics": per_opp_metrics,
        "per_opp_delta": [m["delta"] for m in per_opp_metrics],
    }
    return metrics, artifact
