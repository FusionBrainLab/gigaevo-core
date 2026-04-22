"""Adversarial evaluate.py for Pop A (Constructor) — binary resistance scoring.

Receives:
    opponent_results: list of callables improve(points) -> improved_points  (from Pop B)
    program_output:   (11, 2) np.ndarray  (this Constructor's point configuration)

Intrinsic metrics (quality, actual_fitness) are candidate-level, computed from the
program output alone. Per-opponent resistance is computed and passed to the aggregator.
The ConfigurableAggregator reduces these via heilbron_constructor.yaml.

For sigmoid resistance scoring use pop_a_soft (IV2 soft-fitness variant).
"""

from __future__ import annotations

from helper import get_smallest_triangle_area, get_unit_triangle, is_inside_triangle
import numpy as np

Q_MAX = 0.0365
ALPHA = 0.5


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

    Returns (intrinsic, artifact). Intrinsic contains candidate-level metrics
    (quality, actual_fitness) computed from the program output. Per-opponent
    metrics are in the artifact for the aggregator to reduce downstream.
    """
    points = _validate_config(program_output)
    if points is None:
        return {}, {
            "role": "constructor",
            "per_opp_metrics": [],
            "per_opp_delta": [],
        }

    raw_quality = float(get_smallest_triangle_area(points))
    if raw_quality <= 0:
        return {}, {
            "role": "constructor",
            "per_opp_metrics": [],
            "per_opp_delta": [],
        }

    quality = min(raw_quality / Q_MAX, 1.0)

    if not opponent_results:
        intrinsic = {
            "quality": quality,
            "actual_fitness": raw_quality,
        }
        return intrinsic, {"role": "constructor", "per_opp_metrics": [], "per_opp_delta": []}

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

    intrinsic = {
        "quality": float(quality),
        "actual_fitness": raw_quality,
    }
    artifact = {
        "role": "constructor",
        "per_opp_metrics": per_opp_metrics,
        "per_opp_delta": [m["delta"] for m in per_opp_metrics],
    }
    return intrinsic, artifact
