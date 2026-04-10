"""Adversarial evaluate.py for Pop A (Constructor) — strict GAN variant.

G's fitness = pure resistance from D. No quality component.
Quality is tracked as actual_fitness for paper reporting but is invisible
to MAP-Elites selection and invisible to the LLM prompt.

Receives:
    opponent_results: list of callables improve(points) -> improved_points  (from Pop B)
    program_output:   (11, 2) np.ndarray  (this Constructor's point configuration)

Selection fitness = sigmoid resistance = mean(sigmoid(-delta_i / T)), T=Q_MAX/9
    near 1 when D fails to improve, near 0 when D strongly improves.

    delta_i = post_quality - raw_quality  (signed, no clamp)
    Positive delta → D improved config → resistance < 0.5
    Negative delta → D worsened config → resistance > 0.5
    Zero delta → no change → resistance = 0.5

GAN analogy: G's only gradient comes from D. Quality (actual min_area) emerges
indirectly — configs that are near-optimal are hardest for D to improve.
"""

from __future__ import annotations

import math

from helper import get_smallest_triangle_area, get_unit_triangle, is_inside_triangle
import numpy as np

Q_MAX = 0.0365
_T = Q_MAX / 9  # sigmoid temperature ≈ 0.004

INVALID = {
    "fitness": -1.0,
    "is_valid": 0.0,
    "actual_fitness": -1.0,
    "resistance": -1.0,
    "mean_improvement": -1.0,
    "n_opponents": 0.0,
}


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


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
    """Cross-play: constructor config vs opponent improvers. G fitness = pure resistance."""
    points = _validate_config(program_output)
    if points is None:
        return INVALID

    raw_quality = float(get_smallest_triangle_area(points))
    if raw_quality <= 0:
        return INVALID

    # Cold start: no opponents — resistance is 1.0 (trivially unbeaten).
    # actual_fitness still tracked so paper reporting is not blind at gen 0.
    if not opponent_results:
        return {
            "fitness": 1.0,
            "is_valid": 1.0,
            "actual_fitness": raw_quality,
            "resistance": 1.0,
            "mean_improvement": 0.0,
            "n_opponents": 0.0,
        }

    resistance_scores = []
    deltas = []

    for improve_fn in opponent_results:
        if not callable(improve_fn):
            resistance_scores.append(_sigmoid(0.0))
            deltas.append(0.0)
            continue
        try:
            improved = improve_fn(points.copy())
            improved = _validate_config(improved)
            if improved is None:
                resistance_scores.append(_sigmoid(0.0))
                deltas.append(0.0)
                continue
            post_q = float(get_smallest_triangle_area(improved))
            delta = post_q - raw_quality  # signed: negative when D worsens config
            resistance_scores.append(_sigmoid(-delta / _T))
            deltas.append(delta)
        except Exception:
            resistance_scores.append(_sigmoid(0.0))
            deltas.append(0.0)

    resistance = (
        sum(resistance_scores) / len(resistance_scores) if resistance_scores else 1.0
    )
    mean_delta = sum(deltas) / len(deltas) if deltas else 0.0

    return {
        "fitness": float(resistance),  # MAP-Elites selects on this = pure D signal
        "is_valid": 1.0,
        "actual_fitness": raw_quality,  # paper reporting only — not seen by LLM
        "resistance": float(resistance),
        "mean_improvement": float(mean_delta),
        "n_opponents": float(len(deltas)),
    }
