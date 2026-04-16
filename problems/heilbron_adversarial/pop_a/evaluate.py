"""Adversarial evaluate.py for Pop A (Constructor) — smoothed resistance scoring.

Receives:
    opponent_results: list of callables improve(points) -> improved_points  (from Pop B)
    program_output:   (11, 2) np.ndarray  (this Constructor's point configuration)

Returns: (metrics_dict, artifact_dict)
    metrics_dict — float-only metrics for MAP-Elites + paper.
        Fitness = ALPHA * quality + (1 - ALPHA) * resistance
            quality    = min(min_area / Q_MAX, 1.0)
            resistance = (mean(tanh(-delta_i / Q_MAX)) + 1) / 2 ∈ (0, 1)
                Smooth tanh replaces the prior binary `float(delta <= 0)`
                indicator (see experiments/heilbron/k5-budget-loose/REDESIGN.md).
        actual_fitness = raw min_area (tracked separately for paper reporting).
    artifact_dict — per-opponent fitness deltas for DGTrackerStage. Aligned
        with opponent_results order, NaN when the opponent slot was skipped
        (uncallable, invalid output, exec failed). The artifact slot is
        suppressed in adversarial pipelines (NullArtifactStage) so it does
        not pollute the LLM prompt — only DGTrackerStage reads it.

For sigmoid resistance scoring use pop_a_soft (IV2 soft-fitness variant).
"""

from __future__ import annotations

import math

from helper import get_smallest_triangle_area, get_unit_triangle, is_inside_triangle
import numpy as np

Q_MAX = 0.0365
ALPHA = 0.5

INVALID_METRICS = {
    "fitness": -1000.0,
    "is_valid": 0.0,
    "actual_fitness": -1000.0,
    "quality": -1000.0,
    "resistance": -1000.0,
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


def evaluate(opponent_results: list, program_output: object) -> tuple[dict, dict]:
    """Cross-play: constructor config vs opponent improvers.

    Returns a (metrics, artifact) tuple. The artifact carries per-opponent
    fitness deltas aligned with opponent_results order, consumed by
    DGTrackerStage to populate the (D, G, fitness_delta) tracker.
    """
    n_in = len(opponent_results) if opponent_results else 0
    points = _validate_config(program_output)
    if points is None:
        return INVALID_METRICS, _invalid_artifact(n_in)

    raw_quality = float(get_smallest_triangle_area(points))
    if raw_quality <= 0:
        return INVALID_METRICS, _invalid_artifact(n_in)

    quality = min(raw_quality / Q_MAX, 1.0)

    if not opponent_results:
        # Cold start: no opponents in archive yet. Resistance is at the upper
        # smoothed bound (1.0 = no opponent succeeded). Fitness uses the same
        # ALPHA mix as the main branch to stay scale-consistent.
        metrics = {
            "fitness": ALPHA * quality + (1.0 - ALPHA) * 1.0,
            "is_valid": 1.0,
            "actual_fitness": raw_quality,
            "quality": quality,
            "resistance": 1.0,
            "mean_improvement": 0.0,
            "best_post_improvement": raw_quality,
            "n_opponents": 0.0,
        }
        return metrics, _invalid_artifact(0)

    n = len(opponent_results)
    # Aligned per-opponent arrays: index i ↔ opponent_results[i] ↔ opponent_ids[i].
    # NaN sentinel = "no measurement at this index" (uncallable / invalid output / exec fail).
    # raw_quality is intrinsic to the G points, shared across all opponents.
    per_opp_pre: list[float] = [float("nan")] * n
    per_opp_post: list[float] = [float("nan")] * n
    per_opp_delta: list[float] = [float("nan")] * n

    resistance_scores = []
    deltas = []
    post_qualities = []

    for i, improve_fn in enumerate(opponent_results):
        if not callable(improve_fn):
            # Opponent unrunnable — full resistance (1.0 = upper smoothed bound).
            # Tracker delta = NaN (no measurement) — DGTracker filters non-finite.
            resistance_scores.append(1.0)
            deltas.append(0.0)
            post_qualities.append(raw_quality)
            continue
        try:
            improved = improve_fn(points.copy())
            improved = _validate_config(improved)
            if improved is None:
                # Opponent produced invalid output — full resistance, no measurement.
                resistance_scores.append(1.0)
                deltas.append(0.0)
                post_qualities.append(raw_quality)
                continue
            post_q = float(get_smallest_triangle_area(improved))
            # Keep both signs of delta so smoothing can distinguish "D made it
            # worse" from "D succeeded slightly". The prior max(., 0.0) clip
            # silently merged those into a single binary 0/1 indicator.
            delta = post_q - raw_quality
            # Smoothed ∈ (-1, 1); negative when D succeeded, positive when D failed.
            resistance_scores.append(math.tanh(-delta / Q_MAX))
            deltas.append(delta)
            post_qualities.append(post_q)
            per_opp_pre[i] = raw_quality
            per_opp_post[i] = post_q
            per_opp_delta[i] = delta
        except Exception:
            # Exec failure — same upper-bound signal as unrunnable, no measurement.
            resistance_scores.append(1.0)
            deltas.append(0.0)
            post_qualities.append(raw_quality)

    artifact = {
        "role": "constructor",
        "n_opponents": n,
        "per_opp_pre": per_opp_pre,
        "per_opp_post": per_opp_post,
        "per_opp_delta": per_opp_delta,
    }

    mean_delta = sum(deltas) / len(deltas) if deltas else 0.0
    # resistance_raw mixes smoothed scores ∈ (-1, 1) with failure scores =1.0
    # (already at the upper smoothed bound). Mean stays in [-1, 1]; the rescale
    # then maps to [0, 1] for MAP-Elites.
    resistance_raw = (
        sum(resistance_scores) / len(resistance_scores) if resistance_scores else 1.0
    )
    resistance = (resistance_raw + 1.0) / 2.0
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
    return metrics, artifact
