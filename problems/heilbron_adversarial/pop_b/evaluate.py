"""Adversarial evaluate.py for Pop B (Improver) — smoothed improvement scoring.

Receives:
    opponent_results: list of (11, 2) np.ndarray  (point configs from Pop A)
    program_output:   callable improve(points) -> improved_points

Returns: (metrics_dict, artifact_dict)
    metrics_dict — float-only metrics for MAP-Elites + paper.
        Fitness = (mean(tanh(delta / Q_MAX)) + 1) / 2 ∈ (0, 1)
        Smooth tanh replaces the prior hard-floor max(delta, 0)/Q_MAX clip
        (see experiments/heilbron/k5-budget-loose/REDESIGN.md).
        actual_fitness = best post-improvement min_area achieved.
    artifact_dict — per-opponent fitness deltas for DGTrackerStage. Aligned
        with opponent_results order, NaN when the opponent slot was skipped
        (invalid input, pre==0, exec failed). The artifact slot is suppressed
        in adversarial pipelines (NullArtifactStage) so it does not pollute
        the LLM prompt — only DGTrackerStage reads it.

For sigmoid improvement scoring use pop_b_soft (IV2 soft-fitness variant).
"""

from __future__ import annotations

import math

from helper import get_smallest_triangle_area, get_unit_triangle, is_inside_triangle
import numpy as np

Q_MAX = 0.0365

INVALID_METRICS = {
    "fitness": -1000.0,
    "is_valid": 0.0,
    "actual_fitness": -1000.0,
    "mean_improvement_raw": -1000.0,
    "mean_pre_quality": -1000.0,
    "mean_post_quality": -1000.0,
    "max_post_quality": -1000.0,
    "n_opponents": 0.0,
}


def _invalid_opp_metrics() -> dict[str, float]:
    """Per-opponent primitives for a skipped/invalid opponent slot."""
    nan = float("nan")
    return {
        "pre_q": nan,
        "post_q": nan,
        "delta": nan,
        "score": nan,
        "is_valid": 0.0,
    }


def _invalid_artifact(n: int) -> dict:
    return {
        "role": "improver",
        "n_opponents": n,
        "per_opp_pre": [float("nan")] * n,
        "per_opp_post": [float("nan")] * n,
        "per_opp_delta": [float("nan")] * n,
        "per_opp_metrics": [_invalid_opp_metrics() for _ in range(n)],
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
    """Cross-play: improver vs opponent constructor configs.

    Returns a (metrics, artifact) tuple. The artifact carries per-opponent
    fitness deltas aligned with opponent_results order, consumed by
    DGTrackerStage to populate the (D, G, fitness_delta) tracker.
    """
    improve_fn = program_output
    n_in = len(opponent_results) if opponent_results else 0
    if not callable(improve_fn):
        return INVALID_METRICS, _invalid_artifact(n_in)

    if not opponent_results:
        return INVALID_METRICS, _invalid_artifact(0)

    n = len(opponent_results)
    # Aligned per-opponent arrays: index i ↔ opponent_results[i] ↔ opponent_ids[i].
    # NaN sentinel = "no measurement at this index" (skipped invalid input or pre==0).
    per_opp_pre: list[float] = [float("nan")] * n
    per_opp_post: list[float] = [float("nan")] * n
    per_opp_delta: list[float] = [float("nan")] * n
    per_opp_metrics: list[dict[str, float]] = [_invalid_opp_metrics() for _ in range(n)]

    scores = []
    pre_qualities = []
    post_qualities = []

    def _record_valid(i, pre_q, post_q, delta, smoothed_score):
        per_opp_pre[i] = pre_q
        per_opp_post[i] = post_q
        per_opp_delta[i] = delta
        per_opp_metrics[i] = {
            "pre_q": float(pre_q),
            "post_q": float(post_q),
            "delta": float(delta),
            "score": float(smoothed_score),
            "is_valid": 1.0,
        }

    for i, config in enumerate(opponent_results):
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
                # D-did-nothing branch: tanh(0)=0 → rescales to neutral 0.5.
                # Tracker delta = 0 (no improvement) — DGTracker filters delta<=0.
                scores.append(0.0)
                pre_qualities.append(pre_q)
                post_qualities.append(pre_q)
                _record_valid(i, pre_q, pre_q, 0.0, 0.0)
                continue
            post_q = float(get_smallest_triangle_area(improved))
            delta = post_q - pre_q
            # Smoothed score ∈ (-1, 1); preserves both signs of delta so a D
            # that makes things worse is distinguishable from a D that did
            # nothing, and MAP-Elites does not collapse to a 0.5 point mass.
            smoothed = math.tanh(delta / Q_MAX)
            scores.append(smoothed)
            pre_qualities.append(pre_q)
            post_qualities.append(post_q)
            _record_valid(i, pre_q, post_q, delta, smoothed)
        except Exception:
            # Exec failure: same neutral signal as D-did-nothing.
            scores.append(0.0)
            pre_qualities.append(pre_q)
            post_qualities.append(pre_q)
            _record_valid(i, pre_q, pre_q, 0.0, 0.0)

    artifact = {
        "role": "improver",
        "n_opponents": n,
        "per_opp_pre": per_opp_pre,
        "per_opp_post": per_opp_post,
        "per_opp_delta": per_opp_delta,
        "per_opp_metrics": per_opp_metrics,
    }

    if not scores:
        return INVALID_METRICS, artifact

    mean_score = sum(scores) / len(scores)  # ∈ (-1, 1)
    fitness = (mean_score + 1.0) / 2.0  # rescale to (0, 1) for MAP-Elites
    mean_improvement_raw = sum(
        max(post - pre, 0.0) for pre, post in zip(pre_qualities, post_qualities)
    ) / len(scores)

    metrics = {
        "fitness": float(fitness),
        "is_valid": 1.0,
        "actual_fitness": float(max(post_qualities)),
        "mean_improvement_raw": float(mean_improvement_raw),
        "mean_pre_quality": float(sum(pre_qualities) / len(pre_qualities)),
        "mean_post_quality": float(sum(post_qualities) / len(post_qualities)),
        "max_post_quality": float(max(post_qualities)),
        "n_opponents": float(len(scores)),
    }
    return metrics, artifact
