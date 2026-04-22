"""Adversarial evaluate.py for Pop B (Improver) — binary improvement scoring.

Receives:
    opponent_results: list of (11, 2) np.ndarray  (point configs from Pop A)
    program_output:   callable improve(points) -> improved_points

Emits per-opponent primitives (pre_q, post_q, delta, score, is_valid) in artifact.
The ConfigurableAggregator composes these into program-level metrics via heilbron_improver.yaml.

For sigmoid improvement scoring use pop_b_soft (IV2 soft-fitness variant).
"""

from __future__ import annotations

from helper import get_smallest_triangle_area, get_unit_triangle, is_inside_triangle
import numpy as np

Q_MAX = 0.0365


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


def _invalid_opp_metrics() -> dict[str, float]:
    """Per-opponent primitives for a skipped/invalid opponent slot.

    Shape matches the pop_b schema emitted on the happy path; is_valid=0.0
    gates this record out of the ConfigurableAggregator's reduction.
    """
    return {
        "pre_q": 0.0,
        "post_q": 0.0,
        "delta": 0.0,
        "score": 0.0,
        "is_valid": 0.0,
    }


def evaluate(opponent_results: list, program_output: object):
    """Cross-play: improver vs opponent constructor configs.

    Returns ({}, artifact). Intrinsic metrics are empty (all D metrics are
    per-opponent reductions handled by ConfigurableAggregator).
    The artifact carries per_opp_metrics aligned index-wise with opponent_results
    so the aggregator can reproduce program-level metrics downstream.
    """
    improve_fn = program_output
    n = len(opponent_results)
    per_opp_metrics: list[dict[str, float]] = [_invalid_opp_metrics() for _ in range(n)]

    def _artifact() -> dict:
        return {
            "role": "improver",
            "per_opp_metrics": per_opp_metrics,
            "per_opp_delta": [m["delta"] for m in per_opp_metrics],
        }

    if not callable(improve_fn):
        return {}, _artifact()

    if not opponent_results:
        return {}, _artifact()

    scores = []
    pre_qualities = []
    post_qualities = []

    for idx, config in enumerate(opponent_results):
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
                post_q = pre_q
                delta = 0.0
                score = 0.0
            else:
                post_q = float(get_smallest_triangle_area(improved))
                raw_delta = post_q - pre_q
                delta = max(raw_delta, 0.0)
                score = min(delta / Q_MAX, 1.0)
            scores.append(score)
            pre_qualities.append(pre_q)
            post_qualities.append(post_q)
            per_opp_metrics[idx] = {
                "pre_q": float(pre_q),
                "post_q": float(post_q),
                "delta": float(delta),
                "score": float(score),
                "is_valid": 1.0,
            }
        except Exception:
            scores.append(0.0)
            pre_qualities.append(pre_q)
            post_qualities.append(pre_q)
            per_opp_metrics[idx] = {
                "pre_q": float(pre_q),
                "post_q": float(pre_q),
                "delta": 0.0,
                "score": 0.0,
                "is_valid": 1.0,
            }

    if not scores:
        return {}, _artifact()

    return {}, _artifact()
