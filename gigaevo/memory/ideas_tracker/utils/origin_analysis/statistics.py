"""Pure mathematical helpers for origin analysis. No I/O, no pandas."""
from __future__ import annotations

import bisect
import math


def robust_median(xs: list[float]) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    n = len(ys)
    m = n // 2
    return ys[m] if n % 2 == 1 else 0.5 * (ys[m - 1] + ys[m])


def robust_quantile(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    q = max(0.0, min(1.0, float(q)))
    idx = int(round(q * (len(ys) - 1)))
    idx = max(0, min(len(ys) - 1, idx))
    return ys[idx]


def mad(xs: list[float]) -> float:
    if not xs:
        return float("nan")
    med = robust_median(xs)
    devs = [abs(x - med) for x in xs]
    return robust_median(devs)


def percentile_rank(sorted_vals: list[float], x: float) -> float:
    if not sorted_vals:
        return float("nan")
    k = bisect.bisect_right(sorted_vals, x)
    return k / len(sorted_vals)


def elite_threshold_by_top_k(
    fitness_vals: list[float], elite_pct: float
) -> tuple[float, int]:
    xs = [float(x) for x in fitness_vals if math.isfinite(float(x))]
    if not xs:
        return float("nan"), 0
    xs.sort()
    n = len(xs)
    elite_pct = float(elite_pct)
    elite_pct = max(0.000001, min(1.0, elite_pct))
    elite_k = int(math.ceil(elite_pct * n))
    elite_k = max(1, min(n, elite_k))
    threshold = xs[-elite_k]
    return threshold, elite_k


def nanmedian(vals: list[float]) -> float:
    xs = [float(x) for x in vals if math.isfinite(float(x))]
    return robust_median(xs) if xs else float("nan")


def nanquantile(vals: list[float], q: float) -> float:
    xs = [float(x) for x in vals if math.isfinite(float(x))]
    return robust_quantile(xs, q) if xs else float("nan")


def nanrate_bool(vals: list[float]) -> float:
    xs = [float(x) for x in vals if math.isfinite(float(x))]
    if not xs:
        return float("nan")
    return sum(1 for x in xs if x > 0.5) / len(xs)


def nancount(vals: list[float]) -> int:
    return sum(1 for x in vals if math.isfinite(float(x)))
