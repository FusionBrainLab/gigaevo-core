"""Validate landscapes for deceptiveness against opponent optimizers (Pop B).

Fitness: mean distance of opponent optimizers from the global optimum.
Falls back to static benchmark optimizers when opponent archive is empty.
Uses pipeline=standard with opponent config via env vars.
"""

import math
import random as _random
import signal

from problems.adversarial.shared import (
    exec_entrypoint,
    get_opponent_config,
    sample_opponents,
)

# --- Static fallback optimizers ---


def _random_search(f, bounds, budget):
    best_x, best_val = None, float("inf")
    for _ in range(budget):
        x = [_random.uniform(lo, hi) for lo, hi in bounds]
        val = f(x)
        if val < best_val:
            best_val, best_x = val, x[:]
    return best_x


def _coordinate_descent(f, bounds, budget):
    dim = len(bounds)
    best_x = [(lo + hi) / 2 for lo, hi in bounds]
    best_val = f(best_x)
    evals, step = 1, 0.5
    while evals < budget:
        for d in range(dim):
            if evals >= budget:
                break
            for direction in [1, -1]:
                if evals >= budget:
                    break
                cand = best_x[:]
                lo, hi = bounds[d]
                cand[d] = max(lo, min(hi, cand[d] + direction * step * (hi - lo)))
                val = f(cand)
                evals += 1
                if val < best_val:
                    best_val, best_x = val, cand[:]
        step *= 0.5
        if step < 1e-10:
            break
    return best_x


STATIC_OPTIMIZERS = [_random_search, _coordinate_descent]


def _distance_score(found, optimum, bounds):
    """0.0 = at optimum, 1.0 = at boundary (deceptiveness)."""
    max_dist = math.sqrt(sum((hi - lo) ** 2 for lo, hi in bounds))
    if max_dist < 1e-12:
        return 0.0
    dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(found, optimum)))
    return min(1.0, dist / max_dist)


def _validate_landscape(data):
    """Validate landscape structure: (objective, bounds, optimum, budget)."""
    if not isinstance(data, (tuple, list)) or len(data) != 4:
        raise ValueError("Must return (objective, bounds, optimum, budget)")
    obj_fn, bounds, optimum, budget = data
    if not callable(obj_fn):
        raise ValueError("objective must be callable")
    if len(bounds) < 2 or len(optimum) != len(bounds):
        raise ValueError("bounds/optimum dimension mismatch")
    if not isinstance(budget, int) or budget < 10:
        raise ValueError("budget must be int >= 10")
    # Verify determinism
    val1 = obj_fn(list(optimum))
    val2 = obj_fn(list(optimum))
    if abs(val1 - val2) > 1e-10:
        raise ValueError("objective is not deterministic")
    return obj_fn, bounds, optimum, budget


def _eval_landscape_vs_optimizer(obj_fn, bounds, optimum, budget, opt_fn, timeout=10.0):
    """Run one optimizer, return deceptiveness score."""
    call_count = 0

    def counted_fn(x):
        nonlocal call_count
        call_count += 1
        if call_count > budget:
            raise RuntimeError("Budget exceeded")
        return obj_fn(x)

    def _handler(signum, frame):
        raise TimeoutError("optimizer timeout")

    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        result = opt_fn(counted_fn, bounds, budget)
        if not isinstance(result, (list, tuple)) or len(result) != len(bounds):
            return 1.0  # optimizer failed → landscape "wins"
        return _distance_score(result, optimum, bounds)
    except Exception:
        return 1.0
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def _load_opponent_optimizers():
    """Load optimizers from Pop A's archive."""
    cfg = get_opponent_config()
    if not cfg["prefix"]:
        return []

    opponents = sample_opponents(
        host=cfg["host"],
        port=cfg["port"],
        db=cfg["db"],
        prefix=cfg["prefix"],
        n=5,
    )
    optimizers = []
    for _pid, _fit, code in opponents:
        try:
            result = exec_entrypoint(code, timeout=2.0)
            if callable(result):
                optimizers.append(result)
        except Exception:
            continue
    return optimizers


def validate(data):
    try:
        obj_fn, bounds, optimum, budget = _validate_landscape(data)
    except (ValueError, TypeError):
        return {"is_valid": 0, "fitness": 0.0}

    optimizers = _load_opponent_optimizers() or STATIC_OPTIMIZERS
    scores = [
        _eval_landscape_vs_optimizer(obj_fn, bounds, optimum, budget, opt)
        for opt in optimizers
    ]
    fitness = sum(scores) / len(scores) if scores else 0.0
    return {"is_valid": 1, "fitness": fitness}
