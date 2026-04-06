"""Validate optimizers against adversarial landscapes (Pop A).

Fitness: mean proximity to global optimum across opponent landscapes.
Falls back to static benchmark landscapes when opponent archive is empty.
Uses pipeline=standard with opponent config via env vars.
"""

import math
import signal

from problems.adversarial.shared import (
    exec_entrypoint,
    get_opponent_config,
    sample_opponents,
)

# --- Static fallback landscapes ---


def _rastrigin(x):
    n = len(x)
    return 10 * n + sum(xi**2 - 10 * math.cos(2 * math.pi * xi) for xi in x)


def _schwefel(x):
    n = len(x)
    return 418.9829 * n - sum(xi * math.sin(math.sqrt(abs(xi))) for xi in x)


def _ackley(x):
    n = len(x)
    sum_sq = sum(xi**2 for xi in x) / n
    sum_cos = sum(math.cos(2 * math.pi * xi) for xi in x) / n
    return -20 * math.exp(-0.2 * math.sqrt(sum_sq)) - math.exp(sum_cos) + 20 + math.e


STATIC_LANDSCAPES = [
    {
        "fn": _rastrigin,
        "bounds": [(-5.12, 5.12)] * 5,
        "optimum": [0.0] * 5,
        "budget": 500,
    },
    {
        "fn": _schwefel,
        "bounds": [(-500, 500)] * 5,
        "optimum": [420.9687] * 5,
        "budget": 500,
    },
    {"fn": _ackley, "bounds": [(-5, 5)] * 5, "optimum": [0.0] * 5, "budget": 500},
]


def _distance_score(found, optimum, bounds):
    """1.0 = at optimum, 0.0 = at boundary."""
    max_dist = math.sqrt(sum((hi - lo) ** 2 for lo, hi in bounds))
    if max_dist < 1e-12:
        return 1.0
    dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(found, optimum)))
    return max(0.0, 1.0 - dist / max_dist)


def _run_optimizer_on_landscape(optimizer_fn, landscape, timeout=10.0):
    """Run optimizer on landscape with budget enforcement and timeout."""
    fn, bounds, budget = landscape["fn"], landscape["bounds"], landscape["budget"]
    optimum = landscape["optimum"]
    call_count = 0

    def counted_fn(x):
        nonlocal call_count
        call_count += 1
        if call_count > budget:
            raise RuntimeError(f"Budget exceeded: {call_count} > {budget}")
        return fn(x)

    def _handler(signum, frame):
        raise TimeoutError("optimizer timeout")

    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        result = optimizer_fn(counted_fn, bounds, budget)
        if not isinstance(result, (list, tuple)) or len(result) != len(bounds):
            return 0.0
        return _distance_score(result, optimum, bounds)
    except Exception:
        return 0.0
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def _load_opponent_landscapes():
    """Load adversarial landscapes from Pop B's archive."""
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
    landscapes = []
    for _pid, _fit, code in opponents:
        try:
            result = exec_entrypoint(code, timeout=2.0)
            obj_fn, bounds, optimum, budget = result
            if callable(obj_fn):
                landscapes.append(
                    {
                        "fn": obj_fn,
                        "bounds": bounds,
                        "optimum": optimum,
                        "budget": budget,
                    }
                )
        except Exception:
            continue
    return landscapes


def validate(data):
    if not callable(data):
        return {"is_valid": 0, "fitness": 0.0}

    landscapes = _load_opponent_landscapes() or STATIC_LANDSCAPES
    scores = [_run_optimizer_on_landscape(data, ls) for ls in landscapes]
    fitness = sum(scores) / len(scores) if scores else 0.0
    return {"is_valid": 1 if fitness >= 0 else 0, "fitness": fitness}
