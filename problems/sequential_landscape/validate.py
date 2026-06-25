"""Score an evolved Optimizer against the sequential-landscape ladder.

Contract: entrypoint() returns an Optimizer (instance or class) exposing
optimize(f, bounds, budget) -> list[float]. Fitness is partial credit for
sequential progress: per landscape, the normalized depth reached (0 at the
shallowest minimum, 1 at the global), averaged over the ladder.
"""

import signal

import numpy as np

from problems.sequential_landscape.specs import get_ladder

TIMEOUT = 6.0


def _resolve(data):
    obj = data() if isinstance(data, type) else data
    optimize = getattr(obj, "optimize", None)
    if callable(optimize):
        return optimize
    if callable(obj):
        return obj
    return None


def _instance_score(ls, x) -> float:
    shallow = max(ls.depths)
    span = shallow - ls.global_min_value
    if span < 1e-12:
        return 1.0
    val = ls(x)
    return max(0.0, min(1.0, (shallow - val) / span))


def _run_on_instance(optimize, inst, timeout=TIMEOUT) -> float:
    ls = inst.landscape()
    count = 0

    def counted(x):
        nonlocal count
        count += 1
        if count > inst.budget:
            raise RuntimeError("budget exceeded")
        return ls(x)

    def _handler(signum, frame):
        raise TimeoutError("optimizer timeout")

    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        result = optimize(counted, ls.bounds, inst.budget)
        x = np.asarray(result, dtype=float)
        if x.shape != (ls.dim,):
            return 0.0
        return _instance_score(ls, x)
    except Exception:
        return 0.0
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def validate(data):
    optimize = _resolve(data)
    if optimize is None:
        return {"is_valid": 0.0, "fitness": 0.0}
    scores = [_run_on_instance(optimize, inst) for inst in get_ladder()]
    fitness = sum(scores) / len(scores) if scores else 0.0
    return {"is_valid": 1.0, "fitness": fitness}
