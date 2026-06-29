"""Benchmark classical optimizers down the difficulty ladder.

Prints a per-instance score table (0 = stuck at the shallowest basin, 1 =
reached the global). The breakdown point — where scores collapse as difficulty
rises — is the artifact of the study; compare against the gigaevo-evolved
optimizer's per-instance scores.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import basinhopping, differential_evolution

from problems.sequential_landscape.specs import get_ladder
from problems.sequential_landscape.validate import _instance_score


class _Budget(Exception):
    pass


class _Tracked:
    def __init__(self, f, budget):
        self.f, self.budget, self.n = f, budget, 0
        self.best_x, self.best_v = None, np.inf

    def __call__(self, x):
        if self.n >= self.budget:
            raise _Budget()
        self.n += 1
        v = float(self.f(x))
        if v < self.best_v:
            self.best_v, self.best_x = v, np.asarray(x, float).copy()
        return v


def _wrap(method):
    def run(f, bounds, budget, seed=0):
        t = _Tracked(f, budget)
        try:
            method(t, bounds, budget, seed)
        except _Budget:
            pass
        except Exception:
            pass
        return t.best_x if t.best_x is not None else np.zeros(len(bounds))

    return run


@_wrap
def random_restart(f, bounds, budget, seed=0):
    rng = np.random.default_rng(seed)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    while True:
        f(lo + (hi - lo) * rng.random(len(bounds)))


@_wrap
def gradient_descent(f, bounds, budget, seed=0):
    rng = np.random.default_rng(seed)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    eps, lr = 1e-4, 0.05
    while True:
        x = lo + (hi - lo) * rng.random(len(bounds))
        for _ in range(200):
            base = f(x)
            grad = np.zeros_like(x)
            for i in range(len(x)):
                xp = x.copy()
                xp[i] += eps
                grad[i] = (f(xp) - base) / eps
            x = np.clip(x - lr * grad, lo, hi)


@_wrap
def scipy_basinhopping(f, bounds, budget, seed=0):
    rng = np.random.default_rng(seed)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    x0 = lo + (hi - lo) * rng.random(len(bounds))
    basinhopping(
        f, x0, niter=budget, seed=seed, minimizer_kwargs={"method": "Nelder-Mead"}
    )


@_wrap
def scipy_differential_evolution(f, bounds, budget, seed=0):
    differential_evolution(
        f, bounds, maxiter=budget, tol=1e-9, polish=False, seed=seed, init="sobol"
    )


def _cma_method():
    try:
        import cma
    except ImportError:
        return None

    @_wrap
    def cma_es(f, bounds, budget, seed=0):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        x0 = 0.5 * (lo + hi)
        sigma = 0.25 * float(np.mean(hi - lo))
        es = cma.CMAEvolutionStrategy(
            list(x0),
            sigma,
            {"bounds": [list(lo), list(hi)], "verbose": -9, "seed": seed + 1},
        )
        while not es.stop():
            sols = es.ask()
            es.tell(sols, [f(x) for x in sols])

    return cma_es


def main():
    methods = {
        "random_restart": random_restart,
        "gradient_descent": gradient_descent,
        "basinhopping": scipy_basinhopping,
        "diff_evolution": scipy_differential_evolution,
    }
    cma_es = _cma_method()
    if cma_es is not None:
        methods["cma_es"] = cma_es

    ladder = get_ladder()
    header = "method".ljust(18) + "".join(i.name[:14].ljust(16) for i in ladder)
    print(header)
    print("-" * len(header))
    for mname, method in methods.items():
        row = mname.ljust(18)
        for inst in ladder:
            ls = inst.landscape()
            x = method(ls, ls.bounds, inst.budget)
            row += f"{_instance_score(ls, x):.2f}".ljust(16)
        print(row)


if __name__ == "__main__":
    main()
