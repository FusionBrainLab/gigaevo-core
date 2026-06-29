"""Adversarial insanity gate for the maze ladder.

Certifies each instance on three axes:
  - GAP      : oracle - (best classical) >= GAP_THRESH, i.e. every classical
               optimizer (DE / CMA-ES / basin-hopping / random / gradient) leaves
               most of the root->global path undiscovered. Finding the first basin
               or two is allowed by design; completing the sequence is not.
  - SOLVABLE : the privileged oracle path-follower scores >= SOLVE_THRESH, proving
               the global is reachable by sequential discovery (not impossible/buggy).
  - NO-SKIP  : the wall a straight shortcut between branches must climb dwarfs the
               tallest legitimate barrier.

A landscape any classical method solves is "too easy"; one the oracle can't solve is
"broken". Run as a script for the certification table.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from problems.sequential_landscape.benchmark import (
    _cma_method,
    gradient_descent,
    random_restart,
    scipy_basinhopping,
    scipy_differential_evolution,
)
from problems.sequential_landscape.specs import get_maze_ladder

GAP_THRESH = 0.60
SOLVE_THRESH = 0.95
SEEDS = (0, 1, 2)


def oracle(ls, budget: int) -> np.ndarray:
    """Privileged: descend f locally at each known true-path waypoint in order."""
    waypoints = ls.true_path_points()
    best = waypoints[0]
    for q in waypoints[1:]:
        res = minimize(
            ls,
            q,
            method="Nelder-Mead",
            options={"maxiter": 300, "xatol": 1e-7, "fatol": 1e-10},
        )
        if ls(res.x) < ls(best):
            best = res.x
    return best


def classical_scores(ls, budget: int) -> dict[str, dict]:
    methods = {
        "random": random_restart,
        "gradient": gradient_descent,
        "basinhop": scipy_basinhopping,
        "diff_evo": scipy_differential_evolution,
    }
    cma = _cma_method()
    if cma is not None:
        methods["cma_es"] = cma
    out = {}
    for name, m in methods.items():
        scores = [float(ls.progress(m(ls, ls.bounds, budget, s))) for s in SEEDS]
        out[name] = {"mean": float(np.mean(scores)), "max": float(np.max(scores))}
    return out


def certify(inst) -> dict:
    ls = inst.landscape()
    cls = classical_scores(ls, inst.budget)
    orc = float(ls.progress(oracle(ls, inst.budget)))
    no_skip = ls.worst_shortcut_margin() >= 0.0
    best_mean = max(c["mean"] for c in cls.values())
    best_max = max(c["max"] for c in cls.values())
    gap = orc - best_mean
    return {
        "name": inst.name,
        "dim": ls.dim,
        "path": len(ls.true_path_points()) - 1,
        "classical": cls,
        "oracle": orc,
        "gap": gap,
        "best_mean": best_mean,
        "best_max": best_max,
        "no_skip": bool(no_skip),
        "solvable": orc >= SOLVE_THRESH,
        "insane": gap >= GAP_THRESH and orc >= SOLVE_THRESH and no_skip,
    }


def main():
    ladder = [certify(i) for i in get_maze_ladder()]
    cols = ["random", "gradient", "basinhop", "diff_evo", "cma_es"]
    head = "instance".ljust(14) + "dim ".ljust(5) + "path ".ljust(6)
    head += "".join(c.ljust(10) for c in cols) + "oracle".ljust(8) + "VERDICT"
    print(head + "   (classical = mean over seeds)")
    print("-" * len(head))
    for r in ladder:
        row = r["name"].ljust(14) + str(r["dim"]).ljust(5) + str(r["path"]).ljust(6)
        for c in cols:
            v = r["classical"].get(c)
            row += ("--".ljust(10)) if v is None else f"{v['mean']:.2f}".ljust(10)
        row += f"{r['oracle']:.2f}".ljust(8)
        row += "INSANE ✓" if r["insane"] else "too easy/broken ✗"
        print(row)
    print()
    for r in ladder:
        print(
            f"  {r['name']}: gap={r['gap']:.2f} solvable={r['solvable']} "
            f"no_skip={r['no_skip']} (best classical mean={r['best_mean']:.2f}, "
            f"best single run={r['best_max']:.2f})"
        )


if __name__ == "__main__":
    main()
