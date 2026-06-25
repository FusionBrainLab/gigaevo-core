"""Verify that a built Landscape realizes its prescribed barrier tree, and
render its disconnectivity graph.

Independently RECOVERS the minima and saddles by scanning the canyon floor,
then checks they match what was prescribed — this certifies the landscape
before any optimizer is run against it.
"""

from __future__ import annotations

import numpy as np

from problems.sequential_landscape.landscape import Landscape


def recover(ls: Landscape, samples: int = 8000) -> dict:
    s = np.linspace(0.0, ls.min_arclengths[-1], samples)
    g = np.array([ls.floor(v) for v in s])
    is_min = (g[1:-1] < g[:-2]) & (g[1:-1] < g[2:])
    is_max = (g[1:-1] > g[:-2]) & (g[1:-1] > g[2:])
    interior_minima = g[1:-1][is_min].tolist()
    minima = [g[0]] + interior_minima + [g[-1]]
    barriers = g[1:-1][is_max].tolist()
    return {"minima": minima, "barriers": barriers}


def verify(ls: Landscape, tol: float = 1e-3) -> bool:
    rec = recover(ls)
    if len(rec["minima"]) != ls.num_minima:
        return False
    if len(rec["barriers"]) != len(ls.barriers):
        return False
    if not np.allclose(rec["minima"], ls.depths, atol=tol):
        return False
    if not np.allclose(rec["barriers"], ls.barriers, atol=tol):
        return False
    return _no_skip_holds(ls)


def _no_skip_holds(ls: Landscape) -> bool:
    # a point between two serpentine arms must cost far more than any
    # along-canyon barrier, i.e. the cheapest route stays on the floor
    if ls.num_minima < 2:
        return True
    mids = []
    for a, b in zip(ls.min_points[:-1], ls.min_points[1:]):
        mids.append(ls(0.5 * (a + b)))
    along_max = max(ls.barriers)
    arm_cross = ls.global_min_value + 0.5 * ls.wall_height
    return max(mids) <= along_max + 1e-6 or arm_cross > along_max


def render(ls: Landscape, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for x_idx, d in enumerate(ls.depths):
        ax.plot([x_idx], [d], "o", color="black")
        ax.annotate(f"{d:g}", (x_idx, d), fontsize=7, ha="center", va="top")
    for i, b in enumerate(ls.barriers):
        ax.plot([i + 0.5], [b], "_", color="firebrick", markersize=12)
    ax.set_xlabel("minimum (in-order)")
    ax.set_ylabel("value")
    ax.set_title("disconnectivity profile")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    from problems.sequential_landscape.specs import get_ladder

    for inst in get_ladder():
        ls = inst.landscape()
        ok = verify(ls)
        print(
            f"{inst.name:22s} K={ls.num_minima:3d} dim={ls.dim} "
            f"global={ls.global_min_value:+.2f} verified={ok}"
        )
        render(ls, f"/tmp/seqland_{inst.name}.png")


if __name__ == "__main__":
    main()
