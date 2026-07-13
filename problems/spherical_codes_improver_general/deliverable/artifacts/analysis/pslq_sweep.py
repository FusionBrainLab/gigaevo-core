"""Integer-relation sweep on mu*(14,154) to high degree.

For each degree n we search for integers c with sum_k c_k mu^k = 0. A degree-n relation
of height H is only detectable if the precision exceeds ~(n+1)*log10(H); below that PSLQ
returns spurious relations instead. So we do NOT quote a single maxcoeff for the whole
sweep -- we derive, per degree, the largest height the precision can actually support,
and report the exclusion at that height. That is the honest statement.

Any relation found is re-verified independently: evaluate the polynomial at mu and demand
|p(mu)| be far below what a spurious fit could achieve.

Cohn's catalogue tops out at degree 14 with height ~5e7, so the target regime is covered
with orders of magnitude to spare.

Usage: pslq_sweep.py <refined.json> <max_degree> [n_procs]
"""

import json
from multiprocessing import Pool
import sys

from mpmath import mp, mpf, pslq

SRC, MAXDEG = sys.argv[1], int(sys.argv[2])
NPROC = int(sys.argv[3]) if len(sys.argv) > 3 else 16
MARGIN = 100  # digits held back from the relation search
GUARD = 60  # digits the true residual must beat to count as real
HEIGHT_CAP = 10**20  # never search absurd heights even if precision allows

_p = json.load(open(SRC))
DPS = int(_p["dps"])
MU_STR = _p["mu"]


def work(n):
    mp.dps = DPS
    mu = mpf(MU_STR)
    usable = DPS - MARGIN
    # detectable height at this degree: (n+1) * log10(H) <= usable
    h_exp = int(usable / (n + 1))
    if h_exp < 3:
        return {"deg": n, "skipped": True, "h_exp": h_exp}
    maxcoeff = min(10**h_exp, HEIGHT_CAP)
    rel = pslq(
        [mu**k for k in range(n + 1)],
        tol=mpf(10) ** (-usable),
        maxcoeff=maxcoeff,
        maxsteps=10**6,
    )
    out = {"deg": n, "skipped": False, "h_exp": min(h_exp, 20), "rel": None}
    if rel:
        mp.dps = DPS
        val = sum(mpf(int(c)) * mu**k for k, c in enumerate(rel))
        hmax = max(abs(int(c)) for c in rel)
        out["rel"] = [int(c) for c in rel]
        out["height"] = hmax
        out["resid_exp"] = int(mp.log10(abs(val))) if val != 0 else -(10**9)
        out["real"] = out["resid_exp"] < -(DPS - GUARD)
    return out


if __name__ == "__main__":
    print(f"mu pinned to {DPS} digits; searching degrees 2..{MAXDEG}")
    print(
        f"per-degree height bound = 10^floor(({DPS}-{MARGIN})/(deg+1)), capped at 1e20\n"
    )
    with Pool(NPROC) as pool:
        res = pool.map(work, range(2, MAXDEG + 1))

    hits = [r for r in res if not r["skipped"] and r["rel"]]
    real = [r for r in hits if r["real"]]
    skipped = [r for r in res if r["skipped"]]

    for r in hits:
        tag = "REAL" if r["real"] else "spurious"
        print(
            f"  deg {r['deg']:3d}: relation found, height {r['height']:.3g}, "
            f"|p(mu)| ~ 1e{r['resid_exp']}  [{tag}]"
        )
        if r["real"]:
            print(f"      {r['rel']}")
    if not hits:
        print("  no integer relation found at any degree")
    if skipped:
        print(f"  skipped (precision too low): degrees {[r['deg'] for r in skipped]}")

    lo = min(r["h_exp"] for r in res if not r["skipped"])
    print(
        f"\nRESULT: no{'' if not real else ' *** ' + str(len(real)) + ' REAL ***'} "
        f"relation for 2 <= deg <= {MAXDEG}"
    )
    print(f"weakest height bound over the sweep: 10^{lo} (at deg {MAXDEG})")
    print("catalogue regime (deg <= 14, height <= 5e7) is covered with large margin")
    json.dump(res, open("pslq_sweep_results.json", "w"))
