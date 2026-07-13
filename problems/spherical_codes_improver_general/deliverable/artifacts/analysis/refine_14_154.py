"""Push mu*(14,154) to N digits, then PSLQ to high degree.

The optimum is a nondegenerate KKT vertex (1807 contacts on a 146-point rigid frame,
all lambda > 0), so it is an isolated root of a square analytic system -- Newton is
well posed and the only question is cost.

Cheap-precision trick: mixed-precision ITERATIVE REFINEMENT. The residual is evaluated
in mp at full working precision; the correction comes from a FIXED float64 Jacobian,
pseudo-inverted ONCE (its 91 rotational null directions and any other rank deficiency
are projected out by the pinv). Each iteration then costs one mp residual + one f64
matvec and gains ~16 - log10(cond) digits. Quadratic Newton would need an mp linear
solve per step; this needs none.

The 8 rattlers are dropped: they carry no contact, so they leave the Jacobian rank
deficient while contributing nothing to mu. mu is a property of the frame.

Usage: refine_14_154.py <pinned_kkt.json> <dps> <out_prefix>
"""

import json
import sys
import time

from mpmath import mp, mpf
import numpy as np


def main():
    src, dps, outp = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    mp.dps = dps

    p = json.load(open(src))
    A_all = [tuple(e) for e in p["A"]]
    frame = sorted({i for e in A_all for i in e})
    idx = {g: k for k, g in enumerate(frame)}
    nf, d, m = len(frame), len(p["X"][0]), len(A_all)
    A = [(idx[i], idx[j]) for i, j in A_all]
    X = [[mpf(p["X"][g][t]) for t in range(d)] for g in frame]
    mu = mpf(p["mu"])
    lam = [mpf(str(v)) for v in p["lam"]]
    s = sum(lam)
    lam = [v / s for v in lam]
    print(f"frame {nf}/{len(p['X'])} points, d={d}, |A|={m}, dps={dps}")

    nbrs = [[] for _ in range(nf)]
    for e, (i, j) in enumerate(A):
        nbrs[i].append((e, j))
        nbrs[j].append((e, i))

    # nu from stationarity at the start: 2 nu_k = -sum_e lam_e (x_j . x_k) = -mu * sum lam_e
    nu = [-mu * sum(lam[e] for e, _ in nbrs[k]) / 2 for k in range(nf)]

    nv = nf * d + 1 + m + nf
    off_mu, off_l, off_n = nf * d, nf * d + 1, nf * d + 1 + m

    def residual():
        r = []
        for i in range(nf):
            r.append(sum(X[i][t] * X[i][t] for t in range(d)) - 1)
        for i, j in A:
            r.append(sum(X[i][t] * X[j][t] for t in range(d)) - mu)
        for k in range(nf):
            for t in range(d):
                r.append(
                    sum(lam[e] * X[j][t] for e, j in nbrs[k]) + 2 * nu[k] * X[k][t]
                )
        r.append(sum(lam) - 1)
        return r

    Xf = np.array([[float(x) for x in row] for row in X])
    lamf = np.array([float(v) for v in lam])
    nuf = np.array([float(v) for v in nu])
    J = np.zeros((nf + m + nf * d + 1, nv))
    row = 0
    for i in range(nf):
        J[row, i * d : (i + 1) * d] = 2.0 * Xf[i]
        row += 1
    for e, (i, j) in enumerate(A):
        J[row, i * d : (i + 1) * d] = Xf[j]
        J[row, j * d : (j + 1) * d] = Xf[i]
        J[row, off_mu] = -1.0
        row += 1
    for k in range(nf):
        for e, j in nbrs[k]:
            J[row : row + d, j * d : (j + 1) * d] += lamf[e] * np.eye(d)
            J[row : row + d, off_l + e] = Xf[j]
        J[row : row + d, k * d : (k + 1) * d] += 2.0 * nuf[k] * np.eye(d)
        J[row : row + d, off_n + k] = 2.0 * Xf[k]
        row += d
    J[row, off_l : off_l + m] = 1.0

    t0 = time.time()
    sv = np.linalg.svd(J, compute_uv=False)
    rcond = 1e-10
    Jp = np.linalg.pinv(J, rcond=rcond)
    keep = int((sv > rcond * sv[0]).sum())
    print(
        f"J {J.shape}  rank {keep}/{J.shape[1]}  (null {J.shape[1] - keep}; 91 = rotations)"
    )
    print(f"cond(kept) = {sv[0] / sv[keep - 1]:.3e}   pinv {time.time() - t0:.0f}s")

    prev = None
    for it in range(400):
        r = residual()
        rn = max(abs(v) for v in r)
        dig = -mp.log10(rn) if rn > 0 else mpf(dps)
        print(
            f"  it{it}: max|resid| = {mp.nstr(rn, 4)}  (~{int(dig)} digits)", flush=True
        )
        if rn < mpf(10) ** (-(dps - 15)):
            print("  converged to working precision")
            break
        if prev is not None and rn > prev / 2:
            print("  refinement stalled (f64 Jacobian floor reached)")
            break
        prev = rn
        # SCALE before dropping to f64: the residual reaches 1e-1000+, far under the
        # f64 underflow floor (1e-308). Normalize to O(1), solve, rescale in mp.
        rhat = np.array([float(v / rn) for v in r])
        step = Jp @ (-rhat)
        sc = [rn * mpf(float(step[k])) for k in range(nv)]
        for i in range(nf):
            for t in range(d):
                X[i][t] += sc[i * d + t]
        mu += sc[off_mu]
        for e in range(m):
            lam[e] += sc[off_l + e]
        for k in range(nf):
            nu[k] += sc[off_n + k]

    print(f"\nmu = {mp.nstr(mu, min(dps - 10, 120))}")
    json.dump(
        {
            "mu": mp.nstr(mu, dps),
            "dps": dps,
            "frame": frame,
            "A": [list(e) for e in A_all],
            "X": [[mp.nstr(x, dps) for x in row] for row in X],
        },
        open(f"{outp}.json", "w"),
    )
    print(f"saved {outp}.json")


if __name__ == "__main__":
    main()
