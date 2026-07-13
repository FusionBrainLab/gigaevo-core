"""Exact rational certificate that a packing beats a conjectured optimum.

Rationalize each witness point onto S^{d-1} exactly: pick axis k = argmax x_k,
set integer params a ~ round(D * x_others/(1+x_k)); then
    num = (2*a*D  at other coords, D^2 - |a|^2 at coord k),  den = D^2 + |a|^2
gives num/den exactly unit norm (|num|^2 == den^2, checked). All pairwise inner
products then compared to the conjectured bound in pure integer arithmetic:
    rational bound p/q:   q * dot < p * den_i * den_j
    bound sqrt(r)/s:      dot <= 0  or  (s * dot)^2 < r * (den_i * den_j)^2

Usage: cert_rational_points.py <witness.npy> <d> <n> <bound> <out.json> [D]
  bound = "p/q" (rational) or "sqrt:r/s2" meaning sqrt(r)/s with s2 = s^2
          e.g. (9,36): 1/5      (14,154): sqrt:30/400  (= sqrt(30)/20)
  D = rationalization denominator scale (default 10^7)
"""

import json
import sys

import numpy as np


def rationalize(X, D):
    pts = []
    for x in X:
        k = int(np.argmax(x))
        others = [i for i in range(len(x)) if i != k]
        v = x[others] / (1.0 + x[k])
        a = [int(round(vi * D)) for vi in v]
        a2 = sum(ai * ai for ai in a)
        den = D * D + a2
        num = [0] * len(x)
        for idx, i in enumerate(others):
            num[i] = 2 * a[idx] * D
        num[k] = D * D - a2
        assert sum(c * c for c in num) == den * den, "norm identity failed"
        pts.append((num, den, k, a))
    return pts


def main():
    src, d, n, bound, out = (
        sys.argv[1],
        int(sys.argv[2]),
        int(sys.argv[3]),
        sys.argv[4],
        sys.argv[5],
    )
    X = np.load(src)
    assert X.shape == (n, d), X.shape
    D = int(sys.argv[6]) if len(sys.argv) > 6 else 10**7
    pts = rationalize(X, D)
    print(
        f"{n} points rationalized exactly onto S^{d - 1} (D = {D}); norm identity holds for all"
    )

    if bound.startswith("sqrt:"):
        r, s2 = map(int, bound[5:].split("/"))
        s = int(round(s2**0.5))
        assert s * s == s2
        bound_f = (r**0.5) / s

        def check(dot, dd):
            return dot <= 0 or (s * dot) ** 2 < r * dd * dd

        bound_str = f"sqrt({r})/{s}"
    else:
        p, q = map(int, bound.split("/"))
        bound_f = p / q

        def check(dot, dd):
            return q * dot < p * dd

        bound_str = bound

    viol = 0
    max_ip = -1.0
    for i in range(n):
        ni, di = pts[i][0], pts[i][1]
        for j in range(i + 1, n):
            nj, dj = pts[j][0], pts[j][1]
            dot = sum(a * b for a, b in zip(ni, nj))
            if not check(dot, di * dj):
                viol += 1
            ip = dot / (di * dj)
            if ip > max_ip:
                max_ip = ip

    print(f"max ip = {max_ip:.15f}   conjectured bound {bound_str} = {bound_f:.15f}")
    print(f"margin = {bound_f - max_ip:.3e}   violations: {viol}")
    ok = viol == 0
    status = (
        f"PASSES: mu*({d},{n}) < {bound_str} PROVEN (exact integer arithmetic)"
        if ok
        else "FAILS"
    )
    print(f"\nCERTIFICATE {status}")
    if ok:
        payload = {
            "d": d,
            "n": n,
            "bound": bound_str,
            "D": D,
            "axis": [p[2] for p in pts],
            "a_params": [p[3] for p in pts],
            "max_ip_float": max_ip,
            "witness_file": src,
        }
        with open(out, "w") as f:
            json.dump(payload, f)
        print(f"payload saved: {out}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
