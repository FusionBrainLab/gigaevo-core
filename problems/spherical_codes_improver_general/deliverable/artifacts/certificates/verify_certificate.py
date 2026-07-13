"""Standalone verifier for rational spherical-code certificates.

Consumes ONLY the JSON payload (axis indices + integer parameter vectors + D);
no numpy, no floats in any accept/reject decision, no asserts (explicit exits).
Reconstructs each point num/den with num = (2*a*D at non-axis coords, D^2-|a|^2
at axis), den = D^2+|a|^2, verifies |num|^2 == den^2 exactly, then checks every
pairwise inner product strictly below each bound in arbitrary-precision integer
arithmetic. Reports the exact rational maximum and its attaining pair.

Usage: verify_certificate.py <payload.json> <d> <n> <bound> [bound2 ...]
  bound = "p/q"  or  "sqrt:r/s2" (= sqrt(r)/s, s2=s^2)  or  "root:c2,c1,c0"
          (= larger real root of c2*x^2+c1*x+c0, checked exactly)
Exit 0 iff ALL bounds hold for ALL pairs.
"""

import json
import sys


def fail(msg):
    print(f"VERIFY FAIL: {msg}")
    sys.exit(1)


def isqrt_check(v):
    r = int(v**0.5)
    while r * r > v:
        r -= 1
    while (r + 1) * (r + 1) <= v:
        r += 1
    return r


def make_check(spec):
    """Return (desc, check(dot, dd) -> bool) with dd = den_i*den_j > 0; strict <."""
    if spec.startswith("sqrt:"):
        r, s2 = map(int, spec[5:].split("/"))
        s = isqrt_check(s2)
        if s * s != s2:
            fail(f"bound {spec}: s2 not a perfect square")
        if r <= 0:
            fail(f"bound {spec}: need r > 0")
        return (
            f"sqrt({r})/{s}",
            lambda dot, dd: dot <= 0 or (s * dot) ** 2 < r * dd * dd,
        )
    if spec.startswith("root:"):
        c2, c1, c0 = map(int, spec[5:].split(","))
        disc = c1 * c1 - 4 * c2 * c0
        if c2 <= 0 or disc <= 0:
            fail(f"bound {spec}: need c2 > 0 and positive discriminant")

        # larger root alpha = (-c1 + sqrt(disc)) / (2*c2);  dot/dd < alpha
        # <=> 2*c2*dot + c1*dd < sqrt(disc)*dd  <=> lhs <= 0 or lhs^2 < disc*dd^2
        def chk(dot, dd, c2=c2, c1=c1, disc=disc):
            lhs = 2 * c2 * dot + c1 * dd
            return lhs <= 0 or lhs * lhs < disc * dd * dd

        return f"root({c2}x^2{c1:+d}x{c0:+d})", chk
    p, q = map(int, spec.split("/"))
    if q <= 0:
        fail(f"bound {spec}: need q > 0")
    return spec, lambda dot, dd: q * dot < p * dd


def main():
    if len(sys.argv) < 5:
        fail("usage: verify_certificate.py <payload.json> <d> <n> <bound> [bound2 ...]")
    path, d, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    bounds = [make_check(s) for s in sys.argv[4:]]

    with open(path) as f:
        p = json.load(f)
    for key in ("d", "n", "D", "axis", "a_params"):
        if key not in p:
            fail(f"payload missing key {key}")
    if p["d"] != d or p["n"] != n:
        fail(f"payload (d,n)=({p['d']},{p['n']}) != expected ({d},{n})")
    D = p["D"]
    if not isinstance(D, int) or D <= 0:
        fail("D must be a positive integer")
    axis, aps = p["axis"], p["a_params"]
    if len(axis) != n or len(aps) != n:
        fail(f"expected {n} axis entries and {n} parameter vectors")

    pts = []
    for i, (k, a) in enumerate(zip(axis, aps)):
        if not isinstance(k, int) or not 0 <= k < d:
            fail(f"point {i}: axis {k} out of range")
        if len(a) != d - 1 or not all(isinstance(x, int) for x in a):
            fail(f"point {i}: need {d - 1} integer parameters")
        a2 = sum(x * x for x in a)
        den = D * D + a2
        num = [0] * d
        it = iter(a)
        for j in range(d):
            num[j] = D * D - a2 if j == k else 2 * next(it) * D
        if sum(c * c for c in num) != den * den:
            fail(f"point {i}: norm identity violated")
        pts.append((num, den))
    print(f"{n} points reconstructed; |num|^2 == den^2 holds exactly for all")

    best = None  # (dot, dd, i, j) with max dot/dd, compared by cross-multiplication
    viol = {desc: 0 for desc, _ in bounds}
    for i in range(n):
        ni, di = pts[i]
        for j in range(i + 1, n):
            nj, dj = pts[j]
            dot = sum(x * y for x, y in zip(ni, nj))
            dd = di * dj
            for desc, chk in bounds:
                if not chk(dot, dd):
                    viol[desc] += 1
            if best is None or dot * best[1] > best[0] * dd:
                best = (dot, dd, i, j)

    dot, dd, bi, bj = best
    print(f"exact max inner product = {dot}/{dd}")
    print(f"                        ~ {dot / dd:.15f}  attained by pair ({bi},{bj})")
    ok = True
    for desc, _ in bounds:
        v = viol[desc]
        print(
            f"bound {desc}: {'ALL PAIRS STRICTLY BELOW' if v == 0 else f'{v} VIOLATIONS'}"
        )
        ok = ok and v == 0
    print(f"\nVERIFY {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
