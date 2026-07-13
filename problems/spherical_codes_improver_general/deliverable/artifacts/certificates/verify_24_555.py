"""Standalone exact verifier for mu*(24,555) <= 1/5.  Stdlib only; no floats.

Replays the certificate from its JSON payload:
  * S (272x272 Seidel, symmetric, zero diagonal, entries +-1)  ->  A = I + S/5
  * a 23-line basis J with B = A[J,J] positive definite (exact LDL)
  * the identity A = A[:,J] B^-1 A[J,:] over Q, which forces A PSD of rank 23,
    hence 272 unit vectors in R^23 with pairwise inner products exactly +-1/5
  * 11 further points given by rational stereographic parameters v_i, so their
    24-dim coordinates (q_i, t_i) satisfy q^T B q + t^2 = 1 identically

The 555 points are the 544 vectors +-L_k plus the 11 points, and the code checks
every pairwise inner product against 1/5:
  line/line   |A_ij| = 1/5 by construction
  point/line  |C_ik| = |(Q A[J,:])_ik| <= 1/5
  point/point  G_ij  = (Q B Q^T + t t^T)_ij <= 1/5
Coincident points would show up as an inner product of 1, so the checks also
establish that the 555 points are distinct.

Usage: verify_24_555.py [certificate_24_555_stereographic.json] [certificate_24_555_seidel272.json]
Exits 0 iff the certificate holds.
"""

from fractions import Fraction as F
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
pay_p = (
    Path(sys.argv[1])
    if len(sys.argv) > 1
    else HERE / "certificate_24_555_stereographic.json"
)
sei_p = (
    Path(sys.argv[2])
    if len(sys.argv) > 2
    else HERE / "certificate_24_555_seidel272.json"
)

pay = json.loads(pay_p.read_text())
S = json.loads(sei_p.read_text())
J = pay["J"]
V = [[F(num, den) for num, den in row] for row in pay["v"]]
n, r, m = 272, len(J), len(V)
one5 = F(1, 5)

if len(S) != n or any(len(row) != n for row in S):
    raise SystemExit(f"FAIL: Seidel matrix is not {n}x{n}")
for i in range(n):
    if S[i][i] != 0:
        raise SystemExit(f"FAIL: S[{i}][{i}] != 0")
    for j in range(i + 1, n):
        if S[i][j] != S[j][i]:
            raise SystemExit(f"FAIL: S not symmetric at ({i},{j})")
        if S[i][j] not in (-1, 1):
            raise SystemExit(f"FAIL: S[{i}][{j}] = {S[i][j]} not in {{-1,+1}}")
print(f"S: {n}x{n} symmetric, zero diagonal, entries +-1  [exact]")
print(f"J = {J}")

A = [[(F(1) if i == j else F(S[i][j], 5)) for j in range(n)] for i in range(n)]
B = [[A[a][b] for b in J] for a in J]

M = [row[:] for row in B]
for k in range(r):
    if M[k][k] <= 0:
        raise SystemExit(f"FAIL: LDL pivot {k} = {M[k][k]} is not positive")
    for i in range(k + 1, r):
        f = M[i][k] / M[k][k]
        for j in range(k, r):
            M[i][j] -= f * M[k][j]
print(f"B = A[J,J] positive definite: {r}/{r} positive pivots  [exact]")

aug = [[B[i][j] for j in range(r)] + [F(i == j) for j in range(r)] for i in range(r)]
for c in range(r):
    p = next(i for i in range(c, r) if aug[i][c] != 0)
    aug[c], aug[p] = aug[p], aug[c]
    d = aug[c][c]
    aug[c] = [v / d for v in aug[c]]
    for i in range(r):
        if i != c and aug[i][c] != 0:
            f = aug[i][c]
            aug[i] = [a - f * b for a, b in zip(aug[i], aug[c])]
Binv = [row[r:] for row in aug]

AJ = [[A[i][j] for j in J] for i in range(n)]
T = [[sum(AJ[i][a] * Binv[a][b] for a in range(r)) for b in range(r)] for i in range(n)]
bad = sum(
    sum(T[i][a] * AJ[j][a] for a in range(r)) != A[i][j]
    for i in range(n)
    for j in range(n)
)
if bad:
    raise SystemExit(f"FAIL: identity A = A[:,J] B^-1 A[J,:] fails at {bad} entries")
print(f"identity A = A[:,J] B^-1 A[J,:]: HOLDS exactly  =>  A PSD, rank {r}")
print(f"  => {n} unit vectors in R^{r} with all pairwise |ip| = 1/5")

rho = [
    sum(V[i][a] * sum(B[a][b] * V[i][b] for b in range(r)) for a in range(r))
    for i in range(m)
]
Q = [[2 * V[i][a] / (1 + rho[i]) for a in range(r)] for i in range(m)]
t = [(1 - rho[i]) / (1 + rho[i]) for i in range(m)]
for i in range(m):
    nrm = (
        sum(Q[i][a] * sum(B[a][b] * Q[i][b] for b in range(r)) for a in range(r))
        + t[i] ** 2
    )
    if nrm != 1:
        raise SystemExit(f"FAIL: point {i} has squared norm {nrm} != 1")
print(f"{m} stereographic points: squared norm = 1  [exact]")

Cmax, viol_C = F(-10), 0
for i in range(m):
    for k in range(n):
        c = abs(sum(Q[i][a] * A[J[a]][k] for a in range(r)))
        viol_C += c > one5
        Cmax = max(Cmax, c)
Gmax, viol_G = F(-10), 0
for i in range(m):
    QB = [sum(Q[i][a] * B[a][b] for a in range(r)) for b in range(r)]
    for j in range(i + 1, m):
        g = sum(QB[b] * Q[j][b] for b in range(r)) + t[i] * t[j]
        viol_G += g > one5
        Gmax = max(Gmax, g)
print(f"max |C_ij| (point/line)  = {Cmax} = {float(Cmax):.12f}  violations {viol_C}")
print(f"max  G_ij  (point/point) = {Gmax} = {float(Gmax):.12f}  violations {viol_G}")

if viol_C or viol_G:
    raise SystemExit("\nCERTIFICATE FAILS")
print(f"\n{2 * n + m} points on S^23, all pairwise inner products <= 1/5")
print("CERTIFICATE PASSES: mu*(24,555) <= 1/5 PROVEN (exact rational arithmetic)")
