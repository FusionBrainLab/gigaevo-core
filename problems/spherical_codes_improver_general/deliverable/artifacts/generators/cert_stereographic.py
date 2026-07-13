"""Full rational certificate for mu*(24,555) <= 1/5, Codex-recommended form:
line basis J (23 lines), B = A[J,J] > 0, identity A = A[:,J] B^-1 A[J,:],
11 points via rational stereographic parameters v_i. All checks in Fraction arithmetic."""

from fractions import Fraction as F

import numpy as np

RAW = "/mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/mathemage/gigaevo-core-internal/problems/spherical_codes_improver_general/conjecture_raw"
S = np.load(f"{RAW}/exact_candidate_24_555_seidel272.npy")
X = np.load(f"{RAW}/exact_candidate_24_555.npy")
Lex, P11 = X[:272], X[544:]
n = 272

# --- choose J: greedy max-pivot Cholesky on float A for good conditioning
Af = np.eye(n) + S / 5.0
J, Rm = [], Af.copy()
for _ in range(23):
    k = int(np.argmax(np.diag(Rm)))
    J.append(k)
    Rm = Rm - np.outer(Rm[:, k], Rm[:, k]) / Rm[k, k]
J = sorted(J)
print(f"J = {J}")

one5 = F(1, 5)
A = [[(F(1) if i == j else F(int(S[i, j]), 5)) for j in range(n)] for i in range(n)]
B = [[A[a][b] for b in J] for a in J]

# --- exact LDL^T of B: positive definiteness
import copy

M = copy.deepcopy(B)
piv = []
for k in range(23):
    if M[k][k] <= 0:
        raise SystemExit(f"FAIL: pivot {k} = {M[k][k]} not positive")
    piv.append(M[k][k])
    for r in range(k + 1, 23):
        f = M[r][k] / M[k][k]
        for c in range(k, 23):
            M[r][c] -= f * M[k][c]
print("B positive definite: 23/23 positive pivots  [exact]")

# --- exact inverse of B (Gauss-Jordan over Q)
Maug = [
    [B[i][j] for j in range(23)] + [F(1) if i == j else F(0) for j in range(23)]
    for i in range(23)
]
for col in range(23):
    p = next(r for r in range(col, 23) if Maug[r][col] != 0)
    Maug[col], Maug[p] = Maug[p], Maug[col]
    d = Maug[col][col]
    Maug[col] = [v / d for v in Maug[col]]
    for r in range(23):
        if r != col and Maug[r][col] != 0:
            f = Maug[r][col]
            Maug[r] = [a - f * b for a, b in zip(Maug[r], Maug[col])]
Binv = [row[23:] for row in Maug]

# --- identity A = A[:,J] B^-1 A[J,:]  (proves A PSD, rank 23)
AJ = [[A[i][j] for j in J] for i in range(n)]  # n x 23
T = [
    [sum(AJ[i][a] * Binv[a][b] for a in range(23)) for b in range(23)] for i in range(n)
]  # n x 23
bad = 0
for i in range(n):
    for j in range(n):
        v = sum(T[i][a] * AJ[j][a] for a in range(23))
        if v != A[i][j]:
            bad += 1
print(
    f"identity A = A[:,J] B^-1 A[J,:]: {'HOLDS exactly' if bad == 0 else f'FAILS at {bad} entries'}"
)
if bad:
    raise SystemExit(1)

# --- rational stereographic parameters from the numeric 11 points
LJ = Lex[J]  # 23 x 24 float basis lines
cJ = P11 @ LJ.T  # 11 x 23: ips of points to basis lines
Bf = np.array([[float(B[i][j]) for j in range(23)] for i in range(23)])
q_f = np.linalg.solve(Bf, cJ.T).T  # equatorial coords in line basis
u = np.linalg.svd(Lex - Lex.mean(0) * 0)[2][
    -1
]  # residual dir: smallest right-singular vector
u = u / np.linalg.norm(u)
t_f = P11 @ u
t_f *= np.sign(t_f) if False else 1.0
v_f = q_f / (1.0 + t_f[:, None])  # inverse stereographic
DEN = 10**5
V = [[F(round(v_f[i][a] * DEN), DEN) for a in range(23)] for i in range(11)]

# --- exact reconstruction (2)-(4)
rho = [
    sum(V[i][a] * sum(B[a][b] * V[i][b] for b in range(23)) for a in range(23))
    for i in range(11)
]
Q = [[2 * V[i][a] / (1 + rho[i]) for a in range(23)] for i in range(11)]
t = [(1 - rho[i]) / (1 + rho[i]) for i in range(11)]
# C = Q A[J,:]  (11 x 272 cross ips)
AJrows = [[A[j][k] for k in range(n)] for j in J]  # 23 x n
Cmax = F(-10)
viol_C = 0
for i in range(11):
    for k in range(n):
        c = sum(Q[i][a] * AJrows[a][k] for a in range(23))
        ac = -c if c < 0 else c
        if ac > one5:
            viol_C += 1
        if ac > Cmax:
            Cmax = ac
# G = Q B Q^T + t t^T (11x11)
Gmax = F(-10)
viol_G = 0
for i in range(11):
    QB_i = [sum(Q[i][a] * B[a][b] for a in range(23)) for b in range(23)]
    for j in range(i + 1, 11):
        g = sum(QB_i[b] * Q[j][b] for b in range(23)) + t[i] * t[j]
        if g > one5:
            viol_G += 1
        if g > Gmax:
            Gmax = g
print(
    f"max |C_ij| = {Cmax} = {float(Cmax):.12f}  (<= 1/5: {Cmax <= one5}; violations {viol_C})"
)
print(
    f"max G_ij   = {Gmax} = {float(Gmax):.12f}  (<= 1/5: {Gmax <= one5}; violations {viol_G})"
)
print("norm check: ||p_i|| = 1 identically by construction (stereographic)")
ok = viol_C == 0 and viol_G == 0
print(
    f"\nCERTIFICATE {'PASSES: mu*(24,555) <= 1/5 PROVEN (exact rational arithmetic)' if ok else 'FAILS'}"
)
if ok:
    import json

    payload = {
        "J": [int(j) for j in J],
        # explicit num/den pairs: Fraction auto-reduces, so a shared denominator loses entries
        "v": [
            [[V[i][a].numerator, V[i][a].denominator] for a in range(23)]
            for i in range(11)
        ],
        "max_abs_C": [Cmax.numerator, Cmax.denominator],
        "max_G": [Gmax.numerator, Gmax.denominator],
        "seidel_file": "certificate_24_555_seidel272.json",
    }
    with open(f"{RAW}/certificate_24_555_stereographic.json", "w") as f:
        json.dump(payload, f)
    print(f"payload saved: {RAW}/certificate_24_555_stereographic.json")
