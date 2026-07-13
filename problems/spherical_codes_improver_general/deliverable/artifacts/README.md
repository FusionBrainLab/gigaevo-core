# Three improvements in the spherical-code catalogue — artifact bundle

Certified in exact arithmetic:

| Case | Conjectured optimum | Certified |
|---|---|---|
| (24,555) | root of 39x²+72x−16 ≈ 0.2004565282 | μ*(24,555) ≤ 1/5 |
| (9,36) | 1/5 (Ericson–Zinoviev 1995) | μ*(9,36) < 1/5 |
| (14,154) | √30/20 ≈ 0.2738612788 | μ*(14,154) < √30/20 |

## Verify (stdlib Python 3 only — no numpy, no floating point)

```bash
cd certificates

# (9,36):   mu* < 1/5
python3 verify_certificate.py certificate_9_36_rational.json 9 36 "1/5"

# (14,154): mu* < sqrt(30)/20, and the 17x^2-x-1 root is excluded as well
python3 verify_certificate.py certificate_14_154_rational.json 14 154 "sqrt:30/400" "root:17,-1,-1"

# (24,555): mu* <= 1/5
python3 verify_24_555.py
```

Each verifier exits 0 iff the certificate holds, nonzero on any violation. No
floating-point quantity enters any accept/reject decision in any of them.

## What each certificate is

**(9,36) and (14,154) — pointwise rational certificates.** Every point is an exactly
rational point of the sphere: from an integer vector `a` and an integer `D`, the point is
`num/den` with `num = (2aD, D²−|a|²)` and `den = D²+|a|²`, so `|num|² = den²` holds
identically over ℤ. Every pairwise inner product is then compared to the bound in integer
arithmetic (for `1/5`: `5·dot < denᵢ·denⱼ`; for `√30/20`: `dot ≤ 0` or
`(20·dot)² < 30·(denᵢdenⱼ)²`). Floating point was used only to *select* the integers `a`.

**(24,555) — structural certificate.** The witness is 272 equiangular lines at ±1/5 (the
regular two-graph on 276 vertices minus a 4-subset containing exactly two coherent triples),
giving 544 points, plus 11 further points with strict slack. `verify_24_555.py` checks, over
ℚ: that the Seidel matrix `S` is symmetric with zero diagonal and ±1 off-diagonal; that
`B = A[J,J]` is positive definite for the 23-line basis `J` (exact LDL, 23/23 positive
pivots); that `A = A[:,J]·B⁻¹·A[J,:]` holds exactly — which forces `A = I + S/5` to be PSD of
rank 23, hence the 272 lines embed in ℝ²³; that the 11 stereographic points have squared norm
exactly 1; and that all point/line and point/point inner products are ≤ 1/5. Coincident points
would surface as an inner product of 1, so the checks also establish that the 555 points are
distinct.

## Files

```
certificates/
  verify_certificate.py                 JSON-only verifier for the rational-point certificates
  verify_24_555.py                      JSON-only verifier for the (24,555) structural certificate
  certificate_9_36_rational.json        36 rational points in R^9
  certificate_14_154_rational.json      154 rational points in R^14
  certificate_24_555_stereographic.json line basis J + rational stereographic parameters
  certificate_24_555_seidel272.json     272x272 integer Seidel matrix of the equiangular lines
generators/                             how the certificates were produced (need numpy)
  cert_rational_points.py               float witness -> rational witness
  cert_stereographic.py                 float witness -> (24,555) structural certificate
  exact_candidate_24_555.npy            float witness (272 lines + 11 points)
  exact_candidate_24_555_seidel272.npy  Seidel matrix, .npy form
campaign/
  results.json                          per-case results, all 455 cases, seeds 0-5
  packings_champion.npz                 best packing found per case
  pinned_14_154_kkt.json                (14,154) KKT vertex: ~90 digits, contacts, multipliers
analysis/                               the (14,154) arithmetic study (need numpy + mpmath)
  refine_14_154.py                      KKT pin -> mu at arbitrary precision
  pslq_sweep.py                         integer-relation sweep, per-degree height bounds
  mu_14_154.txt                         mu*(14,154) to 1950 decimal places
```

To reproduce the high-precision value and the integer-relation result:

```
cd analysis
python3 refine_14_154.py ../campaign/pinned_14_154_kkt.json 2000 refine2000
python3 pslq_sweep.py refine2000.json 30
```

The refinement takes about a minute (mixed-precision iterative refinement on the 146-point
rigid frame; the 8 rattlers are dropped, as they carry no contact and only make the Jacobian
rank-deficient). It runs at a 2000-digit working precision and converges to a residual below
1e-1985; two independent runs agree digit-for-digit on the first **1987** decimals, and beyond
that the trailing digits are working-precision noise. `mu_14_154.txt` therefore quotes 1950
digits and no more, and the sweep holds back a 100-digit margin, searching at a 1900-digit
tolerance — 87 digits inside the verified range.

The sweep finds no integer relation at any degree 2..30, at heights up to 1e20 throughout.
Note that it derives a **per-degree** height bound from the available precision rather than
quoting one blanket coefficient cap: a degree-n relation of height H needs ~(n+1)·log10(H)
digits to be detectable at all, and below that threshold PSLQ returns spurious hits rather
than nothing. Any relation it does find is independently re-verified by evaluating the
polynomial at mu. As positive controls the same code recovers 40x^2-3 from sqrt(30)/20, and
recovers the deepest minimal polynomial in Cohn's own table — the degree-14 entry at (3,38) —
exactly, so a null result here reflects the arithmetic and not a failure to search.

`results.json` gives, per case, the conjectured value, the best value found, the winning seed,
and the relative gain. Gains are banded `none < 1e-9 ≤ weak < 1e-6 ≤ moderate < 1e-4 ≤ strong`;
across all 455 cases and 6 seeds the bands are `{strong 3, moderate 0, weak 0, none 452}` — the
three cases above, and nothing else.

## Scope of the claims

These certificates are **upper bounds on μ\*** from explicit witnesses. They disprove the
conjectured optimum values and the optimality of the listed constructions. They do **not**
establish the true optima, and we make no exact-form claim for any of the three. See the
report for what is and is not claimed.
