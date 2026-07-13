# DRAFT — for user review. DO NOT SEND AS-IS.

**To:** Henry Cohn
**Subject:** Three of the conjectured optima can be beaten
**Attach:** `report.pdf`, `spherical_codes_certificates.tar.gz`

---

Dear Henry,

We ran all 455 cases through our optimizer. Three of the conjectured optima can be
strictly beaten; the other 452 reproduced your values (every non-hit gain was below 1e-15).

| Case | Conjectured optimum | What we can certify |
|---|---|---|
| (24,555) | root of 39x²+72x−16 ≈ 0.2004565 | μ* ≤ 1/5 |
| (9,36) | 1/5 (Ericson–Zinoviev 1995) | μ* < 1/5 |
| (14,154) | √30/20 ≈ 0.2738613 | μ* < √30/20 |

All three are certified in exact arithmetic — no floating-point quantity enters any
accept/reject decision. For (9,36) and (14,154) the points are exactly rational points of
the sphere and every inner product is compared to the bound in integer arithmetic. For
(24,555) the witness is 272 equiangular lines at ±1/5 plus 11 further points, and the
embedding is established by an exact rank-23 identity over ℚ.

The attached bundle has the certificates and two verifiers (stdlib Python 3 — they read
the JSON payloads and nothing else). Three commands check everything:

```
cd certificates
python3 verify_certificate.py certificate_9_36_rational.json  9  36 "1/5"
python3 verify_certificate.py certificate_14_154_rational.json 14 154 "sqrt:30/400"
python3 verify_24_555.py
```

Two things you may find interesting, both in the report. For (24,555) we have evidence
that 1/5 is the answer and not merely an upper bound: the 272-line core is locally rigid,
and unconstrained optimization plateaus just above 1/5. And for (14,154) the optimum is a
nondegenerate KKT vertex, so we can pin it to ~1990 digits — and an integer-relation search
on that value finds nothing at any degree up to 30, at heights up to 10²⁰. Your table's
exact values top out at degree 14 with coefficients below 5×10⁷, so whatever μ*(14,154)
is, it does not look like the other entries.

To be clear on scope: what we disprove is the conjectured values and the optimality of the
listed constructions. We do not claim to know the true optima.

Happy to send anything else that would be useful.

Best,
[user to sign]

---

## DRAFT NOTES (not part of the letter)

- Wording per the audit guidance: "disproves the conjectured value / the optimality of the
  E–Z construction", never "disproves a 30-year-old conjecture" (whether E–Z themselves
  claimed optimality is unverified), and never that our numerics are the new exact optima.
- Keep a dated snapshot of the catalogue pages for (24,555), (9,36), (14,154) before
  sending (we have `scratchpad/site_index.html` from 2026-07-12).
- Attachments: `report.pdf` (81 KB) and `spherical_codes_certificates.tar.gz` (597 KB).
- The `(14,154)` certificate also rigorously excludes the root of 17x²−x−1; the extra
  `"root:17,-1,-1"` bound is omitted from the commands above to keep them short, and is
  documented in the bundle README.
- The report covers the (9,36) saddle story (the value we first quoted, 0.19985677170944,
  is a saddle, not the optimum) — deliberately left out of the letter.
- All three commands re-run from a clean copy of the bundle on 2026-07-13; all pass.
