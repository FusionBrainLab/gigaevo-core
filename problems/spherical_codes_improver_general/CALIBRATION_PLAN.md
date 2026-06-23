# Validation calibration — find best params on val (panel), blast on test (full90)

Goal: pick the protocol P* that maximizes mean rel. improvement over Cohn on the
**panel** (val), reproduce the plotted canonical fitness (~0.46) with `validate.py`,
then apply P* to **full90** (test) for champion + E7 + E8 → headline table.

Metric is identical by construction to `validate.py`:
`gain = max(0, (mu_cohn - mu_best)/|mu_cohn|)`, `fitness = 100 * mean_configs gain`.
The only thing that differs across cells is the **search protocol**, not the score.

## Key calibration finding (budget is the dominant lever)
Canonical champion (R=1, B=10, seed=42) on panel:
- `config_timeout=600s`  -> 0.4125%  (all 14 configs wall-bound)
- `config_timeout=3540s` -> 0.4645%  (evo's true setting)  == plotted ~0.46 REPLICATED.

=> "beat 0.46" target = swept protocol on panel at equal (3540s-class) budget must exceed 0.4645%.

Coarse shape sweep (W=480, seed0): smaller M wins.
  M10/sigma1e-6 0.4200 > M10/1e-4 0.4145 > M25/1e-6 0.4127 > M50 ~0.40 > M25/1e-4 0.3942.
  Fewer noising steps per restart -> more restarts fit the wall -> more basin-hops -> better.

Expanded shape sweep (W=900, 2 seeds): best-over-seeds ranking
  M10 0.4269 > M5 0.4212 > M8 0.4198 > M14 0.4125 (all sigma1e-6).
  fresh-every IRRELEVANT (fe5 == fe99999 to 4 dp). High seed variance @W900
  (mean 0.30 vs best/seed 0.43) -> longer wall converges.
  => P* = R-unbounded, M=10, sigma 1:1e-6, fresh=5, dry_patience=0.

FINALIST (panel, P*, W=3540, single seed): 0.4797%  >  canonical 0.4645%.
  At EQUAL per-config budget the swept restart-heavy basin-hopper BEATS the single
  monotone chain by +0.0152pp (+3.3% rel). Wins d in {9,12,15,16}; canonical found a
  better d=10 basin at this single seed (variance; recoverable w/ more seeds).
  "replicate 0.46 then beat it" -> DONE.

Fitness is **wall-limited and monotone in per-config compute**. This is why
"spend more budget" directly buys improvement, and why the test blast uses a large wall.

## Budget allocation (overnight)
1. Canonical anchor @3540s (14w)        — replicate plotted ~0.46.            [running]
2. Quick shape sweep, W=480 (24w)       — coarse first pass.                  [running]
3. Expanded shape sweep, W=900 (≈28w)   — robust best params P* (2 seeds).
4. Stage-2 confirm (fresh policy) at best shape, W=900, 2 seeds.
5. Test blast: P* on full90 for champion+E7+E8, large wall (fair, same proto).
6. Report: TeX table, modevolve-style stats plots, UMAP 2x2; Telegram PDF.

## Expanded shape sweep grid (step 3)
- M (b_steps / noising steps): {10, 20, 40, 80}
- sigma schedule (hi:lo):      {1.0:1e-4, 1.0:1e-6}
- fresh_every:                 5  (explore cadence; pure-exploit tested in step 4)
- restarts:                    100000 (unbounded; bounded by fixed wall)
- wall:                        900s   (fair fixed budget for ranking shapes)
- seeds:                       {0, 1} (variance / robustness)
=> 4 x 2 x 2 = 16 cells x 14 configs x 1 = 224 cell-runs.

Ranking key = best_over_seeds_mean_pct, tie-break mean_gain_pct +- SE.

## Test blast (step 5) — fairness note
All three programs (champion, E7, E8) evaluated under the SAME P* on full90 with the
SAME large wall. The program is the treatment; the protocol is held fixed. Headline =
champion mean rel improvement over Cohn on full90; E7/E8 are program baselines.

## TEST BLAST RESULT (full90, P* = R=100000 / M=10 / sigma 1:1e-6, wall=1800s, single seed)
Headline mean rel. improvement over Cohn, identical protocol for all three:
  champion 0.1250%  (51/90 improved, 90/90 valid)   = 2.08x best prior program
  E7       0.0599%  (53/90 improved, 90/90 valid)
  E8       0.0500%  (34/90 improved, 90/90 valid)
Champion wins by CONCENTRATING gain in high-headroom dims (per-dim % over Cohn):
  d=13 0.343 (vs E7 0.088 / E8 0.118), d=16 0.297 (0.036/0.044), d=12 0.095 (0.047/0.050),
  d=15 0.106 (0.098/0.047). E7 spreads thin (more configs nudged: 53 vs 51) but smaller
  per-config gains; E8 weakest. Champion's edge is DEPTH on the few configs with real slack,
  not breadth.

### improved-count definition (51 not 57)
`improved` flag uses mu_best < mu_cohn - 1e-15 (real improvement). 6 configs reproduce the
Cohn point to ~1e-16 (float-noise tie, gain ~2e-14%); a naive `gain>0` count calls these 57.
Table, headline, and fig_gain_sorted all use the 51 (tolerance-correct) count. Fixed
make_report.py fig title to read the authoritative `improved` field.

## REPORT (delivered)
report/report.pdf (6pp): summary + metric def, calibration narrative (0.46 replicated + beaten),
per-dim table, fig_perdim_gain, fig_gain_sorted, fig_sweep_shape (M x sigma), UMAP 2x2
(fig_umap_2x2), inner-product tail plots (4 extreme-gain configs), full 90-row longtable.
Drop-in paper table = report/table_full90.tex. Built by build_report_pdf.py + make_report.py +
make_umap.py; compiled with tectonic (conda pdflatex fmt broken). Sent to Telegram.
Commit still HELD pending approval.
