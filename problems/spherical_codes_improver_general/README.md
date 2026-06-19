# spherical_codes_improver_general

Evolve a **single, general** spherical-code improver and use it to **boost the Cohn
catalogue** across `d∈[8,16]` — the full-conference extension of the ImprovEvolve paper,
which evolved its improver on `(600,11)` only and froze it for the scan.

**Objective** (per `(N, d)`): place `N` unit vectors in `R^d` minimising the **signed**
maximum pairwise inner product `μ(X) = max_{i<j} ⟨x_i, x_j⟩` (lower is better).

**Warm-start regime.** The grader never cold-starts: it seeds every evaluation with the
best-known **Cohn** configuration and measures how far `improve()` / `perturb()` push `μ`
below that baseline, under **monotone acceptance** (`μ` only decreases). An improver is
floored at the Cohn baseline — non-improvement scores 0, never negative.

Design rationale and the full protocol: `docs/superpowers/specs/2026-06-17-spherical-codes-general-improver-design.md`.

## Files

| File | Role |
|---|---|
| `cohn_catalogue.py` | 90-config `(d,N)` loader; `eval_configs(name)`; live `μ` from downloaded Cohn coordinates |
| `validate.py` | warm-start grader; `validate(Improver) -> ({"fitness","is_valid"}, artifact)` |
| `metrics.yaml` | `fitness` (primary, %) + `is_valid` |
| `task_description.txt` | the LLM-facing problem statement |
| `pipeline.py` | default pipeline with a compact artifact formatter |
| `initial_programs/` | seeds — `paper_evolved.py` (E.7), `paper_evolved_plus.py` (E.8), `riemannian_oblique.py` |
| `run_full_validation.py` | standalone full90 R=3 head-to-head harness (parallel across configs) |
| `cohn_cache/` | generated download cache for Cohn catalogue files; contents are ignored and re-created on demand |

## Eval sets (`SPHERICAL_EVAL_SET`)

| set | size | use |
|---|---|---|
| `smoke` | 6 | wiring check |
| `panel` | 14 | **evolution** — high-headroom configs (most-improvable `N` per dim, from the R=1 baseline headroom maps), all `d∈[8,16]` represented |
| `full90` | 90 | **validation / headline** — Table 7's exact set |

## Env knobs (the grader serves both evolution and paper-matched validation)

| var | default | meaning |
|---|---|---|
| `SPHERICAL_EVAL_SET` | `panel` | which config set to grade (`validate()` default; the full90 harness passes `--eval-set` explicitly) |
| `SPHERICAL_R_ROUNDS` | `1` | Stage-B rounds (paper scan = 3) |
| `SPHERICAL_B_STEPS` | `10` | perturb→improve steps per round (intensities on `geomspace(hi,lo,B)`) |
| `SPHERICAL_INTENSITY_HI` / `_LO` | `1.0` / `1e-4` | intensity schedule endpoints |
| `SPHERICAL_DRY_PATIENCE` | `1` | dry-round early-stop |
| `SPHERICAL_CONFIG_TIMEOUT` | `25.0` | per-config wall-clock cap (s) — bounds large-`n` cost |
| `SPHERICAL_EVAL_TIMEOUT` | `1800.0` | global cap (s); tuned so it never fires in normal operation |
| `SPHERICAL_NORM_TOL` | `1e-12` | unit-row tolerance (float64 required) |
| `SPHERICAL_SEED` | `42` | grader seed (deterministic, reproducible fitness) |

## Cost (CPU, 8 threads; see spec §6)

`improve()` is 5–12 s at medium-large `n` (L-BFGS continuations over an `O(n²)` objective);
JIT compile (~0.5 s) is negligible. Full90 single-thread is ~53 min (numpy) to ~104 min
(E.7) at R=1 — too slow per mutant, hence **panel for evolution**, **full90 for validation**.

## Launch — evolution (panel, Gemini 3.5 Flash to match the paper)

```bash
cd /mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/mathemage/gigaevo-core-internal
export OPENAI_API_KEY=sk-gigaevo            # LiteLLM proxy key (per-endpoint)
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
SPHERICAL_EVAL_SET=panel SPHERICAL_R_ROUNDS=1 SPHERICAL_B_STEPS=8 SPHERICAL_CONFIG_TIMEOUT=45 \
python3 run.py \
  problem.name=spherical_codes_improver_general \
  pipeline=spherical_general \
  redis.db=<DB> <llm + run overrides matching the paper>
```

## Validate — head-to-head (full90, R=3)

Run the evolved champion and the paper baselines through the **same** grader on the **same**
downloaded Cohn cache, then compare the `FITNESS` line:

```bash
cd problems/spherical_codes_improver_general
P=python3
$P run_full_validation.py initial_programs/paper_evolved.py      --label E7        --rounds 3
$P run_full_validation.py initial_programs/paper_evolved_plus.py --label E8        --rounds 3
$P run_full_validation.py /path/to/champion.py                   --label champion  --rounds 3
```

Reports mean improvement over Cohn, per-dimension breakdown, **in-panel vs
out-of-panel** gain with dynamic panel sizes (generalization check), and largest
gains; writes a per-config JSON.
Repeat at `--seed 1 --seed 2` for variance.

## Integrity & drift

- `μ_Cohn` is computed **live** from the frozen snapshot — no hand-transcribed numbers.
- The fitness *is* the true objective recomputed from the returned points (shape, finite,
  unit-norm to `1e-12`); there is **no surrogate to game**.
- The Cohn catalogue can drift. Because `μ_Cohn` and the paper baselines are both evaluated
  on *this* frozen snapshot, the head-to-head is internally consistent regardless of any
  change since the paper accessed it (≈2026-05). The claim is "vs the current best-known".
