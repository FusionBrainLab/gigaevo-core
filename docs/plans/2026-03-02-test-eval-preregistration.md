# Test Evaluation Pre-Registration: P1×P2 Factorial (Runs E/F/G/H)

**Written**: 2026-03-02, ~19:45 UTC
**Experiment state at time of writing**: Generation 3 of 50. No test EM results exist for any program in this experiment.
**Commitment**: This document is committed to git BEFORE any test evaluation is run. The commit hash serves as the timestamp. Amendments must be documented with justification.

---

## 1. Scope

This pre-registration covers test evaluation for the P1×P2 factorial experiment:

| Run | Condition | Redis DB | Redis prefix |
|-----|-----------|----------|-------------|
| E (ctrl) | P1=OFF, P2=OFF | 10 | `chains/hotpotqa/static` |
| F | P1=OFF, P2=ON | 11 | `chains/hotpotqa/static_a` |
| G | P1=ON, P2=OFF | 12 | `chains/hotpotqa/static_r` |
| H | P1=ON, P2=ON | 13 | `chains/hotpotqa/static_ra` |

Test set: `HotpotQA_test.jsonl`, first 300 samples, same file used in all prior experiments.

---

## 2. Triggering Conditions

Test evaluation is triggered **exactly once**, when **all four runs have completed generation 50** (or crashed/stalled — whichever comes first for each run).

**No intermediate test evaluations** at gen 10, gen 25, or any other checkpoint. With n=1 per cell, intermediate evals create cherry-picking risk with no statistical benefit.

**Exception**: If a run crashes before gen 50 and cannot be restarted within 24 hours, test-evaluate it at its final generation. Document the crash generation. Do NOT exclude it from the factorial analysis.

---

## 3. Program Selection

For each of the four runs (E, F, G, H), select **exactly one program**:

> **The program with the highest val EM (fitness) across all generations, as returned by `gen10_test_eval.py --max-gen 9999`.**

This selects `max(fitness)` over all programs with `metadata['iteration'] <= max_gen`. No manual intervention in selection.

**Forbidden**:
- Selecting based on archive membership at gen 50
- Manual inspection of program code before selection
- Using any test EM knowledge to influence selection

---

## 4. Exact Commands

Run all four in sequence, using the **same chain servers as during evolution** (to minimize server-dependent variance):

```bash
export NO_PROXY="localhost,127.0.0.1,INTERNAL_IP,10.225.185.235,INTERNAL_IP,INTERNAL_IP,INTERNAL_IP,10.225.51.251,api.github.com"
export no_proxy="$NO_PROXY"

# Run E (ctrl)
HOTPOTQA_CHAIN_URL=http://localhost:8000/v1 \
  ~/envs/evo_fast/bin/python experiments/hotpotqa_thinking/gen10_test_eval.py \
  --run-label E --redis-db 10 --redis-prefix chains/hotpotqa/static --max-gen 9999 \
  | tee experiments/hotpotqa_p1p2/test_evals/test_eval_E.log

# Run F (P2 only)
HOTPOTQA_CHAIN_URL=http://localhost:8000/v1 \
  ~/envs/evo_fast/bin/python experiments/hotpotqa_thinking/gen10_test_eval.py \
  --run-label F --redis-db 11 --redis-prefix chains/hotpotqa/static_a --max-gen 9999 \
  | tee experiments/hotpotqa_p1p2/test_evals/test_eval_F.log

# Run G (P1 only)
HOTPOTQA_CHAIN_URL=http://10.225.185.235:8001/v1 \
  ~/envs/evo_fast/bin/python experiments/hotpotqa_thinking/gen10_test_eval.py \
  --run-label G --redis-db 12 --redis-prefix chains/hotpotqa/static_r --max-gen 9999 \
  | tee experiments/hotpotqa_p1p2/test_evals/test_eval_G.log

# Run H (P1+P2)
HOTPOTQA_CHAIN_URL=http://10.225.185.235:8000/v1 \
  ~/envs/evo_fast/bin/python experiments/hotpotqa_thinking/gen10_test_eval.py \
  --run-label H --redis-db 13 --redis-prefix chains/hotpotqa/static_ra --max-gen 9999 \
  | tee experiments/hotpotqa_p1p2/test_evals/test_eval_H.log
```

All four must use the **same git commit** of `gen10_test_eval.py`.

---

## 5. Calibration (Before Test Evals)

Before running any test evaluation, re-evaluate the **gen-0 seed program (ddce37b4)** on Run E's fixed-300 validation set to establish post-fix retest noise:

```bash
# One-time retest of seed on val set for noise calibration
HOTPOTQA_CHAIN_URL=http://localhost:8000/v1 \
  ~/envs/evo_fast/bin/python experiments/hotpotqa_thinking/gen10_test_eval.py \
  --run-label E_seed_retest --redis-db 10 --redis-prefix chains/hotpotqa/static --max-gen 0 \
  | tee experiments/hotpotqa_p1p2/test_evals/seed_retest_val.log
```

Compare result to the gen-0 val EM recorded in the run log. The delta is the post-fix single-sample retest noise floor. Report alongside all test EM claims.

This calibration measurement does NOT count as a test evaluation and does not violate this pre-registration (it uses the validation set, not the test set).

---

## 6. Recorded Outputs

Save to `experiments/hotpotqa_p1p2/test_evals/results.json`:

```json
{
  "preregistration_commit": "<git commit hash of this file>",
  "evaluation_date_utc": "<ISO8601>",
  "E": {
    "program_id": "<uuid>",
    "iteration": "<int>",
    "val_em": "<float>",
    "test_em": "<float>",
    "val_test_gap": "<float>",
    "extraction_failure_rate": "<float>",
    "n_test_samples": 300,
    "chain_url": "http://localhost:8000/v1",
    "timestamp_utc": "<ISO8601>"
  },
  "F": { ... },
  "G": { ... },
  "H": { ... },
  "seed_retest": {
    "val_em_original": "<float>",
    "val_em_retest": "<float>",
    "delta": "<float>"
  }
}
```

---

## 7. Analysis Plan

Compute **exactly** these quantities, in this order, using test EM only:

### 7.1 Raw test EMs
Report E_test, F_test, G_test, H_test with 95% CI (±5.49pp at p~0.60, n=300).

### 7.2 Val-test gaps
For each run: `val_em(best_program) - test_em(best_program)`.
For G and H (P1=ON): val_em is on a hash-seeded subset — report but note incomparability with E/F val EM.

### 7.3 Factorial main effects (on test EM only)
- **P1 main effect** = mean(G_test, H_test) − mean(E_test, F_test)
- **P2 main effect** = mean(F_test, H_test) − mean(E_test, G_test)
- **P1×P2 interaction** = (H_test − G_test) − (F_test − E_test)

### 7.4 Decision gates (pre-registered thresholds)
Apply these gates using test EM (NOT val EM):
- **P1 effective**: G_test ≥ E_test AND val-test gap of G/H < 5pp
- **P2 effective**: F_test ≥ E_test + 3pp
- **Ship H**: H_test ≥ 60% AND val-test gap of H < 5pp

### 7.5 Pairwise McNemar tests
For each pair (E, X) where X ∈ {F, G, H}: compute McNemar's test on the 300 per-sample binary correct/incorrect outcomes. Report χ² and p-value. This is the only valid paired statistical test at n=300.

Do NOT compute z-tests on proportions (samples are paired across runs via shared test questions).

---

## 8. Forbidden Actions

1. Do NOT run test evaluation on more than one program per run
2. Do NOT run any test evaluation before all four runs complete (or crash)
3. Do NOT use different versions of `gen10_test_eval.py` for different runs
4. Do NOT re-run a test evaluation because the result "seems wrong" — document anomalies and investigate after all four are recorded
5. Do NOT change this analysis plan after seeing test results
6. Do NOT report val EM comparisons between P1=ON and P1=OFF runs as evidence of P1's effect — only test EM is valid

---

## 9. Amendment Log

*No amendments as of initial commit.*
