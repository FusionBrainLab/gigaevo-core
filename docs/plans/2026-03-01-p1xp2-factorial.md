# Research Plan: P1 x P2 Factorial -- Validation Rotation and Retrieval Diagnostics

**Date**: 2026-03-01
**Investigator**: Dr. Elena Voss (ML Research Methodologist)
**Problem**: `problems/chains/hotpotqa/static/`
**Framework**: GigaEvo evolutionary computation
**Status**: Pre-registration (experiment design, before implementation)
**Builds on**: Thinking-mode 2x2 factorial (PR #66, 2026-03-01)

---

## 1. Research Question

**Can validation set rotation (P1) and per-hop retrieval diagnostics in mutation feedback (P2), alone or in combination, improve test EM beyond 59.3% on HotpotQA static chains with thinking-mode Qwen3-8B, within 50 generations?**

Subsidiary questions:
1. Does P1 reduce the val-test gap below the 5.0pp average observed in the prior experiment?
2. Does P2 improve mutation quality as measured by per-generation fitness gain?
3. Is there a P1 x P2 interaction (does debiased fitness amplify the value of retrieval diagnostics)?

## 2. Hypotheses

### H1 (P1 -- Validation Rotation)
- **H0_1**: Validation set rotation does not reduce the val-test gap (gap >= 4pp on average).
- **H1_1**: Validation set rotation reduces the val-test gap to < 3pp AND improves test EM.

### H2 (P2 -- Retrieval Diagnostics)
- **H0_2**: Adding per-hop gold document recall to mutation feedback does not improve test EM.
- **H1_2**: Retrieval diagnostics improve test EM by >= 3pp vs. control.

### H3 (Interaction)
- **H0_3**: Effects of P1 and P2 are additive (no interaction).
- **H1_3**: P1 and P2 interact positively: debiased fitness amplifies the value of better mutation signal.

## 3. Why a 2x2 Factorial (Not Sequential)

Running P1 then P2 sequentially (2+2 runs) answers 2 questions.
A 2x2 factorial uses 4 runs but answers 3 questions (P1, P2, interaction).

The interaction is worth measuring because there is a plausible mechanism: P2 gives the mutation LLM better diagnostic information about WHY failures occur (retrieval miss vs. reasoning error). But if the fitness signal is biased by overfitting to 300 fixed samples (no P1), the mutation LLM may optimize the right bottleneck for the wrong samples. P1 + P2 together could be substantially better than either alone.

## 4. Design Table

| Run | Label | P1 (Rotation) | P2 (Retrieval Dx) | Seed | DB | Mutation Server | Chain Server |
|-----|-------|---------------|--------------------|------|----|-----------------|--------------|
| E   | Control      | OFF | OFF | ddce37b4 | 10 | INTERNAL_IP:8777  | INTERNAL_IP:8001   |
| F   | P2-only      | OFF | ON  | ddce37b4 | 11 | INTERNAL_IP:8777   | INTERNAL_IP:8000   |
| G   | P1-only      | ON  | OFF | ddce37b4 | 12 | INTERNAL_IP:8777 | 10.225.185.235:8001 |
| H   | P1 + P2      | ON  | ON  | ddce37b4 | 13 | 10.225.51.251:8777  | 10.225.185.235:8000 |

### What Each Cell Tells Us

- **Run E (control)**: Replication of the best prior config (ddce37b4, default pipeline). Answers: is the prior 59.3% test EM (Run D) reproducible, or was it seed variance? Without this control, we cannot attribute improvements to P1 or P2.

- **Run F (P2 only)**: Isolates retrieval diagnostics on the overfitting-prone fixed validation set. If F > E on test, P2 helps even without debiasing. If F ~ E, the feedback was not actionable enough to overcome biased selection.

- **Run G (P1 only)**: Isolates debiased fitness. If G has smaller val-test gap AND higher test EM than E, P1 works as hypothesized. If G has smaller gap but LOWER test EM, rotation introduces too much selection noise.

- **Run H (P1 + P2)**: The combination. If H > max(F, G), there is synergy. If H ~ F + G - E (additive), no interaction. This is the candidate for the production config.

### Factorial Estimands

- Main effect of P1 = mean(G, H) - mean(E, F)
- Main effect of P2 = mean(F, H) - mean(E, G)
- Interaction = (H - G) - (F - E)

## 5. Seed Choice: ddce37b4 for All Runs

**Rationale**: The prior 2x2 already measured the seed factor (ddce37b4 vs. baseline):
- Seed effect on test EM: +1.3pp (small)
- From-scratch evolution (baseline seed): works but wastes 10-15 gens reaching the ddce37b4 vicinity
- ddce37b4 stagnated in Run B (49 gens, never improved) but reached 59.3% in Run D

Starting all 4 runs from ddce37b4 maximizes power in the region we care about (breaking through the 59-60% plateau) and eliminates seed as a confound.

## 6. Intervention Specifications

### P1: Validation Set Rotation

**Mechanism**: Each generation, validation uses a different deterministic 300-sample subset of the 1000 training samples, seeded by generation number.

**Implementation** (`validate_rotating.py`):
```python
import os, random
gen = int(os.environ.get("GIGAEVO_GENERATION", "0"))
rng = random.Random(42 + gen)  # deterministic per generation
all_samples = load_jsonl(DATASET_CONFIG["train_path"])
dataset = [preprocess_sample(s) for s in rng.sample(all_samples, 300)]
```

**Engine change** (one line in `core.py:step()`):
```python
os.environ["GIGAEVO_GENERATION"] = str(self.metrics.total_generations)
```

**Archive consistency**: Archive programs retain their original fitness from initial validation (InputHashCache prevents re-validation on refresh). New mutants in generation N are validated on subset_N. When a mutant competes for an archive cell, it is compared against the incumbent's fitness from a potentially different subset. The expected noise from cross-subset comparison is ~2.0pp std (comparable to the 2.4pp same-program retest noise already observed). This is acceptable; the benefit of preventing systematic overfitting to 300 fixed samples (5pp val-test gap) outweighs the 2pp cross-subset noise.

**Key property**: All 16 mutants within the same generation are evaluated on the SAME 300 samples (same generation seed), so within-generation ranking is exact.

### P2: Per-Hop Retrieval Diagnostics (ASI-lite)

**Mechanism**: After chain execution, compare BM25 retrieval results at steps 1 and 4 against gold `supporting_facts.title` from the dataset. Report per-hop recall and missed gold documents in the failure artifact.

**What this changes**: Currently, the failure artifact contains:
```python
{"question": ..., "gold": ..., "predicted": ...}
```

P2 extends it to:
```python
{
    "question": ...,
    "gold": ...,
    "predicted": ...,
    "hop1_recall": 0.5,   # fraction of gold docs retrieved at step 1
    "hop2_recall": 1.0,   # fraction of gold docs retrieved at step 4
    "hop1_missed": ["Sacramento International Airport"],
    "hop2_missed": [],
}
```

**Implementation** (modify `validate.py` or `validate_asi.py`):
1. Preserve `supporting_facts` in `preprocess_sample()` (currently stripped)
2. After chain execution, parse `step_outputs[0]` and `step_outputs[3]` (the retrieval results) to extract retrieved document titles
3. Compare against `sample["supporting_facts"]["title"]`
4. Add recall metrics to each failure case in the artifact

**Formatter**: Use `HotpotQAFailureFormatter` (reflective pipeline) extended to render retrieval diagnostics:
```
### Case 1
**Question**: Which airport is closer to the island...
**Expected**: Knox County Regional Airport
**Predicted**: Sacramento International Airport
**Hop 1 Retrieval**: 1/2 gold docs found (missed: Sacramento International Airport)
**Hop 2 Retrieval**: 2/2 gold docs found
**Diagnosis**: Retrieval bottleneck at hop 1 -- query generation prompt should be improved
```

**Pipeline for P2 runs**: `pipeline=hotpotqa_reflective` with an enhanced formatter. Since the prior experiment showed the reflective pipeline alone had zero effect, any improvement can be attributed to the enriched content.

**Pipeline for non-P2 runs**: Default pipeline (custom.yaml). This matches the control condition from the prior experiment.

## 7. Controlled Variables

Held constant across all 4 runs:
- Seed program: ddce37b4
- max_generations: 50
- max_mutations_per_generation: 16
- max_elites_per_generation: 8
- num_parents: 1
- primary_resolution: 50
- Chain LLM: Qwen3-8B (thinking mode, temp=0.6, top_p=0.95, top_k=20)
- Mutation LLM: Qwen3-235B-A22B-Thinking-2507
- step_max_tokens: {2: 4096, 3: 2048, 5: 4096, 6: 2048}
- Test set: 300 held-out samples (HotpotQA_test.jsonl)
- Redis: fresh DBs (10-13), no resume
- Mutation prompt template: same for all runs (user.txt unchanged)

**Potential confounds**:
- Server heterogeneity: mitigated by using the same 4-server allocation as prior experiment (known comparable)
- Chain server pairing: E/F share server INTERNAL_IP, G/H share 10.225.185.235. If servers differ, this confounds with P1. Mitigated by verified equivalence in prior experiment.
- LLM non-determinism: irreducible. Accounted for via 2.4pp noise floor.

## 8. Metrics

### Primary
- **Test EM at generation 50** (300 held-out test samples)

### Secondary
- Val-test gap at gen 10, 25, 50 (critical for P1 assessment)
- Val EM trajectory per generation (from logs)
- Per-generation acceptance rate (fraction of 16 mutants entering archive)
- Archive size at gen 10, 25, 50
- Mean hop-1 and hop-2 retrieval recall per generation (P2 runs only)
- Extraction failure rate (should be < 5%)

### Test Evaluation Protocol
Extract best-by-val program at generations 10, 25, and 50. Run test.py with 300 samples. Record: test EM, extraction failures, wall-clock time.

Total test evaluations: 4 runs x 3 checkpoints = 12 evals (~60 min total, marginal).

## 9. Sample Size and Statistical Power

With N=1 per cell, classical hypothesis tests have inadequate power. We address this by:

1. **Pre-registering effect size thresholds** calibrated to the noise floor (2.4pp from Run B same-program retest):
   - Effects < 2.4pp: indistinguishable from noise
   - Effects 2.4-5pp: suggestive, worth replicating
   - Effects > 5pp: likely real

2. **Using factorial estimands** that pool across cells (each main effect estimate uses 2 data points)

3. **External calibration**: 4 prior runs (A-D) provide reference distribution of test EM under different conditions

4. **No p-hacking**: All thresholds defined before seeing results. Analysis plan is fixed.

## 10. Decision Gates (Pre-registered)

### Gate 1: P1 (Rotation)
- **EFFECTIVE** if: val-test gap for G, H averages < 3pp (vs. prior 5pp) AND test EM for G, H >= test EM for E, F
- **HARMFUL** if: test EM decreases by >= 3pp despite gap reduction (noise overwhelms signal)
- **NULL** if: neither gap reduction nor test EM change exceeds noise floor

### Gate 2: P2 (Retrieval Diagnostics)
- **EFFECTIVE** if: mean test EM for F, H exceeds mean test EM for E, G by >= 3pp
- **INCONCLUSIVE** if: effect is 1-3pp
- **NULL** if: no difference

### Gate 3: Combination
- **SHIP** if: Run H test EM >= 60% AND val-test gap < 3pp
- **STRONG SHIP** if: Run H test EM >= 62.3% (GEPA target)
- **ITERATE** if: Run H does not outperform Run E (control)

### Gate 4: Next Steps
| Outcome | Next batch |
|---------|------------|
| P1 works, P2 works | 3 replications of P1+P2 for statistical significance |
| P1 works, P2 null | P1 + alternative mutation improvements (crossover, larger LLM) |
| P1 null, P2 works | P2 + larger val set (all 1000 train, accept slower gens) |
| Both null | Deeper intervention (crossover/merge, full chain topology, accept ceiling) |

## 11. Implementation Plan

### Code Changes

**P1 -- Validation rotation** (2-3 hours):
1. Add `os.environ["GIGAEVO_GENERATION"] = str(self.metrics.total_generations)` to `gigaevo/evolution/engine/core.py:step()` (one line, before phase 2)
2. Create `problems/chains/hotpotqa/static/validate_rotating.py`: copy of validate.py that reads `GIGAEVO_GENERATION` env var and uses `random.Random(42 + gen).sample(all_samples, 300)` instead of `all_samples[:300]`
3. Create pipeline config `config/pipeline/hotpotqa_rotating.yaml` (copy of custom.yaml with path override to validate_rotating.py) -- OR use problem_dir approach

**P2 -- Retrieval diagnostics** (3-4 hours):
1. Modify `preprocess_sample()` in `shared_config.py` to preserve `supporting_facts`
2. Create `validate_asi.py`: extended validate.py that computes per-hop retrieval recall from `ChainResult.step_outputs` and `sample["supporting_facts"]["title"]`, and includes it in the failure artifact
3. Create `HotpotQAASIFormatter` (subclass of HotpotQAFailureFormatter) that renders retrieval diagnostics
4. Create `ReflectiveASIPipelineBuilder` (or pipeline YAML) that uses the ASI formatter

**P1 + P2 (Run H)**: Create `validate_rotating_asi.py` combining both changes.

**Total implementation**: 4-6 hours.

### Validator Files Summary

| Run | Validator file | What it does |
|-----|---------------|-------------|
| E | validate.py (current) | Fixed 300 samples, basic failure cases |
| F | validate_asi.py | Fixed 300 samples, failure cases + retrieval recall |
| G | validate_rotating.py | Rotating 300/1000 samples, basic failure cases |
| H | validate_rotating_asi.py | Rotating 300/1000 samples, failure cases + retrieval recall |

### Pipeline Summary

| Run | Pipeline | Formatter |
|-----|----------|-----------|
| E | custom (default) | FormatterStage (generic) |
| F | hotpotqa_reflective_asi | HotpotQAASIFormatter (retrieval-aware) |
| G | custom (default) | FormatterStage (generic) |
| H | hotpotqa_reflective_asi | HotpotQAASIFormatter (retrieval-aware) |

## 12. Timeline

| Hour | Action |
|------|--------|
| 0-4 | Implement P1 (rotation) and P2 (ASI diagnostics). Unit test on 10 samples. |
| 4-5 | Preflight: flush Redis DBs 10-13, verify servers, create seed dirs, update launch script |
| 5 | Launch all 4 runs simultaneously |
| 10 (~gen 10) | Checkpoint 1: acceptance rates, gen-10 best programs, test evals (4 x ~5 min) |
| 17 (~gen 25) | Checkpoint 2: test evals, val-test gap analysis, retrieval recall stats |
| 24-28 (~gen 50) | Final: test evals for all 4 runs, full analysis |
| 28-32 | Write-up, decision on follow-up |

## 13. Risk Register

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| P1 rotation noise overwhelms signal | Archive quality degrades; test EM drops | MEDIUM | 2pp cross-subset noise is comparable to existing 2.4pp noise floor. Monitor archive churn rate. |
| P2 diagnostics too verbose for mutation LLM | Mutation quality drops; token budget wasted on retrieval info | LOW-MEDIUM | Limit to top-10 failures with retrieval summary. Format is compact (~3 extra lines per case). |
| Supporting_facts parsing errors | Retrieval recall computation fails; P2 artifacts corrupted | LOW | Unit test on 100 samples before launch. Gold docs use title-matching (fuzzy not needed). |
| Run E (control) far from prior 59.3% | Cannot attribute P1/P2 effects vs. seed variance | MEDIUM | Expected range based on prior 4 runs: 56-60% test EM. If E < 54%, investigate. |
| Archive stagnation (0% acceptance 15+ gens) | Insufficient evolution; cannot measure intervention effects | MEDIUM | Monitor acceptance rate. If stagnation at gen 15, the intervention is not breaking the plateau -- informative negative. |
| Chain server crash | Lost progress for affected run | MEDIUM | Monitor hourly. If crash before gen 25, restart fresh on same DB. |

## 14. Alternatives Considered and Rejected

**P1 with program-hash seeding** (each program sees its own fixed 300 subset): Rejected because programs within the same generation would be evaluated on different samples, making within-generation comparison noisy. Generation-based seeding keeps intra-generation comparison exact.

**P1 with archive re-validation** (re-evaluate incumbents each generation on current subset): Rejected because it approximately doubles validation time per generation (~100-150 min extra), making 50 gens infeasible in 24 hours.

**P2 with full chain-of-thought audit** (LLM inspects each reasoning step): Rejected for cost -- requires an extra LLM call per failure case. Retrieval recall is computable from existing outputs at zero additional cost.

**Running 4 replications of P1 only** (ignore P2): Wastes opportunity to study P2. Factorial gives same P1 signal with 2 additional data points.

**Including crossover/merge as a factor**: Rejected by researcher preference (too close to GEPA+Merge). Also changes evolutionary dynamics (num_parents=2).

**Reducing to 20-gen runs for 2 seeds x P1 x P2 (8-cell design)**: Rejected. 20 gens is insufficient for plateau-breaking effects. Prior experiment showed most improvement occurs after gen 15.

## 15. Appendix: Dataset Schema for P2

HotpotQA train samples include:
```json
{
  "id": "...",
  "question": "...",
  "answer": "...",
  "type": "bridge",
  "level": "hard",
  "supporting_facts": {
    "title": ["Sacramento International Airport", "Knox County Regional Airport"],
    "sent_id": [0, 0]
  },
  "context": {
    "title": ["Vinalhaven, Maine", ...],
    "sentences": [["...", "..."], ...]
  }
}
```

Gold documents: `supporting_facts["title"]` -- typically 2 titles for bridge questions.

---

## PROTOCOL AMENDMENT — 2026-03-02 (post-launch, pre-results)

**Amendment author**: Dr. Elena Voss
**Trigger**: Reviewer 2 audit identified discrepancy between pre-registered P1 mechanism and actual implementation.

### Deviation from pre-registered P1 design

**Pre-registered** (Section 6, line 80): Generation-based seeding — all programs in generation N share the same 300-sample subset, seeded by `random.Random(42 + gen)`.

**Actually implemented** (`problems/chains/hotpotqa/static_r/validate.py`, `static_ra/validate.py`): Hash-based seeding — each unique chain_spec is permanently assigned a fixed 300-sample subset via `random.Random(SHA256(chain_spec)[:16] % 2**32).sample(all_1000, 300)`.

**Pre-registered rejection of hash seeding** (Section 14, line 284): "Rejected because programs within the same generation would be evaluated on different samples, making within-generation comparison noisy."

### Rationale for deviation

The pre-registration rejection was based on a within-generation tournament selection mental model that does not apply to GigaEvo's MAP-Elites archive admission mechanism. Archive admission in GigaEvo compares a new mutant against the **cell incumbent** (not against other mutants in the same generation). Within-generation comparison is irrelevant to archive admission decisions. The rejection rationale was therefore incorrect for this system.

Hash-based seeding was chosen because:
1. **Refresh stability**: Archive incumbents are re-validated on refresh (DONE → QUEUED). Hash seeding ensures a program always sees the same subset on refresh (chain_spec is unchanged), keeping incumbent fitness stable. Generation seeding would change the incumbent's fitness each generation refresh, creating archive instability.
2. **Training signal diversity**: Different archive programs have been evaluated on different 300-sample subsets, covering a broader range of failure modes and diversifying mutation LLM feedback across the archive.

### Residual concern

Hash-based seeding means cross-program fitness comparisons (mutant vs. incumbent) involve different 300-sample subsets. The associated cross-subset noise is ~2.0pp std. Combined with LLM retest noise (2.4pp), pairwise archive admission noise is ~3.1pp. This is a genuine trade-off, not eliminated by the hash-based approach.

Cross-run val EM comparisons (e.g., E=67.0% vs. G=67.3%) are on different evaluation subsets for P1-ON runs and should not be treated as directly comparable without re-evaluation on a common set.

### Impact on pre-registered hypotheses

H1_1 (P1 reduces val-test gap to <3pp AND improves test EM) remains the primary hypothesis. The mechanism differs from pre-registration (hash-based subset diversification rather than generation-based rotation) but the anti-overfitting intent is the same. The val-test gap measurement at gen 50 remains the primary outcome measure and is unaffected by the seeding mechanism difference (test set is always the same fixed 300 samples).

### Documentation failure acknowledgment

This deviation should have been documented before launch. The plan was not updated to reflect the implementation decision. All future experiment plans require a code-vs-plan diff review before launch to prevent undocumented deviations.

Retrieval step outputs (steps 1, 4): formatted as `"[1] Title | Text\n[2] Title | Text\n..."` with k=7 documents each.

Title extraction from retrieval output: `re.findall(r"\[\d+\] (.+?) \|", step_output)`.

Recall computation: `len(set(gold_titles) & set(retrieved_titles)) / len(set(gold_titles))`.
