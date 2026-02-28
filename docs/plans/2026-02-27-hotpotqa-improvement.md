# Research Plan: Pushing HotpotQA Static EM Beyond 62.7%

**Date**: 2026-02-27
**Investigator**: Dr. Elena Voss (ML Research Methodologist)
**Problem**: `problems/chains/hotpotqa/static/`
**Framework**: GigaEvo evolutionary computation
**Codebase**: `/workspace-SR008.fs2/mathemage/gigaevo-core`

---

## 1. Research Question

Can we improve the HotpotQA Exact Match (EM) score on the held-out test set beyond 62.7% validation EM (and GEPA's 62.3 EM) through targeted improvements to GigaEvo's evolutionary loop, validation efficiency, and mutation quality?

### Current State

**⚠️ IMPORTANT**: All results must use **non-thinking Qwen3-8B** to match GEPA's evaluation protocol. Previous GigaEvo results (db=5, program ddce37b4, 62.7% val EM) were collected with thinking **enabled** — those numbers are invalid for comparison with GEPA.

| Method | Model | Val EM | Test EM | Thinking | Notes |
|--------|-------|--------|---------|----------|-------|
| GEPA | Qwen3-8B | ? | 62.3% | ❌ disabled | ICLR 2026 Oral |
| MIPROv2 | Qwen3-8B | ? | 55.3% | ❌ disabled | DSPy |
| GRPO | Qwen3-8B | ? | 43.3% | ❌ disabled | RL |
| Baseline | Qwen3-8B | ? | 42.3% | ❌ disabled | Unoptimized |
| GigaEvo (db=5, ddce37b4) | Qwen3-8B | ~~62.7%~~ | ~~55.3%~~ | ✅ enabled | **INVALID** — thinking on |
| GigaEvo ddce37b4 (non-think eval) | Qwen3-8B | 53.3% | **52.3%** | ❌ disabled | Thinking-optimized program, ~10pp gap to GEPA |

**Non-thinking server commands** (see memory/experiments.md for full commands):
- Port 8001: `--chat-template ./qwen3_nonthinking.jinja --max-model-len 16384`
- Port 8000: same template, `--max-model-len 40960`

### Compute Budget
- 3 parallel evolution slots (one Qwen3-235B mutation server each)
- 2 shared chain-execution vLLM endpoints (Qwen3-8B)
- Timeline: ~1 week (7 days)
- At ~21 min per validation (pre-optimization), ~8 mutants/gen: ~3h per generation
- With uncommitted efficiency improvements, validation may drop to ~5-10 min
- Estimated throughput: ~6-8 generations/day/slot with current settings

---

## 2. Pre-Experiment Baseline Establishment (Experiment 0)

### 2.0 Hypothesis
- Null: The current best program (ddce37b4) achieves the same EM on the test set as on the validation set (within noise).
- Alternative: Test EM differs meaningfully from validation EM (either higher or lower), indicating overfitting or underfitting.

### 2.0 Protocol
Before running any evolutionary experiments, establish the definitive baseline:

1. **Commit and apply the uncommitted efficiency changes** (step-batched execution, batch BM25, increased concurrency, reduced max_tokens). These are infrastructure improvements that do not change program behavior.

2. **Run the best program (ddce37b4) on the test set** (300 samples) using `problems/chains/hotpotqa/static/test.py`. Record test EM.

3. **Run the baseline program on both val and test** to establish the null-optimization reference point.

4. **Measure validation speed** with the new step-batched execution. The current 21-min estimate was pre-optimization. The speed improvement determines how many generations we can run in one week.

5. **Run 3 independent evaluations** of the best program on validation (same 300 samples) to measure intra-run variance. LLM outputs are stochastic (temperature=0.6), so repeated evaluations of the same program will yield different EM scores. This variance is the noise floor for all comparisons.

### 2.0 Deliverables
- Baseline table: {program, val_EM, test_EM, val_EM_std (3 runs), wall_clock_time}
- Decision: is 62.7% val EM reproducible? What is the test gap?

**⚠️ Results collected 2026-02-27 (INVALID — port 8001 was running with thinking ENABLED)**:
- ~~Test EM: 55.33%, Val runs: 61.67% / 63.00%, Evolution score: 62.67%~~
- These are inflated by Qwen3 chain-of-thought reasoning; GEPA uses non-thinking
- Port 8001 restarted with `qwen3_nonthinking.jinja` — all results below are fresh

**Fresh baseline (non-thinking, ddce37b4 — was optimized with thinking, expect lower EM)**:
| Run | Split | EM | Extraction failures | Wall-clock | Server |
|-----|-------|------------|---------------------|------------|--------|
| 1 | test | **52.33%** | 0.00% | 2.1 min | 8001 |
| 1 | val | **53.33%** | 0.00% | 2.1 min | 8000 |
| 2 | val | **51.00%** | 0.00% | 2.0 min | 8001 |
| 3 | val | **52.33%** | 0.00% | 2.1 min | 8000 |

**Key findings**:
- Val EM mean=**52.22%**, std=**1.37pp → PASS** (< 3pp threshold)
- Val ≈ Test (52.22% vs 52.33%, Δ≈0pp) — zero overfitting with non-thinking model
- 0% extraction failures across all runs — cleaner outputs, no thinking-token truncation
- ddce37b4 at ~52% is 10pp below GEPA (62.3%) — expected, prompts tuned for thinking mode
- **Efficiency: ~2.1 min / 300 samples** (3× faster than thinking, ~10× faster than pre-optimization estimate)
- Both servers (8000, 8001) give consistent results in non-thinking mode

**Next**: fresh evolution run on new Redis DB with non-thinking servers. Target: close the 10pp gap to GEPA (62.3%).

### 2.0 Pass/Fail
- PASS if variance across 3 runs is < 3 percentage points. If higher, all subsequent experiments need more seeds.
- The test EM provides the true baseline for comparison with GEPA's 62.3.

---

## 3. Experiment 1: Validation Efficiency + Test Metric Logging

### Priority: HIGHEST (enables all subsequent experiments)
### Branch: `exp/hotpotqa-fast-val`

### 3.1 Research Question
Can we reduce per-program validation time without significantly biasing fitness estimates, and can we log test EM alongside validation EM during evolution?

### 3.2 Hypotheses
- Null: Reducing validation samples from 300 to 100 (random subset) introduces bias that changes which programs are selected as elites.
- Alternative: The 100-sample proxy correlates strongly (r > 0.9) with the 300-sample fitness, and the best programs found are the same or better.

### 3.3 What to Change

**3.3a Fast validation (validate2.py integration)**
The uncommitted `validate2.py` already implements:
- 100 random samples (vs. 300)
- Early stopping: abort if EM < baseline (0.44) after first 30 samples
- Same step-batched execution

Change the default pipeline to use `validate2.py` for the main evolution loop, then run the full `validate.py` (300 samples) as a secondary "test" stage that logs metrics but does NOT affect selection.

**3.3b Test EM logging**
Add a secondary validation stage to the DAG pipeline that runs `test.py` on the 300-sample held-out test set. This stage runs AFTER the main validation and insights stages. Its metrics are logged to TensorBoard/Redis under `test_em` but are NOT included in the fitness used for archive selection.

Implementation approach:
- Add a new `CallTestFunction` stage to the pipeline builder
- Wire it as an execution-order dependency after `EnsureMetricsStage`
- Output goes to a new `test_metrics` entry in the program's metrics dict
- Archive selection continues to use `fitness` from `validate.py`/`validate2.py`

**3.3c Timing measurement**
Instrument `validate.py` and `validate2.py` with wall-clock timing. Log per-step latency (BM25, LLM step 2, step 3, step 4, step 5, step 6) to identify the bottleneck.

### 3.4 Experimental Conditions

```
Condition A (control): Standard validate.py (300 samples), no test logging
  python run.py problem.name=chains/hotpotqa/static redis.db=1

Condition B (fast val): validate2.py (100 samples + early stop), no test logging
  python run.py problem.name=chains/hotpotqa/static redis.db=2 \
    pipeline=optuna_opt  # uses validate2.py

Condition C (fast val + test): validate2.py + test_em logged
  python run.py problem.name=chains/hotpotqa/static redis.db=3 \
    pipeline=<new_pipeline_with_test_logging>
```

But since we have only 3 slots and this is an infrastructure experiment, we can run A vs B for 10 generations each and compare:
- Correlation between fast-val fitness and full-val fitness
- Whether the same top programs are identified
- Wall-clock time per generation

### 3.5 Metrics
- Primary: Rank correlation (Spearman's rho) between 100-sample and 300-sample EM across all programs in the archive
- Secondary: Wall-clock time per generation, number of programs evaluated per hour
- Tertiary: Test EM of top program found by each condition

### 3.6 Sample Size
- 10 generations per condition (sufficient to produce ~50-80 evaluated programs)
- 1 run per condition (this is an infrastructure test, not a performance comparison)

### 3.7 Pass/Fail
- PASS if Spearman's rho > 0.85 between fast-val and full-val fitness rankings
- PASS if fast-val achieves >2x speedup in wall-clock time per generation
- Test EM logging is a pure instrumentation change — no pass/fail, just verify it works

---

## 4. Experiment 2: Reducing Mutant Rejection Rate

### Priority: HIGH (addresses the 70% waste)
### Branch: `exp/hotpotqa-rejection-rate`
### Dependency: Experiment 0 (baseline), optionally Experiment 1 (faster iteration)

### 4.1 Research Question
Does reducing the mutant rejection rate (currently ~70%) increase the rate of fitness improvement, controlling for wall-clock time?

### 4.2 Background
The single-island MAP-Elites archive bins programs by fitness with 150 bins on [0, 1]. A mutant is accepted only if its cell is empty or it beats the current occupant. With 60 programs in 150 bins, ~40% of bins are occupied. A mutant must exceed the occupant's fitness to take its place. With incremental improvements, most mutants fall into an occupied bin with a competitive incumbent.

This is a fundamental mismatch: the behavior space is 1D (fitness itself), so the archive is effectively a sorted list where only the best-per-bin survives. This is not exploring diverse strategies — it is just keeping a sliding window of top performers.

### 4.3 Hypotheses
- Null: Reducing the behavior space resolution (fewer bins) does not change the rate of fitness improvement per generation.
- Alternative: Coarser binning (e.g., 20 bins instead of 150) allows more mutants to enter the archive, increasing diversity and the probability of escaping local optima.

### 4.4 What to Change

**4.4a Coarser binning**
Reduce `primary_resolution` from 150 to 20. This gives 20 fitness bins over [0, 1], so each bin covers 0.05 EM range. With 60 programs max in 75-slot archive, most bins will have room for new programs.

**4.4b Extraction failure as second behavior dimension (optional sub-experiment)**
Add `avg_extraction_failures` as a second behavior dimension. This creates a 2D behavior space (fitness x extraction_failures), dramatically increasing the number of cells and allowing programs with different failure modes to coexist.

### 4.5 Experimental Conditions

```
Condition A (baseline): primary_resolution=150 (current)
  python run.py problem.name=chains/hotpotqa/static redis.db=4 \
    primary_resolution=150

Condition B (coarse): primary_resolution=20
  python run.py problem.name=chains/hotpotqa/static redis.db=5 \
    primary_resolution=20

Condition C (2D space): primary_resolution=20, add extraction_failures dimension
  python run.py problem.name=chains/hotpotqa/static redis.db=6 \
    primary_resolution=20 \
    +behavior_space.keys=[fitness,avg_extraction_failures] \
    +behavior_space.resolutions=[20,5]
```

### 4.6 Metrics
- Primary: Best validation EM after 10 generations
- Secondary: Acceptance rate (% of mutants that enter archive), archive diversity (number of unique fitness bins occupied), number of archive replacements per generation
- Tertiary: Test EM of best program

### 4.7 Sample Size
- 10 generations per condition, 1 seed per condition
- Compare conditions at equal generation count (not wall-clock, since acceptance rate affects generation time)

### 4.8 Statistical Analysis
- At 10 generations with 8 mutants each, we expect ~80 evaluated programs per condition
- Comparison is descriptive (fitness curves + acceptance rates) rather than inferential — sample size is too small for formal testing
- The key signal is whether the fitness curve is still rising at generation 10 (vs. plateauing)

### 4.9 Pass/Fail
- PASS if coarser binning (B or C) achieves higher best EM or visibly steeper fitness curve
- PASS if acceptance rate increases from ~30% to >60%
- FAIL if coarser binning leads to archive quality degradation (best EM drops)

---

## 5. Experiment 3: GEPA-Inspired Reflective Mutation

### Priority: MEDIUM-HIGH (addresses mutation quality, the key GEPA advantage)
### Branch: `exp/hotpotqa-reflective-mutation`
### Dependency: Experiment 0 (baseline), Experiment 1 (test logging)

### 5.1 Research Question
Does providing per-sample failure feedback to the mutation LLM (inspired by GEPA's reflective mechanism) improve the quality of mutations compared to the current generic mutation prompt?

### 5.2 Background
GEPA's core advantage is its reflection mechanism: it captures full execution traces, identifies which samples failed and why, and uses this information to propose targeted prompt improvements. GigaEvo's current mutation prompt provides:
- The parent program's code
- Insights (LLM-generated, generic)
- Lineage history (what past mutations achieved)
- Evolutionary statistics

But it does NOT provide:
- Which specific samples the parent got wrong
- What the parent predicted vs. the correct answer
- What the intermediate step outputs looked like for failed cases

### 5.3 Hypotheses
- Null: Adding per-sample failure information to the mutation context does not increase the fraction of mutations that improve fitness.
- Alternative: Failure-aware mutations are more targeted and produce a higher fraction of fitness-improving children.

### 5.4 What to Change

The plumbing for reflective feedback is **already fully wired** in the default pipeline. No framework changes are needed — only problem-level additions.

**5.4a Capture failure details in `validate.py`**
`CallValidatorFunction.parse_output()` already handles `(metrics, artifact)` tuples:
```python
# validate.py: change return to include artifact
failures = [
    {"question": q, "gold": gold, "predicted": pred, "step_outputs": steps}
    for q, gold, pred, steps in failed_cases[:10]
]
return ({"fitness": em, ...}, failures)
```
`FetchArtifact` (already in DAG) extracts `data[1]` and passes it to `FormatterStage`.

**5.4b Create `HotpotQAFailureFormatter` in the problem dir**
Subclass `FormatterStage` and override `format_value()` to render failures as a clean markdown block for the mutation LLM:
```python
# problems/chains/hotpotqa/static/formatter.py
class HotpotQAFailureFormatter(FormatterStage):
    def format_value(self, failures: list[dict]) -> str:
        lines = [f"## Failure Analysis ({len(failures)} failed samples)\n"]
        for i, f in enumerate(failures[:5], 1):
            lines.append(f"### Case {i}")
            lines.append(f"Q: {f['question']}")
            lines.append(f"Expected: {f['gold']}")
            lines.append(f"Predicted: {f['predicted']}")
            if f.get("step_outputs"):
                lines.append(f"Step outputs: {f['step_outputs']}")
        return "\n".join(lines)
```

**5.4c Wire the custom formatter in the pipeline**
The problem's pipeline builder replaces the default `FormatterStage` with `HotpotQAFailureFormatter`. The `FormatterStage → MutationContextStage` data-flow edge is already present; `MutationContextStage` already appends `PreformattedMutationContext` from the `formatted` input. No further wiring needed.

`FormatterStage` is currently always skipped (artifact=None). Once artifact is non-None, the full `FetchArtifact → FormatterStage → MutationContextStage` chain activates automatically.

### 5.5 Experimental Conditions

```
Condition A (baseline): Current mutation prompts, num_parents=2
  python run.py problem.name=chains/hotpotqa/static redis.db=7

Condition B (failure context): Failure examples in prompt, num_parents=2
  python run.py problem.name=chains/hotpotqa/static redis.db=8 \
    pipeline=<reflective_pipeline>

Condition C (failure context + reduced parents): Failure examples, num_parents=1
  python run.py problem.name=chains/hotpotqa/static redis.db=9 \
    pipeline=<reflective_pipeline> \
    algorithm.evolution.num_parents=1
```

**Rationale for Condition C**: Adding failure feedback significantly increases mutation prompt complexity (5 failure cases × ~200 tokens each ≈ +1000 tokens). With `num_parents=2`, the mutation LLM must reconcile two potentially divergent parent programs while also incorporating failure feedback — a more complex cognitive task. `num_parents=1` reduces this complexity, potentially allowing the LLM to focus more on the failure signal. Note: this tests **prompt complexity reduction**, not literal attention dilution — a 235B thinking model has ample capacity for both parents, but simpler prompts may still produce more targeted mutations. C vs B isolates the `num_parents` effect in the presence of failure context.

### 5.6 Metrics
- Primary: Best validation EM after 10 generations
- Secondary: Fraction of mutations that improve over parent fitness, average fitness delta of accepted mutations
- Tertiary: Qualitative analysis — do the mutation LLM's justifications reference the failure examples?
- Quaternary: Mutation prompt token count (track context bloat; C should be ~1000 tokens shorter than B)

### 5.7 Risks and Mitigations
- Risk: Failure examples may be too noisy (stochastic LLM outputs) to provide useful signal
  - Mitigation: Only include failures where the prediction was clearly wrong (not close misses)
- Risk: Longer mutation prompts (Condition B) may dilute the failure signal with two parent programs
  - Mitigation: Condition C tests exactly this — if C > B, reduce to num_parents=1 in future runs
- Risk: num_parents=1 reduces genetic recombination, potentially narrowing search
  - Mitigation: Compare acceptance rates and archive diversity between B and C, not just best EM
- Risk: Implementation complexity in modifying the pipeline
  - Mitigation: Start with the lighter-weight option (5.4c) if time is limited

### 5.8 Pass/Fail

**Pairwise comparison priority** (most to least important):
1. **A vs B** (primary): Does failure context help at all? This is the core research question.
2. **B vs C** (secondary): Does reducing `num_parents` improve failure-context-driven mutation?
3. **A vs C** (exploratory): Confounded (two simultaneous changes); do not draw strong conclusions.

**Thresholds** (pilot study, N=1 per condition — conclusions are directional only):
- PASS if mutation success rate (fraction improving over parent) increases by >5 pp (A vs B)
- PASS if best EM after 10 generations is higher than control by **>3 pp** (raised from 1 pp given 1.4 pp intra-run variance at N=1)
- Qualitative PASS if mutation justifications show evidence of using failure information

**Planned follow-up** (not in this pilot): If Condition C beats B by >3 pp, run a Condition D (num_parents=1, **no** failure context) to disentangle the `num_parents` effect from the failure-context effect. This 2×2 factorial is the correct design but requires a 4th slot not available now.

**Note**: Any conclusion from this 3-condition pilot at N=1 × 10 gens is directional only. Adoption of reflective mutation requires replication with N≥3 seeds.

---

## 6. Experiment 4: Increased Mutation Budget per Generation

### Priority: MEDIUM (simple parameter change, may help if not saturated)
### Branch: `exp/hotpotqa-more-mutants`
### Dependency: Experiment 1 (faster validation is prerequisite for more mutants)

### 6.1 Research Question
Does increasing the number of mutations per generation (from 8 to 16 or 24) improve the rate of fitness gain, given the reduced validation time?

### 6.2 Hypotheses
- Null: Doubling mutations per generation does not change the best EM after equal wall-clock time.
- Alternative: More mutations per generation explore the search space faster and find better prompts.

### 6.3 What to Change
```yaml
max_mutations_per_generation: 16  # or 24
max_elites_per_generation: 8      # select more parents
```

### 6.4 Experimental Conditions
```
Condition A: 8 mutations/gen (current)
Condition B: 16 mutations/gen, 8 elites/gen
Condition C: 24 mutations/gen, 10 elites/gen
```

Run for equal wall-clock time (e.g., 24 hours each) and compare best EM.

### 6.5 Metrics
- Primary: Best validation EM after 24 hours
- Secondary: Programs evaluated per hour, fraction accepted, fitness curve slope

### 6.6 Pass/Fail
- PASS if more mutants produce higher best EM in equal wall-clock time
- This experiment is only viable if Experiment 1 succeeds in reducing validation time

---

## 7. Execution Schedule

Given 3 parallel slots and ~7 days, prioritize as follows:

### Day 1: Baseline + Infrastructure (Experiment 0 + 1)
- **Slot 1**: Commit efficiency changes. Run baseline evaluations (Experiment 0): test EM of best program, validation variance measurement (3 runs).
- **Slot 2**: Implement fast validation pipeline integration (Experiment 1). Run A vs B comparison.
- **Slot 3**: Implement test EM logging. Validate instrumentation.

### Days 2-3: Archive Diversity (Experiment 2)
- **Slot 1**: Run Experiment 2 Condition A (baseline resolution, 10 gens)
- **Slot 2**: Run Experiment 2 Condition B (coarse resolution, 10 gens)
- **Slot 3**: Run Experiment 2 Condition C (2D behavior space, 10 gens)
- Analyze results at end of Day 3.

### Days 4-5: Reflective Mutation (Experiment 3)
- Implement failure context capture (code changes)
- **Slot 1**: Run Experiment 3 Condition A (baseline mutation, 10 gens)
- **Slot 2**: Run Experiment 3 Condition B (reflective mutation, 10 gens)
- **Slot 3**: Run best config from Experiment 2 for extended run (20+ gens)

### Days 6-7: Extended Runs + Mutation Budget (Experiment 4)
- Run best configuration from all prior experiments for extended generation count
- If fast validation worked: try Experiment 4 (more mutants per generation)
- Final test set evaluation of all top programs
- Write up results

### Decision Points
- **After Experiment 0 (DONE)**: Test EM (55.3%) is 7.4pp below val EM (62.7%). Overfitting confirmed. Priority shifts: all experiments must use test EM as the primary metric, not val EM. Experiment 1 (test logging) becomes even more critical.
- **After Experiment 1**: If fast validation has low correlation with full validation, abandon it and revert.
- **After Experiment 2**: Pick the best archive configuration for all subsequent experiments.
- **After Experiment 3**: Decide whether reflective mutation is worth the implementation complexity.

---

## 8. Controlled Variables (Held Constant Across All Experiments Unless Noted)

| Variable | Value | Notes |
|----------|-------|-------|
| Chain execution LLM | Qwen3-8B, temp=0.6, top_p=0.95 | Shared endpoints |
| Mutation LLM | Qwen3-235B-A22B-Thinking-2507 | One per slot |
| BM25 retrieval | k=7, wiki17_abstracts | Frozen in topology |
| Chain topology | 6-step static | Not varied |
| Archive max_size | 75 | Per island config |
| num_parents | 2 | Per mutation; **varied in Exp 3 Condition C (1)** |
| Elite selector | FitnessProportionalEliteSelector | temp=auto |
| Mutation mode | rewrite | Not diff |
| Redis | resume=false (fresh) | Clean state per run |
| max_concurrent_dags | 10 | Runner config |

---

## 9. Logging Protocol

For every experimental run:

1. **Before start**: Record git commit hash, Redis DB number, exact CLI command, start time
2. **During run**: TensorBoard logs (standard GigaEvo output), generation timestamps
3. **After run**: Export using:
   - `tools/top_programs.py --save-dir <experiment_dir>` — top programs with code
   - `tools/comparison.py` — fitness curves across conditions
   - `redis-cli` — archive state snapshot
4. **Per-experiment artifacts** stored in `experiments/hotpotqa_improvement/<experiment_name>/`
5. **Commit results** as individual commits on the experiment branch with commentary

### Naming Convention
```
experiments/hotpotqa_improvement/
  exp0_baseline/
  exp1_fast_val/
  exp2_archive_diversity/
  exp3_reflective_mutation/
  exp4_mutation_budget/
```

---

## 10. Interpretation Guide

### What Would Success Look Like?
- Test EM > 63% on the 300-sample held-out test set
- Clear evidence that one or more interventions (fast val, coarser binning, reflective mutation) contributed to the improvement
- Reproducible: the improvement is not a single lucky seed

### What Would Failure Look Like?
- Test EM is significantly lower than validation EM (overfitting to 300-sample train subset)
- None of the interventions produce a statistically meaningful improvement over the Day 1 baseline
- The fitness curve has plateaued and additional generations/mutations do not help

### What Would Ambiguity Look Like?
- Small improvements (< 1 percentage point) that could be noise
- One intervention helps but introduces a new problem (e.g., faster validation is biased)
- Test EM improves but validation EM does not (or vice versa)

### Fallback Plan
If all experiments fail to improve beyond 62.7% val EM, the most valuable output is:
1. The infrastructure improvements (fast validation, test logging) which accelerate future research
2. The understanding of why the search is stuck (which informs next steps)
3. A fair test-set comparison with GEPA, which may show we are already competitive

---

## 11. Risks and Confounds

| Risk | Impact | Mitigation |
|------|--------|------------|
| LLM endpoint instability | Crashed runs, wasted time | Monitor, auto-resume where possible |
| Validation stochasticity | False positives/negatives in fitness | Measure variance (Exp 0), average over repeats |
| Overfitting to 300 train samples | High val EM, low test EM | **MATERIALIZED**: 62.7% val → 55.3% test (−7.4pp). Test EM logging (Exp 1) is now critical. |
| Efficiency changes alter behavior | Confound all comparisons | Validate that step-batched execution produces identical results (test on baseline program) |
| Mutation LLM non-determinism | High variance across seeds | Fixed seeds where possible; sample size |
| Redis state pollution | Cross-contamination | Fresh Redis DB per run, document DB assignments |
| Chain-execution endpoint contention | Slower validation when 3 runs share 2 endpoints | Schedule to avoid simultaneous validation phases |

---

## 12. Summary of Experiments

| # | Name | Key Change | Primary Metric | Gens | Slots | Priority |
|---|------|-----------|----------------|------|-------|----------|
| 0 | Baseline | None (measure) | Val/Test EM, variance | N/A | 1 | Prerequisite |
| 1 | Fast Validation | 100-sample + early stop + test logging | Val-vs-full correlation, speed | 10 | 2 | Highest |
| 2 | Archive Diversity | Coarser bins, optional 2D space | Best EM, acceptance rate | 10 | 3 | High |
| 3 | Reflective Mutation | Failure examples in prompt ± num_parents=1 | Mutation success rate, best EM | 10 | 3 | Medium-High |
| 4 | Mutation Budget | 16-24 mutants/gen | Best EM per wall-clock hour | 10+ | 1-2 | Medium |

Total: 4 experiments + 1 baseline, fitting within 3 slots over 7 days.
