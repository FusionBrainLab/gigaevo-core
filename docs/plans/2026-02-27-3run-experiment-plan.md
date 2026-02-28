# Research Plan: 3-Run Parallel HotpotQA Evolution Experiment

**Date**: 2026-02-27
**Investigator**: Dr. Elena Voss (ML Research Methodologist)
**Supersedes**: Section 7 (Execution Schedule) of `docs/plans/2026-02-27-hotpotqa-improvement.md`
**Status**: Pre-registration (experiment design, before results)

---

## Deviations from Pre-Registration

| Item | Pre-registered | Actual | Reason |
|------|---------------|--------|--------|
| Run C warm-start seed | `ddce37b4` (52.3% non-thinking val EM) | `warmstart_56pct.py` (56.3% val EM, from Attempt 2 of this experiment) | Attempt 2 produced a stronger non-thinking seed before being aborted; using it for Run C is consistent with the spirit of the warm-start condition and strictly better. The pre-registration referenced ddce37b4 as the best available program at planning time. |
| vLLM version on Run A server | Homogeneous (0.16.0rc2) | Run A server had 0.14.0rc1 at launch; upgraded before Run A was restarted | Discovered via `/version` endpoint check; statistically confirmed no bias (Mann-Whitney p>0.7 across 15 prompts × 3 servers) |
| Run A PID | 1411630 | 1512530 (restarted 2026-02-28 08:18 UTC after vLLM upgrade) | See above; run_a.log covers the restarted run only |

---

## 1. Research Question

**Can GigaEvo's evolutionary prompt optimization for HotpotQA static chains, starting from the non-thinking baseline (42.3% EM), reach or exceed GEPA's 62.3% test EM within 7 days, and which of three candidate interventions (archive resolution, mutation budget, seed quality) contributes most to fitness improvement?**

This is a single coordinated experiment with three parallel runs, each varying one factor. The design prioritizes learning speed: we want to know within 48 hours whether the evolutionary trajectory is promising, and within 7 days whether we can match GEPA.

---

## 2. Hypotheses

### Primary Hypothesis (composite)
- **H0**: None of the three runs reaches 62.3% test EM within 7 days.
- **H1**: At least one run reaches or exceeds 62.3% test EM.

### Per-Run Hypotheses

**Run A (Coarse Archive)**:
- H0_A: Reducing `primary_resolution` from 150 to 10 does not increase the best EM at generation 20 compared to the default resolution.
- H1_A: Coarser binning increases acceptance rate, maintains archive diversity in a useful range, and produces higher best EM.

**Run B (High Mutation Budget)**:
- H0_B: Increasing mutations from 8 to 16 per generation does not improve best EM at equal generation count.
- H1_B: More mutations per generation explore the prompt space faster, producing higher best EM.

**Run C (Warm-Start from ddce37b4)**:
- H0_C: Seeding with ddce37b4 (52.3% test EM, thinking-optimized) does not yield higher final EM than seeding from baseline (42.3%).
- H1_C: The warm start provides a better initial prompt structure that the mutation LLM can adapt to non-thinking mode, yielding faster improvement.

---

## 3. Justification of Design Choices

### 3.1 Why These Three Factors?

The five bottlenecks identified in the context are:
1. 70% rejection rate (archive resolution)
2. Archive refresh cost (scales with archive size)
3. Sequential mutation calls (parallelism)
4. Generic mutation prompts (no failure context)
5. No fast validation

Of these, (1) and (3) can be addressed purely by configuration changes, making them immediately testable. Factor (5) is already partially addressed by the step-batched execution (2.1 min/300 samples). Factor (4) requires code changes (reflective mutation) that are not yet implemented and should be a follow-up experiment, not part of the first run. Factor (2) is partially self-mitigating because the default `InputHashCache` skips validation on refresh when inputs haven't changed -- only lineage/insights LLM calls re-run during refresh.

The seed program question (Run C) is orthogonal to the algorithmic factors and provides a cheap test of whether the ddce37b4 prompt structure is recoverable under non-thinking conditions.

### 3.2 Why Not Vary More Factors?

With 3 runs, we can vary at most 3 factors independently, unless we use a fractional factorial design. However, with N=1 per condition and stochastic LLM-based mutation, we cannot estimate interaction effects reliably. A simple one-factor-at-a-time design with a shared control baseline (Run A) gives us interpretable directional signals:

- Run A = *de facto* baseline (coarse archive + standard mutation budget + baseline seed)
- Run B = Run A + doubled mutation budget
- Run C = Run A + warm-start seed

This means Run A vs Run B isolates the mutation budget effect, and Run A vs Run C isolates the seed effect. All three runs use coarse archive (primary_resolution=10), which we treat as a hygiene improvement rather than an experimental variable, because the 150-bin default is clearly misconfigured for a 1D fitness-only behavior space (see Section 3.3).

### 3.3 Why primary_resolution=10 for All Runs (Not 150)

With 150 bins over [0, 1], each bin spans 0.0067 EM (~0.67 percentage points). The fitness range of interest is roughly 0.42 to 0.63 (~21 pp), spanning ~31 bins. With 8 mutations/gen and ~30% acceptance rate, the archive grows by ~2-3 programs per generation. After 20 generations, we expect ~40-60 programs populating ~31 useful bins -- meaning most bins are occupied and new mutants must strictly beat the incumbent.

With 10 bins over [0, 1], each bin spans 0.10 EM. The 0.42-0.63 range spans ~2-3 bins. This is too coarse if the fitness landscape is rich, but for a 1D fitness-only space it is more honest: it admits that "diversity" in this archive is just keeping a few programs at different fitness levels, which is mainly useful for providing multiple parent candidates to the mutation LLM. The real diversity in this problem comes from prompt text variation, which is not captured by the behavior space.

**Decision: primary_resolution=10 for all runs.** The 150-bin configuration is a known-bad default for this problem. Treating it as a controlled variable (held constant) is more informative than using one run to re-confirm it is bad.

### 3.4 Why Seed from Baseline (42.3%) for Runs A and B?

GEPA achieves 62.3% starting from a simple baseline, not from a pre-optimized prompt. A fair comparison requires the same starting point. The ddce37b4 program was optimized under thinking mode; its prompt structure may contain patterns (e.g., "think step by step" scaffolding) that are counterproductive in non-thinking mode. Run C explicitly tests this -- if it helps, we can adopt it for future runs.

### 3.5 Why Not Use Fast Validation (validate2.py)?

The fast validation (100 samples, early stopping) was designed for Optuna's inner loop, not for the main evolution fitness signal. With 300-sample validation now at 2.1 min (thanks to step-batched execution), the speed benefit of 100-sample validation is modest (~0.7 min saved per program). The risk -- that 100-sample fitness is a noisy proxy that biases selection -- is not worth the small time saving for the primary evolution loop.

**Decision: All runs use standard `validate.py` (300 samples) for the main fitness signal.** We preserve `validate2.py` for Optuna if that pipeline is tested later.

---

## 4. Determining the Number of Generations

### 4.1 Bottom-Up Time Budget Analysis

**Per-generation time estimate (8 mutations, primary_resolution=10):**

| Phase | Duration | Notes |
|-------|----------|-------|
| Wait for idle | ~0 | Engine enters step() when already idle |
| Select elites | <1s | In-memory operation |
| Create 8 mutations (parallel LLM calls) | ~3-5 min | Qwen3-235B, `asyncio.gather` runs all 8 in parallel to one vLLM server; throughput limited by server capacity |
| Wait for mutant DAGs (parallel) | ~2-3 min | 8 programs x 2.1 min validation, up to 10 concurrent DAGs; dominated by LLM server throughput |
| Ingest completed programs | <1s | In-memory archive operations |
| Refresh archive programs | ~1-3 min | Lineage/Insights LLM calls for archive programs with changed inputs; validation cached by InputHashCache |
| Wait for refresh DAGs | ~1-3 min | Only lineage/insights re-run (LLM calls), not validation |

**Estimated total per generation: ~8-14 min** (call it ~12 min average)

Corrections to the user's estimate: The user estimated 25-30 min/gen based on sequential mutation LLM calls. However, `generate_mutations()` in `gigaevo/evolution/engine/mutation.py` (line 88-93) uses `asyncio.gather(*tasks)`, making all mutation calls parallel. With a single Qwen3-235B server handling 8 parallel requests, the wall-clock time is approximately 1 request latency (3-5 min) rather than 8x sequential. The refresh phase is also cheaper than estimated because `InputHashCache` skips validation -- only lineage/insights stages re-run (1-2 LLM calls per program, but only for programs whose inputs changed).

**IMPORTANT CAVEAT**: The actual mutation parallelism depends on the Qwen3-235B vLLM server's `--max-num-seqs` setting. If the server can only handle 1 concurrent request, mutations will be effectively sequential. This must be verified empirically in the first generation. If mutations are sequential, the time estimate rises to ~30-40 min/gen.

**Per-generation time estimate (16 mutations, Run B):**
With 16 mutations, the mutation LLM phase takes longer (server must process 16 requests, possibly with some queueing). If the server handles 8 concurrent sequences, this doubles to ~6-10 min. Validation is still ~2-3 min (all 16 mutants in parallel, 2 chain-execution servers shared). Estimated total: ~12-18 min/gen.

### 4.2 Generations Per Day

| Scenario | Min/gen | Gens/day |
|----------|---------|----------|
| Optimistic (8 mut, parallel) | 8 min | ~180 |
| Realistic (8 mut, partial parallel) | 12 min | ~120 |
| Pessimistic (8 mut, sequential LLM) | 30 min | ~48 |
| Run B (16 mut, realistic) | 15 min | ~96 |

### 4.3 When to Expect Meaningful Signal

Based on the existing run history (db=5, thinking mode):
- Generation 0: baseline seed loaded, 42.3% EM
- Generations 1-3: rapid improvement as first mutations find low-hanging fruit
- Generations 4-7: slower improvement, best reached 62.7% by gen 7

In non-thinking mode, we expect a similar trajectory shape but possibly slower improvement (the search space is the same, but the fitness landscape may differ). GEPA's approach iterates ~30 rounds of reflection per sample, each refining a single prompt. GigaEvo explores broader but shallower -- each generation tries 8 prompt variants.

**Decision: Set max_generations=30 for all runs.**

Justification:
- At 12 min/gen, 30 generations = ~6 hours. This fits 4 full 30-gen runs per day per slot, leaving ample time within the 7-day budget.
- If results plateau early (no improvement for 10 consecutive generations), we can terminate and restart with a modified configuration.
- 30 generations produces ~240 evaluated mutants (8/gen) or ~480 (16/gen), giving enough data to characterize the fitness curve.
- We can extend to 50 or more generations for the best-performing configuration later.

### 4.4 Checkpoint Evaluations

At generations 10, 20, and 30, extract the top program from each run and evaluate on the held-out test set (300 samples) using `problems/chains/hotpotqa/static/test.py`. This gives us test EM at three points, not just the final one, which helps distinguish genuine improvement from overfitting to the validation set.

---

## 5. Experimental Conditions (Exact Commands)

### 5.1 Common Configuration (All Runs)

```yaml
# Held constant across all 3 runs
problem.name: chains/hotpotqa/static
pipeline: auto                        # Uses standard validate.py (300 samples)
primary_resolution: 10                # Coarse archive bins (see Section 3.3)
island_max_size: 75                   # Max archive size
max_elites_per_generation: 5          # Elite selection for mutation parents
num_parents: 2                        # Parents per mutation
mutation_mode: rewrite                # Full program rewrite
max_generations: 30                   # See Section 4.3
max_concurrent_dags: 10               # Parallel validation DAGs
redis.resume: false                   # Fresh state
temperature: 0.6                      # Mutation LLM temperature
```

### 5.2 Run A: Coarse Archive + Standard Budget + Baseline Seed

**Purpose**: Baseline for this experiment batch. Tests the coarse-archive intervention against the fundamental question: can non-thinking evolution reach 60%+ EM?

```bash
nohup bash -c "
export NO_PROXY='localhost,127.0.0.1,INTERNAL_IP'
export no_proxy='localhost,127.0.0.1,INTERNAL_IP'
~/envs/evo_fast/bin/python run.py \
  problem.name=chains/hotpotqa/static \
  redis.db=10 \
  llm_base_url=http://localhost:8000/v1 \
  primary_resolution=10 \
  max_mutations_per_generation=8 \
  max_generations=30
" > experiments/hotpotqa_3run/run_a.log 2>&1 &
```

| Parameter | Value |
|-----------|-------|
| Redis DB | 10 |
| Mutation LLM | INTERNAL_IP:8777 |
| primary_resolution | 10 |
| max_mutations_per_generation | 8 |
| Seed | baseline.py (42.3%) |

### 5.3 Run B: Coarse Archive + Double Budget + Baseline Seed

**Purpose**: Tests whether more mutations per generation (16 vs 8) improves search efficiency. Same archive config and seed as Run A.

```bash
nohup bash -c "
export NO_PROXY='localhost,127.0.0.1,INTERNAL_IP'
export no_proxy='localhost,127.0.0.1,INTERNAL_IP'
~/envs/evo_fast/bin/python run.py \
  problem.name=chains/hotpotqa/static \
  redis.db=11 \
  llm_base_url=http://localhost:8000/v1 \
  primary_resolution=10 \
  max_mutations_per_generation=16 \
  max_elites_per_generation=8 \
  max_generations=30
" > experiments/hotpotqa_3run/run_b.log 2>&1 &
```

| Parameter | Value |
|-----------|-------|
| Redis DB | 11 |
| Mutation LLM | INTERNAL_IP:8777 |
| primary_resolution | 10 |
| max_mutations_per_generation | 16 |
| max_elites_per_generation | 8 |
| Seed | baseline.py (42.3%) |

Note: `max_elites_per_generation` increased from 5 to 8 to provide more parent diversity for 16 mutations (with `num_parents=2` and `RandomParentSelector`, we want at least 8 unique parents available for 16 mutations).

### 5.4 Run C: Coarse Archive + Standard Budget + Warm-Start Seed

**Purpose**: Tests whether starting from ddce37b4 (thinking-optimized, 52.3% non-thinking EM) accelerates convergence compared to baseline seed.

**Implementation**: Copy the ddce37b4 program code into `initial_programs/` as a second seed file. The `DirectoryProgramLoader` will load all `.py` files from `initial_programs/`, so both baseline.py and ddce37b4.py will be loaded as generation-0 programs. The mutation LLM can then select either as a parent.

```bash
# Pre-step: copy ddce37b4 code into initial_programs/
# (exact code to be extracted from Redis db=5 using tools/top_programs.py)

nohup bash -c "
export NO_PROXY='localhost,127.0.0.1,INTERNAL_IP'
export no_proxy='localhost,127.0.0.1,INTERNAL_IP'
~/envs/evo_fast/bin/python run.py \
  problem.name=chains/hotpotqa/static \
  redis.db=12 \
  llm_base_url=http://localhost:8000/v1 \
  primary_resolution=10 \
  max_mutations_per_generation=8 \
  max_generations=30
" > experiments/hotpotqa_3run/run_c.log 2>&1 &
```

| Parameter | Value |
|-----------|-------|
| Redis DB | 12 |
| Mutation LLM | INTERNAL_IP:8777 |
| primary_resolution | 10 |
| max_mutations_per_generation | 8 |
| Seed | baseline.py (42.3%) + ddce37b4.py (52.3%) |

**CRITICAL**: After Run C completes, restore `initial_programs/` to contain only `baseline.py` to avoid contaminating future experiments. Alternatively, use a separate problem directory or a config override to control the seed.

---

## 6. Controlled Variables

| Variable | Value | Rationale |
|----------|-------|-----------|
| Chain topology | 6-step static | Fixed by problem definition |
| Chain execution LLM | Qwen3-8B, non-thinking, temp=0.6 | GEPA comparison protocol |
| Mutation LLM model | Qwen3-235B-A22B-Thinking-2507 | Only model available |
| Mutation LLM temperature | 0.6 | Default config |
| BM25 retrieval | k=7, wiki17_abstracts | Frozen in topology |
| Validation set | 300 train samples | Standard validate.py |
| Test set | 300 held-out samples | For checkpoint evaluation |
| num_parents | 2 | Default config |
| island_max_size | 75 | Default config |
| max_concurrent_dags | 10 | Default runner config |
| Pipeline | auto (standard) | Uses validate.py |
| Cache behavior | InputHashCache (default) | Skips validation on refresh |
| Archive type | Single-island MAP-Elites with DynamicBehaviorSpace | Default |
| Elite selector | FitnessProportionalEliteSelector (auto temp) | Default |
| Redis | resume=false, separate DBs per run | Clean state |

### Potential Confounds

1. **Shared chain-execution servers**: All 3 runs share 2 Qwen3-8B endpoints. If runs hit validation phases simultaneously, they compete for GPU time, slowing each other. Mitigation: 10-concurrent-DAGs limit per run + vLLM's internal batching. Monitor validation wall-clock times for anomalies.

2. **Mutation LLM non-determinism**: Even with the same config, different mutation LLM servers may have slightly different inference behavior (quantization, batching). Mitigation: servers run identical model weights and vLLM version. This is a minor confound.

3. **Seed program loading order**: `DirectoryProgramLoader` loads programs alphabetically. In Run C, both `baseline.py` and `ddce37b4.py` are loaded, but the archive's `FitnessProportionalEliteSelector` will favor ddce37b4 (higher fitness) for mutation parent selection. This is by design for Run C.

4. **Dynamic behavior space expansion**: With primary_resolution=10 and initial bounds [0, 1], the behavior space starts with 10 bins. As programs are added, `DynamicBehaviorSpace` may tighten bounds around observed values, effectively changing the archive structure mid-run. This is a feature (adaptive resolution) but makes cross-run archive comparison less straightforward.

---

## 7. Metrics

### 7.1 Primary Metric
- **Best validation EM** at generations 10, 20, 30 (the maximum fitness in the archive)
- **Best test EM** at generations 10, 20, 30 (evaluated separately on held-out set)

### 7.2 Secondary Metrics
- **Archive acceptance rate**: fraction of mutants added to archive per generation
- **Archive size over time**: number of programs in archive at each generation
- **Fitness curve**: best, mean, and median archive fitness per generation
- **Generation wall-clock time**: total time per generation (phases 1-6)
- **Mutation LLM latency**: time per mutation call (to verify parallelism)
- **Validation wall-clock time**: per-program and per-generation

### 7.3 Diagnostic Metrics
- **Mutation success rate**: fraction of mutants that improve over their parent's fitness
- **Fitness improvement per generation**: delta of best fitness between consecutive generations
- **Plateau detector**: number of consecutive generations with no improvement in best fitness
- **Extraction failure rate**: fraction of samples where answer extraction fails (should be ~0%)

### 7.4 Pass/Fail Thresholds

| Criterion | Threshold | Notes |
|-----------|-----------|-------|
| Primary success | Test EM >= 62.3% | Matches GEPA |
| Strong success | Test EM >= 63.0% | Exceeds GEPA |
| Partial success | Test EM >= 55.0% | Clear improvement over ddce37b4 non-thinking (52.3%) |
| Minimum viability | Test EM >= 50.0% | Better than no evolution |
| Failure | Test EM < 50.0% after 30 gens | Evolution is not working |

---

## 8. Stopping Criteria

### 8.1 Normal Termination
- Run reaches `max_generations=30`.

### 8.2 Early Termination (Manual)
- **Plateau**: If best fitness has not improved for 10 consecutive generations AND best EM < 50%, terminate and investigate.
- **Infrastructure failure**: If a mutation LLM server or chain-execution server crashes and cannot be restarted within 1 hour, pause the affected run.
- **Extreme slowness**: If generation time exceeds 60 min consistently (suggesting mutation LLM bottleneck), reduce `max_mutations_per_generation` or investigate.

### 8.3 Extension (REVISED 2026-02-27)
- **CONSTRAINT**: `redis.resume=true` is broken for MAP-Elites archive reinitialization. The archive does not reconstruct correctly after restart, leading to undefined behavior. Extension via resume is NOT safe.
- **Revised `max_generations`**: Increased from 30 to 50 for all runs, to reduce the need for extensions. At 12 min/gen (optimistic), 50 gens = ~10 hours; at 25 min/gen (pessimistic), 50 gens = ~21 hours. Both fit within 1 day.
- **Continuation path**: If a run finishes gen 50 with best EM > 55% AND still rising, use the "leapfrog" protocol: extract top-3 programs, copy to initial_programs, start fresh run on new Redis DB. This loses archive diversity and lineage context but preserves best solutions.

---

## 9. What to Do After 10 Generations

At generation 10 (~2 hours into the run), perform the following assessment for each run:

### 9.1 Immediate Checks
1. Extract the best program from each run using `tools/top_programs.py --db=N --top=1`.
2. Run test-set evaluation: `~/envs/evo_fast/bin/python problems/chains/hotpotqa/static/test.py` on the best program.
3. Record: best val EM, test EM, archive size, acceptance rate, generation wall-clock times.

### 9.2 Decision Matrix (at Gen 10)

| Best Val EM | Fitness Curve | Action |
|-------------|--------------|--------|
| >= 55% | Rising | Continue to gen 30. Strong signal. |
| 50-55% | Rising | Continue to gen 30. On track. |
| 45-50% | Rising | Continue to gen 20, re-evaluate. |
| 45-50% | Flat (last 5 gens) | Concerning. Check mutation quality (are mutations actually diverse?). Continue to gen 20 but prepare fallback. |
| < 45% | Any | Serious problem. Check: (a) is validation working correctly? (b) are mutations being accepted? (c) is the archive growing? Debug before continuing. |

### 9.3 Fallback Actions (if results are poor at Gen 10)

1. **Check mutation quality**: Examine 5 random mutation LLM outputs. Are they syntactically valid? Do they meaningfully change the prompt structure? Or are they trivial edits?
2. **Check archive state**: How many programs are in the archive? What is the fitness distribution?
3. **Check LLM server health**: Is the mutation LLM responding normally? Are chain-execution servers overloaded?
4. **If all diagnostics pass but fitness is flat**: The problem may be that non-thinking Qwen3-8B has a lower ceiling for this task. This would be a genuine negative result worth documenting.

### 9.4 Comparative Assessment (at Gen 10)

Compare the three runs at equal generation count (gen 10):

| Comparison | What it tells us |
|------------|-----------------|
| Run A vs Run B | Does doubling mutation budget help? If Run B > Run A by > 2 pp, strong signal. |
| Run A vs Run C | Does warm start help? If Run C > Run A by > 2 pp, yes. If Run C < Run A, warm start hurts (thinking-optimized prompts are counterproductive). |
| Run B vs Run C | Mutation budget vs seed quality: which matters more? |

Note: With N=1 per condition and LLM stochasticity (measured at ~1.4 pp std for validation EM), differences < 3 pp are within noise. Only differences > 5 pp are likely to reflect a genuine treatment effect.

---

## 10. Archive Refresh Analysis

### 10.1 How Refresh Works in Practice

Each generation's Phase 5 (`_refresh_archive_programs`) sets all archive programs from DONE to QUEUED, triggering their DAG pipelines. However, the `InputHashCache` (default cache handler) means:

- **ValidateCodeStage**: Cached (code hasn't changed). Skipped.
- **CallProgramFunction**: Cached (code + no context change). Skipped.
- **CallValidatorFunction**: Cached (payload from program hasn't changed). Skipped.
- **FetchMetrics**: Cached. Skipped.
- **ComputeComplexityStage**: Cached (code hasn't changed). Skipped.
- **InsightsStage**: May re-run (if archive composition changed, affecting the "evolutionary statistics" context that feeds into insights). LLM call: ~30-60s.
- **LineageStage**: Re-runs for programs with new children (newly added mutants). LLM call: ~30-60s.
- **MutationContextStage**: Re-runs (aggregates insights + lineage + stats, all of which may have changed).

**Net refresh cost**: For most archive programs, only the lineage/insights/mutation-context stages re-run. These involve 1-2 LLM calls per program, each ~30-60s. With 10-20 programs in the archive and concurrent DAGs, refresh takes ~1-3 min.

**Archive size impact**: With `primary_resolution=10` and `island_max_size=75`, the archive can hold at most 10 programs (one per bin) or 75 (the size limit), whichever is smaller. With 10 bins, at most 10 unique fitness bins exist. The archive will stabilize at ~10 programs quickly.

Wait -- this needs verification. With DynamicBehaviorSpace, the bins tighten around observed values. If the fitness range narrows to [0.42, 0.55], 10 bins over that range means each bin is ~1.3 pp wide. The archive could hold up to 10 programs. With `island_max_size=75`, the max_size limit is not binding.

**Conclusion**: Refresh cost with `primary_resolution=10` is very low (~10 programs x 1-2 min of LLM calls, parallelized = ~2-3 min). This is not a bottleneck.

### 10.2 Monitoring Refresh Cost

Log the wall-clock time of Phase 5+6 (refresh + wait) per generation. If refresh time exceeds 5 min consistently, investigate.

---

## 11. Code Changes Required Before Running

### 11.1 Critical (Must-Have)

**None.** All three runs use existing configuration overrides. No code changes are needed for the experiment itself.

### 11.2 Strongly Recommended (Before Starting)

1. **Create experiment output directory**:
   ```bash
   mkdir -p experiments/hotpotqa_3run/{run_a,run_b,run_c}
   ```

2. **Extract ddce37b4 code for Run C seed**:
   ```bash
   ~/envs/evo_fast/bin/python tools/top_programs.py \
     --db=5 --top=1 --save-dir experiments/hotpotqa_3run/run_c/
   ```
   Then copy the program code to `problems/chains/hotpotqa/static/initial_programs/ddce37b4.py`. After Run C starts and loads programs, restore the directory.

3. **Verify mutation parallelism**: Run a single generation manually and check whether mutation LLM calls overlap in time (from logs). If they are sequential, adjust time estimates.

4. **Verify Redis DBs are clean**:
   ```bash
   redis-cli -n 10 DBSIZE  # should be 0
   redis-cli -n 11 DBSIZE  # should be 0
   redis-cli -n 12 DBSIZE  # should be 0
   ```

### 11.3 Nice-to-Have (Can Do During Run)

1. **Test EM evaluation script**: Create a script that, given a Redis DB number, extracts the top program and runs test.py on it. This will be used at generation checkpoints.

2. **Monitoring dashboard**: A script that polls Redis and reports per-run generation count, archive size, best fitness, and generation wall-clock time.

---

## 12. Logging Protocol

### 12.1 Before Each Run

Record in the experiment log (`experiments/hotpotqa_3run/README.md`):
- Git commit hash: `git rev-parse HEAD`
- Exact CLI command (copy from Section 5)
- Start time (UTC)
- Redis DB number
- Mutation LLM server IP
- Python interpreter: `~/envs/evo_fast/bin/python`

### 12.2 During Each Run

GigaEvo automatically logs to TensorBoard. Additionally:
- Monitor `run_a.log`, `run_b.log`, `run_c.log` for errors
- At generation 10: extract top program, run test evaluation, record results
- At generation 20: same
- At generation 30 (final): same + full archive dump

### 12.3 After Each Run

1. Export top programs:
   ```bash
   ~/envs/evo_fast/bin/python tools/top_programs.py \
     --db=N --top=10 --save-dir experiments/hotpotqa_3run/run_X/
   ```

2. Export fitness curves:
   ```bash
   ~/envs/evo_fast/bin/python tools/comparison.py \
     --dbs 10 11 12 --labels "Run A" "Run B" "Run C" \
     --save experiments/hotpotqa_3run/fitness_curves.png
   ```

3. Record per-generation metrics in a table:
   ```
   | Gen | Run A Best EM | Run B Best EM | Run C Best EM | Run A Archive Size | ...
   ```

4. Commit results as individual commits on the experiment branch.

### 12.4 Naming Convention

```
experiments/hotpotqa_3run/
  README.md                    # Experiment metadata, commands, results summary
  run_a/                       # Run A artifacts
    run_a.log                  # Full stdout/stderr
    top_programs/              # Extracted top programs (code + metrics)
    test_eval_gen10.json       # Test EM at gen 10
    test_eval_gen20.json       # Test EM at gen 20
    test_eval_gen30.json       # Test EM at gen 30
  run_b/                       # Run B artifacts (same structure)
  run_c/                       # Run C artifacts (same structure)
  fitness_curves.png           # Comparative fitness curves
  results_table.md             # Final results table
```

---

## 13. Statistical Analysis Plan

### 13.1 Limitations

With N=1 run per condition, formal statistical testing is not possible. We rely on:
- **Effect size benchmarks**: differences > 5 pp are likely meaningful (given 1.4 pp measured EM std across repeated evaluations of the same program).
- **Fitness curve shapes**: whether the curve is rising, flat, or declining at the end.
- **Consistency between val and test EM**: a val-test gap > 3 pp indicates overfitting.

### 13.2 Planned Comparisons

| Comparison | Method | Interpretation |
|------------|--------|----------------|
| Run A vs Run B (at gen 10, 20, 30) | Best EM difference | If delta > 5 pp, mutation budget matters |
| Run A vs Run C (at gen 10, 20, 30) | Best EM difference | If delta > 5 pp, seed matters |
| Val EM vs Test EM (per run) | Paired difference | If gap > 3 pp, overfitting concern |
| Fitness curve slope (last 10 gens) | Visual + linear fit | Positive slope = still improving |

### 13.3 Multiple Comparisons

With 3 pairwise comparisons (A-B, A-C, B-C) at 3 time points, we have 9 comparisons. Since we are not doing formal hypothesis testing (N=1), we do not apply corrections. Instead, we treat this as exploratory: any strong signal (> 5 pp difference) motivates a follow-up replicated experiment.

### 13.4 Follow-Up Replication

If any run achieves > 60% test EM, we will:
1. Run 3 additional seeds with the same configuration to measure variance.
2. Report mean +/- std across 4 seeds total.
3. Only claim to "beat GEPA" if the mean test EM across 4 seeds exceeds 62.3%.

---

## 14. Expected Timeline

### Day 1 (today)
- **Hour 0-1**: Prepare infrastructure. Verify Redis DBs are clean. Extract ddce37b4 code. Create experiment directories.
- **Hour 1-2**: Start all 3 runs simultaneously.
- **Hour 3-4**: Generation 10 checkpoint. Extract and evaluate top programs. Decision matrix assessment.
- **Hour 5-8**: Runs reach generation 20-30 (if ~12 min/gen) or generation 10-15 (if ~30 min/gen).

### Day 2
- Complete 30 generations for all runs (if ~12 min/gen, done by hour 6-8 of Day 1).
- Analyze results. Run test evaluations.
- Decide on follow-up: extend best run? Start reflective mutation experiment?

### Days 3-7
- Reserved for follow-up experiments based on Day 1-2 results.
- If primary hypothesis fails: investigate failure modes, consider code changes (reflective mutation).
- If primary hypothesis succeeds: replication runs (3 additional seeds for the winning configuration).

---

## 15. Interpretation Guide

### 15.1 What Would Strong Success Look Like?
- At least one run reaches 62%+ test EM by generation 30.
- The fitness curve shows clear, sustained improvement from baseline.
- Val EM and test EM are within 2 pp (no overfitting).
- The winning configuration can be identified (archive resolution, mutation budget, or seed).

### 15.2 What Would Partial Success Look Like?
- Best test EM reaches 55-62%, showing clear improvement but not matching GEPA.
- One factor (e.g., mutation budget) shows clear benefit over others.
- The fitness curve is still rising at gen 30, suggesting more generations would help.

### 15.3 What Would Failure Look Like?
- All runs plateau below 50% test EM.
- The fitness curve flattens by generation 10 with no further improvement.
- Mutations are low quality (small or no fitness improvement over parents).

### 15.4 What Would Ambiguity Look Like?
- Runs reach 55-60% test EM but with val-test gap > 5 pp.
- All three runs perform similarly, making it impossible to distinguish factor effects.
- High variance within runs (fitness oscillates significantly between generations).

### 15.5 Follow-Up Decision Tree

```
IF best test EM >= 62.3%:
  -> Replicate with 3 additional seeds
  -> Report result as "GigaEvo matches/exceeds GEPA"

ELSE IF best test EM >= 55% AND fitness curve still rising:
  -> Extend best run to 50+ generations
  -> Consider adding reflective mutation (code change)

ELSE IF best test EM >= 55% AND fitness curve flat:
  -> Search is stuck. Implement reflective mutation (Experiment 3 from original plan).
  -> Re-run with reflective mutation as the 4th experiment.

ELSE IF best test EM < 55%:
  -> Fundamental issue. Investigate:
     - Is the non-thinking LLM inherently limited on this task?
     - Are mutations diverse enough? (Check mutation LLM outputs.)
     - Is the archive structure appropriate? (Check behavior space dynamics.)
  -> Consider: (a) 2D behavior space, (b) different mutation prompts, (c) human-in-the-loop seed design.
```

---

## 16. Risks and Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Mutation LLM server crash | One run stalls | Medium | Monitor logs; restart server; resume run |
| Chain-execution server overload (3 runs sharing 2 endpoints) | Slower validation for all runs | High | Monitor validation times; stagger run starts by 5 min |
| Mutation calls are sequential (not parallel as code suggests) | Generation time 3x longer; 30 gens takes ~15h instead of 6h | Medium | Verify in first generation; adjust expectations |
| Redis DB collision | Data corruption | Low | Verified empty DBs (10, 11, 12) before starting |
| ddce37b4 seed causes negative transfer in Run C | Run C performs worse than Run A | Medium | This is an experimental finding, not a failure. Document it. |
| Dynamic behavior space collapse | Archive shrinks to 2-3 programs after bounds tighten | Medium | Monitor archive size; if it drops below 5, the resolution is too coarse for the fitness range |
| Overfitting to 300 train samples | High val EM, low test EM | Medium | Test EM checkpoints at gen 10, 20, 30 |
| All runs converge to similar performance | Cannot distinguish factor effects | High | Expected with N=1; this motivates follow-up replication |

---

## 17. Summary Table

| | Run A | Run B | Run C |
|--|-------|-------|-------|
| **Redis DB** | 10 | 11 | 12 |
| **Mutation Server** | INTERNAL_IP | INTERNAL_IP | INTERNAL_IP |
| **primary_resolution** | 10 | 10 | 10 |
| **max_mutations_per_generation** | 8 | **16** | 8 |
| **max_elites_per_generation** | 5 | **8** | 5 |
| **Seed programs** | baseline.py | baseline.py | baseline.py + **ddce37b4.py** |
| **max_generations** | 30 | 30 | 30 |
| **Role** | Baseline | High mutation budget | Warm start |
| **Tests vs Run A** | -- | Mutation budget effect | Seed quality effect |

---

## Appendix A: Key Source Files Referenced

- Evolution engine: `gigaevo/evolution/engine/core.py` (generation loop, 6 phases)
- Mutation generation: `gigaevo/evolution/engine/mutation.py` (parallel via asyncio.gather)
- Archive insertion: `gigaevo/evolution/strategies/island.py` (MapElitesIsland.add)
- Cache handler: `gigaevo/programs/stages/cache_handler.py` (InputHashCache = default)
- Pipeline builder: `gigaevo/entrypoint/default_pipelines.py` (DefaultPipelineBuilder)
- Validation: `problems/chains/hotpotqa/static/validate.py` (300 samples, step-batched)
- Fast validation: `problems/chains/hotpotqa/static/validate2.py` (100 samples, early stop)
- Initial program loader: `gigaevo/problems/initial_loaders.py` (DirectoryProgramLoader)
- Config constants: `config/constants/evolution.yaml`, `config/constants/islands.yaml`
- Behavior space: `gigaevo/evolution/strategies/models.py` (DynamicBehaviorSpace)

## Appendix B: Corrections to User's Initial Estimates

1. **Mutation calls are parallel, not sequential.** `generate_mutations()` uses `asyncio.gather(*tasks)` (mutation.py:93). All 8 mutation LLM calls are issued simultaneously. Server throughput (not sequential latency) determines total mutation time.

2. **Archive refresh is cheaper than estimated.** `InputHashCache` (default) skips stages whose inputs haven't changed. Validation is cached on refresh because program code doesn't change. Only lineage/insights stages re-run (~1-2 LLM calls per program).

3. **Archive size with primary_resolution=10 is self-limiting.** With 10 bins and DynamicBehaviorSpace, the archive holds at most ~10 programs (one per bin). `island_max_size=75` is not binding. Refresh cost is proportional to archive size, so it stays low.

4. **Generation time estimate revised to ~8-14 min** (from 25-30 min), assuming mutation parallelism works in practice. This must be verified empirically.
