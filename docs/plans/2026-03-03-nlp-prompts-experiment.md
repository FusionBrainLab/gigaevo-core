# Research Plan: NLP-Specific Mutation Prompts for HotpotQA

**Date**: 2026-03-03
**Investigator**: Dr. Elena Voss (ML Research Methodologist)
**Problem**: `problems/chains/hotpotqa/static/`
**Framework**: GigaEvo evolutionary computation
**Status**: Pre-registration (before implementation)
**Builds on**: P1xP2 factorial (PR #67, 2026-03-01) -- null result
**Branch**: `exp/hotpotqa-nlp-prompts`
**PR Title**: `feat: NLP-specific mutation prompts for HotpotQA chain evolution`

---

## 1. Research Question

**Does replacing GigaEvo's optimizer-centric prompt templates (insights, lineage, mutation/system) with NLP-chain-specific versions improve test EM on HotpotQA static chains beyond the ddce37b4 seed baseline (60.0%) within 50 generations?**

Subsidiary questions:
1. Does NLP-specific framing in insights/lineage improve mutation acceptance rate?
2. Does the combination of NLP prompts with the best P1xP2 config (P1+P2, Run H pattern) yield further gains?
3. Is the effect concentrated in early generations (better search direction from the start) or late generations (better plateau-breaking)?

## 2. Motivation: Why Mutation Quality Is the Bottleneck

The P1xP2 factorial produced a definitive negative result:

| Run | Condition | Test EM | Val-Test Gap |
|-----|-----------|---------|-------------|
| Seed | ddce37b4 (gen 0) | 60.0% | -1.7pp |
| E | Control | 59.3% | +5.7pp |
| F | P2 (ASI) only | 53.7% | +11.0pp |
| G | P1 (rotation) only | 58.0% | +9.7pp |
| H | P1 + P2 | 61.3% | +4.0pp |

**Key findings**:
- Best test EM (H, 61.3%) is only +1.3pp over seed (60.0%). This is well within the 2.4pp noise floor (p=0.79 on a binomial test, n=300).
- Oracle selection (picking the best among 4 runs post-hoc) yields at most +1.3pp over the seed. Even if we had perfect selection, the programs being generated are not better than the seed.
- The F run (P2 only) *degraded* to 53.7%, suggesting ASI feedback alone can mislead without rotation.
- Conclusion: **selection signal improvements (P1, P2) cannot help if the mutations themselves are not producing better programs. The bottleneck is mutation quality.**

**Why prompt framing matters**: The mutation LLM (Qwen3-235B-A22B-Thinking) receives guidance from three upstream agents:
- **Insights agent**: Analyzes the current program and identifies what to change. Currently framed with optimizer examples: `threshold=0.5 at line 23`, `try/except at line 45 catches ValueError`, `popsize=100 at line 8 limits genetic diversity`.
- **Lineage agent**: Analyzes parent-to-child diffs to extract transferable lessons. Currently gives examples like `Changed threshold from 0.5 to 0.3`, `Removed greedy selection`, `Replaced O(n^2) nested loops with sorted-merge`.
- **Mutation agent system prompt**: Frames the task as "evolutionary optimization of python programs" with archetypes like "Computational Reinvention" and "Solution Space Exploration".

For a prompt-chain evolution problem, these framings are mismatched:
- There are no thresholds, loops, or data structures to optimize -- the "code" is a `def entrypoint()` returning a dict of prompt strings.
- The relevant mutation dimensions are: instruction clarity, reasoning scaffolds, format constraints, query formulation strategy, evidence synthesis approach, answer extraction format.
- The LLM must reason about natural language prompt engineering, not algorithmic optimization.

The `task_description.txt` already provides domain context about the chain architecture, but it appears only in the mutation user prompt. The insights and lineage agents -- which shape the mutation LLM's understanding of *what went wrong* and *what worked before* -- receive only the generic optimizer framing.

## 3. Hypotheses

### H1 (NLP Prompts)
- **H0**: NLP-specific prompt templates do not improve test EM beyond the seed (60.0%). Mean test EM of treatment runs <= 60.0% + 2.4pp (noise floor).
- **H1**: NLP-specific prompt templates improve test EM. At least 2 of 3 treatment runs exceed 61.3% (the best P1xP2 result), and the treatment mean exceeds 62.0%.

### H2 (NLP Prompts + P1+P2)
- **H0**: NLP prompts combined with P1+P2 (Run H config) do not outperform NLP prompts alone.
- **H1**: The combination produces test EM >= 62.3% (GEPA target) in at least 1 run.

## 4. Experimental Design

### 4.1 Allocation Decision: 1 Control + 3 Treatment

**Rationale**: We have 4 server slots and a clear question (does the treatment work?). The key constraints:

1. **Run E (DB 10, control) already provides 1 historical control replicate** at test EM = 59.3%. This was run on the same infrastructure, same seed, same generation count, within 48 hours. Cross-batch variance is a concern but manageable.

2. **The seed itself (60.0% test EM) provides a second reference point** -- every treatment run starts here and must improve on it.

3. **We need 3 treatment replicates** to estimate the treatment mean with reasonable precision. With SD = 2.4pp: SE(mean of 3) = 1.39pp, giving a 95% CI width of +/-2.7pp.

4. **1 new control** run guards against infrastructure drift between P1xP2 batch and this batch. If the new control reproduces E's 59.3% (+/-2.4pp), the historical comparison is validated.

**Why not 0 control + 4 treatment**: If all 4 treatment runs show test EM = 62%, we cannot distinguish "NLP prompts work" from "infrastructure changed" without any concurrent control. One control is the minimum.

**Why not 2+2**: Power at 2 replicates is too low (58% for +3pp effect) and we already have Run E as a historical control.

### 4.2 Treatment Configuration

All treatment runs use:
- NLP-specific prompts (`prompts=hotpotqa` Hydra override)
- Standard pipeline (NOT hotpotqa_asi) -- we do NOT include P2 (ASI) in the base treatment
- Fixed 300-sample validation (NOT P1 rotation) -- we do NOT include P1 either
- Rationale: P1 and P2 showed null-to-negative effects. Adding them would confound the prompt intervention. If NLP prompts show a positive signal, we can test P1+P2 combination in a follow-up.

The control run uses the exact same config as Run E (standard pipeline, fixed 300 validation, default prompts).

### 4.3 Val Set Decision: Fixed 300

**Rationale**: P1 (rotation) showed null effect on test EM and a *worse* val-test gap than Run E in some conditions. Fixed-300 matches the control condition (Run E), enabling clean comparison. The 5.7pp val-test gap in Run E is a known property of this setup; it does not prevent detecting mutation quality improvements (a better mutation engine should improve both val and test EM).

## 5. Design Table

| Run | Label | Prompts | Pipeline | Problem Dir | DB | Mutation Server | Chain Server | Seed |
|-----|-------|---------|----------|-------------|----|-----------------|--------------|------|
| K | Control | default | standard | static | 14 | INTERNAL_IP:8777 | INTERNAL_IP:8001 | ddce37b4 |
| L | NLP-1 | hotpotqa | standard | static | 15 | INTERNAL_IP:8777 | INTERNAL_IP:8000 | ddce37b4 |
| M | NLP-2 | hotpotqa | standard | static | 16 | INTERNAL_IP:8777 | 10.225.185.235:8001 | ddce37b4 |
| N | NLP-3 | hotpotqa | standard | static | 17 | 10.225.51.251:8777 | 10.225.185.235:8000 | ddce37b4 |

**Run labels**: K-N continue the alphabetical sequence (E-H used by P1xP2).

**Server assignment**: Same 4 mutation servers and 4 chain server slots as P1xP2. Control (K) uses the same mutation+chain server pair as Run E to minimize cross-batch confounds.

**Redis DBs**: 14-17 (all free; DBs 10-13 retain P1xP2 data for reference).

## 6. Controlled Variables

Held constant across all 4 runs:
- Seed program: ddce37b4
- max_generations: 50
- max_mutations_per_generation: 16 (effective 8 with AllCombinationsParentSelector + num_parents=1 + max_elites=8)
- max_elites_per_generation: 8
- num_parents: 1
- primary_resolution: 50
- Chain LLM: Qwen3-8B (thinking mode, temp=0.6, top_p=0.95, top_k=20)
- Mutation LLM: Qwen3-235B-A22B-Thinking-2507
- step_max_tokens: {2: 4096, 3: 2048, 5: 4096, 6: 2048}
- Validation: fixed first 300 train samples
- Test set: 300 held-out samples (HotpotQA_test.jsonl)
- Problem directory: `problems/chains/hotpotqa/static` (same for all -- no P1 rotation, no P2 ASI)
- Pipeline: `standard`
- Redis: fresh DBs, no resume

**Only difference between K (control) and L/M/N (treatment)**: `prompts.dir` (null vs. path to `gigaevo/prompts/hotpotqa/`).

## 7. Intervention Specification: NLP-Specific Prompts

### 7.1 What Changes

Create `gigaevo/prompts/hotpotqa/` with NLP-chain-specific versions of 5 files and unchanged copies of 3 files:

| Agent | File | Change Type | Rationale |
|-------|------|-------------|-----------|
| insights | system.txt | **REWRITE** | Replace optimizer examples (threshold=0.5, popsize=100, IndexError) with NLP chain examples (query_formulation, instruction_clarity, evidence_synthesis) |
| insights | user.txt | **EDIT** | Replace "causal mechanism" hint about code optimization with NLP-chain analysis hints |
| lineage | system.txt | **REWRITE** | Replace strategy examples (O(n^2) loops, threshold 0.5->0.3, greedy selection) with prompt evolution examples (tightened format constraint, added entity bridging instruction) |
| lineage | user.txt | **EDIT** | Replace regression checklist (threshold changes, smaller populations) with NLP checklist (did format constraint weaken? did instruction length balloon?) |
| mutation | system.txt | **MINOR EDIT** | Reframe from "evolutionary optimization of python programs" to "evolutionary optimization of multi-step NLP reasoning chains" |
| mutation | user.txt | **COPY UNCHANGED** | task_description.txt already provides domain context |
| scoring | system.txt | **COPY UNCHANGED** | Trait-based scoring is already domain-agnostic |
| scoring | user.txt | **COPY UNCHANGED** | Already domain-agnostic |

### 7.2 Critical Constraint: No Functional Changes to Mutation Logic

The NLP prompts must preserve ALL structural elements of the original prompts:
- Same JSON output schemas (insights array, lineage array, mutation JSON with archetype/justification/code)
- Same tag vocabulary (beneficial, harmful, fragile, rigid, neutral)
- Same severity levels (high, medium, low)
- Same strategy types (imitation, avoidance, generalization, exploration, refinement)
- Same archetype framework (8 archetypes: Precision Optimization through Conservative Exploration)

Only the EXAMPLES and FRAMING change. The LLM's output parsing code is untouched.

### 7.3 Code Change: Fix `load_prompt` Fallback

The `load_prompt` function's docstring promises fallback behavior ("tries prompts_dir first; if the file is missing there, the package default directory is used") but the implementation does not implement this -- it raises `FileNotFoundError` instead of falling back.

**Fix**: Add proper fallback so that a custom prompts_dir only needs to contain the files it overrides. Files not present in the custom directory are loaded from the package default.

This is a zero-risk bug fix that aligns implementation with documented behavior. However, for this experiment we will provide ALL 8 files in the custom directory regardless, so the fix is not on the critical path.

### 7.4 Hydra Config

Create `config/prompts/hotpotqa.yaml`:
```yaml
# @package prompts
# NLP-chain-specific prompt templates for HotpotQA evolution.
# Replaces optimizer-centric examples in insights/lineage with
# prompt-chain-specific examples (query formulation, instruction clarity, etc.).
dir: ${hydra:runtime.cwd}/gigaevo/prompts/hotpotqa
```

Usage: `python run.py prompts=hotpotqa ...`

## 8. Metrics

### Primary
- **Test EM at generation 50** (300 held-out test samples, best-by-val program)

### Secondary
- Val EM trajectory per generation (from logs)
- Val-test gap at gen 10, 25, 50
- Per-generation archive acceptance rate (fraction of 8 mutants entering archive)
- Per-generation mean fitness delta (are mutations producing larger improvements?)
- Archive size at gen 10, 25, 50
- Best-by-val test EM at gen 10 and gen 25 (trajectory checkpoints)

### Diagnostic (treatment only, for understanding mechanism)
- Qualitative review of 5 randomly sampled mutation prompts at gen 10 and gen 40: are the insights/lineage outputs more relevant to NLP chain optimization?

## 9. Sample Size and Statistical Power

### Design
- 1 control + 3 treatment, plus 1 historical control (Run E)
- Noise floor: 2.4pp SD (from prior same-program retests)

### Power Analysis

Comparison: treatment mean (n=3) vs. pooled control mean (n=2, using K + historical E):

- SE(treatment mean) = 2.4 / sqrt(3) = 1.39pp
- SE(control mean) = 2.4 / sqrt(2) = 1.70pp
- SE(difference) = sqrt(1.39^2 + 1.70^2) = 2.19pp

Power to detect effect sizes (two-sided t-test, alpha=0.05, df=3):
- +3pp effect: ~62% power
- +4pp effect: ~78% power
- +5pp effect: ~90% power

This is acceptable. The NLP prompt intervention targets a fundamental mismatch in the system; if it works at all, we expect effects >= 3pp based on the magnitude of the framing mismatch. If the effect is smaller than 3pp, it is genuinely ambiguous and a larger study would be needed.

### Minimum Detectable Effect

With alpha=0.05 and 80% power: MDE = 4.2pp (treatment mean must exceed control mean by at least 4.2pp for statistical significance at 80% power). This corresponds to treatment mean >= 63.5% if control mean is 59.3%.

## 10. Success Criteria (Pre-registered)

### Gate 1: Treatment vs. Seed (Primary)
- **POSITIVE**: Treatment mean test EM > 62.0% (seed + 2pp, above noise floor)
- **STRONG POSITIVE**: Any treatment run test EM >= 62.3% (matches GEPA)
- **NULL**: Treatment mean test EM in [58.0%, 62.0%] (within noise of seed)
- **NEGATIVE**: Treatment mean test EM < 58.0% (NLP framing hurt)

### Gate 2: Treatment vs. Control (Concurrent Comparison)
- **SIGNIFICANT**: Treatment mean - control mean (pooled K+E) >= 4.2pp AND t-test p < 0.05
- **DIRECTIONAL POSITIVE**: Treatment mean > control mean by >= 2.4pp (exceeds noise floor) but p >= 0.05. Worth following up with more replicates.
- **NULL**: Difference < 2.4pp

### Gate 3: Control Consistency Check
- **VALID**: Run K test EM within [56.9%, 61.7%] (Run E +/- 2.4pp). If outside this range, cross-batch comparison with Run E is invalidated and we must use only K as the control reference.

### Gate 4: Val-Test Gap
- **HEALTHY**: Treatment mean val-test gap < 5pp (no worse than Run E)
- **CONCERNING**: Treatment mean val-test gap > 7pp (NLP prompts encouraging overfitting)

### Gate 5: Next Steps Decision Matrix

| Gate 1 | Gate 2 | Next Action |
|--------|--------|------------|
| STRONG POSITIVE | SIGNIFICANT | Replication with 3 seeds (paper-ready). Test NLP prompts + P1+P2 combination. |
| POSITIVE | DIRECTIONAL | 2 more treatment replicates for significance. |
| POSITIVE | NULL | Possible but unlikely. Review prompt quality qualitatively. |
| NULL | NULL | Move to P3 crossover (structural intervention). Accept prompt-only ceiling. |
| NEGATIVE | any | Revert. Investigate what went wrong in prompt outputs qualitatively. |

## 11. Monitoring Plan

### Gen 5 (~2h after launch): Smoke Check
- All 4 runs progressing (check PIDs, Redis key counts)
- No crashes, no stuck runs
- Archive acceptance rate > 0% for all runs
- **Do NOT look at fitness values** -- too early, and looking creates temptation to intervene

### Gen 10 (~5h): First Checkpoint
- Extract best-by-val program for each run
- Run test eval on all 4 (4 x 5 min = 20 min)
- Record: val EM, test EM, acceptance rate, archive size
- **Decision**: If any run has acceptance rate = 0% for gens 5-10, it has stalled. Note this but do NOT intervene (stalling is an informative outcome).
- **NO early stopping**. The experiment runs to gen 50 regardless of gen-10 results.

### Gen 25 (~14h): Midpoint Checkpoint
- Same metrics as gen 10
- Check val-test gap trajectory
- If treatment runs consistently outperform control at gen 25, this is encouraging but NOT conclusive
- Still **no early stopping**

### Gen 50 (~24-28h): Final Evaluation
- Full test eval for all 4 runs (best-by-val at max generation)
- Complete statistical analysis per Section 10
- Write results report

### Early Termination Criteria
- **Infrastructure failure**: If a mutation server or chain server goes down and cannot be restored within 2 hours, that run is lost. Do NOT restart -- record partial results.
- **Run crash**: If a run crashes after gen 25, use gen-25 checkpoint as the final result. If before gen 25, the run is excluded from analysis.
- **No data-based early stopping**: The experiment runs to completion regardless of intermediate results. This prevents bias from peeking.

## 12. Implementation Plan

### Phase 1: Prompt Files (2-3 hours)

1. Create `gigaevo/prompts/hotpotqa/` directory structure:
   ```
   gigaevo/prompts/hotpotqa/
     insights/system.txt   -- REWRITE
     insights/user.txt     -- EDIT
     lineage/system.txt    -- REWRITE
     lineage/user.txt      -- EDIT
     mutation/system.txt    -- MINOR EDIT
     mutation/user.txt      -- COPY from default
     scoring/system.txt     -- COPY from default
     scoring/user.txt       -- COPY from default
   ```

2. Create `config/prompts/hotpotqa.yaml`

3. Fix `load_prompt` fallback behavior (optional, not on critical path since we provide all files)

### Phase 2: Validation (1 hour)

1. Unit test: load all 8 prompts via `load_prompt("insights", "system", prompts_dir="gigaevo/prompts/hotpotqa")` etc. Verify no errors.
2. Dry run: `python run.py problem.name=chains/hotpotqa/static prompts=hotpotqa max_generations=0` -- verify config resolves correctly.
3. Diff review: side-by-side comparison of default vs. NLP prompts to verify structural preservation (same JSON schema, same tags, same archetypes).

### Phase 3: Launch Infrastructure (30 min)

1. Verify all 8 servers are reachable and in thinking mode
2. Flush Redis DBs 14-17
3. Create `experiments/hotpotqa_nlp_prompts/` directory with launch.sh, watchdog, gen_stats
4. Launch all 4 runs

### Phase 4: Monitoring and Evaluation (24-28 hours)

Per monitoring plan above.

## 13. Launch Script Specification

```bash
# Common parameters (same as P1xP2)
COMMON_PARAMS=(
    num_parents=1
    primary_resolution=50
    max_mutations_per_generation=16
    max_elites_per_generation=8
    max_generations=50
    program_loader.problem_dir="$SEED_DIR"
)

# Run K: control (default prompts)
python run.py ${COMMON_PARAMS[@]} \
    problem.name=chains/hotpotqa/static \
    pipeline=standard \
    redis.db=14 \
    llm_base_url="http://localhost:8000/v1"

# Run L: NLP prompts
python run.py ${COMMON_PARAMS[@]} \
    problem.name=chains/hotpotqa/static \
    pipeline=standard \
    prompts=hotpotqa \
    redis.db=15 \
    llm_base_url="http://localhost:8000/v1"

# Run M: NLP prompts
python run.py ${COMMON_PARAMS[@]} \
    problem.name=chains/hotpotqa/static \
    pipeline=standard \
    prompts=hotpotqa \
    redis.db=16 \
    llm_base_url="http://localhost:8000/v1"

# Run N: NLP prompts
python run.py ${COMMON_PARAMS[@]} \
    problem.name=chains/hotpotqa/static \
    pipeline=standard \
    prompts=hotpotqa \
    redis.db=17 \
    llm_base_url="http://10.225.51.251:8777/v1"
```

## 14. Risk Register

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| NLP examples confuse the mutation LLM (wrong output format) | Mutations fail to parse; acceptance rate drops to 0 | LOW | Pre-validate JSON schema preservation. Mutation output parsing is well-tested. |
| NLP framing causes overly conservative mutations (all "refinement") | Archive stagnation; no improvement over seed | MEDIUM | Include exploration examples in prompts. Monitor archetype distribution in mutation logs. |
| NLP prompts are too HotpotQA-specific (overfitting to our chain) | Results do not generalize | LOW for this study | We only care about HotpotQA performance now. Generalization is a future concern. |
| Control (K) differs from historical E by > 2.4pp | Cannot use E as reference; lose 1 control replicate | MEDIUM | Run K on same infrastructure as E. If K differs, use only K as control (reduces power). |
| Chain server restart in non-thinking mode | Affected run produces invalid results | LOW | Preflight thinking-mode check. Watchdog periodic verification. |
| Mutation server crash | Lost run | LOW-MEDIUM | Monitor PIDs. No restart -- record partial results if after gen 25. |

## 15. What This Experiment Does NOT Test

To avoid scope creep, explicitly noting what is excluded:
- **P1 (validation rotation)**: Not included. Showed null effect.
- **P2 (ASI retrieval diagnostics)**: Not included. Showed null-to-negative effect alone (Run F).
- **P3 (crossover / num_parents=2)**: Deferred to after this experiment.
- **task_description.txt changes**: The task description is unchanged. Only the agent prompt templates change.
- **Scoring prompts**: Copied unchanged. Scoring is domain-agnostic and not on the bottleneck path.
- **Mutation user.txt archetype framework**: The 8 archetypes and their selection logic are unchanged. Only the system prompt framing and the upstream agent outputs change.

## 16. Appendix: Prompt Change Summary

### insights/system.txt Changes

**REMOVE** (optimizer examples):
- `threshold=0.5 at line 23 causes 15% of valid candidates to be rejected early`
- `try/except at line 45 catches ValueError but masks useful debug info`
- Categories like `threshold_tuning`, `loop_bounds`, `edge_case`
- Example insights about `population_size`, `boundary_handling`, `selection_pressure`

**ADD** (NLP chain examples):
- `step_3 generates overly verbose queries (~40 words) that dilute BM25 retrieval precision; tightening to <15 words may improve hop-2 recall`
- `step_5 combine_evidence aim repeats step_2 summarization instead of synthesizing across hops; deduplicated synthesis may reduce answer conflicts`
- Categories like `query_formulation`, `instruction_clarity`, `evidence_synthesis`, `answer_extraction`, `format_constraint`, `reasoning_scaffold`
- Example insights about `step_coordination`, `information_flow`, `role_specialization`

### lineage/system.txt Changes

**REMOVE** (optimizer strategy examples):
- `Changed threshold from 0.5 to 0.3, catching 20% more edge cases`
- `Removed greedy selection that was discarding valid candidates too early`
- `Replaced O(n^2) nested loops with sorted-merge approach`

**ADD** (prompt evolution examples):
- `Tightened step_3 stage_action from 45 words to 12 words ("Generate a focused search query"); shorter instruction produced more precise BM25 queries; +0.02 EM`
- `Removed conflicting step_5 rules ("be comprehensive" vs "be concise"); eliminating contradiction improved evidence quality; +0.015 EM`
- `Added explicit entity-bridging instruction to step_2 aim ("identify the bridge entity connecting the question to retrieved passages"); improved hop-2 query relevance; +0.025 EM`

### lineage/user.txt Changes

**REMOVE** (optimizer regression checklist):
- Did a successful heuristic get weakened? (threshold changes, bound restrictions)
- Did exploration get reduced? (smaller populations, tighter constraints, fewer iterations)

**ADD** (NLP chain regression checklist):
- Did a clear step instruction get diluted? (aim/stage_action became vague or contradictory)
- Did a step's output format constraint get weakened? (step_3 query generation no longer constrained to short queries)
- Did system_prompt length balloon? (context dilution across all steps)
- Did step_6 answer extraction format change? (risk of extraction failures)

### mutation/system.txt Changes

**CHANGE**: First line from:
`You are an expert in evolutionary optimization, focusing on performance-driven mutation of python programs.`
To:
`You are an expert in evolutionary optimization of multi-step NLP reasoning chains, focusing on performance-driven mutation of prompt configurations.`

**CHANGE**: ROLE section to mention prompt-chain context.

---

## 17. Timeline

| Hour | Action |
|------|--------|
| 0-2 | Write NLP prompt files. Create Hydra config. |
| 2-3 | Validate: dry run, JSON schema check, diff review. |
| 3-3.5 | Preflight: verify servers, flush DBs 14-17, launch. |
| 3.5-5.5 | Gen 5 smoke check. |
| 5.5-8 | Gen 10 checkpoint. Test evals. |
| 8-17 | Gen 25 checkpoint. Test evals. |
| 17-28 | Gen 50 final evals. Analysis. Report. |

**Total wall-clock**: ~28 hours from implementation start to final results.
**Compute**: 4 runs x ~27 GPU-hours = ~108 GPU-hours.

---

*Pre-registered before any treatment prompt files are written or any results are observed.*
