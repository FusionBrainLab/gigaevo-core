# P3 Crossover / num_parents=2 -- Pre-Planning Analysis and Experimental Design

**Date**: 2026-03-01
**Investigator**: Dr. Elena Voss (ML Research Methodologist)
**Status**: Pre-planning (contingent on P1xP2 factorial results from Runs E/F/G/H)
**Depends on**: `docs/plans/2026-03-01-p1xp2-factorial.md`

---

## 0. Executive Summary

This document analyzes the feasibility, risks, and experimental design for introducing
`num_parents=2` (crossover/merge) into HotpotQA static chain evolution. The core tension
is between the potential benefit of crossover (combining complementary strategies from two
parent programs to escape the ddce37b4 fitness basin) and the cost of doubling the
mutation prompt size, which may degrade mutation quality on a thinking-mode LLM with
constrained context.

**Key findings**:
1. Two-parent prompts are feasible within the 32k context window but leave reduced
   headroom for thinking-mode internal reasoning (~8-10k tokens free).
2. Crossover should be tested AFTER P1xP2 results are in, not in parallel, to avoid
   confounding and to inform the choice of baseline config.
3. The optimal design is a 2-cell experiment (num_parents=1 vs. 2) on top of whichever
   P1/P2 configuration wins, with archive maturity gating (crossover enabled only after
   archive size >= 5).
4. Reducing failure cases from 10 to 5 for 2-parent runs is recommended to stay safely
   within context limits.

---

## 1. Context Budget Analysis

### 1.1 Anatomy of a Single-Parent Mutation Prompt

The mutation LLM (Qwen3-235B-A22B-Thinking-2507, served via vLLM) receives a prompt
assembled by `MutationAgent.build_prompt()` (`gigaevo/llm/agents/mutation.py:160-216`).
The prompt has three parts:

**System prompt** (from `gigaevo/prompts/mutation/system.txt`):
```
You are an expert in evolutionary optimization...
OBJECTIVE: {task_description}
AVAILABLE METRICS: {metrics_description}
```
Estimated size: ~200-400 tokens (short, fixed).

**User prompt** (from `gigaevo/prompts/mutation/user.txt`):
The user prompt template is ~1,800 chars (~450 tokens) of fixed instruction text
(archetype framework, output format, execution principles), plus the `{parent_blocks}`
variable which contains per-parent content.

**Per-parent block** (constructed in `build_prompt()`, lines 178-189):
```
=== Parent {i + 1} ===
```python
{p.code}
```

{formatted_context}
```

The `formatted_context` is assembled by `MutationContextStage`
(`gigaevo/programs/stages/mutation_context.py`) and includes:
- **Metrics block**: ~100 tokens (fitness, extraction failures, validity)
- **Insights block**: ~200-400 tokens (3-6 insights with tags/severity)
- **Family tree / lineage block**: ~300-600 tokens (parent/child transition analyses)
- **Evolutionary statistics block**: ~200-400 tokens (generation history table, 7 rows)
- **Formatted failure cases** (from FormatterStage or HotpotQAFailureFormatter):
  - Without ASI (P2=OFF): ~80 tokens per case x 10 cases = ~800 tokens
  - With ASI (P2=ON): ~120 tokens per case x 10 cases = ~1,200 tokens

**Per-parent code size**:
- Baseline program: 3,736 chars (~930 tokens)
- ddce37b4 evolved program: 8,453 chars (~2,100 tokens)
- Typical evolved programs after 20+ gens: 6,000-10,000 chars (~1,500-2,500 tokens)

### 1.2 Token Budget Breakdown

| Component | 1 parent (no ASI) | 1 parent (with ASI) | 2 parents (no ASI) | 2 parents (with ASI) |
|-----------|-------------------|---------------------|--------------------|-----------------------|
| System prompt | 350 | 350 | 350 | 350 |
| User prompt (fixed) | 450 | 450 | 450 | 450 |
| Parent code (each) | 2,100 | 2,100 | 2,100 x 2 = 4,200 | 2,100 x 2 = 4,200 |
| Metrics (each) | 100 | 100 | 100 x 2 = 200 | 100 x 2 = 200 |
| Insights (each) | 300 | 300 | 300 x 2 = 600 | 300 x 2 = 600 |
| Lineage (each) | 450 | 450 | 450 x 2 = 900 | 450 x 2 = 900 |
| Evo stats (each) | 300 | 300 | 300 x 2 = 600 | 300 x 2 = 600 |
| Failure cases (each) | 800 (10 cases) | 1,200 (10 cases) | 800 x 2 = 1,600 | 1,200 x 2 = 2,400 |
| **Total input** | **~4,850** | **~5,250** | **~8,900** | **~9,700** |

**Qwen3-235B thinking-mode context budget**:
- Total context window: 32,768 tokens (as configured in vLLM; the model supports up to
  131k natively, but vLLM `max_model_len` is typically set to 32k for throughput)
- Thinking tokens (internal CoT): ~8,000-15,000 tokens for complex reasoning
- Output tokens (structured JSON with code): ~3,000-4,000 tokens
- **Available for input**: 32,768 - 15,000 (thinking) - 4,000 (output) = ~13,768 tokens

### 1.3 Feasibility Assessment

| Configuration | Input tokens | Headroom for thinking | Assessment |
|---------------|-------------|----------------------|------------|
| 1 parent, no ASI | ~4,850 | ~23,918 | Comfortable |
| 1 parent, with ASI | ~5,250 | ~23,518 | Comfortable |
| 2 parents, no ASI | ~8,900 | ~19,868 | Safe |
| 2 parents, with ASI | ~9,700 | ~19,068 | Safe but reduced |
| 2 parents, ASI, 10 cases each | ~9,700 | ~19,068 | Tight if programs grow |
| 2 parents, ASI, 5 cases each | ~8,500 | ~20,268 | Comfortable |

**Conclusion**: Two-parent prompts are feasible in all configurations. However, if
programs continue to grow beyond ~10k chars (2,500 tokens) each -- which is plausible
after 50+ generations of evolution -- the 2-parent + ASI + 10 cases configuration
could push input to ~12,000+ tokens, leaving only ~16k for thinking + output. This
is still within bounds but leaves less room for the model's internal reasoning.

**Recommendation**: For num_parents=2 runs, reduce failure cases from 10 to 5 per parent.
This saves ~1,200 tokens (with ASI) or ~800 tokens (without ASI) and keeps the prompt
well within bounds even with large evolved programs. The top-5 failures are generally the
most informative anyway (they represent the most common failure modes).

### 1.4 Implementation Detail: Failure Case Reduction

The `HotpotQAFailureFormatter.format_value()` already takes `failures[:10]`
(`problems/chains/hotpotqa/static/formatter.py:22`). For 2-parent runs, we need a
configuration mechanism:

**Option A** (simple): Create `HotpotQAFailureFormatter5` that slices `[:5]`.
**Option B** (cleaner): Add `max_cases` parameter to formatter, default 10, override to 5
for 2-parent runs.
**Option C** (no code change): Accept the 10-case cost; it fits within context.

Recommendation: Option C for the initial experiment. The token budget analysis shows it
fits. Only implement truncation if we observe degraded mutation quality (measurable as
lower acceptance rate or more invalid programs).

---

## 2. When to Introduce num_parents=2

### 2.1 Archive Maturity Requirement

`AllCombinationsParentSelector` with `num_parents=2` produces `C(N, 2)` combinations
from the selected elites. The number of distinct mutations per generation depends on
archive size:

| Archive size | Elites selected (min(archive, 8)) | C(N, 2) | Capped at 16 | Effective mutations/gen |
|--------------|-----------------------------------|---------|--------------|------------------------|
| 1 | 1 | 0 | 0 | **0 -- BROKEN** |
| 2 | 2 | 1 | 1 | 1 |
| 3 | 3 | 3 | 3 | 3 |
| 5 | 5 | 10 | 10 | 10 |
| 7 | 7 | 21 | 16 | 16 |
| 8+ | 8 | 28 | 16 | 16 |

**Critical observation**: With `num_parents=2` and `archive_size=1` (the initial state
when warm-starting from ddce37b4), `C(1, 2) = 0`. Zero mutations would be generated.
The system would stall at generation 0.

This is a known issue (documented in agent memory: "num_parents=2 + archive_size=1 =
1 mutation/gen. Use num_parents=1 for single-seed starts"). However, the actual behavior
is worse than "1 mutation/gen" -- it is ZERO mutations, because `combinations([x], 2)`
yields an empty iterator.

### 2.2 Design Options

**Option A: num_parents=2 from generation 0, with fallback**
- Requires a code change to `AllCombinationsParentSelector`: when archive_size < 2,
  fall back to single-parent mutation.
- Pro: Clean, no mid-run config changes.
- Con: Early generations (archive 1-3) are effectively single-parent anyway, so the
  crossover benefit only kicks in after archive maturity. First ~5-10 generations are
  wasted from a crossover perspective.

**Option B: Switch mid-run at a fixed generation (e.g., gen 15 or gen 25)**
- Requires: either a config-time schedule or manual intervention.
- Pro: Ensures sufficient archive diversity before crossover begins.
- Con: Introduces a confound (the "switch" event itself). Also requires pausing and
  reconfiguring the run, which risks Redis state issues.

**Option C: num_parents=2 from gen 0, with `num_parents=1` fallback in the selector**
- The `AllCombinationsParentSelector` already handles `len(parents_copy) < num_parents`
  by yielding all parents once (line 75-79 in `parent_selector.py`). When archive has
  1 program and `num_parents=2`, `len([x]) < 2` is True, so it yields `[x]` -- a
  single-parent selection.
- This means `mutate_single` receives `[x]` (1 parent), which the `MutationAgent`
  handles fine (it just builds 1 parent block).
- **BUT**: Only 1 mutation per generation until archive >= 2. With archive=2, C(2,2)=1,
  so still 1 mutation/gen. With archive=3, C(3,2)=3, so 3 mutations/gen.
- This is the current framework behavior -- no code change needed.

**Option D: Hybrid -- start with num_parents=1 for gen 0-N, then switch to num_parents=2**
- GigaEvo does not currently support dynamic config changes mid-run.
- Would require implementing a generation-based scheduler for parent_selector config.
- Over-engineered for a single experiment.

### 2.3 Recommendation

**Use Option C (num_parents=2 from gen 0, relying on the existing fallback).**

Rationale:
1. No code changes required for the selector; the fallback path already works.
2. Early generations with small archive will effectively run as single-parent mutation,
   which matches the P1xP2 factorial's early behavior.
3. The crossover benefit naturally ramps up as the archive grows, which is the desired
   behavior -- crossover is most useful when there are diverse programs to combine.
4. The cost is reduced throughput in early generations (1-3 mutations/gen instead of 8).
   This is acceptable because the ddce37b4 warm-start gives a strong initial program;
   early gens are about exploring nearby variants, not maximizing throughput.

**Mitigation for early-gen throughput loss**: Use `max_elites_per_generation=8` and
`max_mutations_per_generation=16` (same as P1xP2). With num_parents=2, the 16-mutation
cap is reached when archive >= 7 (C(7,2) = 21 > 16). Until then, throughput grows
organically.

---

## 3. Interaction Analysis: P3 vs. P1 and P2

### 3.1 P3 x P1 (Crossover x Validation Rotation) -- CONFOUND RISK

**Problem**: With P1 (validation rotation), two parents may have been evaluated on
DIFFERENT 300-sample subsets (different generations). Their fitness scores are not
directly comparable.

When `MutationAgent.build_prompt()` constructs the 2-parent prompt, each parent's
`formatted_context` includes its metrics (fitness score). The mutation LLM sees:
```
=== Parent 1 ===
## Program Metrics
fitness: 0.627
...

=== Parent 2 ===
## Program Metrics
fitness: 0.610
...
```

With P1, Parent 1's 0.627 might be on an "easy" subset while Parent 2's 0.610 might be
on a "hard" subset. The mutation LLM would naively trust Parent 1 as "better," which is
misleading.

**Severity**: MODERATE. The cross-subset noise is ~2pp (established in prior experiment).
So the comparison is slightly noisy but not catastrophically wrong. The mutation LLM
primarily uses the code and failure cases (not the raw fitness number) to guide its
changes. The fitness number influences archetype selection (exploitation vs. exploration)
but not the specific code changes.

**Mitigation options**:
1. **Do not test P3 with P1 initially.** Run the first P3 experiment with P1=OFF
   (fixed validation set). This isolates the crossover effect cleanly.
2. **Annotate parent fitness with generation number.** The mutation LLM can see which
   generation each parent is from and can infer that fitness comparisons across gens
   are approximate. Low implementation cost, unclear benefit.
3. **Re-evaluate both parents on the current generation's subset before mutation.**
   High cost (~42 min per generation for 2 re-evaluations). Not practical.
4. **Accept the noise.** The LLM's attention is primarily on code structure and failure
   patterns, not fitness numbers. The ~2pp noise is unlikely to change the mutation
   strategy fundamentally.

**Recommendation**: Mitigation 1 for the initial P3 experiment. If P3 shows positive
results without P1, a follow-up can test P3+P1 together.

### 3.2 P3 x P2 (Crossover x ASI Diagnostics) -- CONTEXT BUDGET INTERACTION

**Problem**: With P2, each parent's failure artifact includes per-hop retrieval recall
(~120 tokens/case x 10 cases = 1,200 tokens). With 2 parents, that is 2,400 tokens
of retrieval diagnostics.

**Severity**: LOW. As shown in Section 1.3, the total input stays within ~9,700 tokens
even with both P2 and P3, which is well within the 32k window. The question is whether
the LLM can effectively use retrieval diagnostics from TWO parents simultaneously.

**Potential benefit**: If Parent A has good hop-1 retrieval but bad hop-2, and Parent B
has the reverse, the LLM could synthesize a child that combines A's hop-1 query strategy
with B's hop-2 approach. This is precisely the kind of complementary combination that
crossover is designed to find.

**Recommendation**: Test P3 with P2 in a follow-up experiment ONLY IF P2 shows a positive
effect in the P1xP2 factorial. If P2 is null, there is no reason to compound it with P3.

### 3.3 Decision Matrix: Which Config to Build P3 On

The P3 baseline should be the best-performing configuration from the P1xP2 factorial:

| P1xP2 outcome | P3 baseline config | Rationale |
|----------------|-------------------|-----------|
| H wins (P1+P2) | P1+P2 | Build on the best. But P3 x P1 confound exists (Section 3.1). Consider P2-only base. |
| F wins (P2 only) | P2 only | Clean: crossover benefits from ASI, no rotation confound. |
| G wins (P1 only) | P1 only | But P3 x P1 confound (Section 3.1). Consider control base. |
| E wins (control) | Control | Simplest. No interactions to worry about. |
| Both null | Control | Default to simplest. |

**Most likely scenario** (speculative, pre-registered before P1xP2 results): The control
(E) or P2-only (F) will be the cleanest base for P3.

---

## 4. Experimental Design for P3

### 4.1 Research Question

**Does `num_parents=2` (crossover) improve test EM on HotpotQA static chains beyond
the best `num_parents=1` configuration, within 50 generations from the ddce37b4 warm
start?**

### 4.2 Hypotheses

- **H0**: `num_parents=2` does not improve test EM vs. `num_parents=1` (difference
  < 2.4pp, within noise floor).
- **H1**: `num_parents=2` improves test EM by >= 3pp vs. `num_parents=1`.

Secondary hypotheses:
- **H0_stag**: `num_parents=2` does not reduce the probability of stagnation (defined
  as 0 archive replacements for >= 15 consecutive generations).
- **H1_stag**: `num_parents=2` reduces stagnation probability by enabling the LLM to
  combine diverse strategies.

### 4.3 Experimental Conditions

This is a 2-cell experiment (not factorial). The baseline is the best P1xP2 config.

| Run | Label | num_parents | Other config | DB | Notes |
|-----|-------|-------------|-------------|-----|-------|
| I | Control (1-parent) | 1 | Best P1xP2 config | 14 | Replication of E/F/G/H winner |
| J | Treatment (2-parent) | 2 | Same as Run I | 15 | Only num_parents changes |

**Why not a larger factorial?** The P1xP2 experiment already tests 4 conditions. Adding
num_parents as a third factor would require 8 cells (2x2x2), consuming 8 GPU-weeks. With
N=1 per cell, statistical power would be abysmal. A focused 2-cell comparison is more
efficient: it uses 2 runs to answer 1 question cleanly.

**Why include Run I (replication) instead of reusing the E/F/G/H winner?** Baseline
hygiene. The E/F/G/H runs use a potentially different codebase version, different time
period, and possibly different server conditions. Running the control and treatment
simultaneously on the same infrastructure eliminates these confounds. Additionally, this
provides a second independent replication of the best P1xP2 config, which has statistical
value given N=1 in the factorial.

### 4.4 Controlled Variables

Identical across Runs I and J:
- Seed program: ddce37b4
- max_generations: 50
- max_elites_per_generation: 8
- max_mutations_per_generation: 16
- primary_resolution: 50
- Chain LLM: Qwen3-8B (thinking mode)
- Mutation LLM: Qwen3-235B-A22B-Thinking-2507
- Pipeline: whichever P1/P2 config won (to be determined)
- Validator: matching the pipeline choice
- Redis: fresh DBs (14, 15)
- Test set: 300 held-out samples
- Mutation prompt template: identical (user.txt unchanged)

**Only difference**: `num_parents=1` vs. `num_parents=2` in the Hydra override.

### 4.5 Config Commands

```bash
# Run I (control, num_parents=1)
HOTPOTQA_CHAIN_URL="..." ~/envs/evo_fast/bin/python run.py \
    problem.name=chains/hotpotqa/static \
    num_parents=1 \
    primary_resolution=50 \
    max_mutations_per_generation=16 \
    max_elites_per_generation=8 \
    max_generations=50 \
    redis.db=14 \
    llm_base_url="http://<MUT_SERVER>:8777/v1" \
    program_loader.problem_dir="experiments/hotpotqa_thinking/seeds/ddce37b4" \
    [pipeline=... if P2 won]

# Run J (treatment, num_parents=2)
HOTPOTQA_CHAIN_URL="..." ~/envs/evo_fast/bin/python run.py \
    problem.name=chains/hotpotqa/static \
    num_parents=2 \
    primary_resolution=50 \
    max_mutations_per_generation=16 \
    max_elites_per_generation=8 \
    max_generations=50 \
    redis.db=15 \
    llm_base_url="http://<MUT_SERVER>:8777/v1" \
    program_loader.problem_dir="experiments/hotpotqa_thinking/seeds/ddce37b4" \
    [pipeline=... if P2 won]
```

### 4.6 Throughput Analysis

**Important difference in mutations/generation**:

With `num_parents=1`, `AllCombinationsParentSelector` produces C(N,1) = N mutations
(where N = min(archive_size, max_elites=8)). So at archive maturity (8+), we get
8 mutations/gen, well below the 16 cap.

With `num_parents=2`, C(8,2) = 28, capped at 16. So we get 16 mutations/gen at archive
maturity -- **double the throughput** of num_parents=1.

This is both a feature and a confound:
- **Feature**: More mutations per generation means more exploration per unit time.
  If crossover helps, the doubling of throughput amplifies the effect.
- **Confound**: If Run J outperforms Run I, is it because crossover produces better
  mutations, or because Run J generates twice as many mutations per generation?

**Controlling for throughput**: Two approaches:

**Approach 1 (equal mutations/gen)**: Set `max_mutations_per_generation=8` for Run J
so that both runs produce 8 mutations/gen at maturity. This isolates the crossover
effect from the throughput effect.

**Approach 2 (equal total mutations)**: Run J for 25 gens (25 x 16 = 400 total mutations)
vs. Run I for 50 gens (50 x 8 = 400 total mutations). This equalizes total compute but
introduces a generation-count confound.

**Approach 3 (accept the confound, measure it)**: Run both for 50 gens with the same
`max_mutations_per_generation=16`. Compare at equal generation count (50 gens) and also
at equal total-mutation count (Run I gen 50 vs. Run J gen ~25). If Run J wins at equal
gen AND at equal total mutations, crossover is clearly beneficial. If Run J wins only
at equal gens (where it has 2x throughput), the benefit may be purely throughput.

**Recommendation**: **Approach 1 (equal mutations/gen)**. Set `max_mutations_per_generation=8`
for both runs. With `num_parents=2` and `max_elites=8`, we get C(8,2)=28 combinations
but cap at 8. This cleanly isolates crossover quality from throughput.

If crossover shows a positive effect with equal throughput, a follow-up can test
the throughput bonus by uncapping to 16.

**Revised config**: Both runs use `max_mutations_per_generation=8`.

### 4.7 Note on Mutation Parallelism

All mutations within a generation are generated in parallel
(`gigaevo/evolution/engine/mutation.py:88-93`, `asyncio.gather(*tasks)`). With 8
mutations/gen, all 8 mutation LLM calls run concurrently. Each mutation call to
Qwen3-235B-A22B-Thinking takes ~2-4 minutes. So the per-generation mutation wall-clock
time is ~4 min (bounded by the slowest call), not 8 x 4 = 32 min. This is identical
for both runs under equal throughput.

### 4.8 Metrics

**Primary**: Test EM at generation 50 (300 held-out test samples).

**Secondary**:
- Test EM at gen 10, 25 (early and mid checkpoints)
- Val EM trajectory per generation
- Val-test gap at gen 50
- Per-generation acceptance rate (fraction of 8 mutants entering archive)
- Archive diversity: number of distinct archive cells occupied at gen 50
- Stagnation episodes: count of consecutive-generation stretches with 0 archive replacements
- Mutation prompt size (tokens): mean and max per generation (verify budget analysis)

### 4.9 Sample Size and Statistical Power

With N=1 per condition, we cannot run formal hypothesis tests. We use the same calibrated
threshold approach as the P1xP2 factorial:
- Effects < 2.4pp: indistinguishable from noise
- Effects 2.4-5pp: suggestive, worth replicating
- Effects > 5pp: likely real

**External reference distribution** (from 6 prior runs with num_parents=1, all ddce37b4
seed): Runs B, D, E, F, G, H provide 6 data points for the num_parents=1 condition.
Run J (num_parents=2) can be compared against this distribution. If Run J falls above
the 95th percentile of the 6 num_parents=1 runs, that is strong evidence.

### 4.10 Decision Gates (Pre-registered)

**POSITIVE (crossover helps)**:
- Run J test EM exceeds Run I test EM by >= 3pp AND
- Run J test EM >= 60% (minimum viability threshold) AND
- Run J acceptance rate is not lower than Run I (crossover is not just noisier)

**SUGGESTIVE (worth replicating)**:
- Run J test EM exceeds Run I by 1-3pp, or
- Run J shows significantly less stagnation (>= 5 fewer consecutive 0-acceptance gens)

**NULL (crossover does not help)**:
- Run J test EM within +/- 2.4pp of Run I

**NEGATIVE (crossover hurts)**:
- Run J test EM is >= 3pp below Run I, suggesting context bloat degrades mutation quality

### 4.11 Follow-up Decision Table

| Outcome | Next action |
|---------|-------------|
| Positive | Replicate with 2 more seeds. Then test P3 + throughput bonus (uncap to 16). |
| Suggestive | 2 more replications (same seeds) to resolve ambiguity. |
| Null | Test at gen 50 only (warm-start from gen-25 archive to give archive more diversity). |
| Negative | Investigate: is it context length, or bad crossover combinations? Try with reduced context. |

---

## 5. Timing: Sequential After P1xP2

### 5.1 Why NOT Run P3 in Parallel with E/F/G/H

1. **Baseline selection depends on P1xP2 results.** We do not know which config (E/F/G/H)
   is best until that experiment completes. Running P3 on the wrong baseline wastes
   compute and produces confounded results.

2. **Server contention.** We have 4 mutation servers and 4 chain server ports, all
   currently allocated to Runs E/F/G/H. Adding 2 more runs (I, J) requires either
   sharing servers (introducing latency confounds) or waiting for servers to free up.

3. **Context budget uncertainty.** The token budget analysis in Section 1 is theoretical.
   We should first verify actual prompt sizes from E/F/G/H logs before committing to
   2-parent runs. If evolved programs grow larger than expected, we may need to adjust.

4. **Cognitive load.** Running 6 experiments simultaneously with different configs
   increases the risk of monitoring errors and misattribution.

### 5.2 Recommended Timeline

| Phase | Timeframe | Action |
|-------|-----------|--------|
| 1 | Now - 28h | P1xP2 factorial runs (E/F/G/H) execute |
| 2 | 28-32h | Analyze P1xP2 results, select best config |
| 3 | 32-36h | Implement any P3-specific changes (failure case limit if needed) |
| 4 | 36h | Launch Runs I, J |
| 5 | 36-64h | P3 runs execute (50 gens, ~24-28h each) |
| 6 | 64-68h | Analyze P3 results, write report |

**Total elapsed time**: ~3 days from P1xP2 launch to P3 results.

### 5.3 Redis DB Allocation

- DBs 10-13: P1xP2 factorial (Runs E/F/G/H)
- DB 14: P3 control (Run I, num_parents=1)
- DB 15: P3 treatment (Run J, num_parents=2)

No conflicts. DBs 14-15 are currently available.

### 5.4 Server Allocation

Runs I and J need:
- 2 mutation servers (1 per run) -- reuse 2 of the 4 after E/F/G/H complete
- 2 chain server ports (1 per run) -- reuse from E/F/G/H

Pairing: allocate the same mutation server to I and J's pair mate from E/F/G/H if
possible, to reduce infrastructure variance.

---

## 6. Architecture Notes and Edge Cases

### 6.1 Diff Mode Incompatibility

`LLMMutationOperator.mutate_single()` (`mutation_operator.py:105-108`) raises an error
if `mutation_mode == "diff"` and `len(selected_parents) != 1`. The current HotpotQA
config uses `mutation_mode: rewrite`, so this is not an issue. But it must be verified
before launch.

### 6.2 Parent Block Construction

In `MutationAgent.build_prompt()` (lines 178-189), the loop `for i, p in enumerate(parents)`
naturally handles 2 parents. Each parent gets its own `=== Parent {i+1} ===` block
with code and formatted context. No code changes needed.

### 6.3 Lineage Recording

When `num_parents=2`, `Program.from_mutation_spec(mutation_spec)` records both parent IDs
in `program.lineage.parents`. The `generate_mutations()` function (engine/mutation.py:74-78)
updates both parents' `lineage.children` lists. All lineage stages (AncestorProgramIds,
FamilyTreeMutationContext) iterate over the full parents list. No issues expected.

### 6.4 Archive Cell Competition

With `num_parents=2`, each mutant still occupies exactly one archive cell (determined
by its behavior features). The `SumArchiveSelector` compares the mutant's fitness against
the incumbent in that cell. Crossover does not change the archive dynamics -- only the
distribution of mutants across cells.

### 6.5 FitnessProportionalEliteSelector Interaction

The elite selector picks parents proportional to fitness. With `num_parents=2`, the top
elites are more likely to be paired together (both selected frequently). This creates a
bias toward "best x second-best" crossovers. This is generally desirable (combining top
strategies) but could lead to mode collapse if the top programs are too similar.

**Monitoring**: Track archive diversity (number of occupied cells) per generation.
If diversity drops in Run J vs. Run I, crossover may be producing redundant offspring.

---

## 7. Risk Register

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| gen-0 stall (archive=1, C(1,2)=0) | Run J produces 0 mutations at gen 0 | LOW | Existing fallback in AllCombinationsParentSelector handles this (yields single parent). Verify in preflight. |
| Context overflow | Mutation LLM truncates or hallucinates | LOW | Token budget analysis shows ~9,700 tokens input, well within 32k. Monitor actual sizes from logs. |
| Throughput confound | Cannot attribute improvement to crossover vs. more mutations | MEDIUM | Use equal max_mutations_per_generation=8 for both runs (Section 4.6). |
| P1 rotation confound | Two parents have incomparable fitness scores | MEDIUM | Run initial P3 with P1=OFF (Section 3.1). |
| Similar parents | Top elites are too similar; crossover produces near-clones | MEDIUM | Monitor archive diversity. If problematic, consider using RandomParentSelector instead of AllCombinations. |
| Server variance | Different mutation servers produce different LLM quality | LOW | Assign mutation servers consistently with P1xP2 allocation. |
| Evolved program size growth | Programs grow beyond 10k chars, exceeding token budget | LOW | Monitor prompt sizes. If needed, enable strip_comments_and_docstrings for 2-parent runs. |

---

## 8. What We Do NOT Test (Explicit Scope Exclusions)

1. **Dedicated merge operator**: A custom `MergeOperator` that explicitly instructs the
   LLM to "take hop-1 from Parent A and hop-2 from Parent B" is a promising direction
   but requires new code and is a separate intervention from simply setting num_parents=2.
   The generic mutation prompt with 2 parent blocks already implicitly asks the LLM to
   combine them (archetype "Approach Synthesis" from user.txt). Testing the explicit merge
   operator is a follow-up to P3, not P3 itself.

2. **num_parents > 2**: With `max_elites=8`, C(8,3) = 56 combinations. The context
   budget for 3 parents (~14,000 tokens) is tight and the combinatorial explosion is
   severe. Not tested until 2-parent crossover shows clear value.

3. **Tournament selection for crossover partners**: Currently, AllCombinationsParentSelector
   exhaustively generates all pairs. An alternative would be RandomParentSelector with
   `num_parents=2`, which samples pairs stochastically. This could be tested as a
   follow-up if AllCombinations produces too many redundant pairs.

4. **Hop-level complementarity**: Using per-hop retrieval metrics (from P2/ASI) to
   select complementary parents (e.g., pair a program good at hop-1 with one good at
   hop-2). This requires P2 to be active and is a follow-up optimization.

---

## 9. Pre-Launch Checklist

Before launching Runs I and J:

- [ ] P1xP2 results analyzed; best config selected
- [ ] Verify `num_parents=2` with ddce37b4 seed: dry-run gen 0 on a test Redis DB;
      confirm at least 1 mutation is generated (tests fallback path)
- [ ] Verify mutation prompt size: extract actual token count from a 2-parent prompt
      using the evolved programs from E/F/G/H gen-50 archive
- [ ] Verify `mutation_mode=rewrite` is set (not diff)
- [ ] Flush Redis DBs 14, 15
- [ ] Verify server availability (2 mutation + 2 chain servers free)
- [ ] Update launch script with Run I/J commands
- [ ] Update watchdog with Run I/J PIDs and DB numbers

---

## 10. Summary of Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| When to run P3 | After P1xP2 completes | Need baseline selection from factorial results |
| num_parents=2 from gen 0 or delayed? | From gen 0 | Existing selector fallback handles small archives |
| P1 (rotation) active in P3? | OFF initially | Avoid fitness comparability confound (Section 3.1) |
| P2 (ASI) active in P3? | Only if P2 positive in P1xP2 | No reason to include null interventions |
| max_mutations_per_generation | 8 (equal for both) | Isolate crossover quality from throughput |
| Failure cases per parent | 10 (no change) | Context budget allows it; change only if needed |
| Number of runs | 2 (1 control, 1 treatment) | Focused; relies on external reference distribution |
| Redis DBs | 14, 15 | Available; no conflicts with E/F/G/H |
