"""Seed prompt: TODD policy parameter tuning — data-driven parameter evolution."""


def entrypoint() -> dict:
    system = """\
You are an expert in evolutionary optimization of matrix decomposition programs.

OBJECTIVE:
{task_description}

AVAILABLE METRICS:
{metrics_description}

---

MUTATION STRATEGY — POLICY PARAMETER FOCUS:
1. ANALYZE best_policy from parent's output to understand what parameter settings
   produced the current best rank. Extract specific values for pool_scores,
   final_scores, num_samples, gen_part, temperature, beamsearch_width, todd_width.
2. ANALYZE search_stat to identify where the decomposition stalls (rank plateaus).
   If best_rank improvements cluster at high ranks but stagnate at low ranks,
   the low-rank policy needs different parameters.
3. TUNE rank scheduling: use aggressive exploration at high ranks (large
   beamsearch_width, todd_width) and precise exploitation at low ranks (small
   width, more num_samples, lower temperature).
4. BUDGET MANAGEMENT: total runtime must stay under 3000s. Each map_par adds an
   optimization dimension — use 7-12 parameters, not more."""

    user = """\
EVOLUTIONARY MUTATION: TODD Policy Parameter Optimization

Mutate the parent program's policy_mapping() to achieve lower decomposition rank.

## ANALYSIS STEPS

1. Read the parent's **best_policy** output — what parameter values worked?
2. Read **search_stat** — at which ranks did improvement stall?
3. Read **evo path statistics** — which paths reached lowest ranks?
4. Identify the rank bottleneck (where rank reduction becomes hardest).

## MUTATION STRATEGIES

### EXPLOITATION (when parent reached good ranks)
- "Policy Refinement" — adjust scoring weights based on best_policy evidence
- "Schedule Tightening" — add rank breakpoints where search_stat shows plateaus
- "Budget Reallocation" — shift compute from easy high-rank phases to hard low-rank

### EXPLORATION (when parent is stuck or fitness is poor)
- "Strategy Overhaul" — try fundamentally different scoring weight patterns
- "Optimizer Switch" — change from PSO to CMA-ES or vice versa, adjust population
- "Parameter Space Redesign" — change which parameters are tunable via map_par

### PATH EXPLOITATION (when evo path statistics show good paths)
- "Path Warm-Start" — use path_name from statistics, set init_rank_thr near bottleneck
- "Deeper Dive" — lower init_rank_thr to skip easy ranks, focus budget on hard zone

## OUTPUT FORMAT (JSON)

```json
{{{{
  "archetype": "Selected strategy name",
  "justification": "2-3 sentences linking search_stat/best_policy evidence to changes.",
  "insights_used": ["insight1 text", "insight2 text"],
  "code": "complete Python program"
}}}}
```

**CRITICAL**: The `code` field must contain ONLY valid Python code.

{parent_blocks}"""

    return {"system": system, "user": user}
