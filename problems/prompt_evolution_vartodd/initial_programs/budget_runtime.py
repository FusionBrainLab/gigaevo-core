"""Seed prompt: runtime budget management — maximize useful evaluations."""


def entrypoint() -> dict:
    system = """\
You are an expert in evolutionary optimization of matrix decomposition programs.

OBJECTIVE:
{task_description}

AVAILABLE METRICS:
{metrics_description}

---

MUTATION STRATEGY — RUNTIME BUDGET FOCUS:
The key constraint is the 3000s runtime budget. Every design choice affects how
many FastTODD evaluations fit in that budget:

total_evals ≈ optimizer_iters × population_size × seeds × beamsearch_width

More evaluations = better parameter coverage = lower rank. The mutation LLM must
reason about this budget when changing parameters.

BUDGET TRADEOFFS:
- beamsearch_width: 1→3 triples eval cost but explores 3x more paths per eval
- num_samples: if approaching 2^bd, runtime explodes — keep small
- todd_width: linear cost scaling, diminishing returns above 3
- gen_part: >0.7 is expensive, rarely justified
- Population/particles: more = better coverage, but fewer iterations fit
- Two-stage strategies: split budget wisely (e.g. 60%/40% for init/reinit)"""

    user = """\
EVOLUTIONARY MUTATION: Budget-Aware Decomposition Tuning

Mutate the parent program to use its compute budget more efficiently.

## BUDGET ANALYSIS

From the parent's output, estimate:
1. **total_evals**: how many FastTODD evaluations were performed
2. **best_seen_times**: how many times the best rank was reached (high = good coverage)
3. **rank trajectory**: was improvement still happening when budget ran out?

If best_seen_times is high (>50), the search has converged — explore elsewhere.
If improvement was still happening at budget end, increase evaluation count by
reducing per-eval cost (lower beamsearch_width, fewer samples).

## STRATEGIES

### EFFICIENCY GAINS
- "Budget Compression" — reduce beamsearch_width/todd_width, increase optimizer iters
- "Early Termination" — in __call__(), skip full eval if early seed shows bad rank
- "Staged Budget" — two-phase: cheap exploration then expensive exploitation

### DEPTH INCREASE
- "Deeper Search" — more evaluations with simpler per-eval config
- "Focused Exploitation" — narrow parameter bounds, more CMA-ES iterations

### BREADTH INCREASE
- "Population Boost" — more particles/popsize with fewer iterations
- "Multi-Restart" — multiple short optimizer runs from different starting points

## OUTPUT FORMAT (JSON)

```json
{{{{
  "archetype": "Selected strategy name",
  "justification": "2-3 sentences with budget arithmetic justifying changes.",
  "insights_used": ["insight1 text", "insight2 text"],
  "code": "complete Python program"
}}}}
```

**CRITICAL**: The `code` field must contain ONLY valid Python code.

{parent_blocks}"""

    return {"system": system, "user": user}
