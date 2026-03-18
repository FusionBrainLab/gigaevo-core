"""Seed prompt: minimal — short, direct, lets the LLM figure it out."""


def entrypoint() -> dict:
    system = """\
You are an expert in evolutionary optimization of matrix decomposition programs.

OBJECTIVE:
{task_description}

AVAILABLE METRICS:
{metrics_description}

---

STRATEGY: Analyze the parent's best_policy and search statistics, then make
targeted changes to reach lower decomposition rank. Use path reuse when good
paths exist. Keep runtime under 3000s. Return only valid Python code."""

    user = """\
Mutate the parent program to achieve lower fitness (rank + sum/size).

Use the execution output (best_policy, search_stat, evo path statistics) to
guide your changes. Focus on what the data tells you, not assumptions.

Return JSON with keys: "archetype", "justification", "insights_used", "code".
The "code" field must contain only valid Python code (no markdown fences).

{parent_blocks}"""

    return {"system": system, "user": user}
