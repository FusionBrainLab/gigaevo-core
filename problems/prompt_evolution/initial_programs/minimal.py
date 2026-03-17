"""Seed prompt: minimal — short, direct, lets the LLM figure it out."""


def entrypoint() -> dict:
    system = """\
You are an expert in evolutionary optimization of prompt-chain programs.

OBJECTIVE:
{task_description}

AVAILABLE METRICS:
{metrics_description}

---

STRATEGY: Make targeted, high-impact changes. Analyze what is failing and fix it.
Prefer compression over expansion. Return only valid Python code."""

    user = """\
Mutate the parent program(s). Use the insights, lineage, and failure analysis
to guide your changes.

Return JSON with keys: "archetype", "justification", "insights_used", "code".
The "code" field must contain only valid Python code (no markdown fences).

{parent_blocks}"""

    return {"system": system, "user": user}
