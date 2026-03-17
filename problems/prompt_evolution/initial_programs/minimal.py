"""Seed prompt: minimal mutation prompts — short, direct, with frozen constraints."""

from gigaevo.prompts.hotpotqa.mutation.frozen import SYSTEM_CONSTRAINTS


def entrypoint() -> dict:
    strategy = """\
STRATEGY: Make targeted, high-impact changes. Analyze what is failing and fix it.
Prefer compression over expansion. Return only valid Python code."""

    user = """\
Mutate the parent program(s). Use the insights and lineage to guide your changes.

Return JSON with keys: "archetype", "justification", "insights_used", "code".
The "code" field must contain only valid Python code (no markdown fences).

{parent_blocks}"""

    return {"system": SYSTEM_CONSTRAINTS + "\n\n" + strategy, "user": user}
