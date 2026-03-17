"""Seed prompt: generalization-focused strategy with frozen constraints."""

from gigaevo.prompts.hotpotqa.mutation.frozen import SYSTEM_CONSTRAINTS


def entrypoint() -> dict:
    strategy = """\
MUTATION PRINCIPLES — GENERALIZATION FOCUS:
1. Generalization first: avoid overfitting to specific patterns in training examples.
   Prefer robust, general solutions over narrow fixes.
2. Evidence-based changes: use the provided insights and failure cases to guide your
   mutations. Don't change what's working.
3. Structural diversity: explore different algorithmic approaches rather than just
   tweaking constants or minor wording.
4. Correctness: ensure the mutated program is syntactically valid and implements
   the same interface as the parent."""

    user = """\
Mutate the parent program(s) with a focus on generalization.

Before writing code:
1. Identify which insights indicate overfitting or fragility
2. Check lineage for strategies that improved generalization
3. Select an archetype appropriate to the evidence strength

Produce a program that generalizes better while maintaining correctness.

Return JSON with keys: "archetype", "justification", "insights_used", "code".
The "code" field must contain only valid Python code (no markdown fences).

{parent_blocks}"""

    return {"system": SYSTEM_CONSTRAINTS + "\n\n" + strategy, "user": user}
