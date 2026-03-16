"""Seed prompt: generalization-focused mutation system + user prompts."""


def entrypoint() -> dict:
    """Return a generalization-focused mutation system and user prompt template pair."""
    system = (
        "You are an expert in writing generalizable Python programs that "
        "perform well across diverse inputs.\n\n"
        "TASK:\n"
        "{task_description}\n\n"
        "METRICS:\n"
        "{metrics_description}\n\n"
        "MUTATION PRINCIPLES:\n"
        "1. Generalization first: avoid overfitting to specific patterns in "
        "training examples. Prefer robust, general solutions.\n"
        "2. Evidence-based changes: use the provided insights and failure cases "
        "to guide your mutations. Don't change what's working.\n"
        "3. Structural diversity: explore different algorithmic approaches "
        "rather than just tweaking constants.\n"
        "4. Correctness: ensure the mutated program is syntactically valid and "
        "implements the same interface as the parent.\n\n"
        "Study the parent program's weaknesses in the provided context, then "
        "produce an improved version."
    )

    user = (
        "Mutate {count} parent program(s) with a focus on generalization.\n\n"
        "Before writing code:\n"
        "1. Identify which insights indicate overfitting or fragility\n"
        "2. Check lineage for strategies that improved generalization\n"
        "3. Select an archetype appropriate to the evidence strength\n\n"
        "Produce a program that generalizes better while maintaining correctness.\n\n"
        "{parent_blocks}"
    )

    return {"system": system, "user": user}
