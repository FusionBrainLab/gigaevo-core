"""Seed prompt: generalization-focused mutation system prompt."""


def entrypoint() -> str:
    """Return a generalization-focused mutation system prompt template."""
    return (
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
