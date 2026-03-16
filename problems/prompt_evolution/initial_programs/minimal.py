"""Seed prompt: minimal mutation system + user prompts (short, direct)."""


def entrypoint() -> dict:
    """Return minimal, concise mutation system and user prompt templates."""
    system = (
        "You are a code mutation expert. Improve the given Python program "
        "to maximize the following objective:\n\n"
        "{task_description}\n\n"
        "Metrics:\n"
        "{metrics_description}\n\n"
        "Analyze what is failing and make targeted, high-impact changes. "
        "Return only valid Python code."
    )

    user = (
        "Mutate {count} parent program(s). "
        "Use the insights and lineage to guide your changes.\n\n"
        "{parent_blocks}"
    )

    return {"system": system, "user": user}
