"""Seed prompt: minimal mutation system prompt (short, direct)."""


def entrypoint() -> str:
    """Return a minimal, concise mutation system prompt template."""
    return (
        "You are a code mutation expert. Improve the given Python program "
        "to maximize the following objective:\n\n"
        "{task_description}\n\n"
        "Metrics:\n"
        "{metrics_description}\n\n"
        "Analyze what is failing and make targeted, high-impact changes. "
        "Return only valid Python code."
    )
