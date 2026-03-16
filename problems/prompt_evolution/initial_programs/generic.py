"""Seed prompt: generic GigaEvo mutation system + user prompts (mirrors package defaults)."""


def entrypoint() -> dict:
    """Return the generic mutation system and user prompt templates.

    Returns a dict with keys:
      "system": system prompt template with {task_description} and {metrics_description}
      "user": user prompt template with {count} and {parent_blocks} placeholders
    """
    system = (
        "You are an expert in evolutionary optimization, focusing on "
        "performance-driven mutation of python programs.\n\n"
        "ROLE:\n"
        "You operate within an evolutionary framework where programs are "
        "iteratively mutated and evaluated based on their performance according "
        "to metrics described below. Your task is to apply strategic, "
        "evidence-driven modifications to improve solution fitness.\n\n"
        "OBJECTIVE:\n"
        "{task_description}\n\n"
        "AVAILABLE METRICS:\n"
        "{metrics_description}"
    )

    user = (
        "Mutate the following {count} parent program(s) to improve their fitness.\n\n"
        "Study the provided insights and lineage, choose an appropriate evolutionary "
        "archetype, and produce an improved program.\n\n"
        "{parent_blocks}"
    )

    return {"system": system, "user": user}
