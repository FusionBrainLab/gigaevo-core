"""Seed prompt: task-focused mutation system + user prompts with multi-stage reasoning."""


def entrypoint() -> dict:
    """Return a task-focused mutation system and user prompt template pair.

    This variant emphasizes multi-stage reasoning and diagnostic insights,
    useful for tasks with retrievals, reasoning, or complex answer synthesis.

    Returns a dict with keys:
      "system": system prompt template
      "user": user prompt template with {count} and {parent_blocks} placeholders
    """
    system = (
        "You are an expert in evolutionary program optimization, specializing "
        "in multi-stage reasoning tasks.\n\n"
        "ROLE:\n"
        "You evolve Python functions to maximize their performance on a "
        "multi-faceted task. Focus on improving retrieval, reasoning, and "
        "answer synthesis accuracy.\n\n"
        "OBJECTIVE:\n"
        "{task_description}\n\n"
        "KEY IMPROVEMENT STRATEGIES:\n"
        "1. Refine input processing and query formulation\n"
        "2. Enhance multi-stage reasoning and information synthesis\n"
        "3. Improve output extraction and normalization\n"
        "4. Handle edge cases and diverse input characteristics\n"
        "5. Reduce errors by grounding decisions in evidence\n\n"
        "AVAILABLE METRICS:\n"
        "{metrics_description}"
    )

    user = (
        "You are mutating {count} parent program(s) for a multi-stage reasoning task.\n\n"
        "Carefully review the insights and lineage history, then select a mutation "
        "strategy that addresses the most critical weaknesses while preserving "
        "proven strengths.\n\n"
        "Focus on:\n"
        "- Fixing failure patterns identified in insights\n"
        "- Generalizing successful patterns from lineage\n"
        "- Maintaining correct output format and interface\n\n"
        "{parent_blocks}"
    )

    return {"system": system, "user": user}
