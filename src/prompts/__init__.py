"""Prompt template loading utilities.

All prompts are stored as plain text files organized by agent type.
Prompts use .format() syntax for variable substitution.
"""

from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent


def load_prompt(agent_name: str, prompt_type: str) -> str:
    """Load a prompt template from file.

    Args:
        agent_name: Agent type directory (insights, lineage, scoring, mutation)
        prompt_type: Prompt file type (system, user)

    Returns:
        Template string for .format() substitution

    Example:
        >>> system = load_prompt("insights", "system")
        >>> user = load_prompt("insights", "user")
        >>> formatted = user.format(code="...", metrics="...")
    """
    prompt_file = _PROMPTS_DIR / agent_name / f"{prompt_type}.txt"

    if not prompt_file.exists():
        raise FileNotFoundError(
            f"Prompt not found: {prompt_file}\n"
            f"Looking in: {_PROMPTS_DIR / agent_name}"
        )

    return prompt_file.read_text().strip()


# Simple accessors for common prompts
class InsightsPrompts:
    """Insights agent prompt templates."""

    @staticmethod
    def system() -> str:
        """System prompt for insights analysis."""
        return load_prompt("insights", "system")

    @staticmethod
    def user() -> str:
        """User prompt template for insights analysis."""
        return load_prompt("insights", "user")


class LineagePrompts:
    """Lineage agent prompt templates."""

    @staticmethod
    def system() -> str:
        """System prompt for lineage analysis."""
        return load_prompt("lineage", "system")

    @staticmethod
    def user() -> str:
        """User prompt template for lineage analysis."""
        return load_prompt("lineage", "user")


class ScoringPrompts:
    """Scoring agent prompt templates."""

    @staticmethod
    def system() -> str:
        """System prompt for scoring."""
        return load_prompt("scoring", "system")

    @staticmethod
    def user() -> str:
        """User prompt template for scoring."""
        return load_prompt("scoring", "user")
