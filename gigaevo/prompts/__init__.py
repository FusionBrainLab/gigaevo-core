"""Prompt template loading utilities.

All prompts are stored as plain text files organized by agent type.
Prompts use .format() syntax for variable substitution.

When prompts_dir is given, that directory is tried first; if the file is missing
there, the package default directory is used.
"""

from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent


def load_prompt(
    agent_name: str,
    prompt_type: str,
    prompts_dir: str | Path | None = None,
) -> str:
    """Load a prompt template from file.

    Tries prompts_dir first (if given); if the file is not there, loads from
    the package default directory.

    Args:
        agent_name: Agent type directory (insights, lineage, scoring, mutation)
        prompt_type: Prompt file type (system, user)
        prompts_dir: Optional directory for prompts (e.g. config.prompts.dir).
            Same layout as package: subdirs per agent with system.txt / user.txt.

    Returns:
        Template string for .format() substitution

    Example:
        >>> system = load_prompt("insights", "system")
        >>> user = load_prompt("insights", "user", prompts_dir="/custom/prompts")
    """

    prompt_path = Path(prompts_dir or _PROMPTS_DIR) / agent_name / f"{prompt_type}.txt"
    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Prompt not found: {prompt_path}\nLooking in: {_PROMPTS_DIR / agent_name}"
        )
    return prompt_path.read_text().strip()


# Simple accessors for common prompts
class MutationPrompts:
    """Mutation agent prompt templates."""

    @staticmethod
    def system(prompts_dir: str | Path | None = None) -> str:
        """System prompt for mutation."""
        return load_prompt("mutation", "system", prompts_dir=prompts_dir)

    @staticmethod
    def user(prompts_dir: str | Path | None = None) -> str:
        """User prompt template for mutation."""
        return load_prompt("mutation", "user", prompts_dir=prompts_dir)


class InsightsPrompts:
    """Insights agent prompt templates."""

    @staticmethod
    def system(prompts_dir: str | Path | None = None) -> str:
        """System prompt for insights analysis."""
        return load_prompt("insights", "system", prompts_dir=prompts_dir)

    @staticmethod
    def user(prompts_dir: str | Path | None = None) -> str:
        """User prompt template for insights analysis."""
        return load_prompt("insights", "user", prompts_dir=prompts_dir)


class LineagePrompts:
    """Lineage agent prompt templates."""

    @staticmethod
    def system(prompts_dir: str | Path | None = None) -> str:
        """System prompt for lineage analysis."""
        return load_prompt("lineage", "system", prompts_dir=prompts_dir)

    @staticmethod
    def user(prompts_dir: str | Path | None = None) -> str:
        """User prompt template for lineage analysis."""
        return load_prompt("lineage", "user", prompts_dir=prompts_dir)


class ScoringPrompts:
    """Scoring agent prompt templates."""

    @staticmethod
    def system(prompts_dir: str | Path | None = None) -> str:
        """System prompt for scoring."""
        return load_prompt("scoring", "system", prompts_dir=prompts_dir)

    @staticmethod
    def user(prompts_dir: str | Path | None = None) -> str:
        """User prompt template for scoring."""
        return load_prompt("scoring", "user", prompts_dir=prompts_dir)
