from __future__ import annotations

from pathlib import Path


class ProblemLayout:
    """Standardized problem directory layout and simple scaffolding helpers.

    Centralizes filenames/dirs to avoid hardcoded strings scattered across code.
    """

    # Filenames
    TASK_DESCRIPTION = "task_description.txt"
    TASK_HINTS = "task_hints.txt"
    VALIDATOR = "validate.py"
    MUTATION_SYSTEM_PROMPT = "mutation_system_prompt.txt"
    MUTATION_USER_PROMPT = "mutation_user_prompt.txt"
    CONTEXT_FILE = "context.py"
    METRICS_FILE = "metrics.yaml"

    # Directories
    INITIAL_PROGRAMS_DIR = "initial_programs"

    @classmethod
    def required_files(cls, add_context: bool = False) -> list[str]:
        files = [
            cls.TASK_DESCRIPTION,
            cls.TASK_HINTS,
            cls.VALIDATOR,
            cls.MUTATION_SYSTEM_PROMPT,
            cls.MUTATION_USER_PROMPT,
            cls.METRICS_FILE,
        ]
        if add_context:
            files.append(cls.CONTEXT_FILE)
        return files

    @classmethod
    def required_directories(cls) -> list[str]:
        return [cls.INITIAL_PROGRAMS_DIR]

    @classmethod
    def scaffold(
        cls,
        target_dir: Path,
        *,
        add_context: bool = False,
        overwrite: bool = False,
        task_description: str | None = None,
        task_hints: str | None = None,
        mutation_system_prompt: str | None = None,
        mutation_user_prompt: str | None = None,
    ) -> None:
        """Create a minimal problem directory with placeholder files."""
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / cls.INITIAL_PROGRAMS_DIR).mkdir(parents=True, exist_ok=True)

        templates: dict[str, str] = {
            cls.TASK_DESCRIPTION: task_description or "Describe the optimization objective and constraints.\n",
            cls.TASK_HINTS: task_hints or "Provide short, actionable hints for the model.\n",
            cls.MUTATION_SYSTEM_PROMPT: (
                mutation_system_prompt
                or (
                    "You are an expert in evolutionary optimization.\n\n"
                    "OBJECTIVE:\n{task_definition}\n\n"
                    "STRATEGIC GUIDANCE:\n{task_hints}\n\n"
                    "AVAILABLE METRICS:\n{metrics_description}\n"
                )
            ),
            cls.MUTATION_USER_PROMPT: (
                mutation_user_prompt or "=== Parents ({count}) ===\n{parent_blocks}\n"
            ),
            cls.VALIDATOR: (
                "def validate(output):\n"
                "    # TODO: implement problem-specific validation\n"
                "    return {\"fitness\": 0.0, \"is_valid\": 1}\n"
            ),
            cls.METRICS_FILE: (
                "display_order: [fitness, is_valid]\n"
                "specs:\n"
                "  fitness:\n"
                "    description: Primary objective\n"
                "    decimals: 5\n"
                "    is_primary: true\n"
                "    higher_is_better: true\n"
                "    lower_bound: 0.0\n"
                "    upper_bound: 1.0\n"
                "    include_in_prompts: true\n"
                "    significant_change: 1e-6\n"
                "  is_valid:\n"
                "    description: Program validity (1 valid, 0 invalid)\n"
                "    decimals: 0\n"
                "    is_primary: false\n"
                "    higher_is_better: true\n"
                "    lower_bound: 0.0\n"
                "    upper_bound: 1.0\n"
                "    include_in_prompts: true\n"
            ),
        }

        if add_context:
            templates[cls.CONTEXT_FILE] = (
                "def build_context():\n    return {}\n"
            )

        for name, content in templates.items():
            path = target_dir / name
            if path.exists() and not overwrite:
                continue
            path.write_text(content)


