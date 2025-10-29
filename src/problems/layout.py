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
