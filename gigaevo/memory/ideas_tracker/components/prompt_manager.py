import os
from pathlib import Path


class PromptManager:
    """
    Manages loading and organization of prompt templates for LLM interactions.

    Prompts are stored in subdirectories under prompts/ with format: step_name/prompt_type.txt
    Supports dynamic content insertion via <INSERT> placeholder in user prompts.
    """

    def __init__(self):
        """Initialize PromptManager and scan prompts directory."""
        self.prompts_dir = Path(__file__).resolve().parent / "prompts"
        self._init_prompt_manager()

    def _init_prompt_manager(self) -> None:
        """
        Scan prompts directory and catalog available prompts by step name.

        Creates prompts directory if missing. Builds available_prompts dictionary
        mapping step names to lists of prompt files.
        """
        os.makedirs(self.prompts_dir, exist_ok=True)
        self.available_prompts = {}
        for root, _subdirs, files in os.walk(self.prompts_dir):
            if not files:
                continue
            rel = Path(root).relative_to(self.prompts_dir)
            if len(rel.parts) != 1:
                continue
            step_name = rel.parts[0]
            self.available_prompts.setdefault(step_name, []).extend(files)

    def prompts_list(self) -> dict[str, list[str]]:
        """
        Get available prompts organized by step name.

        Returns:
            Dictionary mapping step names to lists of prompt filenames.
        """
        return {k: list(v) for k, v in self.available_prompts.items()}
    
    def load_prompt_multiple_inserts(self, prompt_name: str, insert_data: dict[str, str]) -> str:
        """
        Load and optionally populate a prompt template with multiple inserts.

        Args:
            prompt_name: Prompt identifier in format "step__type".
            insert_data: Dictionary mapping placeholder names to content.
        """
        prompt_text = self.load_prompt(prompt_name)
        for placeholder, content in insert_data.items():
            prompt_text = prompt_text.replace(f"{placeholder}", content)
        return prompt_text

    def load_prompt(self, prompt_name: str, insert_data: str = "") -> str:
        """
        Load and optionally populate a prompt template.

        Prompt names follow format "step__type" (e.g., "classify__user").
        For user prompts, replaces <INSERT> placeholder with insert_data.

        Args:
            prompt_name: Prompt identifier in format "step__type".
            insert_data: Content to insert into <INSERT> placeholder (user prompts only).

        Returns:
            Loaded prompt text with insertions applied.

        Raises:
            ValueError: If prompt_name doesn't contain '__' separator.
            FileNotFoundError: If prompt file doesn't exist.
        """
        if "__" not in prompt_name:
            raise ValueError(
                f"prompt_name must contain '__' (e.g. 'step__user'), got {prompt_name!r}"
            )
        step_name, prompt_type = prompt_name.split("__", 1)
        prompt_dir = self.prompts_dir / step_name
        prompt_path = prompt_dir / f"{prompt_type}.txt"
        if not prompt_path.is_file():
            raise FileNotFoundError(f"No prompt at {prompt_path}")
        with open(prompt_path, encoding="utf-8") as f:
            prompt_text = f.read()
        if prompt_type == "user":
            prompt_text = prompt_text.replace("<INSERT>", insert_data)
        return prompt_text
