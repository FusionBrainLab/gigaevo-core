import os
from pathlib import Path


class PromptManager:
    def __init__(self):
        self.prompts_dir = Path(__file__).resolve().parent / "prompts"
        self._init_prompt_manager()

    def _init_prompt_manager(self):
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

    def prompts_list(self):
        return {k: list(v) for k, v in self.available_prompts.items()}

    def load_prompt(self, prompt_name: str, insert_data: str = "") -> str:
        if "__" not in prompt_name:
            raise ValueError(
                f"prompt_name must contain '__' (e.g. 'step__user'), got {prompt_name!r}"
            )
        step_name, prompt_type = prompt_name.split("__", 1)
        prompt_dir = self.prompts_dir / step_name
        prompt_path = prompt_dir / f"{prompt_type}.txt"
        if not prompt_path.is_file():
            raise FileNotFoundError(f"No prompt at {prompt_path}")
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_text = f.read()
        if prompt_type == "user":
            prompt_text = prompt_text.replace("<INSERT>", insert_data)
        return prompt_text
