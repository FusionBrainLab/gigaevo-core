from problems.prompts.sudoku.runtime import validate_prompt


def validate(prompt_template: str) -> dict[str, float]:
    return validate_prompt(prompt_template)
