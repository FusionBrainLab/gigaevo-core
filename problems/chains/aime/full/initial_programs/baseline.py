def entrypoint() -> dict:
    return {
        "system_prompt": "",
        "steps": [
            {
                "number": 1,
                "title": "Solve the problem",
                "step_type": "llm",
                "aim": (
                    "Solve the math olympiad problem and provide the answer."
                ),
                "stage_action": (
                    "Please reason step by step, and provide your final answer "
                    "enclosed in a LaTeX \\boxed{...} command."
                ),
                "reasoning_questions": "<none>",
                "example_reasoning": "<none>",
                "dependencies": [],
            },
        ],
    }
