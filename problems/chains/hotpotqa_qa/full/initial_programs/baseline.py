def entrypoint() -> dict:
    return {
        "system_prompt": "",
        "steps": [
            {
                "number": 1,
                "title": "Answer the question",
                "step_type": "llm",
                "aim": ("Answer the question based on the provided passages."),
                "stage_action": (
                    "Answer the following question based on the provided passages. "
                    "Provide your answer in the exact format: Answer: <your answer>"
                ),
                "reasoning_questions": "<none>",
                "example_reasoning": "<none>",
                "dependencies": [],
            },
        ],
    }
