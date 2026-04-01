def entrypoint() -> dict:
    return {
        "system_prompt": "",
        "steps": [
            {
                "number": 1,
                "title": "Answer the MuSiQue question",
                "step_type": "llm",
                "aim": ("Answer the question based only on the provided passages."),
                "stage_action": (
                    "Read the question and passages carefully, then answer strictly from "
                    "the given evidence. "
                    "Provide your answer in the exact format: Answer: <your answer>"
                ),
                "reasoning_questions": "<none>",
                "example_reasoning": "<none>",
                "dependencies": [],
            },
        ],
    }
