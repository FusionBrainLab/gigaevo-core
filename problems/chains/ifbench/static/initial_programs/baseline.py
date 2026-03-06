def entrypoint():
    return {
        "system_prompt": "",
        "steps": [
            # Step 1: Generate initial response to the user query
            {
                "number": 1,
                "title": "Generate initial response",
                "step_type": "llm",
                "aim": "Generate a helpful, complete response to the user query.",
                "stage_action": "Respond to the query.",
                "reasoning_questions": "<none>",
                "example_reasoning": "<none>",
                "dependencies": [],
                "frozen": False,
            },
            # Step 2: Rewrite response to satisfy all instruction constraints
            {
                "number": 2,
                "title": "Rewrite to satisfy constraints",
                "step_type": "llm",
                "aim": "Ensure the response is correct and adheres to the given constraints.",
                "stage_action": (
                    "Rewrite the response to adhere to the given constraints. "
                    "Output only the final rewritten response."
                ),
                "reasoning_questions": "<none>",
                "example_reasoning": "<none>",
                "dependencies": [1],
                "frozen": False,
            },
        ],
    }
