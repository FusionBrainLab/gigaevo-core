def entrypoint() -> dict:
    return {
        "system_prompt": "",
        "steps": [
            {
                "number": 1,
                "title": "Identify relevant evidence",
                "step_type": "llm",
                "aim": "Identify passages and facts that are relevant to the question.",
                "stage_action": (
                    "Read the question and all passages. List the most relevant facts "
                    "and entities needed to answer the question."
                ),
                "reasoning_questions": "<none>",
                "example_reasoning": "<none>",
                "dependencies": [],
                "frozen": False,
            },
            {
                "number": 2,
                "title": "Perform multi-hop reasoning",
                "step_type": "llm",
                "aim": "Connect the extracted facts to infer the final answer.",
                "stage_action": (
                    "Use the relevant facts to reason step by step and derive the "
                    "answer candidate. Keep reasoning concise and evidence-grounded."
                ),
                "reasoning_questions": "<none>",
                "example_reasoning": "<none>",
                "dependencies": [1],
                "frozen": False,
            },
            {
                "number": 3,
                "title": "Produce final answer",
                "step_type": "llm",
                "aim": "Return the final answer in extractable format.",
                "stage_action": (
                    "Based on previous analysis, provide only the final answer in this "
                    "exact format: Answer: <your answer>"
                ),
                "reasoning_questions": "<none>",
                "example_reasoning": "<none>",
                "dependencies": [1, 2],
                "frozen": False,
            },
        ],
    }
