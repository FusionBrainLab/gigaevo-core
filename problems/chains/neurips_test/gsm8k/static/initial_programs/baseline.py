def entrypoint():
    return {
        "system_prompt": "",
        "steps": [
            # Step 1: Understand the problem
            {
                "number": 1,
                "title": "Understand the problem",
                "step_type": "llm",
                "aim": "Identify all key quantities, units, and relationships in the problem.",
                "stage_action": (
                    "Read the math problem carefully. "
                    "List all given numbers, what they represent, and what the question is asking for."
                ),
                "reasoning_questions": "<none>",
                "example_reasoning": "<none>",
                "dependencies": [],
                "frozen": False,
            },
            # Step 2: Solve step by step
            {
                "number": 2,
                "title": "Solve step by step",
                "step_type": "llm",
                "aim": "Compute the answer by working through all arithmetic steps.",
                "stage_action": (
                    "Using the identified quantities and relationships, solve the problem step by step. "
                    "Show each arithmetic operation explicitly. "
                    "Double-check your calculations."
                ),
                "reasoning_questions": "<none>",
                "example_reasoning": "<none>",
                "dependencies": [1],
                "frozen": False,
            },
            # Step 3: State the final answer
            {
                "number": 3,
                "title": "State the final answer",
                "step_type": "llm",
                "aim": "State the final numerical answer clearly.",
                "stage_action": (
                    "Based on your step-by-step solution, state the final answer. "
                    "You MUST end your response with exactly:\n"
                    "Answer: <number>\n"
                    "where <number> is a plain integer or decimal (no units, no commas)."
                ),
                "reasoning_questions": "<none>",
                "example_reasoning": "<none>",
                "dependencies": [2],
                "frozen": False,
            },
        ],
    }
