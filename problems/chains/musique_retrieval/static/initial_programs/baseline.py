def entrypoint():
    return {
        "system_prompt": "",
        "steps": [
            # Step 1: First-hop retrieval (frozen tool step)
            {
                "number": 1,
                "title": "Retrieve first-hop passages.",
                "step_type": "tool",
                "step_config": {
                    "tool_name": "retrieve",
                    "input_mapping": {
                        "query": "$sample.question",
                        "task_id": "$sample.task_id",
                    },
                },
                "dependencies": [],
                "frozen": True,
            },
            # Step 2: Summarize first-hop passages
            {
                "number": 2,
                "title": "Summarize key facts from passages.",
                "step_type": "llm",
                "aim": "Extract key facts from the provided passages relevant to the question.",
                "stage_action": (
                    "Read all passages and identify facts that help answer the question. "
                    "Summarize the most relevant evidence."
                ),
                "reasoning_questions": "<none>",
                "example_reasoning": "<none>",
                "dependencies": [1],
                "frozen": False,
            },
            # Step 3: Generate second-hop query
            {
                "number": 3,
                "title": "Generate second-hop query.",
                "step_type": "llm",
                "aim": "Identify missing information and generate a search query.",
                "stage_action": (
                    "Based on the summary, determine what additional evidence is needed to "
                    "answer the question? Write a concise search query to find the missing evidence.\n"
                    "Provide ONLY the search query, no additional text."
                ),
                "reasoning_questions": "<none>",
                "example_reasoning": "<none>",
                "dependencies": [2],
                "frozen": False,
            },
            # Step 4: Second-hop retrieval (frozen tool step)
            {
                "number": 4,
                "title": "Retrieve second-hop passages.",
                "step_type": "tool",
                "step_config": {
                    "tool_name": "retrieve",
                    "input_mapping": {
                        "query": "$history[-1]",
                        "task_id": "$sample.task_id",
                    },
                },
                "dependencies": [3],
                "frozen": True,
            },
            # Step 5: Combine evidence
            {
                "number": 5,
                "title": "Combine evidence",
                "step_type": "llm",
                "aim": "Combine evidence from both retrieval hops.",
                "stage_action": (
                    "Combine evidence from the first-hop summary and the newly retrieved passages. "
                    "Produce a comprehensive evidence summary."
                ),
                "reasoning_questions": "<none>",
                "example_reasoning": "<none>",
                "dependencies": [2, 4],
                "frozen": False,
            },
            # Step 6: Final answer
            {
                "number": 6,
                "title": "Final answer",
                "step_type": "llm",
                "aim": "Answer the question using all gathered evidence.",
                "stage_action": (
                    "Based on the combined evidence from both hops, answer "
                    "the question. Provide your final answer in the exact format: "
                    "Answer: <your answer>"
                ),
                "reasoning_questions": "<none>",
                "example_reasoning": "<none>",
                "dependencies": [2, 5],
                "frozen": False,
            },
        ],
    }
