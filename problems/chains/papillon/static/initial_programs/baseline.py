def entrypoint() -> dict:
    return {
        "system_prompt": "",
        "steps": [
            {
                "number": 1,
                "title": "Redact PII from query.",
                "step_type": "llm",
                "aim": "Create a privacy-preserving request for a powerful external LLM.",
                "stage_action": (
                    "Given a private user query, create a privacy-preserving version of it for a powerful external LLM. "
                    "The LLM may assist without learning private information about the user.\n"
                    "Provide ONLY the redacted query, no additional text."
                ),
                "reasoning_questions": "<none>",
                "example_reasoning": "<none>",
                "dependencies": [],
                "frozen": False,
            },
            {
                "number": 2,
                "title": "Query external LLM.",
                "step_type": "tool",
                "step_config": {
                    "tool_name": "external_llm",
                    "input_mapping": {"query": "$history[-1]"},
                },
                "dependencies": [1],
                "frozen": True,
            },
            {
                "number": 3,
                "title": "Aggregate final response.",
                "step_type": "llm",
                "aim": (
                    "Respond to the user query by combining "
                    "the external LLM's response with the original and redacted queries."
                ),
                "stage_action": (
                    "Synthesize a complete, helpful answer that addresses the "
                    "user's original request. Output ONLY the final response."
                ),
                "reasoning_questions": "<none>",
                "example_reasoning": "<none>",
                "dependencies": [1, 2],
                "frozen": False,
            },
        ],
    }
