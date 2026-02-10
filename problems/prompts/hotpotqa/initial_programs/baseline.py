PROMPT_TEMPLATE: str = """
Answer the following question based on the provided passages.

Question: {question}

Passages:
{passages}

Provide your answer in the exact format: Answer: <your answer>

Answer: <answer>
""".strip()


def entrypoint():
    return PROMPT_TEMPLATE
