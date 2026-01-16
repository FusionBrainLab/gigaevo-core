PROMPT_TEMPLATE: str = """
Please reason step by step, and provide your final answer enclosed in a LaTeX \\boxed{{...}} command.
Problem: {problem}
Answer:
""".strip()


def entrypoint():
    return PROMPT_TEMPLATE
