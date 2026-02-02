PROMPT_TEMPLATE: str = """
Respond to the query: {prompt}
""".strip()


def entrypoint():
    return PROMPT_TEMPLATE
