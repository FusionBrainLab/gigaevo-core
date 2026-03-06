PROMPT_TEMPLATE: str = """
Reddit comment: {body}

Based on the following rule: "{rule}", determine whether the above comment violates the rule.

The last line of your response should be of the following format: 'Answer: $PROBABILITY' (without quotes) where PROBABILITY is [0, 1] probability of the rule violation by given comment.
""".strip()


def entrypoint():
    return PROMPT_TEMPLATE
