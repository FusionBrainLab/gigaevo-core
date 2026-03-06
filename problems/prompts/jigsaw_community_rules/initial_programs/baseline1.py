PROMPT_TEMPLATE: str = """
Comment:
{body}

Based on the following rule: "{rule}", determine whether the above comment violates the rule.
Subreddit: {subreddit}

To help you decide:
- Here is an example of a comment that **does** violate the rule: "{positive_example_1}"
- Another example of a violating comment: "{positive_example_2}"
- Here is an example of a comment that does **not** violate the rule: "{negative_example_1}"
- Another example of a non-violating comment: "{negative_example_2}"

The last line of your response should be of the following format: 'Answer: $PROBABILITY' (without quotes) where PROBABILITY is [0, 1] probability of the rule violation by given comment.
""".strip()


def entrypoint():
    return PROMPT_TEMPLATE
