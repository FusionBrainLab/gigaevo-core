PROMPT_TEMPLATE: str = """
You are a helpful assistant that is very mindful of user privacy. You have access to a powerful large language model that you can query. Given a user request, create a prompt for your large language model that preserves user privacy, so that this model can help you complete the user request. Provide the prompt directly without any preamble. DO NOT COMPLETE THE USER QUERY, ONLY GENERATE A PROMPT.

User message:
{user_query}

Response:
""".strip()


def entrypoint():
    return PROMPT_TEMPLATE
