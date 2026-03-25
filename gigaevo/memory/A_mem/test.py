from agent.agent_class import LLMService

llm_service = LLMService(
        service="openrouter",
        model_name="qwen/qwen3-235b-a22b",
        api_key='sk-or-v1-REDACTED',
        temperature=0,
        max_tokens=0,
    )

a = llm_service.generate("2+2")
print(a)