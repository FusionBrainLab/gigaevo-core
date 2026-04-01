import asyncio
import os
from typing import Any

from openai import AsyncOpenAI, OpenAI

from gigaevo.memory.ideas_tracker.components.prompt_manager import PromptManager


def _create_llm_clients(
    base_url_: str | None = None,
) -> tuple[OpenAI, AsyncOpenAI, bool]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set. Set it in your environment or .env file."
        )
    env_base_url = (
        os.getenv("OPENAI_BASE_URL")
        or os.getenv("BASE_URL")
        or os.getenv("LLM_BASE_URL")
    )
    base_url = env_base_url or base_url_

    if not base_url and api_key.startswith("sk-or-"):
        base_url = "https://openrouter.ai/api/v1"

    is_openrouter = bool(base_url and "openrouter.ai" in base_url)
    client_kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url

    return OpenAI(**client_kwargs), AsyncOpenAI(**client_kwargs), is_openrouter


class LLMClient:
    def __init__(
        self, model: str, base_url: str | None = None, max_concurrent: int = -1
    ):
        self.model = model
        self.base_url = base_url
        self.llm, self.async_llm, self.is_openrouter = _create_llm_clients(base_url)
        self.prompt_manager = PromptManager()
        self.semaphore = (
            asyncio.Semaphore(max_concurrent) if max_concurrent > 0 else None
        )

    def _create_payload(
        self,
        step_name: str,
        prompt_content: str | dict[str, str] = "",
        reasoning: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        prompt_system_name = f"{step_name}__system"
        prompt_user_name = f"{step_name}__user"
        prompt_system = self.prompt_manager.load_prompt(prompt_name=prompt_system_name)
        if isinstance(prompt_content, dict):
            prompt_user = self.prompt_manager.load_prompt_multiple_inserts(
                prompt_name=prompt_user_name, insert_data=prompt_content
            )
        else:
            prompt_user = self.prompt_manager.load_prompt(
                prompt_name=prompt_user_name, insert_data=prompt_content
            )

        request_kwargs = {
            "messages": [
                {"role": "system", "content": prompt_system},
                {"role": "user", "content": prompt_user},
            ],
            "model": self.model,
            "temperature": 0,
        }

        if self.is_openrouter and reasoning:
            request_kwargs["extra_body"] = {"reasoning": reasoning}
        if not self.is_openrouter and "Qwen3.5" in self.model:
            request_kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": False}
            }

        return request_kwargs

    def call_llm(
        self,
        step_name: str,
        prompt_content: str | dict[str, str] = "",
        reasoning: dict[str, Any] | None = None,
    ) -> str:
        request_kwargs = self._create_payload(step_name, prompt_content, reasoning)
        try:
            result = self._sync_call(request_kwargs)
        except Exception as e:
            print(f"Error calling LLM: {e}")
            return ""
        return result

    async def call_llm_async(
        self,
        step_name: str,
        prompt_content: str | dict[str, str] = "",
        reasoning: dict[str, Any] | None = None,
    ) -> str:

        request_kwargs = self._create_payload(step_name, prompt_content, reasoning)

        async def _do_call() -> str:
            try:
                return await self._async_call(request_kwargs)
            except Exception as e:
                print(f"Error calling LLM: {e}")
                return ""

        if self.semaphore:
            async with self.semaphore:
                return await _do_call()
        else:
            return await _do_call()

    def _sync_call(self, request_kwargs: dict[str, Any]) -> str:
        response = (
            self.llm.chat.completions.create(**request_kwargs)
            .choices[0]
            .message.content
            or ""
        )
        return response

    async def _async_call(self, request_kwargs: dict[str, Any]) -> str:
        response = await self.async_llm.chat.completions.create(**request_kwargs)
        return response.choices[0].message.content or ""
