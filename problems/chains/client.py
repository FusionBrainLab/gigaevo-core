"""LLM client with cost tracking for chain evolution."""

from dataclasses import dataclass
import os
import re
from typing import Any

import httpx
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential


@dataclass
class CallLog:
    """Log entry for a single LLM call."""

    prompt_tokens: int
    completion_tokens: int
    cost: float
    cost_utilization: float


def get_async_client(
    api_key: str | None = None,
    base_url: str = "https://openrouter.ai/api/v1",
) -> AsyncOpenAI:
    """Get async OpenAI client for LLM calls."""
    return AsyncOpenAI(
        api_key=api_key or os.environ.get("OPENAI_API_KEY", "None"),
        base_url=base_url,
        http_client=httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=300,
                max_keepalive_connections=10,
            ),
            timeout=httpx.Timeout(timeout=None, connect=30.0),
        ),
    )


def remove_thinking(text: str) -> str:
    """Strip <think>...</think> blocks from LLM thinking-mode output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


class LLMClient:
    """LLM client with cost tracking."""

    DEFAULT_PRICING: dict[str, dict[str, float]] = {
        "openai/gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
        "Qwen/Qwen3-8B": {"prompt": 0.028, "completion": 0.1104},
    }

    DEFAULT_GENERATION_KWARGS: dict[str, dict[str, Any]] = {
        "openai/gpt-4o-mini": {"max_tokens": 32768, "top_p": 1.0},
        "Qwen/Qwen3-8B": {"max_tokens": 32768, "top_p": 1.0},
    }

    def __init__(
        self,
        model: str,
        max_cost: float = 10.0,
        model_pricing: dict[str, float] | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        client_kwargs: dict[str, str] | None = None,
    ):
        self.model = model
        self.max_cost = max_cost
        self._call_logs: list[CallLog] = []
        self.client = get_async_client(**(client_kwargs or {}))

        self.model_pricing = model_pricing or self._get_default_pricing(model)
        self.generation_kwargs = (
            generation_kwargs or self._get_default_generation_kwargs(model)
        )

    @classmethod
    def _get_default_pricing(cls, model: str) -> dict[str, float]:
        return cls.DEFAULT_PRICING.get(model, {"prompt": 1.0, "completion": 1.0})

    @classmethod
    def _get_default_generation_kwargs(cls, model: str) -> dict[str, Any]:
        return cls.DEFAULT_GENERATION_KWARGS.get(model, {"max_tokens": 32768})

    @property
    def call_logs(self) -> list[CallLog]:
        return self._call_logs

    def clear_logs(self) -> None:
        self._call_logs = []

    def _compute_cost(
        self, prompt_tokens: int, completion_tokens: int
    ) -> tuple[float, float]:
        prompt_price = self.model_pricing.get("prompt", 1.0)
        completion_price = self.model_pricing.get("completion", 1.0)

        prompt_cost = (prompt_tokens / 1_000_000) * prompt_price
        completion_cost = (completion_tokens / 1_000_000) * completion_price
        total_cost = prompt_cost + completion_cost

        utilization = total_cost / self.max_cost if self.max_cost > 0 else 0.0
        return total_cost, utilization

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.1, min=0.1, max=2),
    )
    async def __call__(
        self,
        prompt: str,
        system_message: str | None = None,
        **overrides,
    ) -> str:
        kwargs = {**self.generation_kwargs, **overrides}

        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs,
        )

        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0

        cost, utilization = self._compute_cost(prompt_tokens, completion_tokens)

        self._call_logs.append(
            CallLog(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost=cost,
                cost_utilization=utilization,
            )
        )

        content = response.choices[0].message.content or ""
        content = remove_thinking(content)

        return content

    async def close(self) -> None:
        await self.client.close()

    def copy(self) -> "LLMClient":
        """Create an isolated copy with fresh call logs.

        The copy shares the underlying AsyncOpenAI client (stateless)
        but has independent call logs for parallel processing.
        """
        client = LLMClient.__new__(LLMClient)
        client.model = self.model
        client.max_cost = self.max_cost
        client.model_pricing = self.model_pricing
        client.generation_kwargs = self.generation_kwargs
        client.client = self.client
        client._call_logs = []
        return client
