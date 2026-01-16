"""LLM client with cost tracking."""

from __future__ import annotations

import os
import httpx
from typing import Any, List

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from problems.prompts.types import CallLog


def get_async_client(
    api_key: str | None = None,
    base_url: str = "https://openrouter.ai/api/v1",
    proxy: str | None = None,
) -> AsyncOpenAI:
    """Get async OpenAI client for LLM calls.

    Args:
        api_key: API key (defaults to OPENAI_API_KEY env var)
        base_url: API base URL (defaults to OpenRouter)

    Returns:
        AsyncOpenAI client instance
    """
    http_client = None
    if proxy is not None:
        http_client = httpx.AsyncClient(proxy=proxy)
    return AsyncOpenAI(
        api_key=api_key or os.environ["OPENAI_API_KEY"],
        base_url=base_url,
        http_client=http_client,
    )


class LLMClient:
    """LLM client with cost tracking."""

    # Default pricing per 1M tokens
    DEFAULT_PRICING: dict[str, dict[str, float]] = {
        "openai/gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
        "qwen/qwen3-8b": {"prompt": 0.028, "completion": 0.1104},
    }

    # Default generation kwargs per model
    DEFAULT_GENERATION_KWARGS: dict[str, dict[str, Any]] = {
        "openai/gpt-4o-mini": {"max_tokens": 32768, "top_p": 1.0},
        "qwen/qwen3-8b": {"max_tokens": 32768, "top_p": 1.0},
    }

    def __init__(
        self,
        model: str,
        max_cost: float = 10.0,
        model_pricing: dict[str, float] | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        client_kwargs: dict[str, str] | None = None,
    ):
        """Initialize LLM client.

        Args:
            model: Model name/identifier
            max_cost: Maximum cost budget in dollars per sample
            model_pricing: Custom pricing dict {"prompt": price_per_1M, "completion": price_per_1M}
            generation_kwargs: Custom generation params (top_p, max_tokens, etc.)
            client_kwargs: kwargs for AsyncOpenAI client (api_key, base_url)
        """
        self.model = model
        self.max_cost = max_cost
        self._call_logs: List[CallLog] = []
        self.client = get_async_client(**(client_kwargs or {}))

        # Model pricing (per 1M tokens) - user-provided or default
        self.model_pricing = model_pricing or self._get_default_pricing(model)

        # Generation kwargs - user-provided or default for model
        self.generation_kwargs = (
            generation_kwargs or self._get_default_generation_kwargs(model)
        )

    @classmethod
    def _get_default_pricing(cls, model: str) -> dict[str, float]:
        """Get default pricing for common models."""
        return cls.DEFAULT_PRICING.get(model, {"prompt": 1.0, "completion": 1.0})

    @classmethod
    def _get_default_generation_kwargs(cls, model: str) -> dict[str, Any]:
        """Get default generation kwargs for common models."""
        return cls.DEFAULT_GENERATION_KWARGS.get(model, {"max_tokens": 1024})

    @property
    def call_logs(self) -> List[CallLog]:
        """Get list of all call logs."""
        return self._call_logs

    def clear_logs(self) -> None:
        """Clear all call logs."""
        self._call_logs = []

    def _compute_cost(
        self, prompt_tokens: int, completion_tokens: int
    ) -> tuple[float, float]:
        """Compute cost in dollars and utilization fraction.

        Args:
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens

        Returns:
            Tuple of (cost_in_dollars, cost_utilization)
        """
        prompt_price = self.model_pricing.get("prompt", 1.0)
        completion_price = self.model_pricing.get("completion", 1.0)

        prompt_cost = (prompt_tokens / 1_000_000) * prompt_price
        completion_cost = (completion_tokens / 1_000_000) * completion_price
        total_cost = prompt_cost + completion_cost

        utilization = total_cost / self.max_cost if self.max_cost > 0 else 0.0

        return total_cost, utilization

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def __call__(self, prompt: str) -> str:
        """Make LLM call with cost tracking.

        Args:
            prompt: The prompt to send

        Returns:
            Raw response string

        Raises:
            ValueError: If cost budget is exceeded
        """
        kwargs = self.generation_kwargs

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )

        # Extract token usage
        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0

        # Compute and log cost
        cost, utilization = self._compute_cost(prompt_tokens, completion_tokens)

        self._call_logs.append(
            CallLog(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost=cost,
                cost_utilization=utilization,
            )
        )

        # Check if we exceeded budget after this call
        total_util = sum(log.cost_utilization for log in self._call_logs)
        if total_util > 1.0:
            raise ValueError(f"Cost budget exceeded: {total_util:.2%}")

        # Return raw response content
        return response.choices[0].message.content or ""

    async def close(self) -> None:
        """Close the underlying client."""
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
        client.client = self.client  # Share client (stateless)
        client._call_logs = []  # Fresh logs

        return client
