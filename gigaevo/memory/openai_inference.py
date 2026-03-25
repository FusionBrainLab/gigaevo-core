from __future__ import annotations

from typing import Any


class OpenAIInferenceService:
    """OpenAI-compatible inference adapter with A-MEM LLMService-like interface."""

    def __init__(
        self,
        *,
        model_name: str,
        api_key: str,
        base_url: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning: dict[str, Any] | None = None,
        openrouter_referer: str | None = None,
        openrouter_title: str | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required")

        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning = reasoning if isinstance(reasoning, dict) else {}
        self.openrouter_referer = openrouter_referer
        self.openrouter_title = openrouter_title

        resolved_base_url = self._resolve_base_url(base_url, api_key)
        self._is_openrouter = bool(
            resolved_base_url and "openrouter.ai" in resolved_base_url
        )

        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if resolved_base_url:
            client_kwargs["base_url"] = resolved_base_url

        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "Install the 'openai' package to use OpenAIInferenceService"
            ) from exc

        self.client = OpenAI(**client_kwargs)

    @staticmethod
    def _resolve_base_url(base_url: str | None, api_key: str) -> str | None:
        text = (base_url or "").strip()
        if text:
            return text
        if api_key.startswith("sk-or-"):
            return "https://openrouter.ai/api/v1"
        return None

    @staticmethod
    def _extract_content_text(message_content: Any) -> str:
        if isinstance(message_content, str):
            return message_content
        if not isinstance(message_content, list):
            return ""

        parts: list[str] = []
        for part in message_content:
            if isinstance(part, dict):
                text = part.get("text")
            else:
                text = getattr(part, "text", None)
            if isinstance(text, str):
                parts.append(text)
        return "".join(parts)

    @staticmethod
    def _extract_total_tokens(usage: dict[str, Any]) -> int | None:
        total = usage.get("total_tokens")
        if isinstance(total, int):
            return total

        prompt = usage.get("prompt_tokens")
        if prompt is None:
            prompt = usage.get("input_tokens")
        completion = usage.get("completion_tokens")
        if completion is None:
            completion = usage.get("output_tokens")

        if isinstance(prompt, int) or isinstance(completion, int):
            return int(prompt or 0) + int(completion or 0)
        return None

    def generate(self, data: str) -> tuple[str, Any, int | None, float | None]:
        extra_headers: dict[str, str] = {}
        if self._is_openrouter and self.openrouter_referer:
            extra_headers["HTTP-Referer"] = self.openrouter_referer
        if self._is_openrouter and self.openrouter_title:
            extra_headers["X-Title"] = self.openrouter_title

        request_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": data}],
        }
        if self.temperature is not None:
            request_kwargs["temperature"] = self.temperature
        if self.max_tokens is not None and self.max_tokens > 0:
            request_kwargs["max_tokens"] = self.max_tokens
        if extra_headers:
            request_kwargs["extra_headers"] = extra_headers

        if self._is_openrouter:
            extra_body: dict[str, Any] = {"usage": {"include": True}}
            if self.reasoning:
                extra_body["reasoning"] = self.reasoning
            request_kwargs["extra_body"] = extra_body

        response = self.client.chat.completions.create(**request_kwargs)
        response_payload = response.model_dump()
        content = ""
        if response.choices:
            content = self._extract_content_text(response.choices[0].message.content)

        usage = response_payload.get("usage", {}) or {}
        token_count = self._extract_total_tokens(usage)
        cost = usage.get("cost")
        if cost is not None:
            try:
                cost = float(cost)
            except (TypeError, ValueError):
                cost = None
        return content, response_payload, token_count, cost
