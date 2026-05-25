"""LangChain-backed LLM service for GAM agentic memory.

Thin adapter that satisfies LLMServiceProtocol by delegating to
langchain_openai.ChatOpenAI, so GAM shares the same plumbing
(with_structured_output, Langfuse tracing, per-provider dispatch) as the
rest of gigaevo's LLM stack instead of running a parallel raw-SDK pipeline.

Structured output method is per-deployment configurable via the
``structured_output_method`` constructor arg (sourced from
``gigaevo.memory.config.STRUCTURED_OUTPUT_METHOD`` at call sites). When
unset, the kwarg is omitted so LangChain picks the provider-appropriate
default — mirrors the ``kwargs.setdefault`` convention in
``gigaevo/llm/models.py``.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from loguru import logger
from pydantic import BaseModel


class LangChainLLMService:
    """LLMServiceProtocol implementation backed by langchain_openai.ChatOpenAI."""

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
        structured_output_method: str | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required")

        self.model_name = model_name
        self._structured_output_method = structured_output_method
        resolved_base_url = self._resolve_base_url(base_url, api_key)
        self._is_openrouter = bool(
            resolved_base_url and "openrouter.ai" in resolved_base_url
        )

        kwargs: dict[str, Any] = {
            "model": model_name,
            "api_key": api_key,
        }
        if resolved_base_url:
            kwargs["base_url"] = resolved_base_url
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None and max_tokens > 0:
            kwargs["max_tokens"] = max_tokens

        if self._is_openrouter:
            headers: dict[str, str] = {}
            if openrouter_referer:
                headers["HTTP-Referer"] = openrouter_referer
            if openrouter_title:
                headers["X-Title"] = openrouter_title
            if headers:
                kwargs["default_headers"] = headers

            extra_body: dict[str, Any] = {"usage": {"include": True}}
            if reasoning:
                extra_body["reasoning"] = reasoning
            kwargs["extra_body"] = extra_body

        self.client = ChatOpenAI(**kwargs)

    @staticmethod
    def _resolve_base_url(base_url: str | None, api_key: str) -> str | None:
        text = (base_url or "").strip()
        if text:
            return text
        if api_key.startswith("sk-or-"):
            return "https://openrouter.ai/api/v1"
        return None

    @staticmethod
    def _extract_content_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return ""
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
            else:
                text = getattr(part, "text", None)
            if isinstance(text, str):
                parts.append(text)
        return "".join(parts)

    @staticmethod
    def _extract_tokens(raw: Any) -> int | None:
        usage = getattr(raw, "usage_metadata", None)
        if isinstance(usage, dict):
            total = usage.get("total_tokens")
            if isinstance(total, int):
                return total
            prompt = usage.get("input_tokens") or usage.get("prompt_tokens")
            completion = usage.get("output_tokens") or usage.get("completion_tokens")
            if isinstance(prompt, int) or isinstance(completion, int):
                return int(prompt or 0) + int(completion or 0)

        meta = getattr(raw, "response_metadata", None) or {}
        token_usage = meta.get("token_usage") or meta.get("usage") or {}
        if isinstance(token_usage, dict):
            total = token_usage.get("total_tokens")
            if isinstance(total, int):
                return total
        return None

    @staticmethod
    def _extract_cost(raw: Any) -> float | None:
        meta = getattr(raw, "response_metadata", None) or {}
        cost = meta.get("cost")
        if cost is None:
            usage = meta.get("usage") or {}
            cost = usage.get("cost") if isinstance(usage, dict) else None
        if cost is None:
            return None
        try:
            return float(cost)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _parsed_to_text(parsed: Any) -> str:
        if isinstance(parsed, BaseModel):
            return parsed.model_dump_json()
        if isinstance(parsed, (dict, list)):
            return json.dumps(parsed)
        if parsed is None:
            return ""
        return str(parsed)

    def generate(
        self,
        data: str,
        *,
        schema: dict[str, Any] | None = None,
    ) -> tuple[str, Any, int | None, float | None]:
        """Send a prompt to the underlying ChatOpenAI and return a 4-tuple.

        When ``schema`` is provided, the call routes through
        ``with_structured_output(schema, include_raw=True)``; the ``method``
        kwarg is forwarded only when ``structured_output_method`` was set
        on the constructor (otherwise LangChain picks its
        provider-appropriate default).
        """
        messages = [HumanMessage(content=data)]

        if schema is not None:
            kwargs: dict[str, Any] = {"include_raw": True}
            if self._structured_output_method is not None:
                kwargs["method"] = self._structured_output_method
            structured = self.client.with_structured_output(schema, **kwargs)
            envelope = structured.invoke(messages)
            raw = envelope.get("raw") if isinstance(envelope, dict) else None
            parsed = envelope.get("parsed") if isinstance(envelope, dict) else None
            parsing_error = (
                envelope.get("parsing_error") if isinstance(envelope, dict) else None
            )
            if parsing_error is not None and parsed is None:
                logger.warning(
                    "[Memory][LLM] structured-output parsing failed: {}",
                    parsing_error,
                )
            content = self._parsed_to_text(parsed)
        else:
            raw = self.client.invoke(messages)
            content = self._extract_content_text(getattr(raw, "content", ""))

        return content, raw, self._extract_tokens(raw), self._extract_cost(raw)
