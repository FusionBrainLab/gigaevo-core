"""
LLM client for the IdeasTracker module.

LLMClient wraps an OpenAI-compatible API with prompt-file loading.
Prompts are stored in prompts/{step}/system.txt and prompts/{step}/user.txt
adjacent to this file. _PromptLoader is a private implementation detail.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from loguru import logger
from openai import AsyncOpenAI, OpenAI


class _PromptLoader:
    """Loads prompt text files from the prompts/ directory next to llm.py."""

    def __init__(self) -> None:
        self._dir = Path(__file__).resolve().parent / "prompts"

    def load(
        self, step: str, prompt_type: str, insert: str | dict[str, str] = ""
    ) -> str:
        """
        Load prompts/{step}/{prompt_type}.txt and optionally fill placeholders.

        For user prompts with a string insert, replaces <INSERT>.
        For user prompts with a dict insert, replaces each key with its value.
        """
        path = self._dir / step / f"{prompt_type}.txt"
        if not path.is_file():
            raise FileNotFoundError(f"No prompt at {path}")
        text = path.read_text(encoding="utf-8")
        if prompt_type == "user":
            if isinstance(insert, dict):
                for placeholder, content in insert.items():
                    text = text.replace(placeholder, content)
            else:
                text = text.replace("<INSERT>", insert)
        return text


def _init_clients(base_url: str | None) -> tuple[OpenAI, AsyncOpenAI, bool]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set.")
    env_base = (
        os.getenv("OPENAI_BASE_URL")
        or os.getenv("BASE_URL")
        or os.getenv("LLM_BASE_URL")
    )
    effective_url = env_base or base_url
    if not effective_url and api_key.startswith("sk-or-"):
        effective_url = "https://openrouter.ai/api/v1"
    is_openrouter = bool(effective_url and "openrouter.ai" in effective_url)
    kwargs: dict[str, Any] = {"api_key": api_key}
    if effective_url:
        kwargs["base_url"] = effective_url
    return OpenAI(**kwargs), AsyncOpenAI(**kwargs), is_openrouter


class LLMClient:
    """
    OpenAI-compatible LLM client with prompt-file loading.

    Prompts are loaded from prompts/{step}/system.txt and prompts/{step}/user.txt
    next to this file. Supports sync and async calls with optional concurrency limiting.

    Args:
        model: Model identifier (e.g. "google/gemini-3-flash-preview").
        base_url: Optional API base URL override. Falls back to OPENAI_BASE_URL env var.
        max_concurrent: Max parallel async calls. -1 means unlimited.
    """

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        max_concurrent: int = -1,
    ) -> None:
        self.model = model
        self._sync, self._async, self._is_openrouter = _init_clients(
            str(base_url).strip() if base_url is not None else None
        )
        self._prompts = _PromptLoader()
        self._semaphore = (
            asyncio.Semaphore(max_concurrent) if max_concurrent > 0 else None
        )

    def _build_request(
        self, step: str, content: str | dict[str, str], reasoning: dict | None
    ) -> dict[str, Any]:
        system = self._prompts.load(step, "system")
        user = self._prompts.load(step, "user", content)
        kwargs: dict[str, Any] = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "model": self.model,
            "temperature": 0,
        }
        if self._is_openrouter and reasoning:
            kwargs["extra_body"] = {"reasoning": reasoning}
        if not self._is_openrouter and "Qwen3.5" in self.model:
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        return kwargs

    def call(
        self,
        step: str,
        content: str | dict[str, str] = "",
        reasoning: dict | None = None,
    ) -> str:
        """Synchronous LLM call for the given prompt step."""
        try:
            request = self._build_request(step, content, reasoning)
            return (
                self._sync.chat.completions.create(**request).choices[0].message.content
                or ""
            )
        except Exception as e:
            logger.error(f"LLMClient.call({step!r}) failed: {e}")
            return ""

    async def call_async(
        self,
        step: str,
        content: str | dict[str, str] = "",
        reasoning: dict | None = None,
    ) -> str:
        """Asynchronous LLM call for the given prompt step."""
        request = self._build_request(step, content, reasoning)

        async def _do() -> str:
            try:
                resp = await self._async.chat.completions.create(**request)
                return resp.choices[0].message.content or ""
            except Exception as e:
                logger.error(f"LLMClient.call_async({step!r}) failed: {e}")
                return ""

        if self._semaphore:
            async with self._semaphore:
                return await _do()
        return await _do()
