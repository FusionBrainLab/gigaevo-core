"""
Prompt-step LLM calls for the IdeasTracker module.

Each analysis step has a prompt pair at prompts/{step}/{system,user}.txt next
to this file. The functions here render that pair into chat messages and call
the memory MultiModelRouter, which owns transport, credentials, temperature,
reasoning settings, Langfuse tracing, and token accounting (metrics under
``llm/tokens/<router name>/...``).
"""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

from loguru import logger
from pydantic import BaseModel

from gigaevo.llm.models import MultiModelRouter

TSchema = TypeVar("TSchema", bound=BaseModel)

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def read_prompt(step: str, prompt_type: str) -> str:
    path = PROMPTS_DIR / step / f"{prompt_type}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"No prompt at {path}")
    return path.read_text(encoding="utf-8")


def render_messages(
    step: str, content: str | dict[str, str] = ""
) -> list[tuple[str, str]]:
    """
    Render prompts/{step} into (system, user) chat messages.

    A dict ``content`` replaces each key with its value in the user prompt;
    a string ``content`` replaces the <INSERT> placeholder.
    """
    system = read_prompt(step, "system")
    user = read_prompt(step, "user")
    if isinstance(content, dict):
        for placeholder, value in content.items():
            user = user.replace(placeholder, value)
    else:
        user = user.replace("<INSERT>", content)
    return [("system", system), ("user", user)]


def call_step(
    llm: MultiModelRouter, step: str, content: str | dict[str, str] = ""
) -> str:
    """Plain-text LLM call for a prompt step; returns "" on failure."""
    try:
        return llm.invoke(render_messages(step, content)).text
    except Exception as exc:
        logger.error("[Memory][IdeaTracker][LLM] call({!r}) failed: {}", step, exc)
        return ""


async def call_step_async(
    llm: MultiModelRouter, step: str, content: str | dict[str, str] = ""
) -> str:
    """Async plain-text LLM call for a prompt step; returns "" on failure."""
    try:
        return (await llm.ainvoke(render_messages(step, content))).text
    except Exception as exc:
        logger.error("[Memory][IdeaTracker][LLM] call({!r}) failed: {}", step, exc)
        return ""


def call_step_structured(
    llm: MultiModelRouter,
    step: str,
    schema: type[TSchema],
    content: str | dict[str, str] = "",
) -> TSchema:
    """Structured LLM call for a prompt step; raises on transport or parse failure."""
    structured = llm.with_structured_output(schema, method="function_calling")
    return structured.invoke(render_messages(step, content))


async def call_step_structured_async(
    llm: MultiModelRouter,
    step: str,
    schema: type[TSchema],
    content: str | dict[str, str] = "",
) -> TSchema:
    """Async structured LLM call; raises on transport or parse failure."""
    structured = llm.with_structured_output(schema, method="function_calling")
    return await structured.ainvoke(render_messages(step, content))
