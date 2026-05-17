"""Strict-construction wrapper around ``langchain_openai.ChatOpenAI``.

``ChatOpenAI`` silently reroutes unknown kwargs into ``model_kwargs`` and
ships them to the OpenAI HTTP endpoint, so a YAML typo only surfaces as
an opaque remote error. :func:`strict_chat_openai` validates every
kwarg against the union of ``ChatOpenAI.model_fields`` and its Pydantic
aliases before construction; unknown kwargs and an unresolved
``api_key=None`` raise :class:`StrictChatOpenAIError` at the call site.
"""

from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI


class StrictChatOpenAIError(ValueError):
    """Raised when :func:`strict_chat_openai` rejects its inputs."""


def _allowed_kwargs() -> frozenset[str]:
    """Field names plus every non-None Pydantic alias on ``ChatOpenAI``.

    Computed once at import time. ``populate_by_name=True`` means
    LangChain accepts either form, so the allowlist must include both.
    """
    names: set[str] = set(ChatOpenAI.model_fields)
    aliases: set[str] = {
        field.alias
        for field in ChatOpenAI.model_fields.values()
        if field.alias is not None
    }
    return frozenset(names | aliases)


_ALLOWED: frozenset[str] = _allowed_kwargs()

# Both spellings of the OpenAI API key kwarg. ``api_key`` is the public
# alias documented in LangChain; ``openai_api_key`` is the underlying
# field name. Either may carry the resolved env-var value.
_API_KEY_KWARGS: tuple[str, ...] = ("api_key", "openai_api_key")


def strict_chat_openai(**kwargs: Any) -> ChatOpenAI:
    """Construct a :class:`ChatOpenAI` instance rejecting unknown kwargs.

    Raises
    ------
    StrictChatOpenAIError
        If any kwarg is not a known ``ChatOpenAI`` field or alias, or if
        the OpenAI API key resolves to ``None`` (env var unset).
    """
    unknown = sorted(set(kwargs) - _ALLOWED)
    if unknown:
        raise StrictChatOpenAIError(
            f"Unknown ChatOpenAI kwarg(s): {unknown}. Allowed: {sorted(_ALLOWED)}."
        )

    for key in _API_KEY_KWARGS:
        if key in kwargs and kwargs[key] is None:
            raise StrictChatOpenAIError(
                "OPENAI_API_KEY environment variable must be set; got None "
                f"(via kwarg '{key}'). Export OPENAI_API_KEY before running, "
                "or override api_key explicitly in the YAML."
            )

    return ChatOpenAI(**kwargs)
