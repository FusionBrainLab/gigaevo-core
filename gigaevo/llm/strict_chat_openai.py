"""Strict-construction wrapper around ``langchain_openai.ChatOpenAI``.

``ChatOpenAI`` declares ``model_config['extra'] = 'ignore'`` and silently
reroutes unknown kwargs into ``model_kwargs``, then ships them verbatim to
the OpenAI HTTP endpoint. A typo such as ``tempetature: 0.5`` therefore
survives composition, survives instantiation, and only fails (or is
ignored) at the remote API. The fault surfaces as an opaque OpenAI error
far from the YAML site that introduced it.

This module exposes a thin factory, :func:`strict_chat_openai`, that
validates every kwarg against the union of ``ChatOpenAI.model_fields``
and the Pydantic aliases (``api_key`` ↔ ``openai_api_key``,
``max_tokens`` ↔ ``max_completion_tokens``, ``request_timeout`` ↔
``timeout`` etc.) *before* construction. Unknown kwargs raise
:class:`StrictChatOpenAIError` naming the offender. The factory also
refuses an explicit ``api_key=None`` (the resolved value of
``${oc.env:OPENAI_API_KEY,null}`` when the env var is unset) with an
application-layer message that names the missing variable.

The returned object is an unmodified :class:`ChatOpenAI` instance so the
LangChain runtime contract is preserved.
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
