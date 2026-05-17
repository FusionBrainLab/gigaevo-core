"""Tests for the strict-construction wrapper around ``ChatOpenAI``.

The wrapper closes two failure modes that the underlying class exposes:
unknown kwargs silently fall through into ``model_kwargs``, and a
``None`` ``api_key`` (often the result of an unset env var) constructs
an instance that fails opaquely later in the OpenAI client. Both
surface here as :class:`StrictChatOpenAIError`.
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI
import pytest

from gigaevo.llm.strict_chat_openai import (
    StrictChatOpenAIError,
    strict_chat_openai,
)


def test_known_kwargs_pass_through() -> None:
    """Common, well-spelled kwargs construct a ``ChatOpenAI`` instance."""
    instance = strict_chat_openai(
        model="gpt-4o-mini",
        api_key="sk-test",
        temperature=0.5,
        max_tokens=100,
    )
    assert isinstance(instance, ChatOpenAI)
    assert instance.model_name == "gpt-4o-mini"
    assert instance.temperature == 0.5


def test_known_aliases_pass_through() -> None:
    """``request_timeout`` (alias of ``timeout``) and ``base_url`` work."""
    instance = strict_chat_openai(
        model="gpt-4o-mini",
        api_key="sk-test",
        request_timeout=30,
        base_url="https://example.invalid/v1",
    )
    assert isinstance(instance, ChatOpenAI)


def test_field_name_form_passes_through() -> None:
    """Field-name spellings (not aliases) are also accepted."""
    instance = strict_chat_openai(
        model_name="gpt-4o-mini",
        openai_api_key="sk-test",
        max_completion_tokens=100,
    )
    assert isinstance(instance, ChatOpenAI)
    assert instance.model_name == "gpt-4o-mini"


def test_unknown_kwarg_raises() -> None:
    """A typo raises :class:`StrictChatOpenAIError` naming the offender."""
    with pytest.raises(StrictChatOpenAIError) as excinfo:
        strict_chat_openai(
            model="gpt-4o-mini",
            api_key="sk-test",
            tempetature=0.5,
        )
    assert "tempetature" in str(excinfo.value)


def test_unknown_kwarg_is_value_error() -> None:
    """``StrictChatOpenAIError`` is catchable as ``ValueError``."""
    with pytest.raises(ValueError):
        strict_chat_openai(
            model="gpt-4o-mini",
            api_key="sk-test",
            bogus_argument=1,
        )


def test_missing_api_key_raises_clear_error() -> None:
    """Explicit ``api_key=None`` raises a message naming ``OPENAI_API_KEY``."""
    with pytest.raises(StrictChatOpenAIError) as excinfo:
        strict_chat_openai(model="gpt-4o-mini", api_key=None)
    assert "OPENAI_API_KEY" in str(excinfo.value)


def test_missing_openai_api_key_field_form_raises() -> None:
    """The field-name form ``openai_api_key=None`` is also rejected."""
    with pytest.raises(StrictChatOpenAIError) as excinfo:
        strict_chat_openai(model="gpt-4o-mini", openai_api_key=None)
    assert "OPENAI_API_KEY" in str(excinfo.value)
