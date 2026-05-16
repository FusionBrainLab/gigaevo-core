"""Tests for the strict-construction wrapper around ``ChatOpenAI``.

The wrapper closes two failure modes that the underlying class exposes:
unknown kwargs silently fall through into ``model_kwargs`` and the
``${oc.env:OPENAI_API_KEY}`` interpolation resolves to ``None`` when the
env var is unset. Both behaviours used to manifest deep inside the OpenAI
client or as opaque Hydra interpolation tracebacks.
"""

from __future__ import annotations

from hydra.errors import InstantiationException
from hydra.utils import instantiate
from langchain_openai import ChatOpenAI
from omegaconf import OmegaConf
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


def test_via_hydra_instantiate_typo_raises() -> None:
    """``hydra.utils.instantiate`` surfaces the typo at construction time.

    Hydra wraps target exceptions in ``InstantiationException`` and chains
    the original via ``__cause__``; the typed wrapper is therefore the
    cause, not the directly raised exception.
    """
    cfg = OmegaConf.create(
        {
            "_target_": "gigaevo.llm.strict_chat_openai.strict_chat_openai",
            "model": "gpt-4o-mini",
            "api_key": "sk-test",
            "tempetature": 0.5,
        }
    )
    with pytest.raises(InstantiationException) as excinfo:
        instantiate(cfg)
    cause = excinfo.value.__cause__
    assert isinstance(cause, StrictChatOpenAIError)
    assert "tempetature" in str(cause)


def test_via_hydra_instantiate_missing_env_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``${oc.env:OPENAI_API_KEY,null}`` unset surfaces as a typed error."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg = OmegaConf.create(
        {
            "_target_": "gigaevo.llm.strict_chat_openai.strict_chat_openai",
            "model": "gpt-4o-mini",
            "api_key": "${oc.env:OPENAI_API_KEY,null}",
        }
    )
    with pytest.raises(InstantiationException) as excinfo:
        instantiate(cfg)
    cause = excinfo.value.__cause__
    assert isinstance(cause, StrictChatOpenAIError)
    assert "OPENAI_API_KEY" in str(cause)


def test_via_hydra_instantiate_present_env_constructs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the env var is set, instantiation succeeds via Hydra."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-from-env")
    cfg = OmegaConf.create(
        {
            "_target_": "gigaevo.llm.strict_chat_openai.strict_chat_openai",
            "model": "gpt-4o-mini",
            "api_key": "${oc.env:OPENAI_API_KEY,null}",
        }
    )
    instance = instantiate(cfg)
    assert isinstance(instance, ChatOpenAI)
