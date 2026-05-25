"""Tests for LangChainLLMService.

This adapter satisfies LLMServiceProtocol by delegating to langchain_openai's
ChatOpenAI, so GAM shares the same plumbing (with_structured_output,
Langfuse tracing, per-provider dispatch) as the rest of gigaevo's LLM stack.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from gigaevo.memory.langchain_llm_service import LangChainLLMService

# ===========================================================================
# _resolve_base_url (static, pure) — base-url + sk-or- inference
# ===========================================================================


class TestResolveBaseUrl:
    def test_explicit_url(self):
        assert (
            LangChainLLMService._resolve_base_url("http://localhost:8000", "sk-abc")
            == "http://localhost:8000"
        )

    def test_explicit_url_with_whitespace(self):
        assert (
            LangChainLLMService._resolve_base_url("  http://localhost:8000  ", "sk-abc")
            == "http://localhost:8000"
        )

    def test_openrouter_key_detection(self):
        assert (
            LangChainLLMService._resolve_base_url(None, "sk-or-abc123")
            == "https://openrouter.ai/api/v1"
        )

    def test_regular_key_no_base_url(self):
        assert LangChainLLMService._resolve_base_url(None, "sk-abc123") is None

    def test_empty_base_url_with_openrouter_key(self):
        assert (
            LangChainLLMService._resolve_base_url("", "sk-or-abc")
            == "https://openrouter.ai/api/v1"
        )


# ===========================================================================
# Constructor — wires kwargs into ChatOpenAI
# ===========================================================================


class TestInit:
    def test_missing_api_key_raises(self):
        with pytest.raises(ValueError, match="api_key is required"):
            LangChainLLMService(model_name="gpt-4", api_key="")

    def test_none_api_key_raises(self):
        with pytest.raises((ValueError, TypeError)):
            LangChainLLMService(model_name="gpt-4", api_key=None)  # type: ignore[arg-type]

    @patch("gigaevo.memory.langchain_llm_service.ChatOpenAI")
    def test_basic_wiring(self, mock_chat_cls):
        LangChainLLMService(
            model_name="gpt-4",
            api_key="sk-test",
            base_url="http://localhost:8000",
            temperature=0.0,
            max_tokens=512,
        )
        mock_chat_cls.assert_called_once()
        kwargs = mock_chat_cls.call_args.kwargs
        assert kwargs["model"] == "gpt-4"
        assert kwargs["api_key"] == "sk-test"
        assert kwargs["base_url"] == "http://localhost:8000"
        assert kwargs["temperature"] == 0.0
        assert kwargs["max_tokens"] == 512

    @patch("gigaevo.memory.langchain_llm_service.ChatOpenAI")
    def test_zero_max_tokens_not_passed(self, mock_chat_cls):
        # The legacy adapter treated max_tokens=0 as "unset"; preserve that.
        LangChainLLMService(model_name="gpt-4", api_key="sk-test", max_tokens=0)
        kwargs = mock_chat_cls.call_args.kwargs
        assert "max_tokens" not in kwargs

    @patch("gigaevo.memory.langchain_llm_service.ChatOpenAI")
    def test_openrouter_referer_and_title_set_default_headers(self, mock_chat_cls):
        LangChainLLMService(
            model_name="gpt-4",
            api_key="sk-or-test",
            openrouter_referer="http://example.com",
            openrouter_title="Test App",
        )
        kwargs = mock_chat_cls.call_args.kwargs
        assert kwargs["default_headers"]["HTTP-Referer"] == "http://example.com"
        assert kwargs["default_headers"]["X-Title"] == "Test App"

    @patch("gigaevo.memory.langchain_llm_service.ChatOpenAI")
    def test_openrouter_extra_body_carries_reasoning_and_usage(self, mock_chat_cls):
        LangChainLLMService(
            model_name="gpt-4",
            api_key="sk-or-test",
            reasoning={"effort": "high"},
        )
        kwargs = mock_chat_cls.call_args.kwargs
        assert kwargs["extra_body"]["reasoning"] == {"effort": "high"}
        assert kwargs["extra_body"]["usage"] == {"include": True}

    @patch("gigaevo.memory.langchain_llm_service.ChatOpenAI")
    def test_non_openrouter_keeps_extra_body_empty(self, mock_chat_cls):
        LangChainLLMService(
            model_name="gpt-4",
            api_key="sk-test",
            base_url="http://localhost:8000",
            reasoning={"effort": "high"},
        )
        kwargs = mock_chat_cls.call_args.kwargs
        # Reasoning is an OpenRouter-only knob; suppress it for non-OpenRouter
        # endpoints (LiteLLM proxy would reject the unknown field).
        assert "extra_body" not in kwargs or "reasoning" not in kwargs.get(
            "extra_body", {}
        )

    @patch("gigaevo.memory.langchain_llm_service.ChatOpenAI")
    def test_non_openrouter_does_not_set_referer_headers(self, mock_chat_cls):
        LangChainLLMService(
            model_name="gpt-4",
            api_key="sk-test",
            base_url="http://localhost:8000",
            openrouter_referer="http://example.com",
            openrouter_title="Test App",
        )
        kwargs = mock_chat_cls.call_args.kwargs
        assert "default_headers" not in kwargs


# ===========================================================================
# generate() — free-form path (no schema)
# ===========================================================================


def _raw_message(*, content: str, total_tokens: int | None = None, cost=None):
    msg = MagicMock()
    msg.content = content
    if total_tokens is not None:
        msg.usage_metadata = {
            "total_tokens": total_tokens,
            "input_tokens": total_tokens // 2,
            "output_tokens": total_tokens - total_tokens // 2,
        }
    else:
        msg.usage_metadata = None
    response_meta: dict = {}
    if cost is not None:
        response_meta["cost"] = cost
    msg.response_metadata = response_meta
    return msg


class TestGeneratePlain:
    @patch("gigaevo.memory.langchain_llm_service.ChatOpenAI")
    def test_returns_4_tuple(self, mock_chat_cls):
        mock_chat = MagicMock()
        mock_chat_cls.return_value = mock_chat
        svc = LangChainLLMService(
            model_name="gpt-4",
            api_key="sk-test",
            base_url="http://localhost:8000",
        )
        mock_chat.invoke = MagicMock(
            return_value=_raw_message(content="Hello world", total_tokens=42)
        )

        content, raw, tokens, cost = svc.generate("test prompt")
        assert content == "Hello world"
        assert tokens == 42
        assert cost is None

    @patch("gigaevo.memory.langchain_llm_service.ChatOpenAI")
    def test_invoke_called_with_human_message(self, mock_chat_cls):
        from langchain_core.messages import HumanMessage

        mock_chat = MagicMock()
        mock_chat_cls.return_value = mock_chat
        svc = LangChainLLMService(model_name="gpt-4", api_key="sk-test")
        mock_chat.invoke = MagicMock(return_value=_raw_message(content="ok"))

        svc.generate("the prompt")
        args, _ = mock_chat.invoke.call_args
        messages = args[0]
        assert isinstance(messages, list) and len(messages) == 1
        assert isinstance(messages[0], HumanMessage)
        assert messages[0].content == "the prompt"

    @patch("gigaevo.memory.langchain_llm_service.ChatOpenAI")
    def test_no_schema_does_not_call_with_structured_output(self, mock_chat_cls):
        mock_chat = MagicMock()
        mock_chat_cls.return_value = mock_chat
        svc = LangChainLLMService(model_name="gpt-4", api_key="sk-test")
        mock_chat.invoke = MagicMock(return_value=_raw_message(content="ok"))

        svc.generate("test")
        mock_chat.with_structured_output.assert_not_called()

    @patch("gigaevo.memory.langchain_llm_service.ChatOpenAI")
    def test_cost_extracted_from_response_metadata(self, mock_chat_cls):
        mock_chat = MagicMock()
        mock_chat_cls.return_value = mock_chat
        svc = LangChainLLMService(model_name="gpt-4", api_key="sk-test")
        mock_chat.invoke = MagicMock(
            return_value=_raw_message(content="ok", total_tokens=10, cost=0.005)
        )

        _, _, _, cost = svc.generate("test")
        assert cost == 0.005


# ===========================================================================
# generate(schema=dict) — structured output is wire-bound; `method` is
# forwarded only when configured (else LangChain's default applies).
# ===========================================================================


class TestGenerateWithSchema:
    @patch("gigaevo.memory.langchain_llm_service.ChatOpenAI")
    def test_method_kwarg_omitted_when_unset(self, mock_chat_cls):
        mock_chat = MagicMock()
        mock_chat_cls.return_value = mock_chat
        svc = LangChainLLMService(model_name="gpt-4", api_key="sk-test")

        structured = MagicMock()
        structured.invoke = MagicMock(
            return_value={
                "raw": _raw_message(content="", total_tokens=7),
                "parsed": {"answer": "ok"},
                "parsing_error": None,
            }
        )
        mock_chat.with_structured_output = MagicMock(return_value=structured)

        schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        }
        svc.generate("test prompt", schema=schema)

        call = mock_chat.with_structured_output.call_args
        passed_schema = call.args[0] if call.args else call.kwargs.get("schema")
        assert passed_schema is schema
        assert "method" not in call.kwargs
        assert call.kwargs.get("include_raw") is True

    @patch("gigaevo.memory.langchain_llm_service.ChatOpenAI")
    def test_method_kwarg_forwarded_when_configured(self, mock_chat_cls):
        mock_chat = MagicMock()
        mock_chat_cls.return_value = mock_chat
        svc = LangChainLLMService(
            model_name="gpt-4",
            api_key="sk-test",
            structured_output_method="function_calling",
        )

        structured = MagicMock()
        structured.invoke = MagicMock(
            return_value={
                "raw": _raw_message(content="", total_tokens=1),
                "parsed": {"answer": "ok"},
                "parsing_error": None,
            }
        )
        mock_chat.with_structured_output = MagicMock(return_value=structured)

        svc.generate("test", schema={"type": "object"})

        call = mock_chat.with_structured_output.call_args
        assert call.kwargs.get("method") == "function_calling"
        assert call.kwargs.get("include_raw") is True

    @patch("gigaevo.memory.langchain_llm_service.ChatOpenAI")
    def test_method_kwarg_accepts_json_schema(self, mock_chat_cls):
        mock_chat = MagicMock()
        mock_chat_cls.return_value = mock_chat
        svc = LangChainLLMService(
            model_name="gpt-4",
            api_key="sk-test",
            structured_output_method="json_schema",
        )

        structured = MagicMock()
        structured.invoke = MagicMock(
            return_value={
                "raw": _raw_message(content="", total_tokens=1),
                "parsed": {"answer": "ok"},
                "parsing_error": None,
            }
        )
        mock_chat.with_structured_output = MagicMock(return_value=structured)

        svc.generate("test", schema={"type": "object"})

        call = mock_chat.with_structured_output.call_args
        assert call.kwargs.get("method") == "json_schema"

    @patch("gigaevo.memory.langchain_llm_service.ChatOpenAI")
    def test_returns_parsed_dict_as_json_text(self, mock_chat_cls):
        mock_chat = MagicMock()
        mock_chat_cls.return_value = mock_chat
        svc = LangChainLLMService(model_name="gpt-4", api_key="sk-test")

        structured = MagicMock()
        structured.invoke = MagicMock(
            return_value={
                "raw": _raw_message(content="", total_tokens=7),
                "parsed": {"answer": "ok"},
                "parsing_error": None,
            }
        )
        mock_chat.with_structured_output = MagicMock(return_value=structured)

        content, _, tokens, _ = svc.generate("test", schema={"type": "object"})
        assert json.loads(content) == {"answer": "ok"}
        assert tokens == 7

    @patch("gigaevo.memory.langchain_llm_service.ChatOpenAI")
    def test_parsing_error_returns_empty_content(self, mock_chat_cls):
        mock_chat = MagicMock()
        mock_chat_cls.return_value = mock_chat
        svc = LangChainLLMService(model_name="gpt-4", api_key="sk-test")

        raw = _raw_message(content="garbled", total_tokens=3)
        structured = MagicMock()
        structured.invoke = MagicMock(
            return_value={
                "raw": raw,
                "parsed": None,
                "parsing_error": Exception("model refused tool call"),
            }
        )
        mock_chat.with_structured_output = MagicMock(return_value=structured)

        content, _, _, _ = svc.generate("test", schema={"type": "object"})
        assert content == ""

    @patch("gigaevo.memory.langchain_llm_service.ChatOpenAI")
    def test_pydantic_parsed_is_serialised(self, mock_chat_cls):
        from pydantic import BaseModel

        class Decision(BaseModel):
            answer: str
            confidence: float = 0.9

        mock_chat = MagicMock()
        mock_chat_cls.return_value = mock_chat
        svc = LangChainLLMService(model_name="gpt-4", api_key="sk-test")

        structured = MagicMock()
        structured.invoke = MagicMock(
            return_value={
                "raw": _raw_message(content="", total_tokens=4),
                "parsed": Decision(answer="ok"),
                "parsing_error": None,
            }
        )
        mock_chat.with_structured_output = MagicMock(return_value=structured)

        content, _, _, _ = svc.generate("test", schema={"type": "object"})
        assert json.loads(content) == {"answer": "ok", "confidence": 0.9}
