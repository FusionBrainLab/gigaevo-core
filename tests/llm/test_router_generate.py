"""MultiModelRouter.generate(): the LLMServiceProtocol entrypoint for memory.

The memory subsystem (A-MEM/GAM, dedup, search) calls
``generate(prompt, schema=...)`` and reads only the text slot of the returned
4-tuple; token accounting happens inside the router's TokenTracker. The
structured path delegates to ``with_structured_output`` so parse failures
raise instead of silently degrading.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from pydantic import BaseModel
import pytest

from gigaevo.llm.models import MultiModelRouter


def _mock_model(name: str = "mem-model") -> MagicMock:
    m = MagicMock()
    m.model_name = name
    return m


def _router(model: MagicMock, **kwargs) -> MultiModelRouter:
    return MultiModelRouter([model], [1.0], name="memory-test", **kwargs)


class TestGeneratePlain:
    def test_returns_content_and_raw_response(self):
        model = _mock_model()
        raw = MagicMock()
        raw.content = "hello"
        model.invoke.return_value = raw

        text, response, tokens, cost = _router(model).generate("prompt")

        assert text == "hello"
        assert response is raw
        assert tokens is None
        assert cost is None

    def test_prompt_forwarded_to_model(self):
        model = _mock_model()
        model.invoke.return_value = MagicMock(content="x")

        _router(model).generate("the prompt")

        model.invoke.assert_called_once()
        assert model.invoke.call_args.args[0] == "the prompt"

    def test_list_content_parts_joined(self):
        model = _mock_model()
        raw = MagicMock()
        raw.content = [
            {"type": "text", "text": "part one "},
            {"type": "text", "text": "part two"},
            {"type": "image", "url": "ignored"},
        ]
        model.invoke.return_value = raw

        text, _, _, _ = _router(model).generate("prompt")

        assert text == "part one part two"

    def test_no_structured_wrapper_without_schema(self):
        model = _mock_model()
        model.invoke.return_value = MagicMock(content="x")

        _router(model).generate("prompt")

        model.with_structured_output.assert_not_called()


class TestGenerateWithSchema:
    SCHEMA = {"type": "object", "properties": {"answer": {"type": "integer"}}}

    def _model_returning(self, envelope: dict) -> MagicMock:
        model = _mock_model()
        structured = MagicMock()
        structured.invoke.return_value = envelope
        model.with_structured_output.return_value = structured
        return model

    def test_delegates_to_structured_wrapper(self):
        model = self._model_returning(
            {"raw": MagicMock(), "parsed": {"answer": 42}, "parsing_error": None}
        )

        text, response, tokens, cost = _router(model).generate(
            "prompt", schema=self.SCHEMA
        )

        model.with_structured_output.assert_called_once()
        assert model.with_structured_output.call_args.kwargs["include_raw"] is True
        assert json.loads(text) == {"answer": 42}
        assert response == {"answer": 42}
        assert tokens is None
        assert cost is None

    def test_method_omitted_when_unset(self):
        model = self._model_returning(
            {"raw": MagicMock(), "parsed": {}, "parsing_error": None}
        )

        _router(model).generate("prompt", schema=self.SCHEMA)

        assert "method" not in model.with_structured_output.call_args.kwargs

    def test_method_forwarded_from_constructor(self):
        model = self._model_returning(
            {"raw": MagicMock(), "parsed": {}, "parsing_error": None}
        )

        _router(model, structured_output_method="function_calling").generate(
            "prompt", schema=self.SCHEMA
        )

        assert (
            model.with_structured_output.call_args.kwargs["method"]
            == "function_calling"
        )

    def test_pydantic_parsed_serialized(self):
        class Out(BaseModel):
            answer: int

        model = self._model_returning(
            {"raw": MagicMock(), "parsed": Out(answer=7), "parsing_error": None}
        )

        text, _, _, _ = _router(model).generate("prompt", schema=self.SCHEMA)

        assert json.loads(text) == {"answer": 7}

    def test_parse_failure_raises(self):
        raw = MagicMock()
        raw.content = "not parseable"
        model = self._model_returning(
            {"raw": raw, "parsed": None, "parsing_error": Exception("boom")}
        )

        with pytest.raises(ValueError, match="parse failed"):
            _router(model).generate("prompt", schema=self.SCHEMA)
