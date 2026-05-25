"""Tests for AMemGenerator schema-on-the-wire enforcement.

These tests pin the contract that when a schema is provided, AMemGenerator
hands it to the underlying LLM service (so it lands on the wire as tools/
response_format) rather than concatenating it as text into the prompt.

That way, the schema becomes a server-enforced contract instead of a polite
suggestion the model may or may not honour.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from gigaevo.memory._vendor.GAM_root.gam.generator.amem_generator import (
    AMemGenerator,
)


class _RecordingLLMService:
    """Captures (prompt, schema) on each ``generate`` call."""

    def __init__(self, *, content: str) -> None:
        self.calls: list[dict[str, Any]] = []
        self._content = content

    def generate(self, data: str, *, schema: dict[str, Any] | None = None):
        self.calls.append({"data": data, "schema": schema})
        return self._content, MagicMock(), None, None


def _schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }


class TestSchemaDoesNotLeakIntoPrompt:
    def test_schema_does_not_appear_in_prompt_text(self):
        svc = _RecordingLLMService(content='{"answer": "x"}')
        gen = AMemGenerator({"llm_service": svc})

        gen.generate_single(prompt="user question", schema=_schema())

        assert len(svc.calls) == 1
        sent = svc.calls[0]["data"]
        # The schema must not have been concatenated into the prompt — it has
        # to travel via the kwarg so the wire payload enforces it.
        assert "JSON schema" not in sent
        assert "additionalProperties" not in sent
        assert "json_schema" not in sent

    def test_schema_kwarg_is_forwarded_to_llm_service(self):
        svc = _RecordingLLMService(content='{"answer": "x"}')
        gen = AMemGenerator({"llm_service": svc})
        schema = _schema()

        gen.generate_single(prompt="ask", schema=schema)

        assert svc.calls[0]["schema"] is schema

    def test_no_schema_means_no_schema_kwarg_or_text_injection(self):
        svc = _RecordingLLMService(content="free-form reply")
        gen = AMemGenerator({"llm_service": svc})

        out = gen.generate_single(prompt="ask")

        assert svc.calls[0]["schema"] is None
        assert svc.calls[0]["data"] == "ask"
        assert out["text"] == "free-form reply"
        assert out["json"] is None


class TestSchemaResponseParsing:
    def test_json_field_parses_clean_tool_call_arguments(self):
        payload = '{"answer": "ok"}'
        svc = _RecordingLLMService(content=payload)
        gen = AMemGenerator({"llm_service": svc})

        out = gen.generate_single(prompt="ask", schema=_schema())

        assert out["text"] == payload
        assert out["json"] == {"answer": "ok"}

    def test_json_field_none_when_content_is_empty(self):
        svc = _RecordingLLMService(content="")
        gen = AMemGenerator({"llm_service": svc})

        out = gen.generate_single(prompt="ask", schema=_schema())

        assert out["json"] is None

    def test_json_field_raises_on_invalid_json(self):
        # Wire-enforced schema means the response is always valid JSON; if it
        # isn't, that's a contract violation and we fail loudly instead of
        # silently brace-matching prose.
        svc = _RecordingLLMService(content="not json at all")
        gen = AMemGenerator({"llm_service": svc})

        with pytest.raises(json.JSONDecodeError):
            gen.generate_single(prompt="ask", schema=_schema())


class TestRejectsExtraParams:
    def test_extra_params_still_rejected(self):
        svc = _RecordingLLMService(content="{}")
        gen = AMemGenerator({"llm_service": svc})

        with pytest.raises(ValueError, match="extra_params"):
            gen.generate_single(
                prompt="ask", schema=_schema(), extra_params={"top_p": 0.9}
            )
