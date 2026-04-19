"""Tests for LLM_CALL canonical-event emission.

Every LLM invocation routes through `LangGraphAgent.acall_llm()` — the
single emission seam for LLM_CALL. Success or failure, exactly one
`[LLM_CALL] {json}` line must land per invocation, with:
- ok: bool
- latency_ms: float >= 0
- error_type: populated on failure, None on success
"""

from __future__ import annotations

import json
import re
from unittest.mock import AsyncMock, MagicMock

from loguru import logger
import pytest

from gigaevo.llm.agents.base import LangGraphAgent


class _TestAgent(LangGraphAgent):
    """Minimal concrete agent for exercising acall_llm only."""

    # Duck-typed; acall_llm only reads state["messages"] and writes llm_response.
    StateSchema = dict

    def __init__(self, llm):
        # Skip graph construction (LangGraph needs a real TypedDict) —
        # we only test acall_llm.
        self.llm = llm

    def build_prompt(self, state):  # pragma: no cover — unused
        return state

    def parse_response(self, state):  # pragma: no cover — unused
        return state

    async def arun(self, *args, **kwargs):  # pragma: no cover — unused
        return None


@pytest.fixture
def log_sink():
    captured: list[str] = []

    def sink(message):
        captured.append(str(message))

    sink_id = logger.add(sink, level="DEBUG", format="{message}")
    yield captured
    logger.remove(sink_id)


def _llm_call_lines(captured):
    return [m for m in captured if "[LLM_CALL]" in m]


def _body(line: str) -> dict:
    m = re.search(r"\{.*\}\s*$", line)
    assert m, f"no JSON body in {line!r}"
    return json.loads(m.group(0))


class TestLLMCallEmits:
    async def test_success_emits_llm_call_ok(self, log_sink):
        llm = MagicMock()
        llm.ainvoke = AsyncMock(return_value=MagicMock(content="ok"))
        llm.model_name = "fake-model"
        agent = _TestAgent(llm)

        state = {"messages": ["hi"]}
        await agent.acall_llm(state)

        lines = _llm_call_lines(log_sink)
        assert len(lines) == 1, f"expected 1 LLM_CALL line, got {lines}"
        body = _body(lines[0])
        assert body["event"] == "LLM_CALL"
        assert body["ok"] is True
        assert body["latency_ms"] >= 0.0
        assert body["error_type"] is None

    async def test_failure_emits_llm_call_with_error_type(self, log_sink):
        llm = MagicMock()
        llm.ainvoke = AsyncMock(side_effect=RuntimeError("nope"))
        llm.model_name = "fake-model"
        agent = _TestAgent(llm)

        state = {"messages": ["hi"]}
        with pytest.raises(RuntimeError, match="nope"):
            await agent.acall_llm(state)

        lines = _llm_call_lines(log_sink)
        assert len(lines) == 1, f"expected 1 LLM_CALL line, got {lines}"
        body = _body(lines[0])
        assert body["event"] == "LLM_CALL"
        assert body["ok"] is False
        assert body["error_type"] == "RuntimeError"
