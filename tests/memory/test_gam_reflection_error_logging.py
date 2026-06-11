"""Reflection LLM failures must keep the safe fallback AND log a diagnostic
with the exception type and traceback.

Regression for the live-run error logged as a bare
"Error in experimental reflection: 'NoneType' object is not iterable" —
the message without type/traceback made the root cause undiagnosable.
"""

from __future__ import annotations

from loguru import logger

from gigaevo.memory._vendor.GAM_root.gam.agents.research_agent import ResearchAgent


class _ExplodingGenerator:
    """Reproduces the live failure shape: a TypeError raised deep inside the
    LLM call stack (e.g. langchain iterating a None ``choices``)."""

    def generate_single(self, **kwargs):
        list(None)


def _make_agent() -> ResearchAgent:
    return ResearchAgent(page_store=object(), generator=_ExplodingGenerator())


class TestReflectionExperimentalErrorLogging:
    def test_fallback_decision_preserved(self):
        decision = _make_agent()._reflection_experimental("req", [])
        assert decision.mode == "continue"
        assert decision.top_ideas == []
        assert decision.additional_queries == []

    def test_log_carries_exception_type_and_traceback(self):
        captured: list[str] = []
        sink_id = logger.add(captured.append, level="ERROR")
        try:
            _make_agent()._reflection_experimental("req", [])
        finally:
            logger.remove(sink_id)

        text = "".join(captured)
        assert "TypeError" in text
        assert "'NoneType' object is not iterable" in text
        assert "generate_single" in text
        assert "[GAM][experimental]" in text
