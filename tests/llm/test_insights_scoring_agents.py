"""Tests for InsightsAgent.build_prompt/parse_response and ScoringAgent.build_prompt/parse_response.

These tests exercise the prompt-building and response-parsing logic without
calling a real LLM.  The graph is NOT invoked — nodes are called directly.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from langchain_core.messages import HumanMessage, SystemMessage
import pytest

from gigaevo.llm.agents.insights import InsightsAgent, ProgramInsight, ProgramInsights
from gigaevo.llm.agents.scoring import ProgramScore, ScoringAgent
from gigaevo.programs.core_types import ProgramStageResult, StageError, StageState
from gigaevo.programs.metrics.context import MetricsContext, MetricSpec
from gigaevo.programs.metrics.formatter import MetricsFormatter
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_ctx() -> MetricsContext:
    return MetricsContext(
        specs={
            "score": MetricSpec(
                description="primary score",
                is_primary=True,
                higher_is_better=True,
                lower_bound=0.0,
                upper_bound=1.0,
                sentinel_value=-1.0,
            )
        }
    )


def _mock_llm():
    m = MagicMock()
    m.with_structured_output.return_value = m
    return m


def _make_program(
    metrics: dict | None = None,
    code: str = "def solve(): pass",
) -> Program:
    p = Program(code=code, state=ProgramState.RUNNING)
    if metrics:
        p.add_metrics(metrics)
    return p


# ---------------------------------------------------------------------------
# InsightsAgent
# ---------------------------------------------------------------------------


def _make_insights_agent(
    system_template: str = "System prompt",
    user_template: str = "Code: {code}\nMetrics: {metrics}\n{error_section}\nMax: {max_insights}",
    max_insights: int = 5,
) -> InsightsAgent:
    return InsightsAgent(
        llm=_mock_llm(),
        system_prompt_template=system_template,
        user_prompt_template=user_template,
        max_insights=max_insights,
        metrics_formatter=MetricsFormatter(_make_ctx()),
    )


class TestInsightsAgentBuildPrompt:
    def test_two_messages_created(self):
        agent = _make_insights_agent()
        prog = _make_program(metrics={"score": 0.8})
        state = {
            "program": prog,
            "messages": [],
            "llm_response": None,
            "insights": None,
            "metadata": {},
        }
        result = agent.build_prompt(state)
        assert len(result["messages"]) == 2
        assert isinstance(result["messages"][0], SystemMessage)
        assert isinstance(result["messages"][1], HumanMessage)

    def test_system_message_uses_template_verbatim(self):
        agent = _make_insights_agent(system_template="MY_EXACT_SYSTEM")
        prog = _make_program()
        state = {
            "program": prog,
            "messages": [],
            "llm_response": None,
            "insights": None,
            "metadata": {},
        }
        result = agent.build_prompt(state)
        assert result["messages"][0].content == "MY_EXACT_SYSTEM"

    def test_user_message_includes_code(self):
        agent = _make_insights_agent()
        prog = _make_program(code="def unique_fn_42(): return 99")
        state = {
            "program": prog,
            "messages": [],
            "llm_response": None,
            "insights": None,
            "metadata": {},
        }
        result = agent.build_prompt(state)
        assert "unique_fn_42" in result["messages"][1].content

    def test_no_metrics_inserts_placeholder(self):
        agent = _make_insights_agent()
        prog = _make_program(metrics=None)  # no metrics
        state = {
            "program": prog,
            "messages": [],
            "llm_response": None,
            "insights": None,
            "metadata": {},
        }
        result = agent.build_prompt(state)
        assert "No metrics available" in result["messages"][1].content

    def test_error_section_included_when_stage_failed(self):
        agent = _make_insights_agent(user_template="Code: {code}\n{error_section}")
        prog = _make_program()
        prog.stage_results["validate"] = ProgramStageResult(
            status=StageState.FAILED,
            error=StageError(type="TypeError", message="boom", stage="validate"),
        )
        state = {
            "program": prog,
            "messages": [],
            "llm_response": None,
            "insights": None,
            "metadata": {},
        }
        result = agent.build_prompt(state)
        # Stage name and error type appear in the error section
        assert "validate" in result["messages"][1].content
        assert "TypeError" in result["messages"][1].content

    def test_no_real_errors_shows_placeholder(self):
        """With no failed stages, format_errors returns a placeholder (never empty).

        format_errors always returns a non-empty string so error_section is
        always populated — verify the user message is non-trivially long.
        """
        agent = _make_insights_agent()
        prog = _make_program()  # no stage results
        state = {
            "program": prog,
            "messages": [],
            "llm_response": None,
            "insights": None,
            "metadata": {},
        }
        result = agent.build_prompt(state)
        assert len(result["messages"][1].content) > 0

    def test_max_insights_in_user_message(self):
        agent = _make_insights_agent(max_insights=3)
        prog = _make_program()
        state = {
            "program": prog,
            "messages": [],
            "llm_response": None,
            "insights": None,
            "metadata": {},
        }
        result = agent.build_prompt(state)
        assert "3" in result["messages"][1].content


class TestInsightsAgentParseResponse:
    def test_insights_field_populated(self):
        agent = _make_insights_agent()
        insights = ProgramInsights(
            insights=[
                ProgramInsight(
                    type="performance",
                    insight="Use caching",
                    tag="cache",
                    severity="low",
                )
            ]
        )
        state = {
            "program": MagicMock(),
            "messages": [],
            "llm_response": insights,
            "insights": None,
            "metadata": {},
        }
        result = agent.parse_response(state)
        assert result["insights"] is insights

    def test_insights_passed_through_unchanged(self):
        agent = _make_insights_agent()
        insights = ProgramInsights(insights=[])
        state = {
            "program": MagicMock(),
            "messages": [],
            "llm_response": insights,
            "insights": None,
            "metadata": {},
        }
        result = agent.parse_response(state)
        assert result["insights"].insights == []


# ---------------------------------------------------------------------------
# ScoringAgent
# ---------------------------------------------------------------------------


def _make_scoring_agent(
    system_prompt: str = "Score this program",
    user_template: str = "Code: {code}\nTrait: {trait_description}\nMax: {max_score}",
    trait_description: str = "novelty",
    max_score: float = 1.0,
) -> ScoringAgent:
    return ScoringAgent(
        llm=_mock_llm(),
        system_prompt=system_prompt,
        user_prompt_template=user_template,
        trait_description=trait_description,
        max_score=max_score,
    )


class TestScoringAgentBuildPrompt:
    def test_two_messages_created(self):
        agent = _make_scoring_agent()
        prog = _make_program()
        state = {
            "program": prog,
            "trait_description": "novelty",
            "max_score": 1.0,
            "messages": [],
            "llm_response": None,
            "score": 0.0,
            "metadata": {},
        }
        result = agent.build_prompt(state)
        assert len(result["messages"]) == 2
        assert isinstance(result["messages"][0], SystemMessage)
        assert isinstance(result["messages"][1], HumanMessage)

    def test_system_message_verbatim(self):
        agent = _make_scoring_agent(system_prompt="MY_SCORING_SYSTEM")
        prog = _make_program()
        state = {
            "program": prog,
            "trait_description": "t",
            "max_score": 1.0,
            "messages": [],
            "llm_response": None,
            "score": 0.0,
            "metadata": {},
        }
        result = agent.build_prompt(state)
        assert result["messages"][0].content == "MY_SCORING_SYSTEM"

    def test_user_message_includes_code(self):
        agent = _make_scoring_agent()
        prog = _make_program(code="def unique_impl(): pass")
        state = {
            "program": prog,
            "trait_description": "novelty",
            "max_score": 1.0,
            "messages": [],
            "llm_response": None,
            "score": 0.0,
            "metadata": {},
        }
        result = agent.build_prompt(state)
        assert "unique_impl" in result["messages"][1].content

    def test_user_message_includes_trait_and_max_score(self):
        agent = _make_scoring_agent()
        prog = _make_program()
        state = {
            "program": prog,
            "trait_description": "MY_TRAIT",
            "max_score": 7.5,
            "messages": [],
            "llm_response": None,
            "score": 0.0,
            "metadata": {},
        }
        result = agent.build_prompt(state)
        content = result["messages"][1].content
        assert "MY_TRAIT" in content
        assert "7.5" in content


class TestScoringAgentParseResponse:
    def test_score_extracted_from_program_score(self):
        agent = _make_scoring_agent(max_score=1.0)
        state = {
            "program": MagicMock(),
            "trait_description": "t",
            "max_score": 1.0,
            "messages": [],
            "llm_response": ProgramScore(score=0.8),
            "score": 0.0,
            "metadata": {},
        }
        result = agent.parse_response(state)
        assert result["score"] == pytest.approx(0.8)

    def test_score_clipped_to_max_score(self):
        agent = _make_scoring_agent(max_score=1.0)
        state = {
            "program": MagicMock(),
            "trait_description": "t",
            "max_score": 1.0,
            "messages": [],
            "llm_response": ProgramScore(score=5.0),  # exceeds max
            "score": 0.0,
            "metadata": {},
        }
        result = agent.parse_response(state)
        assert result["score"] == pytest.approx(1.0)

    def test_score_below_max_not_clipped(self):
        agent = _make_scoring_agent(max_score=10.0)
        state = {
            "program": MagicMock(),
            "trait_description": "t",
            "max_score": 10.0,
            "messages": [],
            "llm_response": ProgramScore(score=7.3),
            "score": 0.0,
            "metadata": {},
        }
        result = agent.parse_response(state)
        assert result["score"] == pytest.approx(7.3)

    def test_non_program_score_raises(self):
        agent = _make_scoring_agent()
        state = {
            "program": MagicMock(),
            "trait_description": "t",
            "max_score": 1.0,
            "messages": [],
            "llm_response": "not a ProgramScore",
            "score": 0.0,
            "metadata": {},
        }
        with pytest.raises(ValueError, match="Expected ProgramScore"):
            agent.parse_response(state)
