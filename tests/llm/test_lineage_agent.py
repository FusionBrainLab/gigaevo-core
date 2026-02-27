"""Tests for LineageAgent covering diff computation, prompt building, response parsing, and arun."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from gigaevo.llm.agents.lineage import (
    LineageAgent,
    LineageState,
    TransitionAnalysis,
    TransitionInsight,
    TransitionInsights,
)
from gigaevo.programs.metrics.context import MetricsContext, MetricSpec
from gigaevo.programs.metrics.formatter import MetricsFormatter
from gigaevo.programs.program import Program

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_metrics_context(
    *,
    primary_key: str = "fitness",
    primary_description: str = "Main objective",
    higher_is_better: bool = True,
) -> MetricsContext:
    """Create a minimal MetricsContext for testing."""
    return MetricsContext(
        specs={
            primary_key: MetricSpec(
                description=primary_description,
                is_primary=True,
                higher_is_better=higher_is_better,
            ),
        }
    )


def _make_formatter(
    *,
    higher_is_better: bool = True,
) -> MetricsFormatter:
    ctx = _make_metrics_context(higher_is_better=higher_is_better)
    return MetricsFormatter(ctx)


def _make_program(
    code: str = "def solve(): return 42",
    metrics: dict[str, float] | None = None,
) -> Program:
    p = Program(code=code)
    if metrics:
        p.add_metrics(metrics)
    return p


def _mock_llm() -> MagicMock:
    """Create a mock LLM that supports with_structured_output."""
    mock = MagicMock()
    # with_structured_output returns a new mock that will serve as the structured LLM
    structured_mock = MagicMock()
    mock.with_structured_output.return_value = structured_mock
    return mock


def _make_agent(
    *,
    llm: MagicMock | None = None,
    higher_is_better: bool = True,
    system_prompt: str = "You are a lineage analyst.",
    user_prompt_template: str = (
        "Task: {task_description}\n"
        "Metric: {metric_name} ({metric_description})\n"
        "Delta: {delta} ({higher_is_better_text})\n"
        "Interpretation: {delta_interpretation}\n"
        "Parent errors: {parent_errors}\n"
        "Child errors: {child_errors}\n"
        "Additional: {additional_metrics}\n"
        "Diff:\n{diff_blocks}\n"
        "Parent code:\n{parent_code}"
    ),
    task_description: str = "Maximize fitness",
) -> LineageAgent:
    if llm is None:
        llm = _mock_llm()
    formatter = _make_formatter(higher_is_better=higher_is_better)
    return LineageAgent(
        llm=llm,
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
        task_description=task_description,
        metrics_formatter=formatter,
    )


def _sample_insights(n: int = 3) -> TransitionInsights:
    return TransitionInsights(
        insights=[
            TransitionInsight(
                strategy="imitation",
                description=f"Insight {i}",
            )
            for i in range(n)
        ]
    )


# ---------------------------------------------------------------------------
# TestComputeDiffBlocks
# ---------------------------------------------------------------------------


class TestComputeDiffBlocks:
    """Tests for LineageAgent._compute_diff_blocks."""

    def test_identical_code_returns_empty(self):
        agent = _make_agent()
        result = agent._compute_diff_blocks("def f(): return 1", "def f(): return 1")
        assert result == []

    def test_identical_code_with_whitespace_returns_empty(self):
        agent = _make_agent()
        result = agent._compute_diff_blocks(
            "  def f(): return 1  ", "\ndef f(): return 1\n"
        )
        assert result == []

    def test_single_change_produces_one_hunk(self):
        agent = _make_agent()
        parent = "def f():\n    return 1"
        child = "def f():\n    return 2"
        blocks = agent._compute_diff_blocks(parent, child)
        assert len(blocks) == 1
        assert "-    return 1" in blocks[0]
        assert "+    return 2" in blocks[0]

    def test_multiple_hunks_separated(self):
        """Widely separated changes produce multiple diff hunks."""
        agent = _make_agent()
        # Build code with enough spacing that two changes form separate hunks
        # unified diff uses 3 context lines by default, so we need >6 lines between changes
        shared_lines = [f"    x{i} = {i}" for i in range(10)]
        parent_lines = ["def f():"] + shared_lines + ["    return 0"]
        child_lines = ["def g():"] + shared_lines + ["    return 99"]
        parent = "\n".join(parent_lines)
        child = "\n".join(child_lines)
        blocks = agent._compute_diff_blocks(parent, child)
        assert len(blocks) >= 2

    def test_addition_only(self):
        agent = _make_agent()
        parent = "def f():\n    pass"
        child = "def f():\n    x = 1\n    pass"
        blocks = agent._compute_diff_blocks(parent, child)
        assert len(blocks) >= 1
        joined = "\n".join(blocks)
        assert "+    x = 1" in joined

    def test_deletion_only(self):
        agent = _make_agent()
        parent = "def f():\n    x = 1\n    pass"
        child = "def f():\n    pass"
        blocks = agent._compute_diff_blocks(parent, child)
        assert len(blocks) >= 1
        joined = "\n".join(blocks)
        assert "-    x = 1" in joined


# ---------------------------------------------------------------------------
# TestBuildPrompt
# ---------------------------------------------------------------------------


class TestBuildPrompt:
    """Tests for LineageAgent.build_prompt."""

    def _make_state(
        self,
        parent_code: str = "def f(): return 1",
        child_code: str = "def f(): return 2",
        parent_fitness: float = 0.5,
        child_fitness: float = 0.7,
    ) -> LineageState:
        parent = _make_program(code=parent_code, metrics={"fitness": parent_fitness})
        child = _make_program(code=child_code, metrics={"fitness": child_fitness})
        return {
            "parent": parent,
            "child": child,
            "messages": [],
            "llm_response": None,
            "delta": 0.0,
            "diff_blocks": [],
            "insights": [],
            "full_analysis": {},
            "metadata": {},
        }

    def test_messages_contain_system_and_human(self):
        agent = _make_agent(system_prompt="You are a lineage analyst.")
        state = self._make_state()
        result = agent.build_prompt(state)
        messages = result["messages"]
        assert len(messages) == 2
        assert messages[0].type == "system"
        assert messages[0].content == "You are a lineage analyst."
        assert messages[1].type == "human"
        # Key template vars must appear in human message
        assert "Maximize fitness" in messages[1].content
        assert "fitness" in messages[1].content

    def test_delta_computed_correctly(self):
        agent = _make_agent()
        state = self._make_state(parent_fitness=0.3, child_fitness=0.8)
        result = agent.build_prompt(state)
        assert result["delta"] == pytest.approx(0.5)

    def test_no_diff_placeholder(self):
        """When parent and child code are identical, use the placeholder text."""
        agent = _make_agent()
        state = self._make_state(
            parent_code="def f(): return 1",
            child_code="def f(): return 1",
        )
        result = agent.build_prompt(state)
        user_msg = result["messages"][1].content
        assert "(No code differences detected)" in user_msg

    def test_block_labels_and_fences_in_prompt(self):
        """Diff blocks are labeled 'Block 1' with ```diff fences in the prompt."""
        agent = _make_agent()
        state = self._make_state(
            parent_code="def f(): return 1",
            child_code="def f(): return 2",
        )
        result = agent.build_prompt(state)
        user_msg = result["messages"][1].content
        assert "--- Block 1 ---" in user_msg
        assert "```diff" in user_msg

    def test_improvement_interpretation_higher_is_better(self):
        """Positive delta with higher_is_better=True is IMPROVEMENT."""
        agent = _make_agent(higher_is_better=True)
        state = self._make_state(parent_fitness=0.3, child_fitness=0.8)
        result = agent.build_prompt(state)
        user_msg = result["messages"][1].content
        assert "IMPROVEMENT" in user_msg

    def test_regression_interpretation_higher_is_better(self):
        """Negative delta with higher_is_better=True is REGRESSION."""
        agent = _make_agent(higher_is_better=True)
        state = self._make_state(parent_fitness=0.8, child_fitness=0.3)
        result = agent.build_prompt(state)
        user_msg = result["messages"][1].content
        assert "REGRESSION" in user_msg

    def test_improvement_interpretation_lower_is_better(self):
        """Negative delta with higher_is_better=False is IMPROVEMENT."""
        agent = _make_agent(higher_is_better=False)
        state = self._make_state(parent_fitness=0.8, child_fitness=0.3)
        result = agent.build_prompt(state)
        user_msg = result["messages"][1].content
        assert "IMPROVEMENT" in user_msg


# ---------------------------------------------------------------------------
# TestParseResponse
# ---------------------------------------------------------------------------


class TestParseResponse:
    """Tests for LineageAgent.parse_response."""

    def _make_state_with_response(
        self,
        llm_response,
        diff_blocks: list[str] | None = None,
    ) -> LineageState:
        parent = _make_program(metrics={"fitness": 0.5})
        child = _make_program(metrics={"fitness": 0.7})
        return {
            "parent": parent,
            "child": child,
            "messages": [],
            "llm_response": llm_response,
            "delta": 0.2,
            "diff_blocks": diff_blocks or ["@@ some diff"],
            "insights": [],
            "full_analysis": {},
            "metadata": {},
        }

    def test_valid_insights_parsed(self):
        agent = _make_agent()
        insights = _sample_insights(3)
        state = self._make_state_with_response(insights)
        result = agent.parse_response(state)

        assert result["insights"] is insights
        assert isinstance(result["full_analysis"], TransitionAnalysis)
        assert len(result["full_analysis"].insights.insights) == 3

    def test_wrong_type_raises_value_error(self):
        agent = _make_agent()
        # Pass a plain string instead of TransitionInsights
        state = self._make_state_with_response("not a TransitionInsights")
        with pytest.raises(ValueError, match="Expected TransitionInsights"):
            agent.parse_response(state)

    def test_diff_blocks_preserved_in_analysis(self):
        agent = _make_agent()
        insights = _sample_insights(3)
        diff = ["@@ hunk1", "@@ hunk2"]
        state = self._make_state_with_response(insights, diff_blocks=diff)
        result = agent.parse_response(state)
        assert result["full_analysis"].diff_blocks == diff

    def test_parent_child_ids_in_analysis(self):
        agent = _make_agent()
        insights = _sample_insights(3)
        state = self._make_state_with_response(insights)
        result = agent.parse_response(state)
        analysis = result["full_analysis"]
        assert analysis.from_id == state["parent"].id
        assert analysis.to_id == state["child"].id


# ---------------------------------------------------------------------------
# TestArun
# ---------------------------------------------------------------------------


class TestArun:
    """Tests for LineageAgent.arun end-to-end with mocked graph."""

    @pytest.fixture
    def mock_insights(self):
        return _sample_insights(3)

    async def test_single_parent(self, mock_insights):
        agent = _make_agent()
        parent = _make_program(code="def f(): return 1", metrics={"fitness": 0.5})
        child = _make_program(code="def f(): return 2", metrics={"fitness": 0.8})

        # Mock the graph.ainvoke to return a completed state
        fake_analysis = TransitionAnalysis(
            **{
                "from": parent.id,
                "to": child.id,
            },
            parent_metrics=parent.metrics,
            child_metrics=child.metrics,
            diff_blocks=["@@ diff"],
            insights=mock_insights,
        )
        agent.graph = MagicMock()
        agent.graph.ainvoke = AsyncMock(return_value={"full_analysis": fake_analysis})

        results = await agent.arun(parents=[parent], program=child)

        assert len(results) == 1
        assert results[0].from_id == parent.id
        assert results[0].to_id == child.id
        agent.graph.ainvoke.assert_called_once()

    async def test_multiple_parents(self, mock_insights):
        agent = _make_agent()
        parent1 = _make_program(code="def f(): return 1", metrics={"fitness": 0.3})
        parent2 = _make_program(code="def f(): return 10", metrics={"fitness": 0.6})
        child = _make_program(code="def f(): return 99", metrics={"fitness": 0.9})

        def _fake_invoke(state):
            p = state["parent"]
            return {
                "full_analysis": TransitionAnalysis(
                    **{"from": p.id, "to": child.id},
                    parent_metrics=p.metrics,
                    child_metrics=child.metrics,
                    diff_blocks=[],
                    insights=mock_insights,
                )
            }

        agent.graph = MagicMock()
        agent.graph.ainvoke = AsyncMock(side_effect=_fake_invoke)

        results = await agent.arun(parents=[parent1, parent2], program=child)

        assert len(results) == 2
        assert results[0].from_id == parent1.id
        assert results[1].from_id == parent2.id
        assert agent.graph.ainvoke.call_count == 2

    async def test_regression_interpretation(self, mock_insights):
        """Ensure arun works when child fitness is lower (regression scenario)."""
        agent = _make_agent()
        parent = _make_program(code="def f(): return 99", metrics={"fitness": 0.9})
        child = _make_program(code="def f(): return 1", metrics={"fitness": 0.2})

        fake_analysis = TransitionAnalysis(
            **{"from": parent.id, "to": child.id},
            parent_metrics=parent.metrics,
            child_metrics=child.metrics,
            diff_blocks=[],
            insights=mock_insights,
        )
        agent.graph = MagicMock()
        agent.graph.ainvoke = AsyncMock(return_value={"full_analysis": fake_analysis})

        results = await agent.arun(parents=[parent], program=child)
        assert len(results) == 1
        # Verify the child metrics indicate regression
        assert (
            results[0].child_metrics["fitness"] < results[0].parent_metrics["fitness"]
        )


# ---------------------------------------------------------------------------
# TestTransitionAnalysis
# ---------------------------------------------------------------------------


class TestTransitionAnalysis:
    """Tests for TransitionAnalysis Pydantic model."""

    def test_alias_fields(self):
        """TransitionAnalysis uses 'from'/'to' aliases for from_id/to_id."""
        insights = _sample_insights(3)
        analysis = TransitionAnalysis(
            **{"from": "parent-id", "to": "child-id"},
            parent_metrics={"fitness": 0.5},
            child_metrics={"fitness": 0.8},
            diff_blocks=["@@ diff"],
            insights=insights,
        )
        assert analysis.from_id == "parent-id"
        assert analysis.to_id == "child-id"

    def test_populate_by_name(self):
        """TransitionAnalysis supports both alias and field name due to populate_by_name."""
        insights = _sample_insights(3)
        analysis = TransitionAnalysis(
            from_id="parent-id",
            to_id="child-id",
            parent_metrics={"fitness": 0.5},
            child_metrics={"fitness": 0.8},
            diff_blocks=[],
            insights=insights,
        )
        assert analysis.from_id == "parent-id"
        assert analysis.to_id == "child-id"

    def test_insights_validation(self):
        """TransitionInsights enforces min_length=3 and max_length=5."""
        from pydantic import ValidationError

        # Too few insights
        with pytest.raises(ValidationError):
            TransitionInsights(
                insights=[
                    TransitionInsight(strategy="imitation", description="One"),
                    TransitionInsight(strategy="avoidance", description="Two"),
                ]
            )

        # Exactly 3 is fine
        result = TransitionInsights(
            insights=[
                TransitionInsight(strategy="imitation", description=f"Insight {i}")
                for i in range(3)
            ]
        )
        assert len(result.insights) == 3

        # Exactly 5 is fine
        result = TransitionInsights(
            insights=[
                TransitionInsight(strategy="imitation", description=f"Insight {i}")
                for i in range(5)
            ]
        )
        assert len(result.insights) == 5

        # Too many insights
        with pytest.raises(ValidationError):
            TransitionInsights(
                insights=[
                    TransitionInsight(strategy="imitation", description=f"Insight {i}")
                    for i in range(6)
                ]
            )


# ---------------------------------------------------------------------------
# Additional tests from audit
# ---------------------------------------------------------------------------


class TestBuildPromptEdgeCases:
    """Missing interpretation cases and prompt content verification."""

    def _make_state(
        self,
        parent_fitness: float = 0.5,
        child_fitness: float = 0.7,
        parent_code: str = "def f(): return 1",
        child_code: str = "def f(): return 2",
    ) -> LineageState:
        parent = _make_program(code=parent_code, metrics={"fitness": parent_fitness})
        child = _make_program(code=child_code, metrics={"fitness": child_fitness})
        return {
            "parent": parent,
            "child": child,
            "messages": [],
            "llm_response": None,
            "delta": 0.0,
            "diff_blocks": [],
            "insights": [],
            "full_analysis": {},
            "metadata": {},
        }

    def test_regression_lower_is_better_positive_delta(self):
        """Positive delta with higher_is_better=False is REGRESSION."""
        agent = _make_agent(higher_is_better=False)
        state = self._make_state(parent_fitness=0.3, child_fitness=0.8)
        result = agent.build_prompt(state)
        user_msg = result["messages"][1].content
        assert "REGRESSION" in user_msg

    def test_zero_delta_is_regression(self):
        """Zero delta (no change) should be classified as REGRESSION."""
        agent = _make_agent(higher_is_better=True)
        state = self._make_state(parent_fitness=0.5, child_fitness=0.5)
        result = agent.build_prompt(state)
        user_msg = result["messages"][1].content
        assert "REGRESSION" in user_msg
        assert result["delta"] == pytest.approx(0.0)

    def test_higher_is_better_direction_text(self):
        agent = _make_agent(higher_is_better=True)
        state = self._make_state()
        result = agent.build_prompt(state)
        assert "higher is better" in result["messages"][1].content

    def test_lower_is_better_direction_text(self):
        agent = _make_agent(higher_is_better=False)
        state = self._make_state()
        result = agent.build_prompt(state)
        assert "lower is better" in result["messages"][1].content


class TestParseResponseMetrics:
    """Verify parent/child metrics are correctly assigned in TransitionAnalysis."""

    def test_metrics_not_swapped_in_analysis(self):
        agent = _make_agent()
        insights = _sample_insights(3)
        parent = _make_program(metrics={"fitness": 0.3})
        child = _make_program(metrics={"fitness": 0.9})
        state: LineageState = {
            "parent": parent,
            "child": child,
            "messages": [],
            "llm_response": insights,
            "delta": 0.6,
            "diff_blocks": [],
            "insights": [],
            "full_analysis": {},
            "metadata": {},
        }
        result = agent.parse_response(state)
        assert result["full_analysis"].parent_metrics["fitness"] == pytest.approx(0.3)
        assert result["full_analysis"].child_metrics["fitness"] == pytest.approx(0.9)


class TestArunEdgeCases:
    """Edge cases for arun: empty parents, initial state verification."""

    async def test_empty_parents_returns_empty(self):
        agent = _make_agent()
        child = _make_program(metrics={"fitness": 0.5})
        agent.graph = MagicMock()
        agent.graph.ainvoke = AsyncMock()
        results = await agent.arun(parents=[], program=child)
        assert results == []
        agent.graph.ainvoke.assert_not_called()

    async def test_initial_state_passed_to_graph(self):
        """Verify the initial state dict passed to ainvoke has correct keys."""
        agent = _make_agent()
        parent = _make_program(metrics={"fitness": 0.5})
        child = _make_program(metrics={"fitness": 0.8})
        captured_state = {}

        async def _capture(state):
            captured_state.update(state)
            return {
                "full_analysis": TransitionAnalysis(
                    **{"from": parent.id, "to": child.id},
                    parent_metrics=parent.metrics,
                    child_metrics=child.metrics,
                    diff_blocks=[],
                    insights=_sample_insights(3),
                )
            }

        agent.graph = MagicMock()
        agent.graph.ainvoke = AsyncMock(side_effect=_capture)
        await agent.arun(parents=[parent], program=child)

        assert captured_state["parent"] is parent
        assert captured_state["child"] is child
        assert captured_state["metadata"]["parent_id"] == parent.id
        assert captured_state["metadata"]["child_id"] == child.id
