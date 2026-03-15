"""Tests for agent factory functions (llm/agents/factories.py).

Each factory is tested for:
- Correct agent type returned
- Key attributes (system prompt, mode, max_insights) passed through
- Custom prompts_dir support
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from gigaevo.llm.agents.factories import (
    create_insights_agent,
    create_lineage_agent,
    create_mutation_agent,
    create_scoring_agent,
)
from gigaevo.llm.agents.insights import InsightsAgent
from gigaevo.llm.agents.lineage import LineageAgent
from gigaevo.llm.agents.mutation import MutationAgent
from gigaevo.llm.agents.scoring import ScoringAgent
from gigaevo.programs.metrics.context import MetricsContext, MetricSpec

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


# ---------------------------------------------------------------------------
# create_mutation_agent
# ---------------------------------------------------------------------------


class TestCreateMutationAgent:
    def test_returns_mutation_agent(self):
        agent = create_mutation_agent(
            llm=_mock_llm(),
            task_description="maximize score",
            metrics_context=_make_ctx(),
        )
        assert isinstance(agent, MutationAgent)

    def test_system_prompt_contains_task_description(self):
        agent = create_mutation_agent(
            llm=_mock_llm(),
            task_description="MY_UNIQUE_TASK_XYZ",
            metrics_context=_make_ctx(),
        )
        assert "MY_UNIQUE_TASK_XYZ" in agent.system_prompt

    def test_default_mutation_mode_is_rewrite(self):
        agent = create_mutation_agent(
            llm=_mock_llm(),
            task_description="task",
            metrics_context=_make_ctx(),
        )
        assert agent.mutation_mode == "rewrite"

    def test_custom_mutation_mode_passed_through(self):
        agent = create_mutation_agent(
            llm=_mock_llm(),
            task_description="task",
            metrics_context=_make_ctx(),
            mutation_mode="diff",
        )
        assert agent.mutation_mode == "diff"

    def test_user_prompt_template_set(self):
        agent = create_mutation_agent(
            llm=_mock_llm(),
            task_description="task",
            metrics_context=_make_ctx(),
        )
        assert isinstance(agent.user_prompt_template, str)
        assert len(agent.user_prompt_template) > 0

    def test_custom_prompts_dir(self, tmp_path: Path):
        custom_dir = tmp_path / "mutation"
        custom_dir.mkdir()
        (custom_dir / "system.txt").write_text(
            "Custom system: {task_description}\n{metrics_description}"
        )
        (custom_dir / "user.txt").write_text("User: {code}")
        agent = create_mutation_agent(
            llm=_mock_llm(),
            task_description="CUSTOM",
            metrics_context=_make_ctx(),
            prompts_dir=tmp_path,
        )
        assert "CUSTOM" in agent.system_prompt


# ---------------------------------------------------------------------------
# create_insights_agent
# ---------------------------------------------------------------------------


class TestCreateInsightsAgent:
    def test_returns_insights_agent(self):
        agent = create_insights_agent(
            llm=_mock_llm(),
            task_description="maximize score",
            metrics_context=_make_ctx(),
        )
        assert isinstance(agent, InsightsAgent)

    def test_default_max_insights_is_7(self):
        agent = create_insights_agent(
            llm=_mock_llm(),
            task_description="task",
            metrics_context=_make_ctx(),
        )
        assert agent.max_insights == 7

    def test_custom_max_insights_passed_through(self):
        agent = create_insights_agent(
            llm=_mock_llm(),
            task_description="task",
            metrics_context=_make_ctx(),
            max_insights=3,
        )
        assert agent.max_insights == 3

    def test_user_prompt_template_set(self):
        agent = create_insights_agent(
            llm=_mock_llm(),
            task_description="task",
            metrics_context=_make_ctx(),
        )
        assert isinstance(agent.user_prompt_template, str)

    def test_custom_prompts_dir(self, tmp_path: Path):
        custom_dir = tmp_path / "insights"
        custom_dir.mkdir()
        (custom_dir / "system.txt").write_text(
            "Custom insights: {task_description}\n{max_insights}\n{metrics_description}"
        )
        (custom_dir / "user.txt").write_text(
            "Code: {code}\n{metrics}\n{error_section}\n{max_insights}"
        )
        agent = create_insights_agent(
            llm=_mock_llm(),
            task_description="MY_TASK",
            metrics_context=_make_ctx(),
            prompts_dir=tmp_path,
        )
        assert isinstance(agent, InsightsAgent)


# ---------------------------------------------------------------------------
# create_lineage_agent
# ---------------------------------------------------------------------------


class TestCreateLineageAgent:
    def test_returns_lineage_agent(self):
        agent = create_lineage_agent(
            llm=_mock_llm(),
            task_description="maximize score",
            metrics_context=_make_ctx(),
        )
        assert isinstance(agent, LineageAgent)

    def test_task_description_passed_through(self):
        agent = create_lineage_agent(
            llm=_mock_llm(),
            task_description="MY_LINEAGE_TASK",
            metrics_context=_make_ctx(),
        )
        assert agent.task_description == "MY_LINEAGE_TASK"

    def test_user_prompt_template_set(self):
        agent = create_lineage_agent(
            llm=_mock_llm(),
            task_description="task",
            metrics_context=_make_ctx(),
        )
        assert isinstance(agent.user_prompt_template, str)

    def test_custom_prompts_dir(self, tmp_path: Path):
        custom_dir = tmp_path / "lineage"
        custom_dir.mkdir()
        (custom_dir / "system.txt").write_text("lineage system")
        (custom_dir / "user.txt").write_text("lineage user")
        agent = create_lineage_agent(
            llm=_mock_llm(),
            task_description="task",
            metrics_context=_make_ctx(),
            prompts_dir=tmp_path,
        )
        assert isinstance(agent, LineageAgent)


# ---------------------------------------------------------------------------
# create_scoring_agent
# ---------------------------------------------------------------------------


class TestCreateScoringAgent:
    def test_returns_scoring_agent(self):
        agent = create_scoring_agent(
            llm=_mock_llm(),
            trait_description="code novelty",
            max_score=1.0,
        )
        assert isinstance(agent, ScoringAgent)

    def test_trait_description_passed_through(self):
        agent = create_scoring_agent(
            llm=_mock_llm(),
            trait_description="MY_TRAIT",
            max_score=5.0,
        )
        assert agent.trait_description == "MY_TRAIT"

    def test_max_score_passed_through(self):
        agent = create_scoring_agent(
            llm=_mock_llm(),
            trait_description="trait",
            max_score=10.0,
        )
        assert agent.max_score == pytest.approx(10.0)

    def test_custom_prompts_dir(self, tmp_path: Path):
        custom_dir = tmp_path / "scoring"
        custom_dir.mkdir()
        (custom_dir / "system.txt").write_text("scoring system")
        (custom_dir / "user.txt").write_text("scoring user")
        agent = create_scoring_agent(
            llm=_mock_llm(),
            trait_description="trait",
            max_score=1.0,
            prompts_dir=tmp_path,
        )
        assert isinstance(agent, ScoringAgent)
