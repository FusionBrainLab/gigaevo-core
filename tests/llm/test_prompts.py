"""Tests for gigaevo/prompts/__init__.py — load_prompt and Prompts accessor classes."""

from __future__ import annotations

from pathlib import Path

import pytest

from gigaevo.prompts import (
    InsightsPrompts,
    LineagePrompts,
    MutationPrompts,
    ScoringPrompts,
    load_prompt,
)

# ---------------------------------------------------------------------------
# load_prompt — core function
# ---------------------------------------------------------------------------


class TestLoadPrompt:
    def test_loads_default_mutation_system(self):
        template = load_prompt("mutation", "system")
        assert isinstance(template, str) and len(template) > 0

    def test_loads_default_mutation_user(self):
        template = load_prompt("mutation", "user")
        assert isinstance(template, str) and len(template) > 0

    def test_loads_default_insights_system(self):
        template = load_prompt("insights", "system")
        assert isinstance(template, str) and len(template) > 0

    def test_loads_default_insights_user(self):
        template = load_prompt("insights", "user")
        assert isinstance(template, str) and len(template) > 0

    def test_loads_default_lineage_system(self):
        template = load_prompt("lineage", "system")
        assert isinstance(template, str) and len(template) > 0

    def test_loads_default_lineage_user(self):
        template = load_prompt("lineage", "user")
        assert isinstance(template, str) and len(template) > 0

    def test_loads_default_scoring_system(self):
        template = load_prompt("scoring", "system")
        assert isinstance(template, str) and len(template) > 0

    def test_loads_default_scoring_user(self):
        template = load_prompt("scoring", "user")
        assert isinstance(template, str) and len(template) > 0

    def test_result_is_stripped(self):
        """load_prompt strips leading/trailing whitespace from file content."""
        template = load_prompt("mutation", "system")
        assert template == template.strip()

    def test_missing_prompt_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="Prompt not found"):
            load_prompt("nonexistent_agent", "system")

    def test_missing_prompt_type_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_prompt("mutation", "nonexistent_type")

    # custom prompts_dir —————————————————————————————————————————————————

    def test_custom_dir_overrides_default(self, tmp_path: Path):
        custom_dir = tmp_path / "mutation"
        custom_dir.mkdir()
        (custom_dir / "system.txt").write_text("custom system prompt")
        result = load_prompt("mutation", "system", prompts_dir=tmp_path)
        assert result == "custom system prompt"

    def test_custom_dir_strips_whitespace(self, tmp_path: Path):
        custom_dir = tmp_path / "mutation"
        custom_dir.mkdir()
        (custom_dir / "system.txt").write_text("  trimmed  \n")
        result = load_prompt("mutation", "system", prompts_dir=tmp_path)
        assert result == "trimmed"

    def test_custom_dir_falls_back_to_default_if_file_missing(self, tmp_path: Path):
        """Custom dir given but file missing → falls back to package default."""
        result = load_prompt("mutation", "system", prompts_dir=tmp_path)
        default = load_prompt("mutation", "system")
        assert result == default

    def test_custom_dir_does_not_affect_other_agents(self, tmp_path: Path):
        """Custom dir with a file for mutation doesn't affect insights."""
        (tmp_path / "mutation").mkdir()
        (tmp_path / "mutation" / "system.txt").write_text("custom")
        insights = load_prompt("insights", "system", prompts_dir=tmp_path)
        default_insights = load_prompt("insights", "system")
        assert insights == default_insights

    def test_prompts_dir_as_string(self, tmp_path: Path):
        """prompts_dir can be a plain str (not just Path)."""
        custom_dir = tmp_path / "mutation"
        custom_dir.mkdir()
        (custom_dir / "system.txt").write_text("str path prompt")
        result = load_prompt("mutation", "system", prompts_dir=str(tmp_path))
        assert result == "str path prompt"


# ---------------------------------------------------------------------------
# MutationPrompts accessor class
# ---------------------------------------------------------------------------


class TestMutationPrompts:
    def test_system_returns_non_empty_string(self):
        s = MutationPrompts.system()
        assert isinstance(s, str) and len(s) > 0

    def test_user_returns_non_empty_string(self):
        u = MutationPrompts.user()
        assert isinstance(u, str) and len(u) > 0

    def test_system_custom_dir(self, tmp_path: Path):
        (tmp_path / "mutation").mkdir()
        (tmp_path / "mutation" / "system.txt").write_text("override")
        assert MutationPrompts.system(prompts_dir=tmp_path) == "override"

    def test_user_custom_dir(self, tmp_path: Path):
        (tmp_path / "mutation").mkdir()
        (tmp_path / "mutation" / "user.txt").write_text("user override")
        assert MutationPrompts.user(prompts_dir=tmp_path) == "user override"


# ---------------------------------------------------------------------------
# InsightsPrompts
# ---------------------------------------------------------------------------


class TestInsightsPrompts:
    def test_system_returns_non_empty_string(self):
        assert isinstance(InsightsPrompts.system(), str)

    def test_user_returns_non_empty_string(self):
        assert isinstance(InsightsPrompts.user(), str)

    def test_custom_dir_fallback(self, tmp_path: Path):
        """No custom file → falls back to package default."""
        default = InsightsPrompts.system()
        result = InsightsPrompts.system(prompts_dir=tmp_path)
        assert result == default

    def test_custom_dir_override(self, tmp_path: Path):
        (tmp_path / "insights").mkdir()
        (tmp_path / "insights" / "system.txt").write_text("insights override")
        assert InsightsPrompts.system(prompts_dir=tmp_path) == "insights override"


# ---------------------------------------------------------------------------
# LineagePrompts
# ---------------------------------------------------------------------------


class TestLineagePrompts:
    def test_system_returns_non_empty_string(self):
        assert isinstance(LineagePrompts.system(), str)

    def test_user_returns_non_empty_string(self):
        assert isinstance(LineagePrompts.user(), str)

    def test_custom_dir_override(self, tmp_path: Path):
        (tmp_path / "lineage").mkdir()
        (tmp_path / "lineage" / "user.txt").write_text("lineage user override")
        assert LineagePrompts.user(prompts_dir=tmp_path) == "lineage user override"


# ---------------------------------------------------------------------------
# ScoringPrompts
# ---------------------------------------------------------------------------


class TestScoringPrompts:
    def test_system_returns_non_empty_string(self):
        assert isinstance(ScoringPrompts.system(), str)

    def test_user_returns_non_empty_string(self):
        assert isinstance(ScoringPrompts.user(), str)

    def test_custom_dir_override(self, tmp_path: Path):
        (tmp_path / "scoring").mkdir()
        (tmp_path / "scoring" / "system.txt").write_text("scoring override")
        assert ScoringPrompts.system(prompts_dir=tmp_path) == "scoring override"
