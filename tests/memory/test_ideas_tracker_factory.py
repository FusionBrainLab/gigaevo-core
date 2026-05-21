"""Tests for the analyzer factory that translates Hydra kwargs into
ClassifyingAnalyzer instances."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from gigaevo.memory.ideas_tracker.analyzers import ClassifyingAnalyzer


@pytest.fixture(autouse=True)
def _stub_llm_clients(monkeypatch):
    """Stub the OpenAI client builder so factory tests don't need a real key.

    These tests verify the factory wires Hydra kwargs into the right analyzer
    type with the right fields. LLM construction is incidental — mock it.
    """
    import gigaevo.memory.ideas_tracker.llm as _llm_mod

    def _fake_init_clients(base_url):
        return MagicMock(), MagicMock(), False

    monkeypatch.setattr(_llm_mod, "_init_clients", _fake_init_clients)


def _factory():
    """Re-imported inside tests so RED phase surfaces a clean ImportError."""
    from gigaevo.memory.ideas_tracker.ideas_tracker import (
        _build_analyzer_from_hydra_fields,
    )

    return _build_analyzer_from_hydra_fields


class TestBuildAnalyzerDefault:
    def test_default_type_returns_classifying(self):
        build = _factory()
        analyzer = build(
            analyzer_type="default",
            analyzer_model="google/gemini-3-flash-preview",
            analyzer_base_url="https://openrouter.ai/api/v1",
            analyzer_reasoning={"effort": "minimal"},
            analyzer_fast_settings=None,
            description_rewriting=True,
        )
        assert isinstance(analyzer, ClassifyingAnalyzer)
        assert analyzer.model == "google/gemini-3-flash-preview"

    def test_empty_base_url_becomes_none(self):
        build = _factory()
        analyzer = build(
            analyzer_type="default",
            analyzer_model="google/gemini-3-flash-preview",
            analyzer_base_url="   ",
            analyzer_reasoning=None,
            analyzer_fast_settings=None,
            description_rewriting=True,
        )
        assert isinstance(analyzer, ClassifyingAnalyzer)

    def test_reasoning_passed_through(self):
        build = _factory()
        analyzer = build(
            analyzer_type="default",
            analyzer_model="m",
            analyzer_base_url="",
            analyzer_reasoning={"effort": "high", "extra": "x"},
            analyzer_fast_settings=None,
            description_rewriting=True,
        )
        assert analyzer._reasoning == {"effort": "high", "extra": "x"}

    def test_description_rewriting_flag_propagates(self):
        build = _factory()
        analyzer_on = build(
            analyzer_type="default",
            analyzer_model="m",
            analyzer_base_url="",
            analyzer_reasoning=None,
            analyzer_fast_settings=None,
            description_rewriting=True,
        )
        analyzer_off = build(
            analyzer_type="default",
            analyzer_model="m",
            analyzer_base_url="",
            analyzer_reasoning=None,
            analyzer_fast_settings=None,
            description_rewriting=False,
        )
        assert analyzer_on._description_rewriting is True
        assert analyzer_off._description_rewriting is False


class TestBuildAnalyzerNormalization:
    @pytest.mark.parametrize(
        "kind,expected",
        [
            ("default", ClassifyingAnalyzer),
            ("DEFAULT", ClassifyingAnalyzer),
            ("Default", ClassifyingAnalyzer),
            (" default ", ClassifyingAnalyzer),
            ("", ClassifyingAnalyzer),
            (None, ClassifyingAnalyzer),
        ],
    )
    def test_case_and_whitespace_normalization(self, kind, expected):
        build = _factory()
        analyzer = build(
            analyzer_type=kind,  # type: ignore[arg-type]
            analyzer_model="m",
            analyzer_base_url="",
            analyzer_reasoning=None,
            analyzer_fast_settings=None,
            description_rewriting=True,
        )
        assert isinstance(analyzer, expected)

    def test_unknown_type_falls_back_to_default(self):
        """The current factory falls through to ClassifyingAnalyzer on any
        non-fast value rather than raising. Pin that behavior."""
        build = _factory()
        analyzer = build(
            analyzer_type="wizardry",
            analyzer_model="m",
            analyzer_base_url="",
            analyzer_reasoning=None,
            analyzer_fast_settings=None,
            description_rewriting=True,
        )
        assert isinstance(analyzer, ClassifyingAnalyzer)
