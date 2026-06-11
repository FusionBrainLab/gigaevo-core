"""Tests for the analyzer factory that translates Hydra kwargs into
analyzer instances wired to the memory LLM router."""

from __future__ import annotations

import pytest

from gigaevo.memory.ideas_tracker.analyzers import ClassifyingAnalyzer
from tests.fakes.llm_router import FakeMemoryRouter


def _factory():
    from gigaevo.memory.ideas_tracker.ideas_tracker import (
        _build_analyzer_from_hydra_fields,
    )

    return _build_analyzer_from_hydra_fields


class TestBuildAnalyzerDefault:
    def test_default_type_returns_classifying(self):
        build = _factory()
        llm = FakeMemoryRouter()
        analyzer = build(
            analyzer_type="default",
            llm=llm,
            analyzer_fast_settings=None,
            description_rewriting=True,
        )
        assert isinstance(analyzer, ClassifyingAnalyzer)
        assert analyzer.llm is llm

    def test_description_rewriting_flag_propagates(self):
        build = _factory()
        analyzer_on = build(
            analyzer_type="default",
            llm=FakeMemoryRouter(),
            analyzer_fast_settings=None,
            description_rewriting=True,
        )
        analyzer_off = build(
            analyzer_type="default",
            llm=FakeMemoryRouter(),
            analyzer_fast_settings=None,
            description_rewriting=False,
        )
        assert analyzer_on._description_rewriting is True
        assert analyzer_off._description_rewriting is False

    def test_max_concurrent_classifications_propagates(self):
        build = _factory()
        analyzer = build(
            analyzer_type="default",
            llm=FakeMemoryRouter(),
            analyzer_fast_settings=None,
            description_rewriting=True,
            analyzer_max_concurrent_classifications=3,
        )
        assert analyzer._max_concurrent_classifications == 3


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
            llm=FakeMemoryRouter(),
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
            llm=FakeMemoryRouter(),
            analyzer_fast_settings=None,
            description_rewriting=True,
        )
        assert isinstance(analyzer, ClassifyingAnalyzer)
