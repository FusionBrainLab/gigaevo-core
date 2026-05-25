"""Tests for stale-only enrichment in run_increment."""

from __future__ import annotations

import uuid

import pytest

from gigaevo.memory.ideas_tracker.ideas_tracker import (
    IdeaTracker,
    _select_ideas_needing_enrichment,
)
from gigaevo.memory.ideas_tracker.models import (
    AnalysisResult,
    Idea,
    IdeaExplanation,
    IdeaUpdate,
)
from gigaevo.programs.metrics.context import VALIDITY_KEY
from gigaevo.programs.program import Program


def _idea(*, id_: str, entries: list[str], keywords: list[str] | None = None) -> Idea:
    return Idea(
        id=id_,
        description=f"test idea {id_}",
        keywords=keywords or [],
        explanation=IdeaExplanation(entries=entries),
    )


class _StubAnalyzer:
    model = "stub-model"

    def __init__(self, scripted: list[AnalysisResult]) -> None:
        self._scripted = list(scripted)

    def analyze(self, records, bank):  # type: ignore[no-untyped-def]
        return self._scripted.pop(0) if self._scripted else AnalysisResult()

    async def analyze_async(self, records, bank):  # type: ignore[no-untyped-def]
        return self._scripted.pop(0) if self._scripted else AnalysisResult()

    def call(self, step: str, content: str | dict[str, str] = "") -> str:
        return '{"summary": "stub task summary"}'

    async def call_async(self, step: str, content: str | dict[str, str] = "") -> str:
        return '{"summary": "stub task summary"}'


def _child_program(idx: int) -> Program:
    pid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"enrich-child-{idx}"))
    parent_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, "enrich-parent"))
    prog = Program(id=pid, code=f"# child {idx}")
    prog.lineage.parents = [parent_id]
    prog.metrics[VALIDITY_KEY] = 1.0
    prog.metrics["fitness"] = 0.5
    return prog


def _build_tracker(analyzer: _StubAnalyzer, logs_dir) -> IdeaTracker:
    return IdeaTracker(
        analyzer=analyzer,
        memory_write_enabled=False,
        memory_usage_tracking_enabled=False,
        task_description="t",
        logs_dir=logs_dir,
    )


def _install_enrichment_spy(monkeypatch) -> list[list[str]]:
    captured: list[list[str]] = []

    async def spy(ideas, *_a, **_k):  # type: ignore[no-untyped-def]
        captured.append([i.id for i in ideas])
        return list(ideas)

    monkeypatch.setattr(
        "gigaevo.memory.ideas_tracker.ideas_tracker."
        "_enrich_ideas_with_keywords_and_summaries",
        spy,
    )
    return captured


@pytest.mark.asyncio
async def test_run_increment_enriches_all_ideas_on_first_call(monkeypatch, tmp_path):
    idea_a = _idea(id_="a", entries=["seed-a"])
    idea_b = _idea(id_="b", entries=["seed-b"])
    analyzer = _StubAnalyzer([AnalysisResult(new_ideas=[idea_a, idea_b])])
    tracker = _build_tracker(analyzer, tmp_path)
    captured = _install_enrichment_spy(monkeypatch)

    await tracker.run_increment([_child_program(1)])

    assert len(captured) == 1
    assert sorted(captured[0]) == ["a", "b"]


@pytest.mark.asyncio
async def test_run_increment_only_re_enriches_grown_ideas(monkeypatch, tmp_path):
    idea_a = _idea(id_="a", entries=["seed-a"])
    idea_b = _idea(id_="b", entries=["seed-b"])
    res1 = AnalysisResult(new_ideas=[idea_a, idea_b])
    res2 = AnalysisResult(updates=[IdeaUpdate(idea_id="a", motivation="grew-a")])
    analyzer = _StubAnalyzer([res1, res2])
    tracker = _build_tracker(analyzer, tmp_path)
    captured = _install_enrichment_spy(monkeypatch)

    await tracker.run_increment([_child_program(1)])
    await tracker.run_increment([_child_program(2)])

    assert len(captured) == 2
    assert sorted(captured[0]) == ["a", "b"]
    assert captured[1] == ["a"]


@pytest.mark.asyncio
async def test_run_increment_skips_enrichment_when_nothing_changed(
    monkeypatch, tmp_path
):
    idea_a = _idea(id_="a", entries=["seed-a"])
    res1 = AnalysisResult(new_ideas=[idea_a])
    res2 = AnalysisResult()
    analyzer = _StubAnalyzer([res1, res2])
    tracker = _build_tracker(analyzer, tmp_path)
    captured = _install_enrichment_spy(monkeypatch)

    await tracker.run_increment([_child_program(1)])
    await tracker.run_increment([_child_program(2)])

    assert len(captured) == 1
    assert captured[0] == ["a"]


class TestSelectIdeasNeedingEnrichment:
    def test_unseen_idea_is_selected(self):
        """Idea not in last_entry_count → must be enriched (first sighting)."""
        idea = _idea(id_="a", entries=[])
        selected = _select_ideas_needing_enrichment([idea], last_entry_count={})
        assert [i.id for i in selected] == ["a"]

    def test_idea_with_unchanged_entries_is_skipped(self):
        """Idea already enriched with same entry count → skipped."""
        idea = _idea(id_="a", entries=["e1"], keywords=["k"])
        selected = _select_ideas_needing_enrichment([idea], last_entry_count={"a": 1})
        assert selected == []

    def test_idea_with_grown_entries_is_re_selected(self):
        """Entry list grew since last enrichment → must re-enrich."""
        idea = _idea(id_="a", entries=["e1", "e2"], keywords=["k"])
        selected = _select_ideas_needing_enrichment([idea], last_entry_count={"a": 1})
        assert [i.id for i in selected] == ["a"]

    def test_idea_with_shrunk_entries_is_re_selected(self):
        """Defensive: if entry count decreased (rebuild?), re-enrich."""
        idea = _idea(id_="a", entries=[], keywords=["k"])
        selected = _select_ideas_needing_enrichment([idea], last_entry_count={"a": 2})
        assert [i.id for i in selected] == ["a"]

    def test_mixed_bank_filters_correctly(self):
        """Bank with stale, fresh, and unseen ideas → only stale+unseen returned."""
        fresh = _idea(id_="fresh", entries=["e"], keywords=["k"])
        stale = _idea(id_="stale", entries=["e1", "e2"], keywords=["old_k"])
        unseen = _idea(id_="unseen", entries=[])
        last = {"fresh": 1, "stale": 1}

        selected = _select_ideas_needing_enrichment(
            [fresh, stale, unseen], last_entry_count=last
        )

        ids = {i.id for i in selected}
        assert ids == {"stale", "unseen"}

    def test_empty_bank_returns_empty(self):
        assert _select_ideas_needing_enrichment([], last_entry_count={}) == []

    def test_predicate_does_not_mutate_inputs(self):
        """Pure function: must not write into last_entry_count or ideas."""
        idea = _idea(id_="a", entries=["e"])
        last = {"a": 1}
        snapshot = dict(last)
        _select_ideas_needing_enrichment([idea], last_entry_count=last)
        assert last == snapshot
