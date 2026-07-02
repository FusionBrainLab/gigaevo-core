"""Shortlisting: research-query assembly and the store-backed shortlister."""

from __future__ import annotations

import pytest

from gigaevo.evolution.mutation.constants import MUTATION_CONTEXT_METADATA_KEY
from gigaevo.memory.read.shortlist import ResearchShortlister, build_research_query
from gigaevo.memory.storage.base import ResearchRequest, ResearchResult


class _Parent:
    def __init__(self, code: str, context: str = "") -> None:
        self.code = code
        self.metadata = {MUTATION_CONTEXT_METADATA_KEY: context} if context else {}


class _RecordingStore:
    def __init__(self, result: ResearchResult | None = None) -> None:
        self.requests: list[ResearchRequest] = []
        self._result = result or ResearchResult()

    async def research(self, request: ResearchRequest) -> ResearchResult:
        self.requests.append(request)
        return self._result


class _ExplodingStore:
    async def research(self, request: ResearchRequest) -> ResearchResult:
        raise RuntimeError("store down")


class TestBuildResearchQuery:
    def test_contains_all_mutation_inputs(self):
        query = build_research_query(
            parents=[_Parent("def f(): pass", context="snapshot A")],
            mutation_mode="crossover",
            task_description="pack points",
            metrics_description="fitness: min distance",
        )
        assert "TASK DESCRIPTION:\npack points" in query
        assert "AVAILABLE METRICS:\nfitness: min distance" in query
        assert "MUTATION MODE:\ncrossover" in query
        assert "=== Parent 1 ===" in query
        assert "def f(): pass" in query
        assert "snapshot A" in query
        assert query.endswith("select none if no card overlaps.")

    def test_empty_fields_get_placeholders(self):
        query = build_research_query(
            parents=[],
            mutation_mode="",
            task_description="  ",
            metrics_description="",
        )
        assert "TASK DESCRIPTION:\n<empty>" in query
        assert "AVAILABLE METRICS:\n<empty>" in query
        assert "MUTATION MODE:\nrewrite" in query

    def test_explicit_parent_contexts_override_metadata(self):
        query = build_research_query(
            parents=[_Parent("code0", context="stale"), _Parent("code1")],
            mutation_mode="rewrite",
            task_description="t",
            metrics_description="m",
            parent_contexts=["fresh ctx", ""],
        )
        assert "fresh ctx" in query
        assert "stale" not in query
        assert "=== Parent 2 ===" in query

    def test_metadata_context_used_when_no_override(self):
        query = build_research_query(
            parents=[_Parent("code0", context="from metadata")],
            mutation_mode="rewrite",
            task_description="t",
            metrics_description="m",
        )
        assert "from metadata" in query


class TestResearchShortlister:
    @pytest.mark.asyncio
    async def test_threads_query_and_exclusions_to_store(self, make_card):
        card = make_card()
        store = _RecordingStore(ResearchResult(cards=(card,), iterations=2))
        shortlister = ResearchShortlister(store)
        result = await shortlister.shortlist(
            parents=[_Parent("code")],
            mutation_mode="rewrite",
            task_description="task",
            metrics_description="metrics",
            exclude_ids=frozenset({"m-old"}),
        )
        assert result.cards == (card,)
        (request,) = store.requests
        assert request.exclude_ids == frozenset({"m-old"})
        assert "MUTATION INPUTS" in request.query

    @pytest.mark.asyncio
    async def test_store_failure_degrades_to_empty(self):
        shortlister = ResearchShortlister(_ExplodingStore())
        result = await shortlister.shortlist(
            parents=[_Parent("code")],
            mutation_mode="rewrite",
            task_description="task",
            metrics_description="metrics",
        )
        assert result == ResearchResult()
