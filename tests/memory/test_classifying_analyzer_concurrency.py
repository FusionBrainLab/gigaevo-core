"""Tests for parallel per-record classification in ClassifyingAnalyzer."""

from __future__ import annotations

import pytest

from gigaevo.memory.ideas_tracker.analyzers import ClassifyingAnalyzer
from gigaevo.memory.ideas_tracker.idea_bank import IdeaBank
from gigaevo.memory.ideas_tracker.models import (
    AnalysisResult,
    Idea,
    ProgramRecord,
)
from gigaevo.memory.ideas_tracker.schemas import ClassifyExtResponse
from tests.fakes.llm_router import FakeMemoryRouter


def _empty_classification(schema=None, messages=None) -> ClassifyExtResponse:
    return ClassifyExtResponse(new_ideas=[], present_ideas=[], updated_ideas=[])


def _records(n: int) -> list[ProgramRecord]:
    return [
        ProgramRecord(
            id=f"p{i}",
            fitness=0.5,
            generation=1,
            parents=["root"],
            improvements=[{"description": f"incoming-{i}", "explanation": "why"}],
        )
        for i in range(n)
    ]


def _seeded_bank() -> IdeaBank:
    bank = IdeaBank()
    bank.add(Idea(id="aaaa0000-0000-0000-0000-000000000001", description="seed-1"))
    return bank


@pytest.mark.asyncio
async def test_analyze_async_runs_records_concurrently():
    bank = _seeded_bank()
    records = _records(4)
    llm = FakeMemoryRouter(
        respond=_empty_classification, delay_s=0.05, allow_sync=False
    )

    analyzer = ClassifyingAnalyzer(llm=llm, max_concurrent_classifications=4)

    result = await analyzer.analyze_async(records, bank)

    assert len(result.new_ideas) == 4
    assert llm.max_in_flight >= 2, (
        f"expected concurrent classification, max_in_flight={llm.max_in_flight}"
    )
    assert len(llm.calls) == 4


@pytest.mark.asyncio
async def test_analyze_async_respects_concurrency_cap():
    bank = _seeded_bank()
    records = _records(6)
    llm = FakeMemoryRouter(
        respond=_empty_classification, delay_s=0.05, allow_sync=False
    )

    analyzer = ClassifyingAnalyzer(llm=llm, max_concurrent_classifications=2)

    await analyzer.analyze_async(records, bank)

    assert llm.max_in_flight <= 2
    assert len(llm.calls) == 6


@pytest.mark.asyncio
async def test_analyze_async_matches_sync_shape():
    bank = _seeded_bank()
    records = _records(3)

    a_sync = ClassifyingAnalyzer(llm=FakeMemoryRouter(respond=_empty_classification))
    a_async = ClassifyingAnalyzer(
        llm=FakeMemoryRouter(respond=_empty_classification, allow_sync=False)
    )

    sync_result = a_sync.analyze(records, _seeded_bank())
    async_result = await a_async.analyze_async(records, bank)

    assert sorted(i.description for i in sync_result.new_ideas) == sorted(
        i.description for i in async_result.new_ideas
    )
    assert len(sync_result.updates) == len(async_result.updates)


@pytest.mark.asyncio
async def test_analyze_async_empty_records_makes_no_llm_calls():
    bank = _seeded_bank()
    llm = FakeMemoryRouter(respond=_empty_classification, allow_sync=False)

    analyzer = ClassifyingAnalyzer(llm=llm)

    result = await analyzer.analyze_async([], bank)

    assert result == AnalysisResult()
    assert llm.calls == []


@pytest.mark.asyncio
async def test_analyze_async_record_without_improvements_skips_llm():
    bank = _seeded_bank()
    records = [
        ProgramRecord(
            id="p-noop",
            fitness=0.5,
            generation=1,
            parents=["root"],
            improvements=[],
        )
    ]
    llm = FakeMemoryRouter(respond=_empty_classification, allow_sync=False)

    analyzer = ClassifyingAnalyzer(llm=llm)

    result = await analyzer.analyze_async(records, bank)

    assert result.new_ideas == []
    assert result.updates == []
