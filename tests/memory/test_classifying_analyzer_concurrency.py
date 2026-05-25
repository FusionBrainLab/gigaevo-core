"""Tests for parallel per-record classification in ClassifyingAnalyzer."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from gigaevo.memory.ideas_tracker.analyzers import ClassifyingAnalyzer
from gigaevo.memory.ideas_tracker.idea_bank import IdeaBank
from gigaevo.memory.ideas_tracker.models import (
    AnalysisResult,
    Idea,
    ProgramRecord,
)

EMPTY_CLASSIFICATION = '{"present_ideas": [], "new_ideas": [], "updated_ideas": []}'


class _ConcurrencyTrackingLLM:
    """Stub LLM that counts simultaneous in-flight ``call_async`` invocations."""

    def __init__(self, response: str, delay_s: float = 0.02) -> None:
        self.response = response
        self.delay_s = delay_s
        self.in_flight = 0
        self.max_in_flight = 0
        self.call_log: list[str] = []
        self._lock = asyncio.Lock()

    def call(self, step, content="", reasoning=None):  # type: ignore[no-untyped-def]
        raise AssertionError("sync call() invoked in async test path")

    async def call_async(self, step, content="", reasoning=None):  # type: ignore[no-untyped-def]
        async with self._lock:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(self.delay_s)
            return self.response
        finally:
            async with self._lock:
                self.in_flight -= 1
                self.call_log.append(step)


def _build_analyzer(*, max_concurrent_classifications: int = 8) -> ClassifyingAnalyzer:
    with patch(
        "gigaevo.memory.ideas_tracker.llm._init_clients",
        return_value=(MagicMock(), MagicMock(), False),
    ):
        return ClassifyingAnalyzer(
            model="mock-model",
            max_concurrent_classifications=max_concurrent_classifications,
        )


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
    llm = _ConcurrencyTrackingLLM(response=EMPTY_CLASSIFICATION, delay_s=0.05)

    analyzer = _build_analyzer(max_concurrent_classifications=4)
    analyzer._llm = llm  # type: ignore[assignment]

    result = await analyzer.analyze_async(records, bank)

    assert len(result.new_ideas) == 4
    assert llm.max_in_flight >= 2, (
        f"expected concurrent classification, max_in_flight={llm.max_in_flight}"
    )
    assert len(llm.call_log) == 4


@pytest.mark.asyncio
async def test_analyze_async_respects_concurrency_cap():
    bank = _seeded_bank()
    records = _records(6)
    llm = _ConcurrencyTrackingLLM(response=EMPTY_CLASSIFICATION, delay_s=0.05)

    analyzer = _build_analyzer(max_concurrent_classifications=2)
    analyzer._llm = llm  # type: ignore[assignment]

    await analyzer.analyze_async(records, bank)

    assert llm.max_in_flight <= 2
    assert len(llm.call_log) == 6


@pytest.mark.asyncio
async def test_analyze_async_matches_sync_shape():
    bank = _seeded_bank()
    records = _records(3)

    sync_llm = MagicMock()
    sync_llm.call = MagicMock(return_value=EMPTY_CLASSIFICATION)
    async_llm = _ConcurrencyTrackingLLM(response=EMPTY_CLASSIFICATION, delay_s=0)

    a_sync = _build_analyzer()
    a_sync._llm = sync_llm  # type: ignore[assignment]
    a_async = _build_analyzer()
    a_async._llm = async_llm  # type: ignore[assignment]

    sync_result = a_sync.analyze(records, _seeded_bank())
    async_result = await a_async.analyze_async(records, bank)

    assert sorted(i.description for i in sync_result.new_ideas) == sorted(
        i.description for i in async_result.new_ideas
    )
    assert len(sync_result.updates) == len(async_result.updates)


@pytest.mark.asyncio
async def test_analyze_async_empty_records_makes_no_llm_calls():
    bank = _seeded_bank()
    llm = _ConcurrencyTrackingLLM(response=EMPTY_CLASSIFICATION, delay_s=0)

    analyzer = _build_analyzer()
    analyzer._llm = llm  # type: ignore[assignment]

    result = await analyzer.analyze_async([], bank)

    assert result == AnalysisResult()
    assert llm.call_log == []


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
    llm = _ConcurrencyTrackingLLM(response=EMPTY_CLASSIFICATION, delay_s=0)

    analyzer = _build_analyzer()
    analyzer._llm = llm  # type: ignore[assignment]

    result = await analyzer.analyze_async(records, bank)

    assert result.new_ideas == []
    assert result.updates == []
    assert llm.call_log == []
