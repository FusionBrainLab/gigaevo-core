"""Total-LLM-failure behavior of ClassifyingAnalyzer.

When every classification attempt for a bank chunk fails, the pending items
must be DROPPED (they reappear in later sweeps), not minted as NEW ideas —
otherwise an LLM outage floods the bank with unvetted duplicates. A
successful-but-empty classification response stays the legitimate
"this idea is new" path, and an empty bank (cold start, no chunks) still
mints NEW without any LLM call.
"""

from __future__ import annotations

import pytest

from gigaevo.memory.ideas_tracker.analyzers import ClassifyingAnalyzer, _PendingIdeas
from gigaevo.memory.ideas_tracker.idea_bank import IdeaBank
from gigaevo.memory.ideas_tracker.models import (
    AnalysisResult,
    Idea,
    Improvement,
    ProgramRecord,
)
from tests.fakes.llm_router import FakeMemoryRouter


def _always_raise(schema=None, messages=None):
    raise RuntimeError("analyzer LLM down")


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


def test_sync_total_failure_drops_items_instead_of_minting_new():
    analyzer = ClassifyingAnalyzer(
        llm=FakeMemoryRouter(respond=_always_raise), retry_attempts=2
    )

    result = analyzer.analyze(_records(2), _seeded_bank())

    assert result.new_ideas == []
    assert result.updates == []
    assert result.failed_program_ids == ["p0", "p1"]


@pytest.mark.asyncio
async def test_async_total_failure_drops_items_instead_of_minting_new():
    analyzer = ClassifyingAnalyzer(
        llm=FakeMemoryRouter(respond=_always_raise, allow_sync=False),
        retry_attempts=2,
    )

    result = await analyzer.analyze_async(_records(2), _seeded_bank())

    assert result.new_ideas == []
    assert result.updates == []
    assert sorted(result.failed_program_ids) == ["p0", "p1"]


def test_partial_classification_failure_emits_nothing_for_retry():
    """If classification failed with items still pending, emitting the
    classified subset double-applies those updates when the tracker un-sees
    the program and retries the whole record next sweep."""
    analyzer = ClassifyingAnalyzer(llm=FakeMemoryRouter(), retry_attempts=1)
    pending = _PendingIdeas.from_improvements(
        [
            Improvement(description="idea one", explanation="m1"),
            Improvement(description="idea two", explanation="m2"),
        ]
    )
    pending.mark_classified(1, "existing-idea", rewrite=False)
    pending.refresh_mapping()
    pending.classification_failed = True
    record = _records(1)[0]
    result = AnalysisResult()

    analyzer._apply_pending_to_result(pending, record, result)

    assert result.updates == []
    assert result.new_ideas == []
    assert result.failed_program_ids == [record.id]


def test_fully_classified_despite_chunk_failure_emits_normally():
    """A chunk failure followed by full classification on a later chunk is a
    complete result — emit it, no retry."""
    analyzer = ClassifyingAnalyzer(llm=FakeMemoryRouter(), retry_attempts=1)
    pending = _PendingIdeas.from_improvements(
        [Improvement(description="idea one", explanation="m1")]
    )
    pending.mark_classified(1, "existing-idea", rewrite=False)
    pending.classification_failed = True
    record = _records(1)[0]
    result = AnalysisResult()

    analyzer._apply_pending_to_result(pending, record, result)

    assert len(result.updates) == 1
    assert result.failed_program_ids == []


def test_mark_classified_handles_duplicate_descriptions():
    """Two pending items with identical descriptions must classify one each;
    matching the first item regardless of state strands the second forever."""
    pending = _PendingIdeas.from_improvements(
        [
            Improvement(description="same lever", explanation="a"),
            Improvement(description="same lever", explanation="b"),
        ]
    )
    pending.mark_classified(1, "idea-1", rewrite=False)
    pending.refresh_mapping()
    pending.mark_classified(1, "idea-2", rewrite=False)

    assert pending.unclassified_count == 0
    assert [i.target_id for i in pending.items] == ["idea-1", "idea-2"]


def test_cold_start_empty_bank_still_mints_new_without_llm():
    llm = FakeMemoryRouter(respond=_always_raise)
    analyzer = ClassifyingAnalyzer(llm=llm, retry_attempts=2)

    result = analyzer.analyze(_records(3), IdeaBank())

    assert len(result.new_ideas) == 3
    assert result.failed_program_ids == []
    assert llm.calls == []
