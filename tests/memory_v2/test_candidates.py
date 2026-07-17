from __future__ import annotations

import asyncio
import json
import math

import pytest

from gigaevo.memory.cards import Card
from gigaevo.memory.events import memory_event_context
from gigaevo.memory.storage.base import ResearchResult
from gigaevo.memory_v2.candidates import (
    AgenticCandidateSource,
    WholeBankCandidateSource,
)
from gigaevo.memory_v2.models import RetrievalRecord
from gigaevo.programs.program import Program

_PARENT_ID = "00000000-0000-4000-8000-000000000001"


def _cards(count: int) -> tuple[Card, ...]:
    return tuple(
        Card(id=f"card-{index}", task_key="task", description=f"idea {index}")
        for index in range(count)
    )


class _Store:
    def __init__(self, cards: tuple[Card, ...]) -> None:
        self.cards = cards

    def snapshot(self) -> tuple[Card, ...]:
        return self.cards


class _Shortlister:
    def __init__(
        self,
        result: ResearchResult | None = None,
        *,
        failure: Exception | None = None,
    ) -> None:
        self.result = result or ResearchResult()
        self.failure = failure
        self.calls: list[dict[str, object]] = []

    async def shortlist(self, **kwargs) -> ResearchResult:
        self.calls.append(kwargs)
        if self.failure is not None:
            raise self.failure
        return self.result


class _MutatingShortlister(_Shortlister):
    def __init__(self, store: _Store, result: ResearchResult) -> None:
        super().__init__(result)
        self.store = store

    async def shortlist(self, **kwargs) -> ResearchResult:
        result = await super().shortlist(**kwargs)
        self.store.cards = self.store.cards[1:]
        return result


class _BlockingShortlister(_Shortlister):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.cancelled = False

    async def shortlist(self, **kwargs) -> ResearchResult:
        self.calls.append(kwargs)
        self.started.set()
        try:
            await asyncio.Event().wait()
        finally:
            self.cancelled = True


class _Excluder:
    def __init__(self, *card_ids: str) -> None:
        self.card_ids = frozenset(card_ids)

    def exclude_for(self, program: Program) -> frozenset[str]:
        del program
        return self.card_ids


@pytest.mark.asyncio
async def test_empty_bank_skips_agentic_research() -> None:
    shortlister = _Shortlister()
    source = AgenticCandidateSource(
        store=_Store(()),  # type: ignore[arg-type]
        shortlister=shortlister,
    )

    slate = await source.candidate_snapshot(
        Program(id=_PARENT_ID, code="pass"),
        task_key="task",
        task_description="task",
        metrics_description="score",
        parent_context=None,
        pending_by_bank_card={},
        max_pending_per_card=2,
        rng_key="0" * 64,
    )

    assert slate.candidates == ()
    assert slate.retrieval.status == "empty"
    assert shortlister.calls == []


@pytest.mark.asyncio
async def test_whole_bank_record_requires_probability_one() -> None:
    cards = _cards(2)
    source = WholeBankCandidateSource(store=_Store(cards))  # type: ignore[arg-type]
    slate = await source.candidate_snapshot(
        Program(id=_PARENT_ID, code="pass"),
        task_key="task",
        task_description="task",
        metrics_description="score",
        parent_context=None,
        pending_by_bank_card={},
        max_pending_per_card=2,
        rng_key="1" * 64,
    )

    payload = slate.retrieval.model_dump(mode="python", exclude_computed_fields=True)
    payload["random_slate_probability"] = 0.5
    with pytest.raises(ValueError, match="probability one"):
        RetrievalRecord.model_validate(payload)


@pytest.mark.asyncio
async def test_agentic_core_has_uniform_discovery_support() -> None:
    cards = _cards(10)
    shortlister = _Shortlister(
        ResearchResult(cards=cards[:4], summary="relevant", iterations=1)
    )
    source = AgenticCandidateSource(
        store=_Store(cards),  # type: ignore[arg-type]
        shortlister=shortlister,
        max_candidates=6,
        exploration_candidates=2,
        mutation_mode="diff",
    )
    kwargs = dict(
        task_key="task",
        task_description="packing task",
        metrics_description="maximize score",
        parent_context="live MAP-Elites state",
        pending_by_bank_card={},
        max_pending_per_card=2,
        rng_key="a" * 64,
    )

    first = await source.candidate_snapshot(
        Program(id=_PARENT_ID, code="def solve(): pass", iteration=3), **kwargs
    )
    second = await source.candidate_snapshot(
        Program(id=_PARENT_ID, code="def solve(): pass", iteration=3), **kwargs
    )

    record = first.retrieval
    assert record.status == "agentic"
    assert record.core_bank_card_ids == tuple(card.id for card in cards[:4])
    assert len(record.exploration_bank_card_ids) == 2
    assert record.candidate_bank_card_ids == (
        record.core_bank_card_ids + record.exploration_bank_card_ids
    )
    assert record == second.retrieval
    assert record.conditional_tail_inclusion_probability == pytest.approx(2 / 6)
    assert record.random_slate_probability == pytest.approx(1 / math.comb(6, 2))
    payload = record.model_dump(mode="python", exclude_computed_fields=True)
    payload["exploration_bank_card_ids"] = tuple(
        reversed(record.exploration_bank_card_ids)
    )
    payload["candidate_bank_card_ids"] = (
        record.core_bank_card_ids + payload["exploration_bank_card_ids"]
    )
    with pytest.raises(ValueError, match="RNG key"):
        RetrievalRecord.model_validate(payload)
    call = shortlister.calls[0]
    assert call["task_description"] == "packing task"
    assert call["metrics_description"] == "maximize score"
    assert call["parent_contexts"] == ["live MAP-Elites state"]
    assert call["mutation_mode"] == "diff"


@pytest.mark.asyncio
async def test_core_priority_caps_random_tail_without_backfilling() -> None:
    cards = _cards(10)
    source = AgenticCandidateSource(
        store=_Store(cards),  # type: ignore[arg-type]
        shortlister=_Shortlister(ResearchResult(cards=cards[:1], iterations=1)),
        max_candidates=6,
        exploration_candidates=2,
        selection_logic="core_priority",
    )

    slate = await source.candidate_snapshot(
        Program(id=_PARENT_ID, code="pass"),
        task_key="task",
        task_description="task",
        metrics_description="score",
        parent_context=None,
        pending_by_bank_card={},
        max_pending_per_card=2,
        rng_key="7" * 64,
    )

    record = slate.retrieval
    assert record.specification.name == "agentic_research_core_priority"
    assert record.core_bank_card_ids == (cards[0].id,)
    assert len(record.exploration_bank_card_ids) == 2
    assert len(record.candidate_bank_card_ids) == 3
    assert record.conditional_tail_inclusion_probability == pytest.approx(2 / 9)
    assert record.random_slate_probability == pytest.approx(1 / math.comb(9, 2))


@pytest.mark.asyncio
async def test_prepared_research_is_reused_without_a_second_llm_call() -> None:
    cards = _cards(4)
    shortlister = _Shortlister(ResearchResult(cards=cards[:2], iterations=1))
    source = AgenticCandidateSource(
        store=_Store(cards),  # type: ignore[arg-type]
        shortlister=shortlister,
        max_candidates=3,
        exploration_candidates=1,
    )
    program = Program(id=_PARENT_ID, code="pass")
    common = dict(
        task_key="task",
        task_description="task",
        metrics_description="score",
        parent_context=None,
        pending_by_bank_card={},
        max_pending_per_card=2,
    )

    research = await source.prepare(program, **common)
    slate = await source.candidate_snapshot(
        program,
        **common,
        rng_key="9" * 64,
        research=research,
    )

    assert len(shortlister.calls) == 1
    assert slate.retrieval.core_bank_card_ids == tuple(card.id for card in cards[:2])


@pytest.mark.asyncio
async def test_agentic_research_timeout_fails_open_to_uniform_slate() -> None:
    cards = _cards(4)
    shortlister = _BlockingShortlister()
    source = AgenticCandidateSource(
        store=_Store(cards),  # type: ignore[arg-type]
        shortlister=shortlister,
        max_candidates=3,
        exploration_candidates=1,
        research_timeout_seconds=0.05,
    )

    slate = await source.candidate_snapshot(
        Program(id=_PARENT_ID, code="pass"),
        task_key="task",
        task_description="task",
        metrics_description="score",
        parent_context=None,
        pending_by_bank_card={},
        max_pending_per_card=2,
        rng_key="8" * 64,
    )

    assert shortlister.cancelled
    assert slate.retrieval.status == "uniform_fallback"
    assert len(slate.candidates) == 3


@pytest.mark.asyncio
async def test_agentic_research_timeout_emits_completion_event(tmp_path) -> None:
    cards = _cards(2)
    source = AgenticCandidateSource(
        store=_Store(cards),  # type: ignore[arg-type]
        shortlister=_BlockingShortlister(),
        research_timeout_seconds=0.01,
    )
    events_file = tmp_path / "events.jsonl"

    with memory_event_context(
        event_path=events_file,
        program_id=_PARENT_ID,
        parent_ids=(_PARENT_ID,),
    ):
        result = await source.prepare(
            Program(id=_PARENT_ID, code="pass"),
            task_key="task",
            task_description="task",
            metrics_description="score",
            parent_context=None,
            pending_by_bank_card={},
            max_pending_per_card=2,
        )

    assert result == ResearchResult()
    rows = [json.loads(line) for line in events_file.read_text().splitlines()]
    assert [row["event"] for row in rows] == ["MEMORY_RESEARCH"]
    assert rows[0]["outcome"] == "failed"
    assert rows[0]["program_id"] == _PARENT_ID
    assert "exceeded" in rows[0]["error"]


@pytest.mark.asyncio
async def test_external_cancellation_still_propagates() -> None:
    shortlister = _BlockingShortlister()
    source = AgenticCandidateSource(
        store=_Store(_cards(2)),  # type: ignore[arg-type]
        shortlister=shortlister,
        research_timeout_seconds=10.0,
    )
    task = asyncio.create_task(
        source.prepare(
            Program(id=_PARENT_ID, code="pass"),
            task_key="task",
            task_description="task",
            metrics_description="score",
            parent_context=None,
            pending_by_bank_card={},
            max_pending_per_card=2,
        )
    )
    await shortlister.started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert shortlister.cancelled


@pytest.mark.asyncio
async def test_agentic_research_refreshes_a_changed_live_bank() -> None:
    cards = _cards(5)
    store = _Store(cards)
    source = AgenticCandidateSource(
        store=store,  # type: ignore[arg-type]
        shortlister=_MutatingShortlister(
            store, ResearchResult(cards=cards[:3], iterations=1)
        ),
        max_candidates=4,
        exploration_candidates=1,
    )

    slate = await source.candidate_snapshot(
        Program(id=_PARENT_ID, code="pass"),
        task_key="task",
        task_description="task",
        metrics_description="score",
        parent_context=None,
        pending_by_bank_card={},
        max_pending_per_card=2,
        rng_key="2" * 64,
    )

    assert slate.retrieval.eligible_bank_card_ids == tuple(
        card.id for card in cards[1:]
    )
    assert cards[0].id not in slate.retrieval.candidate_bank_card_ids
    assert tuple(card.id for card in slate.lineage_registry) == tuple(
        card.id for card in cards[1:]
    )


@pytest.mark.asyncio
async def test_agentic_failure_uses_a_replayable_uniform_slate() -> None:
    cards = _cards(8)
    source = AgenticCandidateSource(
        store=_Store(cards),  # type: ignore[arg-type]
        shortlister=_Shortlister(failure=RuntimeError("research unavailable")),
        max_candidates=5,
        exploration_candidates=2,
    )

    slate = await source.candidate_snapshot(
        Program(id=_PARENT_ID, code="pass"),
        task_key="task",
        task_description="task",
        metrics_description="score",
        parent_context=None,
        pending_by_bank_card={},
        max_pending_per_card=2,
        rng_key="b" * 64,
    )

    record = slate.retrieval
    assert record.status == "uniform_fallback"
    assert record.core_bank_card_ids == ()
    assert len(record.exploration_bank_card_ids) == 5
    assert len(record.candidate_bank_card_ids) == 5
    assert record.conditional_tail_inclusion_probability == pytest.approx(5 / 8)
    assert record.random_slate_probability == pytest.approx(1 / math.comb(8, 5))


@pytest.mark.asyncio
async def test_core_priority_failure_keeps_only_declared_random_budget() -> None:
    cards = _cards(8)
    source = AgenticCandidateSource(
        store=_Store(cards),  # type: ignore[arg-type]
        shortlister=_Shortlister(failure=RuntimeError("research unavailable")),
        max_candidates=5,
        exploration_candidates=2,
        selection_logic="core_priority",
    )

    slate = await source.candidate_snapshot(
        Program(id=_PARENT_ID, code="pass"),
        task_key="task",
        task_description="task",
        metrics_description="score",
        parent_context=None,
        pending_by_bank_card={},
        max_pending_per_card=2,
        rng_key="d" * 64,
    )

    record = slate.retrieval
    assert record.status == "uniform_fallback"
    assert len(record.exploration_bank_card_ids) == 2
    assert len(record.candidate_bank_card_ids) == 2
    assert record.conditional_tail_inclusion_probability == pytest.approx(2 / 8)
    assert record.random_slate_probability == pytest.approx(1 / math.comb(8, 2))


@pytest.mark.asyncio
async def test_exclusion_and_pending_filter_before_agentic_research() -> None:
    cards = _cards(6)
    shortlister = _Shortlister(ResearchResult(cards=cards, iterations=1))
    source = AgenticCandidateSource(
        store=_Store(cards),  # type: ignore[arg-type]
        shortlister=shortlister,
        excluder=_Excluder("card-0"),
        max_candidates=4,
        exploration_candidates=1,
    )

    slate = await source.candidate_snapshot(
        Program(id=_PARENT_ID, code="pass"),
        task_key="task",
        task_description="task",
        metrics_description="score",
        parent_context="state",
        pending_by_bank_card={"card-1": 2},
        max_pending_per_card=2,
        rng_key="c" * 64,
    )

    assert slate.retrieval.eligible_bank_card_ids == (
        "card-2",
        "card-3",
        "card-4",
        "card-5",
    )
    assert set(slate.retrieval.candidate_bank_card_ids) <= {
        "card-2",
        "card-3",
        "card-4",
        "card-5",
    }
    assert shortlister.calls[0]["exclude_ids"] >= {"card-0", "card-1"}


@pytest.mark.asyncio
async def test_task_and_kind_filters_apply_before_agentic_research() -> None:
    eligible = Card(id="eligible", task_key="task", description="usable idea")
    cross_task = Card(id="cross-task", task_key="other", description="other task")
    disallowed_kind = Card(
        id="program-card",
        kind="program",
        program_id="program-1",
        task_key="task",
        description="program exemplar",
    )
    cards = (eligible, cross_task, disallowed_kind)
    shortlister = _Shortlister(ResearchResult(cards=cards, iterations=1))
    source = AgenticCandidateSource(
        store=_Store(cards),  # type: ignore[arg-type]
        shortlister=shortlister,
        allow_cross_task=False,
        allowed_kinds=("insight",),
        max_candidates=2,
        exploration_candidates=1,
    )

    slate = await source.candidate_snapshot(
        Program(id=_PARENT_ID, code="pass"),
        task_key="task",
        task_description="task",
        metrics_description="score",
        parent_context=None,
        pending_by_bank_card={},
        max_pending_per_card=2,
        rng_key="f" * 64,
    )

    assert slate.retrieval.eligible_bank_card_ids == (eligible.id,)
    assert slate.retrieval.core_bank_card_ids == (eligible.id,)
    assert shortlister.calls[0]["exclude_ids"] >= {
        cross_task.id,
        disallowed_kind.id,
    }


@pytest.mark.asyncio
async def test_pending_counts_follow_merged_card_lineage() -> None:
    survivor = Card(
        id="card-survivor",
        absorbed_ids=["card-absorbed"],
        task_key="task",
        description="merged idea",
    )
    shortlister = _Shortlister(ResearchResult(cards=(survivor,), iterations=1))
    source = AgenticCandidateSource(
        store=_Store((survivor,)),  # type: ignore[arg-type]
        shortlister=shortlister,
        max_candidates=2,
        exploration_candidates=1,
    )

    blocked = await source.candidate_snapshot(
        Program(id=_PARENT_ID, code="pass"),
        task_key="task",
        task_description="task",
        metrics_description="score",
        parent_context=None,
        pending_by_bank_card={"card-absorbed": 2},
        max_pending_per_card=2,
        rng_key="d" * 64,
    )
    released = await source.candidate_snapshot(
        Program(id=_PARENT_ID, code="pass"),
        task_key="task",
        task_description="task",
        metrics_description="score",
        parent_context=None,
        pending_by_bank_card={},
        max_pending_per_card=2,
        rng_key="e" * 64,
    )

    assert blocked.retrieval.status == "empty"
    assert released.retrieval.eligible_bank_card_ids == (survivor.id,)
    assert released.retrieval.status == "agentic"
