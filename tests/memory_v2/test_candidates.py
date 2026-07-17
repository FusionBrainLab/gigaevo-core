from __future__ import annotations

import asyncio

import pytest

from gigaevo.memory.cards import Card
from gigaevo.memory.events import MemoryResearch
from gigaevo.memory.storage.base import ResearchFailure, ResearchResult
from gigaevo.memory_v2.candidates import (
    AgenticApplicabilityProvider,
    WholeBankCandidateSource,
)
from gigaevo.memory_v2.models import ApplicabilityStatus, CandidateUniverseRecord
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

    async def shortlist(self, **kwargs: object) -> ResearchResult:
        self.calls.append(kwargs)
        if self.failure is not None:
            raise self.failure
        return self.result


class _MutatingShortlister(_Shortlister):
    def __init__(self, store: _Store, result: ResearchResult) -> None:
        super().__init__(result)
        self.store = store

    async def shortlist(self, **kwargs: object) -> ResearchResult:
        result = await super().shortlist(**kwargs)
        self.store.cards = self.store.cards[1:]
        return result


class _BlockingShortlister(_Shortlister):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.cancelled = False

    async def shortlist(self, **kwargs: object) -> ResearchResult:
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


async def _snapshot(
    source: WholeBankCandidateSource,
    *,
    research: ResearchResult | None = None,
    pending_by_bank_card: dict[str, int] | None = None,
):
    return await source.candidate_snapshot(
        Program(id=_PARENT_ID, code="pass"),
        task_key="task",
        task_description="task",
        metrics_description="score",
        parent_context="live state",
        pending_by_bank_card=pending_by_bank_card or {},
        max_pending_per_card=1,
        research=research,
    )


@pytest.mark.asyncio
async def test_empty_bank_skips_agentic_assessment() -> None:
    shortlister = _Shortlister()
    source = WholeBankCandidateSource(
        store=_Store(()),  # type: ignore[arg-type]
        applicability=AgenticApplicabilityProvider(shortlister=shortlister),
    )

    slate = await _snapshot(source)

    assert slate.candidates == ()
    assert slate.candidate_universe.status == "empty"
    assert slate.applicability.status == "empty"
    assert shortlister.calls == []


@pytest.mark.asyncio
async def test_null_applicability_exposes_the_complete_eligible_bank() -> None:
    cards = _cards(3)
    source = WholeBankCandidateSource(store=_Store(cards))  # type: ignore[arg-type]

    slate = await _snapshot(source)

    assert tuple(card.id for card in slate.candidates) == tuple(
        card.id for card in cards
    )
    assert slate.candidate_universe.eligible_bank_card_ids == tuple(
        card.id for card in cards
    )
    assert slate.applicability.status == "disabled"
    payload = slate.candidate_universe.model_dump(
        mode="python", exclude_computed_fields=True
    )
    payload["eligible_bank_card_ids"] = tuple(card.id for card in reversed(cards))
    with pytest.raises(ValueError, match="sorted"):
        CandidateUniverseRecord.model_validate(payload)


@pytest.mark.asyncio
async def test_agentic_assessment_labels_a_subset_without_gating_the_bank() -> None:
    cards = _cards(5)
    shortlister = _Shortlister(
        ResearchResult(cards=cards[:2], summary="mechanism fit", iterations=1)
    )
    source = WholeBankCandidateSource(
        store=_Store(cards),  # type: ignore[arg-type]
        applicability=AgenticApplicabilityProvider(
            shortlister=shortlister,
            mutation_mode="diff",
        ),
    )
    program = Program(id=_PARENT_ID, code="pass")
    research = await source.prepare(
        program,
        task_key="task",
        task_description="task",
        metrics_description="score",
        parent_context="live state",
        pending_by_bank_card={},
        max_pending_per_card=1,
    )
    slate = await _snapshot(source, research=research)

    assert len(shortlister.calls) == 1
    assert tuple(card.id for card in slate.candidates) == tuple(
        card.id for card in cards
    )
    assert slate.candidate_universe.eligible_bank_card_ids == tuple(
        card.id for card in cards
    )
    assert slate.applicability.status == "assessed"
    assert slate.applicability.applicable_bank_card_ids == tuple(
        card.id for card in cards[:2]
    )
    assert slate.applicability.specification.mutation_mode == "diff"


@pytest.mark.asyncio
async def test_failed_agentic_assessment_is_a_neutral_full_bank_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[MemoryResearch] = []
    monkeypatch.setattr("gigaevo.memory_v2.candidates.emit_memory_event", events.append)
    cards = _cards(4)
    shortlister = _Shortlister(failure=RuntimeError("research unavailable"))
    source = WholeBankCandidateSource(
        store=_Store(cards),  # type: ignore[arg-type]
        applicability=AgenticApplicabilityProvider(shortlister=shortlister),
    )

    slate = await _snapshot(source)

    assert tuple(card.id for card in slate.candidates) == tuple(
        card.id for card in cards
    )
    assert slate.applicability.status == "failed"
    assert slate.applicability.applicable_bank_card_ids == ()
    assert slate.applicability.failure is ResearchFailure.SHORTLISTER_EXCEPTION
    payload = slate.applicability.model_dump(mode="json")
    assert payload["status"] == ApplicabilityStatus.FAILED.value
    assert payload["failure"] == ResearchFailure.SHORTLISTER_EXCEPTION.value
    assert len(events) == 1
    assert events[0].outcome == "failed"
    assert events[0].error == "research unavailable"


@pytest.mark.asyncio
async def test_agentic_timeout_is_a_neutral_full_bank_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[MemoryResearch] = []
    monkeypatch.setattr("gigaevo.memory_v2.candidates.emit_memory_event", events.append)
    cards = _cards(2)
    shortlister = _BlockingShortlister()
    source = WholeBankCandidateSource(
        store=_Store(cards),  # type: ignore[arg-type]
        applicability=AgenticApplicabilityProvider(
            shortlister=shortlister,
            research_timeout_seconds=0.01,
        ),
    )

    slate = await _snapshot(source)

    assert slate.candidate_universe.status == "eligible_bank"
    assert slate.applicability.status == "failed"
    assert slate.applicability.failure is ResearchFailure.TIMEOUT
    assert shortlister.cancelled
    assert len(events) == 1
    assert events[0].outcome == "failed"
    assert events[0].error


@pytest.mark.asyncio
async def test_refresh_drops_stale_labels_but_keeps_all_live_eligible_cards() -> None:
    cards = _cards(4)
    store = _Store(cards)
    source = WholeBankCandidateSource(
        store=store,  # type: ignore[arg-type]
        applicability=AgenticApplicabilityProvider(
            shortlister=_MutatingShortlister(
                store,
                ResearchResult(cards=cards[:3], iterations=1),
            )
        ),
    )

    slate = await _snapshot(source)

    assert tuple(card.id for card in slate.candidates) == tuple(
        card.id for card in cards[1:]
    )
    assert slate.applicability.applicable_bank_card_ids == tuple(
        card.id for card in cards[1:3]
    )


@pytest.mark.asyncio
async def test_eligibility_filters_before_assessment_and_posterior_selection() -> None:
    cards = _cards(4)
    shortlister = _Shortlister(ResearchResult(cards=cards, iterations=1))
    source = WholeBankCandidateSource(
        store=_Store(cards),  # type: ignore[arg-type]
        applicability=AgenticApplicabilityProvider(shortlister=shortlister),
        excluder=_Excluder(cards[0].id),
    )

    slate = await _snapshot(source, pending_by_bank_card={cards[1].id: 1})

    assert tuple(card.id for card in slate.candidates) == tuple(
        card.id for card in cards[2:]
    )
    assert slate.applicability.applicable_bank_card_ids == tuple(
        card.id for card in cards[2:]
    )
    assert shortlister.calls[0]["exclude_ids"] >= {cards[0].id, cards[1].id}


@pytest.mark.asyncio
async def test_external_cancellation_propagates() -> None:
    shortlister = _BlockingShortlister()
    source = WholeBankCandidateSource(
        store=_Store(_cards(2)),  # type: ignore[arg-type]
        applicability=AgenticApplicabilityProvider(
            shortlister=shortlister,
            research_timeout_seconds=10.0,
        ),
    )
    task = asyncio.create_task(_snapshot(source))
    await shortlister.started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert shortlister.cancelled
