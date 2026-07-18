from __future__ import annotations

import pytest

from gigaevo.llm.agents.admission_novelty import NoveltyVerdict
from gigaevo.llm.agents.card_author import AuthoredCard, CardAuthorResponse
from gigaevo.llm.agents.equivalence import EquivalenceResponse
from gigaevo.llm.agents.program_author import ProgramAuthorResponse
from gigaevo.memory.cards import Card, CardKind, ContextualGain, DecisionContext
from gigaevo.memory.storage.base import ScoredCard
from gigaevo.memory.write.admission import CardAdmissionGate, WriteOutcome
from gigaevo.memory.write.decisions import ArchiveStatus, WriteDecision
from gigaevo.memory.write.eviction import NullEvictor
from gigaevo.memory.write.librarian import Librarian


def authored(description: str = "When C holds, try A because M.") -> AuthoredCard:
    return AuthoredCard(
        description=description,
        explanation_summary="A changes the limiting mechanism under C.",
    )


class FakeAuthor:
    def __init__(self, response: CardAuthorResponse) -> None:
        self.response = response
        self.calls: list[dict] = []
        self.error: Exception | None = None

    async def arun(self, **kwargs) -> CardAuthorResponse:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class FakeEquivalence:
    def __init__(self, response: EquivalenceResponse) -> None:
        self.response = response
        self.calls: list[dict] = []
        self.error: Exception | None = None

    async def arun(self, **kwargs) -> EquivalenceResponse:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class FakeProgramAuthor:
    def __init__(self, response: ProgramAuthorResponse | None = None) -> None:
        self.response = response or ProgramAuthorResponse(
            decision=WriteDecision.NEW, card=authored("When P holds, try S because M.")
        )
        self.calls: list[dict] = []

    async def arun(self, **kwargs) -> ProgramAuthorResponse:
        self.calls.append(kwargs)
        return self.response


class FakeNoveltyJudge:
    def __init__(self, keep: bool) -> None:
        self.keep = keep
        self.calls: list[dict] = []

    async def arun(self, **kwargs) -> NoveltyVerdict:
        self.calls.append(kwargs)
        return NoveltyVerdict(keep=self.keep, reason="test")


def make_librarian(
    store,
    *,
    author: FakeAuthor | None = None,
    equivalence: FakeEquivalence | None = None,
    program_author: FakeProgramAuthor | None = None,
    admission_judge=None,
    task_key: str = "task-a",
) -> Librarian:
    return Librarian(
        author=author
        or FakeAuthor(CardAuthorResponse(decision=WriteDecision.NEW, card=authored())),
        equivalence=equivalence
        or FakeEquivalence(EquivalenceResponse(decision=WriteDecision.NEW)),
        program_author=program_author or FakeProgramAuthor(),
        gate=CardAdmissionGate(store=store, evictor=NullEvictor()),
        store=store,
        neighbors=store,
        top_k=5,
        task_key=task_key,
        task_description="full task",
        task_description_summary="task summary",
        admission_judge=admission_judge,
    )


async def ingest_idea(librarian: Librarian, **overrides):
    params = {
        "base_parent_code": "x = 0",
        "child_id": "child-1",
        "child_code": "x = 1",
        "mutation_report": "Change: x\nMutator explanation: improve x",
        "parent_fitness": 0.4,
        "child_fitness": 0.6,
        "signed_gain": 0.2,
        "higher_is_better": True,
        "archive_status": ArchiveStatus.ARCHIVED,
        "founding_gain": ContextualGain(
            context=DecisionContext(parent_metrics={"fitness": 0.4}),
            gain=0.2,
            founding=True,
        ),
    }
    params.update(overrides)
    return await librarian.ingest_idea(**params)


@pytest.mark.asyncio
async def test_drop_is_a_valid_zero_write_path(store) -> None:
    author = FakeAuthor(CardAuthorResponse(decision=WriteDecision.DROP))
    equivalence = FakeEquivalence(EquivalenceResponse(decision=WriteDecision.NEW))
    librarian = make_librarian(store, author=author, equivalence=equivalence)

    assert await ingest_idea(librarian) is None
    assert store.snapshot() == ()
    assert equivalence.calls == []


@pytest.mark.asyncio
async def test_author_receives_complete_outcome_and_only_one_card_is_added(
    store,
) -> None:
    author = FakeAuthor(CardAuthorResponse(decision=WriteDecision.NEW, card=authored()))
    librarian = make_librarian(store, author=author)

    result = await ingest_idea(librarian)

    assert result is not None
    assert result.outcome is WriteOutcome.ADDED
    assert len(store.snapshot()) == 1
    call = author.calls[0]
    assert call["parent_fitness"] == 0.4
    assert call["child_fitness"] == 0.6
    assert call["signed_gain"] == 0.2
    assert call["higher_is_better"] is True
    assert call["archive_status"] is ArchiveStatus.ARCHIVED
    assert "Mutator explanation" in call["mutation_report"]


@pytest.mark.asyncio
async def test_retrieval_uses_authored_action_not_raw_mutator_report(store) -> None:
    existing = Card(
        id="mem-existing",
        task_key="task-a",
        description="existing",
        explanation_summary="existing why",
    )
    store.save(existing)
    queries: list[tuple[str, int, CardKind | None, str | None]] = []

    def nearest(text, k, kind=None, task_key=None):
        queries.append((text, k, kind, task_key))
        return [ScoredCard(card=existing, distance=0.1)]

    store.nearest = nearest
    equivalence = FakeEquivalence(EquivalenceResponse(decision=WriteDecision.NEW))
    librarian = make_librarian(store, equivalence=equivalence)

    await ingest_idea(librarian, mutation_report="RAW_NOTE_MARKER")

    assert len(queries) == 1
    assert "When C holds, try A because M." in queries[0][0]
    assert "RAW_NOTE_MARKER" not in queries[0][0]
    assert queries[0][2] is CardKind.INSIGHT
    assert queries[0][3] == "task-a"


@pytest.mark.asyncio
async def test_equivalent_insight_keeps_payload_and_appends_provenance(store) -> None:
    existing = Card(
        id="mem-existing",
        task_key="task-a",
        description="stable treatment",
        explanation_summary="stable why",
        programs=("old-child",),
    )
    store.save(existing)
    store.hits = [ScoredCard(card=existing, distance=0.1)]
    equivalence = FakeEquivalence(
        EquivalenceResponse(decision=WriteDecision.EQUIVALENT, target_id=existing.id)
    )
    librarian = make_librarian(store, equivalence=equivalence)

    result = await ingest_idea(librarian)

    assert result is not None
    assert result.card_id == existing.id
    assert len(store.snapshot()) == 1
    updated = store.get(existing.id)
    assert updated is not None
    assert updated.description == "stable treatment"
    assert updated.explanation_summary == "stable why"
    assert updated.programs == ("old-child", "child-1")
    assert equivalence.calls[0]["candidate"].description.startswith("When C")


@pytest.mark.asyncio
async def test_foreign_task_neighbor_cannot_be_an_equivalence_target(store) -> None:
    foreign = Card(
        id="foreign",
        task_key="task-b",
        description="same words",
        explanation_summary="same why",
    )
    store.save(foreign)
    store.hits = [ScoredCard(card=foreign, distance=0.01)]
    equivalence = FakeEquivalence(EquivalenceResponse(decision=WriteDecision.NEW))
    librarian = make_librarian(store, equivalence=equivalence)

    result = await ingest_idea(librarian)

    assert result is not None
    assert result.outcome is WriteOutcome.ADDED
    assert equivalence.calls == []


@pytest.mark.asyncio
async def test_unoffered_target_and_equivalence_failure_fail_open_to_new(store) -> None:
    existing = Card(
        id="offered",
        task_key="task-a",
        description="near action",
        explanation_summary="why",
    )
    store.save(existing)
    store.hits = [ScoredCard(card=existing, distance=0.1)]
    invalid = FakeEquivalence(
        EquivalenceResponse(decision=WriteDecision.EQUIVALENT, target_id="not-offered")
    )
    first = make_librarian(store, equivalence=invalid)
    first_result = await ingest_idea(first, child_id="child-1")
    assert first_result is not None
    assert first_result.outcome is WriteOutcome.ADDED

    failing = FakeEquivalence(EquivalenceResponse(decision=WriteDecision.NEW))
    failing.error = RuntimeError("llm down")
    second = make_librarian(store, equivalence=failing)
    second_result = await ingest_idea(second, child_id="child-2")
    assert second_result is not None
    assert second_result.outcome is WriteOutcome.ADDED


@pytest.mark.asyncio
async def test_author_failure_propagates_without_banking_raw_note(store) -> None:
    author = FakeAuthor(CardAuthorResponse(decision=WriteDecision.DROP))
    author.error = RuntimeError("llm down")
    librarian = make_librarian(store, author=author)

    with pytest.raises(RuntimeError, match="llm down"):
        await ingest_idea(librarian, mutation_report="raw uncurated note")
    assert store.snapshot() == ()


@pytest.mark.asyncio
async def test_program_equivalence_keeps_family_id_and_best_representative(
    store,
) -> None:
    target = Card(
        id="program-family",
        kind=CardKind.PROGRAM,
        task_key="task-a",
        program_id="old",
        programs=("old",),
        description="stable holistic strategy",
        explanation_summary="stable mechanism",
        fitness=0.5,
    )
    store.save(target)
    store.hits = [ScoredCard(card=target, distance=0.1)]
    equivalence = FakeEquivalence(
        EquivalenceResponse(decision=WriteDecision.EQUIVALENT, target_id=target.id)
    )
    program_author = FakeProgramAuthor()
    librarian = make_librarian(
        store, equivalence=equivalence, program_author=program_author
    )

    result = await librarian.ingest_program(
        program_id="better",
        code="better code",
        fitness=0.8,
        archive_rank=1,
        higher_is_better=True,
        store_code=True,
    )

    assert result.card_id == target.id
    assert len(store.snapshot()) == 1
    family = store.get(target.id)
    assert family is not None
    assert family.id == "program-family"
    assert family.description == "stable holistic strategy"
    assert family.program_id == "better"
    assert family.fitness == 0.8
    assert family.code == "better code"
    assert family.programs == ("old", "better")


@pytest.mark.asyncio
async def test_worse_equivalent_program_only_appends_provenance_and_is_cached(store):
    target = Card(
        id="program-family",
        kind=CardKind.PROGRAM,
        task_key="task-a",
        program_id="best",
        programs=("best",),
        description="stable strategy",
        explanation_summary="stable why",
        fitness=0.9,
    )
    store.save(target)
    store.hits = [ScoredCard(card=target, distance=0.1)]
    equivalence = FakeEquivalence(
        EquivalenceResponse(decision=WriteDecision.EQUIVALENT, target_id=target.id)
    )
    program_author = FakeProgramAuthor()
    librarian = make_librarian(
        store, equivalence=equivalence, program_author=program_author
    )

    await librarian.ingest_program(
        program_id="worse",
        code="worse",
        fitness=0.4,
        archive_rank=3,
        higher_is_better=True,
    )
    cached = await librarian.ingest_program(
        program_id="worse",
        code="worse",
        fitness=0.4,
        archive_rank=3,
        higher_is_better=True,
    )

    family = store.get(target.id)
    assert family is not None
    assert family.program_id == "best"
    assert family.fitness == 0.9
    assert family.programs == ("best", "worse")
    assert len(program_author.calls) == 1
    assert cached.outcome is WriteOutcome.DISCARDED


@pytest.mark.asyncio
async def test_novelty_judge_only_gates_new_insights(store) -> None:
    judge = FakeNoveltyJudge(keep=False)
    librarian = make_librarian(store, admission_judge=judge)

    result = await ingest_idea(librarian)

    assert result is not None
    assert result.outcome is WriteOutcome.REJECTED_NOVELTY
    assert store.snapshot() == ()
    assert len(judge.calls) == 1
