"""Librarian ingest routing: reconcile decisions, exemplar twin dedup."""

from __future__ import annotations

import pytest

from gigaevo.llm.agents.admission_novelty import NoveltyVerdict
from gigaevo.llm.agents.program_author import ProgramAuthorResponse
from gigaevo.llm.agents.reconcile import LibrarianCard, ReconcileItem, ReconcileResponse
from gigaevo.memory.cards import Card, CardKind, ContextualGain, DecisionContext
from gigaevo.memory.storage.base import ScoredCard
from gigaevo.memory.write.admission import CardAdmissionGate
from gigaevo.memory.write.eviction import NullEvictor
from gigaevo.memory.write.librarian import Librarian, _strictly_better


def founding_event(gain: float = 0.2, parent_id: str = "parent-1") -> ContextualGain:
    return ContextualGain(
        context=DecisionContext(parent_id=parent_id),
        gain=gain,
        founding=True,
    )


class MarkEvictor:
    """Evicts exactly the ids in ``harmful`` (harm-gate stand-in)."""

    def __init__(self, harmful: set[str]) -> None:
        self._harmful = harmful

    def should_evict(self, card: Card) -> bool:
        return card.id in self._harmful

    def sweep(self, cards) -> list[str]:
        return [card.id for card in cards if self.should_evict(card)]


class FakeReconcileAgent:
    def __init__(self, response: ReconcileResponse | None = None) -> None:
        self.response = response or ReconcileResponse(items=[])
        self.calls: list[dict] = []
        self.raise_on_call = False

    async def arun(self, **kwargs) -> ReconcileResponse:
        self.calls.append(kwargs)
        if self.raise_on_call:
            raise RuntimeError("llm down")
        return self.response


class FakeProgramAuthor:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def arun(self, **kwargs) -> ProgramAuthorResponse:
        self.calls.append(kwargs)
        return ProgramAuthorResponse(
            description="authored exemplar", keywords=["fresh"]
        )


class FakeAdmissionJudge:
    def __init__(self, *, keep: bool = True) -> None:
        self.keep = keep
        self.calls: list[dict] = []
        self.raise_on_call = False

    async def arun(self, **kwargs) -> NoveltyVerdict:
        self.calls.append(kwargs)
        if self.raise_on_call:
            raise RuntimeError("judge down")
        return NoveltyVerdict(keep=self.keep, reason="test verdict")


def item(decision: str, *, description: str = "an idea", target_id: str = ""):
    return ReconcileItem(
        decision=decision,
        card=LibrarianCard(description=description),
        target_id=target_id,
    )


@pytest.fixture
def author():
    return FakeProgramAuthor()


@pytest.fixture
def make_librarian(store, author):
    def _make(agent: FakeReconcileAgent, **overrides) -> Librarian:
        params = {
            "agent": agent,
            "program_author": author,
            "gate": CardAdmissionGate(store=store, evictor=NullEvictor()),
            "store": store,
            "neighbors": store,
            "top_k": 5,
            "max_cards": 3,
            "task_description": "task",
            "task_description_summary": "summary",
        }
        params.update(overrides)
        return Librarian(**params)

    return _make


async def ingest(librarian: Librarian, note: str = "note") -> list[str]:
    return await librarian.ingest_idea(
        base_parent_code="x = 0", child_id="child-1", child_code="x = 1", note=note
    )


async def test_near_duplicate_routed_through_agent(store, make_card, make_librarian):
    near = make_card()
    store.save(near)
    store.hits = [ScoredCard(card=near, distance=0.01)]
    agent = FakeReconcileAgent(
        ReconcileResponse(items=[item("DUPLICATE", target_id=near.id)])
    )
    librarian = make_librarian(agent)
    assert await ingest(librarian) == [near.id]
    assert len(agent.calls) == 1
    assert agent.calls[0]["neighbors"] == [near]
    assert store.get(near.id).programs == ("child-1",)


async def test_nearest_failure_still_reaches_agent(store, make_librarian, monkeypatch):
    def broken_nearest(text, k, kind=None):
        raise RuntimeError("index down")

    monkeypatch.setattr(store, "nearest", broken_nearest)
    agent = FakeReconcileAgent(ReconcileResponse(items=[item("NEW")]))
    librarian = make_librarian(agent)
    ids = await ingest(librarian)
    assert len(agent.calls) == 1
    assert agent.calls[0]["neighbors"] == []
    assert len(ids) == 1
    assert store.get(ids[0]).description == "an idea"


async def test_agent_failure_admits_note_verbatim(store, make_librarian):
    agent = FakeReconcileAgent()
    agent.raise_on_call = True
    librarian = make_librarian(agent)
    ids = await ingest(librarian, note="raw mutation note")
    assert len(ids) == 1
    banked = store.get(ids[0])
    assert banked.description == "raw mutation note"
    assert banked.programs == ("child-1",)
    assert banked.task_description == "task"


async def test_new_decision_admits_authored_card(store, make_librarian):
    agent = FakeReconcileAgent(ReconcileResponse(items=[item("NEW")]))
    librarian = make_librarian(agent)
    ids = await ingest(librarian)
    assert len(ids) == 1
    assert store.get(ids[0]).description == "an idea"


async def test_novelty_gate_admits_novel_new_card(store, make_librarian):
    judge = FakeAdmissionJudge(keep=True)
    agent = FakeReconcileAgent(ReconcileResponse(items=[item("NEW")]))
    librarian = make_librarian(agent, admission_judge=judge)
    ids = await ingest(librarian)
    assert len(ids) == 1
    assert store.get(ids[0]).description == "an idea"
    assert len(judge.calls) == 1


async def test_novelty_gate_rejects_prior_known_new_card(store, make_librarian):
    judge = FakeAdmissionJudge(keep=False)
    agent = FakeReconcileAgent(ReconcileResponse(items=[item("NEW")]))
    librarian = make_librarian(agent, admission_judge=judge)
    ids = await ingest(librarian)
    assert ids == []
    assert store.snapshot() == ()
    assert len(judge.calls) == 1


async def test_novelty_gate_receives_card_prose(store, make_librarian):
    judge = FakeAdmissionJudge(keep=True)
    agent = FakeReconcileAgent(
        ReconcileResponse(items=[item("NEW", description="novel lever")])
    )
    librarian = make_librarian(agent, admission_judge=judge)
    await ingest(librarian)
    assert judge.calls[0]["description"] == "novel lever"
    assert "explanation_summary" in judge.calls[0]


async def test_novelty_gate_failure_admits_fail_open(store, make_librarian):
    judge = FakeAdmissionJudge(keep=False)
    judge.raise_on_call = True
    agent = FakeReconcileAgent(ReconcileResponse(items=[item("NEW")]))
    librarian = make_librarian(agent, admission_judge=judge)
    ids = await ingest(librarian)
    assert len(ids) == 1
    assert store.get(ids[0]).description == "an idea"


async def test_novelty_gate_rejects_fallback_new_from_missing_target(
    store, make_librarian
):
    # A DUPLICATE whose target is gone re-authors as NEW; that fallback NEW must
    # also pass the novelty gate, or prior-known cards leak in through it.
    judge = FakeAdmissionJudge(keep=False)
    agent = FakeReconcileAgent(
        ReconcileResponse(items=[item("DUPLICATE", target_id="gone")])
    )
    librarian = make_librarian(agent, admission_judge=judge)
    ids = await ingest(librarian)
    assert ids == []
    assert store.snapshot() == ()


async def test_novelty_gate_skipped_on_reconcile_failure_verbatim(
    store, make_librarian
):
    # The reconcile-failed verbatim path must never be gated: it is the
    # never-silent-drop degrade path, and the judge would likely fail too.
    judge = FakeAdmissionJudge(keep=False)
    agent = FakeReconcileAgent()
    agent.raise_on_call = True
    librarian = make_librarian(agent, admission_judge=judge)
    ids = await ingest(librarian, note="raw note")
    assert len(ids) == 1
    assert store.get(ids[0]).description == "raw note"
    assert judge.calls == []


async def test_novelty_gate_does_not_gate_program_exemplars(
    store, make_card, make_librarian
):
    judge = FakeAdmissionJudge(keep=False)
    librarian = make_librarian(FakeReconcileAgent(), admission_judge=judge)
    card = program_card(make_card, card_id="program-p1", fitness=0.5)
    assert librarian.admit_program(card, higher_is_better=True) == card.id
    assert store.get(card.id) is not None
    assert judge.calls == []


async def test_duplicate_decision_bumps_target(store, make_card, make_librarian):
    target = make_card()
    store.save(target)
    agent = FakeReconcileAgent(
        ReconcileResponse(items=[item("DUPLICATE", target_id=target.id)])
    )
    librarian = make_librarian(agent)
    assert await ingest(librarian) == [target.id]
    assert store.get(target.id).programs == ("child-1",)


async def test_duplicate_missing_target_falls_back_to_admit(store, make_librarian):
    agent = FakeReconcileAgent(
        ReconcileResponse(items=[item("DUPLICATE", target_id="gone")])
    )
    librarian = make_librarian(agent)
    ids = await ingest(librarian)
    assert len(ids) == 1
    assert store.get(ids[0]).description == "an idea"


async def test_merge_decision_folds_onto_target(store, make_card, make_librarian):
    target = make_card()
    store.save(target)
    agent = FakeReconcileAgent(
        ReconcileResponse(
            items=[item("MERGE", description="union prose", target_id=target.id)]
        )
    )
    librarian = make_librarian(agent)
    assert await ingest(librarian) == [target.id]
    assert store.get(target.id).description == "union prose"


async def test_merge_empty_target_falls_back_to_admit(store, make_librarian):
    agent = FakeReconcileAgent(ReconcileResponse(items=[item("MERGE")]))
    librarian = make_librarian(agent)
    ids = await ingest(librarian)
    assert len(ids) == 1
    assert store.get(ids[0]).description == "an idea"


async def test_merge_ruled_harmful_is_not_reauthored_as_new(
    store, make_card, make_librarian
):
    # A MERGE whose union the harm gate rejects must be DROPPED — never laundered
    # back in as a fresh NEW card. This is the verdict a benign missing-target
    # merge (which DOES fall back to NEW) must be distinguished from: the harm
    # gate already judged and deleted the target, so re-authoring resurrects it.
    target = make_card()
    store.save(target)
    gate = CardAdmissionGate(store=store, evictor=MarkEvictor({target.id}))
    agent = FakeReconcileAgent(
        ReconcileResponse(
            items=[item("MERGE", description="union prose", target_id=target.id)]
        )
    )
    librarian = make_librarian(agent, gate=gate)
    ids = await ingest(librarian)
    assert ids == []
    assert store.get(target.id) is None
    assert store.snapshot() == ()


async def test_new_card_born_with_founding_event(store, make_librarian):
    agent = FakeReconcileAgent(ReconcileResponse(items=[item("NEW")]))
    librarian = make_librarian(agent)
    founding = founding_event()
    ids = await librarian.ingest_idea(
        base_parent_code="x = 0",
        child_id="child-1",
        child_code="x = 1",
        note="note",
        founding_gain=founding,
    )
    assert len(ids) == 1
    assert store.get(ids[0]).gain_events == (founding,)


async def test_new_card_without_founding_gain_has_empty_events(store, make_librarian):
    agent = FakeReconcileAgent(ReconcileResponse(items=[item("NEW")]))
    librarian = make_librarian(agent)
    ids = await ingest(librarian)
    assert store.get(ids[0]).gain_events == ()


async def test_merge_unions_founding_onto_target(store, make_card, make_librarian):
    target = make_card()
    store.save(target)
    agent = FakeReconcileAgent(
        ReconcileResponse(
            items=[item("MERGE", description="union prose", target_id=target.id)]
        )
    )
    librarian = make_librarian(agent)
    founding = founding_event()
    ids = await librarian.ingest_idea(
        base_parent_code="x = 0",
        child_id="child-1",
        child_code="x = 1",
        note="note",
        founding_gain=founding,
    )
    assert ids == [target.id]
    assert store.get(target.id).gain_events == (founding,)


async def test_verbatim_fallback_carries_founding_event(store, make_librarian):
    agent = FakeReconcileAgent()
    agent.raise_on_call = True
    librarian = make_librarian(agent)
    founding = founding_event()
    ids = await librarian.ingest_idea(
        base_parent_code="x = 0",
        child_id="child-1",
        child_code="x = 1",
        note="raw note",
        founding_gain=founding,
    )
    assert len(ids) == 1
    assert store.get(ids[0]).gain_events == (founding,)


async def test_max_cards_truncates_items(store, make_librarian):
    agent = FakeReconcileAgent(
        ReconcileResponse(
            items=[item("NEW", description=f"idea {i}") for i in range(5)]
        )
    )
    librarian = make_librarian(agent, max_cards=2)
    ids = await ingest(librarian)
    assert len(ids) == 2


async def test_author_program_cache_hit_skips_llm(
    store, make_card, make_librarian, author
):
    banked = make_card(
        id="program-p1",
        kind=CardKind.PROGRAM,
        program_id="p1",
        description="cached exemplar",
        code="x = 1",
        fitness=0.4,
    )
    store.save(banked)
    librarian = make_librarian(FakeReconcileAgent())
    resp = await librarian.author_program(program_id="p1", code="x = 1", fitness=0.4)
    assert resp.description == "cached exemplar"
    assert author.calls == []

    fresh = await librarian.author_program(program_id="p2", code="y = 2", fitness=0.5)
    assert fresh.description == "authored exemplar"
    assert len(author.calls) == 1


def program_card(make_card, *, card_id: str, fitness: float) -> Card:
    return make_card(
        id=card_id,
        kind=CardKind.PROGRAM,
        program_id=card_id.removeprefix("program-"),
        description="same strategy",
        code="x = 1",
        fitness=fitness,
    )


async def test_admit_program_without_twin_admits(store, make_card, make_librarian):
    librarian = make_librarian(FakeReconcileAgent())
    card = program_card(make_card, card_id="program-p1", fitness=0.5)
    assert librarian.admit_program(card, higher_is_better=True) == card.id
    assert store.get(card.id) is not None


async def test_admit_program_replaces_strictly_worse_twin(
    store, make_card, make_librarian
):
    twin = program_card(make_card, card_id="program-old", fitness=0.3)
    store.save(twin)
    store.hits = [ScoredCard(card=twin, distance=0.01)]
    librarian = make_librarian(FakeReconcileAgent())
    incoming = program_card(make_card, card_id="program-new", fitness=0.6)
    assert librarian.admit_program(incoming, higher_is_better=True) == incoming.id
    assert store.get(twin.id) is None
    assert store.get(incoming.id) is not None


async def test_admit_program_drops_when_not_strictly_better(
    store, make_card, make_librarian
):
    twin = program_card(make_card, card_id="program-old", fitness=0.6)
    store.save(twin)
    store.hits = [ScoredCard(card=twin, distance=0.01)]
    librarian = make_librarian(FakeReconcileAgent())
    incoming = program_card(make_card, card_id="program-new", fitness=0.6)
    assert librarian.admit_program(incoming, higher_is_better=True) == ""
    assert store.get(twin.id) is not None
    assert store.get(incoming.id) is None


async def test_admit_program_dedupes_identical_code_regardless_of_prose(
    store, make_card, make_librarian
):
    # Byte-identical code, independently-authored (different) prose, and NO
    # neighbor surfaced: the prose-cosine gate missed this exact case (two
    # program cards of the same code banked as twins). Code identity must catch
    # it, and must not depend on nearest() returning anything.
    twin = make_card(
        id="program-old",
        kind=CardKind.PROGRAM,
        program_id="old",
        description="grid init raises min-area",
        code="def solve():\n    return 1\n",
        fitness=0.5,
    )
    store.save(twin)
    store.hits = []
    librarian = make_librarian(FakeReconcileAgent())
    incoming = make_card(
        id="program-new",
        kind=CardKind.PROGRAM,
        program_id="new",
        description="component substitution borrowed from parent 1",
        code="def solve():\n    return 1\n",
        fitness=0.5,
    )
    assert librarian.admit_program(incoming, higher_is_better=True) == ""
    assert store.get("program-old") is not None
    assert store.get("program-new") is None


async def test_admit_program_keeps_distinct_code(store, make_card, make_librarian):
    first = make_card(
        id="program-a",
        kind=CardKind.PROGRAM,
        program_id="a",
        description="strategy A",
        code="def solve():\n    return 1\n",
        fitness=0.5,
    )
    store.save(first)
    librarian = make_librarian(FakeReconcileAgent())
    second = make_card(
        id="program-b",
        kind=CardKind.PROGRAM,
        program_id="b",
        description="strategy B",
        code="def solve():\n    return 2\n",
        fitness=0.5,
    )
    assert librarian.admit_program(second, higher_is_better=True) == "program-b"
    assert store.get("program-a") is not None
    assert store.get("program-b") is not None


def test_strictly_better_direction_table():
    assert _strictly_better(None, 0.5, True) is False
    assert _strictly_better(0.5, None, True) is True
    assert _strictly_better(0.6, 0.5, True) is True
    assert _strictly_better(0.5, 0.5, True) is False
    assert _strictly_better(0.4, 0.5, False) is True
    assert _strictly_better(0.6, 0.5, False) is False
