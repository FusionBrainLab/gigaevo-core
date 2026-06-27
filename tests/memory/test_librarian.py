"""Tests for the Librarian: the LLM-first idea-write path.

Behavioral surface only — the librarian orchestrates a NeighborSource, a
ReconcileAgent, a ProgramAuthorAgent, and a CardAdmissionGate. We fake each
collaborator and assert on what reaches the gate, never on internal calls.
"""

from __future__ import annotations

import pytest

from gigaevo.llm.agents.program_author import ProgramAuthorResponse
from gigaevo.llm.agents.reconcile import (
    LibrarianCard,
    ReconcileItem,
    ReconcileResponse,
)
from gigaevo.memory.ideas_tracker.librarian import Librarian
from gigaevo.memory.shared_memory.models import MemoryCard, ProgramCard


class _FakeCardStore:
    def __init__(self) -> None:
        self.cards: dict = {}


class _FakeStore:
    def __init__(self) -> None:
        self.card_store = _FakeCardStore()


class _FakeGate:
    def __init__(self) -> None:
        self.admitted: list = []
        self.merged: list = []
        self.bumped: list = []

    def admit(self, card) -> str:  # noqa: ANN001
        self.admitted.append(card)
        return card.id or "mem-new"

    def merge(self, target_id, card) -> str:  # noqa: ANN001
        self.merged.append((target_id, card))
        return target_id

    def bump_provenance(self, target_id, child_id) -> str:  # noqa: ANN001
        self.bumped.append((target_id, child_id))
        return target_id


class _FakeNeighbors:
    def __init__(self, hits) -> None:  # noqa: ANN001
        self._hits = hits

    def nearest(self, note, k):  # noqa: ANN001
        return self._hits


class _FakeAgent:
    def __init__(self, response: ReconcileResponse) -> None:
        self._response = response
        self.calls = 0

    async def arun(self, **kwargs):  # noqa: ANN003
        self.calls += 1
        return self._response


class _FailingAgent:
    def __init__(self) -> None:
        self.calls = 0

    async def arun(self, **kwargs):  # noqa: ANN003
        self.calls += 1
        raise RuntimeError("llm down")


class _FakeProgramAuthor:
    def __init__(self, response: ProgramAuthorResponse) -> None:
        self._response = response
        self.calls = 0

    async def arun(self, **kwargs):  # noqa: ANN003
        self.calls += 1
        return self._response


def _lib(agent, gate, store, neighbors, program_author=None, **kw) -> Librarian:  # noqa: ANN001
    return Librarian(
        agent=agent,
        program_author=program_author
        or _FakeProgramAuthor(ProgramAuthorResponse(description="d")),
        gate=gate,
        store=store,
        neighbors=neighbors,
        **kw,
    )


async def _ingest(lib: Librarian, **over) -> list[str]:  # noqa: ANN003
    kwargs = dict(
        base_parent_id="p",
        base_parent_code="def f(): return 1",
        child_id="c",
        child_code="def f(): return 2",
        note="bumped k",
    )
    kwargs.update(over)
    return await lib.ingest_idea(**kwargs)


@pytest.mark.asyncio
async def test_new_card_is_authored_and_admitted() -> None:
    gate, store, agent = (
        _FakeGate(),
        _FakeStore(),
        _FakeAgent(
            ReconcileResponse(
                items=[
                    ReconcileItem(
                        decision="NEW",
                        card=LibrarianCard(
                            description="widen spectral gap", keywords=["spectral"]
                        ),
                    )
                ]
            )
        ),
    )
    ids = await _ingest(_lib(agent, gate, store, _FakeNeighbors([])))
    assert ids == ["mem-new"]
    assert agent.calls == 1
    assert len(gate.admitted) == 1
    assert gate.admitted[0].description == "widen spectral gap"
    assert gate.admitted[0].keywords == ["spectral"]
    assert gate.admitted[0].programs == ["c"]


@pytest.mark.asyncio
async def test_empty_items_means_drop() -> None:
    gate, store, agent = (
        _FakeGate(),
        _FakeStore(),
        _FakeAgent(ReconcileResponse(items=[])),
    )
    ids = await _ingest(_lib(agent, gate, store, _FakeNeighbors([])))
    assert ids == []
    assert gate.admitted == []
    assert gate.merged == []


@pytest.mark.asyncio
async def test_merge_routes_to_gate_merge() -> None:
    gate, store, agent = (
        _FakeGate(),
        _FakeStore(),
        _FakeAgent(
            ReconcileResponse(
                items=[
                    ReconcileItem(
                        decision="MERGE",
                        card=LibrarianCard(description="union prose"),
                        target_id="mem-T",
                    )
                ]
            )
        ),
    )
    ids = await _ingest(_lib(agent, gate, store, _FakeNeighbors([])))
    assert ids == ["mem-T"]
    assert gate.merged and gate.merged[0][0] == "mem-T"
    assert gate.admitted == []


@pytest.mark.asyncio
async def test_duplicate_without_target_is_dropped() -> None:
    gate, store, agent = (
        _FakeGate(),
        _FakeStore(),
        _FakeAgent(
            ReconcileResponse(
                items=[
                    ReconcileItem(
                        decision="DUPLICATE",
                        card=LibrarianCard(description="dup"),
                        target_id="",
                    )
                ]
            )
        ),
    )
    ids = await _ingest(_lib(agent, gate, store, _FakeNeighbors([])))
    assert ids == []
    assert gate.merged == []
    assert gate.admitted == []


@pytest.mark.asyncio
async def test_pre_gate_bumps_provenance_without_calling_llm() -> None:
    gate, store = _FakeGate(), _FakeStore()
    near = MemoryCard(id="mem-near", description="same lever", keywords=[])
    agent = _FakeAgent(ReconcileResponse(items=[]))
    ids = await _ingest(
        _lib(agent, gate, store, _FakeNeighbors([(near, 0.01)])), child_id="c"
    )
    assert agent.calls == 0
    assert gate.bumped == [("mem-near", "c")]
    assert ids == ["mem-near"]
    assert gate.admitted == []


@pytest.mark.asyncio
async def test_far_neighbor_does_not_short_circuit() -> None:
    gate, store = _FakeGate(), _FakeStore()
    far = MemoryCard(id="mem-far", description="other lever", keywords=[])
    agent = _FakeAgent(
        ReconcileResponse(
            items=[ReconcileItem(decision="NEW", card=LibrarianCard(description="x"))]
        )
    )
    ids = await _ingest(_lib(agent, gate, store, _FakeNeighbors([(far, 0.5)])))
    assert agent.calls == 1
    assert gate.bumped == []
    assert ids == ["mem-new"]


@pytest.mark.asyncio
async def test_llm_failure_admits_note_verbatim_not_drop() -> None:
    gate, store, agent = _FakeGate(), _FakeStore(), _FailingAgent()
    lib = _lib(
        agent,
        gate,
        store,
        _FakeNeighbors([]),
        task_description="full task text",
        task_description_summary="short summary",
    )
    ids = await _ingest(lib, note="real lever")
    assert agent.calls == 1
    assert len(gate.admitted) == 1
    assert gate.admitted[0].description == "real lever"
    assert gate.admitted[0].programs == ["c"]
    assert gate.admitted[0].task_description == "full task text"
    assert gate.admitted[0].task_description_summary == "short summary"
    assert ids == ["mem-new"]


@pytest.mark.asyncio
async def test_idea_card_carries_full_task_and_summary() -> None:
    gate, store = _FakeGate(), _FakeStore()
    agent = _FakeAgent(
        ReconcileResponse(
            items=[ReconcileItem(decision="NEW", card=LibrarianCard(description="x"))]
        )
    )
    lib = _lib(
        agent,
        gate,
        store,
        _FakeNeighbors([]),
        task_description="full task text",
        task_description_summary="short summary",
    )
    await _ingest(lib)
    assert gate.admitted[0].task_description == "full task text"
    assert gate.admitted[0].task_description_summary == "short summary"


@pytest.mark.asyncio
async def test_cardinality_capped_at_max_cards() -> None:
    gate, store = _FakeGate(), _FakeStore()
    items = [
        ReconcileItem(decision="NEW", card=LibrarianCard(description=f"lever {i}"))
        for i in range(5)
    ]
    agent = _FakeAgent(ReconcileResponse(items=items))
    ids = await _ingest(_lib(agent, gate, store, _FakeNeighbors([])))
    assert len(gate.admitted) == 3
    assert len(ids) == 3


@pytest.mark.asyncio
async def test_author_program_authors_new_program() -> None:
    gate, store = _FakeGate(), _FakeStore()
    author = _FakeProgramAuthor(
        ProgramAuthorResponse(description="greedy spectral", keywords=["spectral"])
    )
    lib = _lib(
        _FakeAgent(ReconcileResponse(items=[])),
        gate,
        store,
        _FakeNeighbors([]),
        program_author=author,
    )
    resp = await lib.author_program(program_id="42", code="def s(): ...", fitness=0.53)
    assert resp.description == "greedy spectral"
    assert resp.keywords == ["spectral"]
    assert author.calls == 1


@pytest.mark.asyncio
async def test_author_program_cached_by_program_id_skips_llm() -> None:
    gate, store = _FakeGate(), _FakeStore()
    store.card_store.cards["program-42"] = ProgramCard(
        id="program-42",
        program_id="42",
        description="already authored",
        code="def s(): ...",
        keywords=["cached-kw"],
    )
    author = _FakeProgramAuthor(ProgramAuthorResponse(description="greedy spectral"))
    lib = _lib(
        _FakeAgent(ReconcileResponse(items=[])),
        gate,
        store,
        _FakeNeighbors([]),
        program_author=author,
    )
    resp = await lib.author_program(program_id="42", code="x", fitness=0.5)
    assert resp.description == "already authored"
    assert resp.keywords == ["cached-kw"]
    assert author.calls == 0
