"""Tests for the Librarian: the LLM-first idea-write path.

Behavioral surface only — the librarian orchestrates a NeighborSource, a
ReconcileAgent, a ProgramAuthorAgent, and a CardAdmissionGate. We fake each
collaborator and assert on what reaches the gate, never on internal calls.
"""

from __future__ import annotations

import json

import pytest

from gigaevo.llm.agents.program_author import ProgramAuthorResponse
from gigaevo.llm.agents.reconcile import (
    LibrarianCard,
    ReconcileItem,
    ReconcileResponse,
)
from gigaevo.memory.core.events import memory_event_context
from gigaevo.memory.ideas_tracker.librarian import Librarian
from gigaevo.memory.shared_memory.models import MemoryCard, ProgramCard


class _FakeCardStore:
    def __init__(self) -> None:
        self.cards: dict = {}


class _FakeStore:
    def __init__(self) -> None:
        self.card_store = _FakeCardStore()
        self.deleted: list = []

    def get_card(self, card_id: str):  # noqa: ANN201
        return self.card_store.cards.get(card_id)

    def all_cards_snapshot(self) -> dict:
        return dict(self.card_store.cards)

    def delete(self, card_id: str) -> None:
        self.deleted.append(card_id)
        self.card_store.cards.pop(card_id, None)


class _FakeGate:
    """Records what reaches the gate. With ``store`` set, merge/bump mirror the
    real CardAdmissionGate: a target absent from the bank yields ``""`` (the
    no-op the librarian must treat as 'author the idea anyway, never drop it')."""

    def __init__(self, store=None) -> None:  # noqa: ANN001
        self._store = store
        self.admitted: list = []
        self.merged: list = []
        self.bumped: list = []

    def _target_missing(self, target_id) -> bool:  # noqa: ANN001
        return self._store is not None and not isinstance(
            self._store.card_store.cards.get(target_id), MemoryCard
        )

    def admit(self, card) -> str:  # noqa: ANN001
        self.admitted.append(card)
        return card.id or "mem-new"

    def merge(self, target_id, card) -> str:  # noqa: ANN001
        if self._target_missing(target_id):
            return ""
        self.merged.append((target_id, card))
        return target_id

    def bump_provenance(self, target_id, child_id) -> str:  # noqa: ANN001
        if self._target_missing(target_id):
            return ""
        self.bumped.append((target_id, child_id))
        return target_id


class _FakeNeighbors:
    """Mirrors the production ChromaNeighborSource type contract: ``nearest``
    surfaces only cards of the requested ``card_type`` — so idea dedup gets
    ``MemoryCard`` hits and exemplar twin dedup gets ``ProgramCard`` hits."""

    def __init__(self, hits) -> None:  # noqa: ANN001
        self._hits = hits

    def nearest(self, note, k, card_type):  # noqa: ANN001
        return [(c, d) for c, d in self._hits if isinstance(c, card_type)]


class _QueryNeighbors:
    """Query-aware NeighborSource fake keyed on the exact query text.

    The pre-gate queries on the raw mutation NOTE; the post-authoring re-query
    queries on the AUTHORED description. Those are different strings, so this
    fake can surface different neighbors for each — the very asymmetry the
    post-authoring dedup closes. Unknown queries return no hits."""

    def __init__(self, by_query) -> None:  # noqa: ANN001
        self._by_query = by_query

    def nearest(self, text, k, card_type):  # noqa: ANN001
        hits = self._by_query.get((text or "").strip(), [])
        return [(c, d) for c, d in hits if isinstance(c, card_type)][:k]


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
async def test_duplicate_with_target_bumps_provenance_not_merge() -> None:
    gate, store, agent = (
        _FakeGate(),
        _FakeStore(),
        _FakeAgent(
            ReconcileResponse(
                items=[
                    ReconcileItem(
                        decision="DUPLICATE",
                        card=LibrarianCard(description="dup prose"),
                        target_id="mem-T",
                    )
                ]
            )
        ),
    )
    ids = await _ingest(_lib(agent, gate, store, _FakeNeighbors([])), child_id="c")
    assert gate.bumped == [("mem-T", "c")]
    assert gate.merged == []
    assert gate.admitted == []
    assert ids == ["mem-T"]


@pytest.mark.asyncio
async def test_duplicate_without_target_admits_as_new() -> None:
    # A DUPLICATE the agent never named a target for cannot bump anything; the
    # idea must still land (authored as NEW), never be silently dropped.
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
    assert ids == ["mem-new"]
    assert gate.merged == []
    assert gate.bumped == []
    assert [c.description for c in gate.admitted] == ["dup"]


@pytest.mark.asyncio
async def test_duplicate_with_stale_target_admits_as_new() -> None:
    # The agent named a DUPLICATE target that has since left the bank, so the
    # gate's bump is a no-op (returns ""). The idea must be authored as NEW
    # rather than dropped — the gate is faithful (store-aware) here.
    store = _FakeStore()
    gate = _FakeGate(store)
    agent = _FakeAgent(
        ReconcileResponse(
            items=[
                ReconcileItem(
                    decision="DUPLICATE",
                    card=LibrarianCard(description="dup of a ghost"),
                    target_id="mem-ghost",
                )
            ]
        )
    )
    ids = await _ingest(_lib(agent, gate, store, _FakeNeighbors([])))
    assert ids == ["mem-new"]
    assert gate.bumped == []
    assert [c.description for c in gate.admitted] == ["dup of a ghost"]


@pytest.mark.asyncio
async def test_merge_with_stale_target_admits_as_new() -> None:
    # Same for MERGE: a target that has left the bank makes the merge a no-op
    # (returns ""), so the unioned prose must be authored as NEW, not lost.
    store = _FakeStore()
    gate = _FakeGate(store)
    agent = _FakeAgent(
        ReconcileResponse(
            items=[
                ReconcileItem(
                    decision="MERGE",
                    card=LibrarianCard(description="union onto a ghost"),
                    target_id="mem-ghost",
                )
            ]
        )
    )
    ids = await _ingest(_lib(agent, gate, store, _FakeNeighbors([])))
    assert ids == ["mem-new"]
    assert gate.merged == []
    assert [c.description for c in gate.admitted] == ["union onto a ghost"]


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
async def test_new_authored_description_near_existing_bumps_provenance() -> None:
    """The confirmed dedup miss: the pre-gate compares the raw NOTE against
    indexed AUTHORED card docs (cross-domain), so an idea the agent then authors
    into an existing card's description slips past it as NEW. A post-authoring
    re-query on the authored description catches it and downgrades NEW ->
    provenance bump rather than banking a twin."""
    gate, store = _FakeGate(), _FakeStore()
    near = MemoryCard(id="mem-near", description="same lever", keywords=[])
    agent = _FakeAgent(
        ReconcileResponse(
            items=[
                ReconcileItem(
                    decision="NEW", card=LibrarianCard(description="same lever")
                )
            ]
        )
    )
    # Pre-gate queries the note ("bumped k") -> miss; post-gate queries the
    # authored description ("same lever") -> hit within eps.
    neighbors = _QueryNeighbors({"same lever": [(near, 0.01)]})
    ids = await _ingest(_lib(agent, gate, store, neighbors), child_id="c")
    assert agent.calls == 1
    assert gate.bumped == [("mem-near", "c")]
    assert gate.admitted == []
    assert ids == ["mem-near"]


@pytest.mark.asyncio
async def test_authored_hit_beyond_online_eps_admits_as_new() -> None:
    """The post-authoring gate uses the tight online eps (0.05), not the
    consolidation recall eps (0.2): a 0.15 hit is NOT a near-dup here, so the
    card is admitted as NEW (consolidation's LLM arbiter handles the looser
    band, never a silent bump)."""
    gate, store = _FakeGate(), _FakeStore()
    near = MemoryCard(id="mem-near", description="adjacent lever", keywords=[])
    agent = _FakeAgent(
        ReconcileResponse(
            items=[
                ReconcileItem(
                    decision="NEW", card=LibrarianCard(description="adjacent lever")
                )
            ]
        )
    )
    neighbors = _QueryNeighbors({"adjacent lever": [(near, 0.15)]})
    ids = await _ingest(_lib(agent, gate, store, neighbors))
    assert gate.bumped == []
    assert [c.description for c in gate.admitted] == ["adjacent lever"]
    assert ids == ["mem-new"]


@pytest.mark.asyncio
async def test_post_authoring_dup_with_stale_target_admits_as_new() -> None:
    """The post-authoring near-dup target may have left the bank between the
    re-query and the bump (the gate's bump is then a no-op, returns ""). The
    idea must be authored as NEW, never dropped — the NEW branch carries its own
    admit fallback (the pre-existing fallback only covers DUPLICATE/MERGE)."""
    store = _FakeStore()
    gate = _FakeGate(store)  # store-aware: a target absent from the bank -> ""
    ghost = MemoryCard(id="mem-ghost", description="ghost lever", keywords=[])
    agent = _FakeAgent(
        ReconcileResponse(
            items=[
                ReconcileItem(
                    decision="NEW", card=LibrarianCard(description="ghost lever")
                )
            ]
        )
    )
    neighbors = _QueryNeighbors({"ghost lever": [(ghost, 0.01)]})
    ids = await _ingest(_lib(agent, gate, store, neighbors))
    assert gate.bumped == []
    assert [c.description for c in gate.admitted] == ["ghost lever"]
    assert ids == ["mem-new"]


@pytest.mark.asyncio
async def test_post_authoring_neighbor_failure_admits_as_new(tmp_path) -> None:
    """A post-authoring re-query failure (Chroma hiccup) must be observable and
    must NOT drop the idea — same contract as the pre-gate retrieval."""
    gate, store = _FakeGate(), _FakeStore()
    agent = _FakeAgent(
        ReconcileResponse(
            items=[ReconcileItem(decision="NEW", card=LibrarianCard(description="x"))]
        )
    )

    class _NoteOkDescBoom:
        """Pre-gate note query succeeds (empty); post-authoring description
        query raises."""

        def nearest(self, text, k, card_type):  # noqa: ANN001
            if (text or "").strip() == "bumped k":
                return []
            raise RuntimeError("chroma down")

    path = tmp_path / "memory_events.jsonl"
    with memory_event_context(event_path=path):
        ids = await _ingest(_lib(agent, gate, store, _NoteOkDescBoom()))

    assert ids == ["mem-new"]
    assert [c.description for c in gate.admitted] == ["x"]
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    assert any(r["event_type"] == "neighbor.retrieval_failed" for r in rows)


@pytest.mark.asyncio
async def test_two_new_items_same_authored_description_second_bumps_first() -> None:
    """Intra-batch twins: two NEW items the agent authors into the same
    description in one reconcile response. The first admits; the second's
    post-authoring re-query sees it (admit syncs to the index before returning)
    and bumps it instead of banking a second twin."""

    class _ReflectingNeighbors:
        def __init__(self) -> None:
            self._cards: list = []

        def register(self, card) -> None:  # noqa: ANN001
            self._cards.append(card)

        def nearest(self, text, k, card_type):  # noqa: ANN001
            t = (text or "").strip()
            return [
                (c, 0.0)
                for c in self._cards
                if isinstance(c, card_type) and (c.description or "").strip() == t
            ][:k]

    class _RegisteringGate:
        """admit assigns an id and registers the card so a later re-query in the
        same call can see it (mirrors admit -> Chroma sync)."""

        def __init__(self, neighbors) -> None:  # noqa: ANN001
            self._neighbors = neighbors
            self.admitted: list = []
            self.bumped: list = []
            self._n = 0

        def admit(self, card) -> str:  # noqa: ANN001
            self._n += 1
            saved = card.model_copy(update={"id": f"mem-{self._n}"})
            self.admitted.append(saved)
            self._neighbors.register(saved)
            return saved.id

        def bump_provenance(self, target_id, child_id) -> str:  # noqa: ANN001
            self.bumped.append((target_id, child_id))
            return target_id

    neighbors = _ReflectingNeighbors()
    gate = _RegisteringGate(neighbors)
    agent = _FakeAgent(
        ReconcileResponse(
            items=[
                ReconcileItem(
                    decision="NEW", card=LibrarianCard(description="dup lever")
                ),
                ReconcileItem(
                    decision="NEW", card=LibrarianCard(description="dup lever")
                ),
            ]
        )
    )
    ids = await _ingest(_lib(agent, gate, _FakeStore(), neighbors), child_id="c")
    assert [c.description for c in gate.admitted] == ["dup lever"]
    assert gate.bumped == [("mem-1", "c")]
    assert ids == ["mem-1", "mem-1"]


@pytest.mark.asyncio
async def test_post_authoring_program_card_hit_is_ignored() -> None:
    """The post-authoring re-query is MemoryCard-only (idea dedup), so a program
    exemplar within eps must not trigger a bump — the idea is authored as NEW."""
    gate, store = _FakeGate(), _FakeStore()
    program_neighbor = ProgramCard(
        id="program-7",
        program_id="7",
        description="widen gap",
        code="def s(): ...",
    )
    agent = _FakeAgent(
        ReconcileResponse(
            items=[
                ReconcileItem(
                    decision="NEW", card=LibrarianCard(description="widen gap")
                )
            ]
        )
    )
    neighbors = _QueryNeighbors({"widen gap": [(program_neighbor, 0.0)]})
    ids = await _ingest(_lib(agent, gate, store, neighbors))
    assert gate.bumped == []
    assert [c.description for c in gate.admitted] == ["widen gap"]
    assert ids == ["mem-new"]


@pytest.mark.asyncio
async def test_program_card_neighbor_is_skipped_not_treated_as_duplicate() -> None:
    """A program exemplar card within eps must NOT short-circuit the idea write.

    ProgramCards live in the same bank as idea cards and leak into the neighbor
    source. Bumping a ProgramCard's provenance is a no-op that would silently
    drop the idea, so program cards are filtered out of the neighbor set and the
    idea is authored through the reconcile path instead.
    """
    gate, store = _FakeGate(), _FakeStore()
    program_neighbor = ProgramCard(
        id="program-7",
        program_id="7",
        description="greedy spectral exemplar",
        code="def s(): ...",
    )
    agent = _FakeAgent(
        ReconcileResponse(
            items=[
                ReconcileItem(
                    decision="NEW", card=LibrarianCard(description="widen gap")
                )
            ]
        )
    )
    ids = await _ingest(
        _lib(agent, gate, store, _FakeNeighbors([(program_neighbor, 0.0)])),
        child_id="c",
    )
    assert gate.bumped == []
    assert agent.calls == 1
    assert len(gate.admitted) == 1
    assert gate.admitted[0].description == "widen gap"
    assert ids == ["mem-new"]


@pytest.mark.asyncio
async def test_neighbor_failure_emits_event_and_still_admits(tmp_path) -> None:
    """A NeighborSource failure must be observable and must NOT drop the idea.

    Chroma can fail (cold index, disk hiccup). The librarian swallows the error
    so the write path survives, but a silent swallow hides a degraded dedup
    surface; the failure is recorded as a canonical memory event and the idea is
    still authored through the reconcile path with an empty neighbor set.
    """
    gate, store = _FakeGate(), _FakeStore()
    agent = _FakeAgent(
        ReconcileResponse(
            items=[ReconcileItem(decision="NEW", card=LibrarianCard(description="x"))]
        )
    )

    class _BoomNeighbors:
        def nearest(self, note, k, card_type):  # noqa: ANN001
            raise RuntimeError("chroma down")

    path = tmp_path / "memory_events.jsonl"
    with memory_event_context(event_path=path):
        ids = await _ingest(_lib(agent, gate, store, _BoomNeighbors()))

    assert ids == ["mem-new"]
    assert agent.calls == 1
    assert path.exists()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    assert any(r["event_type"] == "neighbor.retrieval_failed" for r in rows)


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
async def test_new_card_carries_authored_explanation_summary() -> None:
    # The librarian must thread the reconcile agent's authored explanation_summary
    # onto the admitted card so A-MEM's explanation_summary Chroma channel is fed.
    gate, store = _FakeGate(), _FakeStore()
    agent = _FakeAgent(
        ReconcileResponse(
            items=[
                ReconcileItem(
                    decision="NEW",
                    card=LibrarianCard(
                        description="clamp the update step",
                        explanation_summary="raw steps diverge where the landscape is steepest",
                    ),
                )
            ]
        )
    )
    await _ingest(_lib(agent, gate, store, _FakeNeighbors([])))
    assert len(gate.admitted) == 1
    assert (
        gate.admitted[0].explanation_summary
        == "raw steps diverge where the landscape is steepest"
    )


@pytest.mark.asyncio
async def test_author_program_cached_preserves_explanation_summary() -> None:
    gate, store = _FakeGate(), _FakeStore()
    store.card_store.cards["program-42"] = ProgramCard(
        id="program-42",
        program_id="42",
        description="already authored",
        code="def s(): ...",
        explanation_summary="why this exemplar scores",
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
    assert resp.explanation_summary == "why this exemplar scores"
    assert author.calls == 0


def _prog(pid, *, fitness=None, description="strat X") -> ProgramCard:  # noqa: ANN001
    return ProgramCard(
        id=f"program-{pid}",
        program_id=str(pid),
        description=description,
        code="def s(): ...",
        fitness=fitness,
    )


def test_admit_program_with_no_twin_admits() -> None:
    gate, store = _FakeGate(), _FakeStore()
    lib = _lib(_FakeAgent(ReconcileResponse(items=[])), gate, store, _FakeNeighbors([]))
    fid = lib.admit_program(_prog("a", fitness=0.5), higher_is_better=True)
    assert fid == "program-a"
    assert [c.id for c in gate.admitted] == ["program-a"]
    assert store.deleted == []


def test_admit_program_skips_equal_fitness_twin() -> None:
    # The strict-better contract: an equally-good same-strategy twin is kept and
    # the redundant incoming card is dropped (no churn).
    gate, store = _FakeGate(), _FakeStore()
    twin = _prog("a", fitness=0.9)
    store.card_store.cards["program-a"] = twin
    lib = _lib(
        _FakeAgent(ReconcileResponse(items=[])),
        gate,
        store,
        _FakeNeighbors([(twin, 0.0)]),
    )
    fid = lib.admit_program(_prog("b", fitness=0.9), higher_is_better=True)
    assert fid == ""
    assert gate.admitted == []
    assert store.deleted == []
    assert store.get_card("program-a") is twin


def test_admit_program_skips_worse_incoming_twin() -> None:
    gate, store = _FakeGate(), _FakeStore()
    twin = _prog("a", fitness=0.9)
    store.card_store.cards["program-a"] = twin
    lib = _lib(
        _FakeAgent(ReconcileResponse(items=[])),
        gate,
        store,
        _FakeNeighbors([(twin, 0.0)]),
    )
    fid = lib.admit_program(_prog("b", fitness=0.5), higher_is_better=True)
    assert fid == ""
    assert gate.admitted == []
    assert store.deleted == []


def test_admit_program_replaces_worse_twin_when_strictly_better() -> None:
    gate, store = _FakeGate(), _FakeStore()
    twin = _prog("a", fitness=0.5)
    store.card_store.cards["program-a"] = twin
    lib = _lib(
        _FakeAgent(ReconcileResponse(items=[])),
        gate,
        store,
        _FakeNeighbors([(twin, 0.0)]),
    )
    fid = lib.admit_program(_prog("b", fitness=0.9), higher_is_better=True)
    assert fid == "program-b"
    assert store.deleted == ["program-a"]
    assert [c.id for c in gate.admitted] == ["program-b"]


def test_admit_program_ignores_idea_card_neighbor() -> None:
    gate, store = _FakeGate(), _FakeStore()
    idea = MemoryCard(id="mem-x", description="strat X", keywords=[])
    lib = _lib(
        _FakeAgent(ReconcileResponse(items=[])),
        gate,
        store,
        _FakeNeighbors([(idea, 0.0)]),
    )
    fid = lib.admit_program(_prog("b", fitness=0.5), higher_is_better=True)
    assert fid == "program-b"
    assert [c.id for c in gate.admitted] == ["program-b"]
    assert store.deleted == []


def test_admit_program_ignores_far_program_twin() -> None:
    gate, store = _FakeGate(), _FakeStore()
    far = _prog("a", fitness=0.9, description="different strat")
    lib = _lib(
        _FakeAgent(ReconcileResponse(items=[])),
        gate,
        store,
        _FakeNeighbors([(far, 0.5)]),
    )
    fid = lib.admit_program(_prog("b", fitness=0.5), higher_is_better=True)
    assert fid == "program-b"
    assert [c.id for c in gate.admitted] == ["program-b"]
    assert store.deleted == []


def test_admit_program_readmit_same_id_is_not_a_twin() -> None:
    # Re-admitting an exemplar already in the bank must not see ITSELF as a
    # duplicate; it flows to the gate as an UPDATE, not a self-replace/skip.
    gate, store = _FakeGate(), _FakeStore()
    self_card = _prog("a", fitness=0.5)
    store.card_store.cards["program-a"] = self_card
    lib = _lib(
        _FakeAgent(ReconcileResponse(items=[])),
        gate,
        store,
        _FakeNeighbors([(self_card, 0.0)]),
    )
    fid = lib.admit_program(_prog("a", fitness=0.6), higher_is_better=True)
    assert fid == "program-a"
    assert [c.id for c in gate.admitted] == ["program-a"]
    assert store.deleted == []


def test_admit_program_lower_is_better_replaces_on_lower_fitness() -> None:
    gate, store = _FakeGate(), _FakeStore()
    twin = _prog("a", fitness=0.9)
    store.card_store.cards["program-a"] = twin
    lib = _lib(
        _FakeAgent(ReconcileResponse(items=[])),
        gate,
        store,
        _FakeNeighbors([(twin, 0.0)]),
    )
    fid = lib.admit_program(_prog("b", fitness=0.5), higher_is_better=False)
    assert fid == "program-b"
    assert store.deleted == ["program-a"]


def test_admit_program_survives_neighbor_failure() -> None:
    gate, store = _FakeGate(), _FakeStore()

    class _BoomNeighbors:
        def nearest(self, note, k, card_type):  # noqa: ANN001
            raise RuntimeError("chroma down")

    lib = _lib(_FakeAgent(ReconcileResponse(items=[])), gate, store, _BoomNeighbors())
    fid = lib.admit_program(_prog("b", fitness=0.5), higher_is_better=True)
    assert fid == "program-b"
    assert [c.id for c in gate.admitted] == ["program-b"]


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
