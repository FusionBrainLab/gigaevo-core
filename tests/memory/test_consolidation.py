"""Behavior tests for the periodic consolidation pass.

``consolidate`` runs the same NeighborSource nearest-card primitive over the
whole bank and folds each near-dup pair into a single canonical card via the
gate. We fake every collaborator and assert on the bank state and the merge
count, never on internal calls. The fake NeighborSource is faithful: it ranks
the *live* store (deleted cards disappear) and encodes proximity in a group
prefix on the description, so two cards collapse iff they share a group.
"""

from __future__ import annotations

import pytest

from gigaevo.llm.agents.consolidate_cards import ConsolidateDecision
from gigaevo.llm.agents.reconcile import LibrarianCard
from gigaevo.memory.context import ContextualGain, DecisionContext
from gigaevo.memory.ideas_tracker.consolidation import consolidate
from gigaevo.memory.shared_memory.card_merge import merge_cards
from gigaevo.memory.shared_memory.models import CardT, MemoryCard


def _gain(value: float) -> ContextualGain:
    return ContextualGain(
        context=DecisionContext(parent_metrics={"f": value}), gain=value
    )


class _FakeCardStore:
    def __init__(self) -> None:
        self.cards: dict[str, MemoryCard] = {}


class _FakeStore:
    def __init__(self) -> None:
        self.card_store = _FakeCardStore()

    def get_card(self, card_id: str) -> MemoryCard | None:
        return self.card_store.cards.get(card_id)

    def all_cards_snapshot(self) -> dict[str, MemoryCard]:
        return dict(self.card_store.cards)

    def delete(self, card_id: str) -> bool:
        return self.card_store.cards.pop(card_id, None) is not None


class _FakeGate:
    """Twin of CardAdmissionGate.merge: field-preserving union onto the target.

    Records the submitted card's id as the real gate does (its ledger
    ``incoming_id`` — what happened to the SUBMITTED card, not the target).
    """

    def __init__(self, store: _FakeStore) -> None:
        self._store = store
        self.merged: list[tuple[str, MemoryCard]] = []
        self.submitted_ids: list[str] = []

    def merge(self, target_id: str, card: MemoryCard) -> str:
        target = self._store.card_store.cards.get(target_id)
        if target is None:
            return ""
        self.submitted_ids.append(card.id)
        merged = merge_cards(target, card, replace_description=True)
        self._store.card_store.cards[target_id] = merged
        self.merged.append((target_id, merged))
        return target_id


class _HarmGate:
    """Twin of CardAdmissionGate.merge when the union is harmful: evicts the
    target and returns "" without committing — never touches the partner."""

    def __init__(self, store: _FakeStore) -> None:
        self._store = store

    def merge(self, target_id: str, card: MemoryCard) -> str:
        self._store.delete(target_id)
        return ""


class _RaiseOnDeleteGate:
    """Twin of CardAdmissionGate.merge on the harmful-union path: it evicts the
    target via ``store.delete`` and that delete RAISES (e.g. an API HTTP error)
    before any fold. No partner evidence reached a survivor, so the partner must
    survive even though merge raised."""

    def __init__(self, store: _FakeStore) -> None:
        self._store = store

    def merge(self, target_id: str, card: MemoryCard) -> str:
        raise RuntimeError("backend delete failed on harmful-union eviction")


class _FakeNeighbors:
    """Ranks the live store by a group prefix on the description.

    Same group -> near (0.01); different group -> far (0.9). Mirrors the real
    ChromaNeighborSource: ranks every card incl. the query itself, ascending.
    """

    def __init__(self, store: _FakeStore) -> None:
        self._store = store

    def nearest(
        self, note: str, k: int, card_type: type[CardT]
    ) -> list[tuple[CardT, float]]:
        group = note.split(":", 1)[0]
        scored = [
            (c, 0.01 if (c.description or "").split(":", 1)[0] == group else 0.9)
            for c in self._store.card_store.cards.values()
            if isinstance(c, card_type) and (c.description or "").strip()
        ]
        scored.sort(key=lambda pair: pair[1])
        return scored[:k]


class _FakeMergeAgent:
    """Always MERGEs: synthesizes union prose, keeping the survivor's group prefix."""

    def __init__(self) -> None:
        self.calls = 0

    async def arun(self, *, card_a, card_b):  # noqa: ANN001
        self.calls += 1
        group = (card_a.description or "").split(":", 1)[0]
        return ConsolidateDecision(
            merge=True,
            card=LibrarianCard(
                description=f"{group}: union prose",
                keywords=["union"],
                explanation_summary=f"{group}: union why",
            ),
        )


class _AbstainAgent:
    """Always abstains: the two surfaced candidates are not the same lever."""

    def __init__(self) -> None:
        self.calls = 0

    async def arun(self, *, card_a, card_b):  # noqa: ANN001
        self.calls += 1
        return ConsolidateDecision(merge=False, card=None)


class _AbstainOnIdAgent:
    """MERGEs any pair EXCEPT one touching ``loner_id`` — that card drifted into
    eps proximity of the cluster but names a distinct lever, so the agent refuses
    to fold it with anyone. Lets a test surface a distinct closest candidate and
    assert the pass moves on to fold a genuine farther duplicate."""

    def __init__(self, loner_id: str) -> None:
        self.calls = 0
        self._loner = loner_id

    async def arun(self, *, card_a, card_b):  # noqa: ANN001
        self.calls += 1
        if self._loner in (card_a.id, card_b.id):
            return ConsolidateDecision(merge=False, card=None)
        group = (card_a.description or "").split(":", 1)[0]
        return ConsolidateDecision(
            merge=True,
            card=LibrarianCard(description=f"{group}: union prose", keywords=["union"]),
        )


class _FailOnNthMergeAgent:
    """Twin of _FakeMergeAgent that raises on the Nth call (LLM mid-pass death)."""

    def __init__(self, fail_on: int) -> None:
        self.calls = 0
        self._fail_on = fail_on

    async def arun(self, *, card_a, card_b):  # noqa: ANN001
        self.calls += 1
        if self.calls >= self._fail_on:
            raise RuntimeError("merge llm down")
        group = (card_a.description or "").split(":", 1)[0]
        return ConsolidateDecision(
            merge=True,
            card=LibrarianCard(description=f"{group}: union prose", keywords=["union"]),
        )


def _card(cid: str, group: str, programs: list[str] | None = None) -> MemoryCard:
    return MemoryCard(
        id=cid,
        description=f"{group}: lever {cid}",
        keywords=[group],
        programs=programs or [cid],
    )


def _stack(*cards: MemoryCard):
    store = _FakeStore()
    for c in cards:
        store.card_store.cards[c.id] = c
    return store, _FakeGate(store), _FakeNeighbors(store), _FakeMergeAgent()


@pytest.mark.asyncio
async def test_two_near_dups_collapse_into_one_canonical_card() -> None:
    store, gate, neighbors, agent = _stack(
        _card("mem-a", "g1", ["p1"]), _card("mem-b", "g1", ["p2"])
    )
    merges = await consolidate(store=store, gate=gate, neighbors=neighbors, agent=agent)
    assert merges == 1
    assert agent.calls == 1
    assert "mem-b" not in store.card_store.cards
    survivor = store.card_store.cards["mem-a"]
    assert survivor.description == "g1: union prose"
    assert set(survivor.programs) == {"p1", "p2"}


@pytest.mark.asyncio
async def test_consolidation_records_absorbed_partner_id_as_merge_incoming() -> None:
    # The write ledger reports what happened to the SUBMITTED card via its id
    # (the gate's incoming_id). The card folded away is the partner, so the
    # submitted card must carry the partner's id — else the deleted partner has
    # no ledger/replay trace and the row reads as the survivor merged into itself.
    survivor = _card("mem-a", "g1", ["p1"])
    partner = _card("mem-b", "g1", ["p2"])
    store, gate, neighbors, agent = _stack(survivor, partner)
    merges = await consolidate(store=store, gate=gate, neighbors=neighbors, agent=agent)
    assert merges == 1
    assert "mem-b" not in store.card_store.cards  # partner absorbed + deleted
    assert "mem-a" in store.card_store.cards  # survivor kept
    assert gate.submitted_ids == ["mem-b"]


@pytest.mark.asyncio
async def test_consolidation_unions_gain_events_onto_survivor() -> None:
    a = _card("mem-a", "g1", ["p1"]).model_copy(update={"gain_events": [_gain(0.1)]})
    b = _card("mem-b", "g1", ["p2"]).model_copy(update={"gain_events": [_gain(0.2)]})
    store, gate, neighbors, agent = _stack(a, b)
    merges = await consolidate(store=store, gate=gate, neighbors=neighbors, agent=agent)
    assert merges == 1
    survivor = store.card_store.cards["mem-a"]
    assert survivor.gain_events == [_gain(0.1), _gain(0.2)]


@pytest.mark.asyncio
async def test_consolidation_trusts_agent_curated_keywords() -> None:
    # The consolidate agent authors the union card's keywords as the curated
    # few-most-distinctive set across both cards. The fold must take that set
    # verbatim, not re-union it with both cards' raw keyword lists — re-unioning
    # would re-bloat the survivor and undo the de-duplication the prompt asks for.
    a = _card("mem-a", "g1", ["p1"])
    b = _card("mem-b", "g1", ["p2"]).model_copy(update={"keywords": ["g1", "extra-kw"]})
    store, gate, neighbors, agent = _stack(a, b)
    merges = await consolidate(store=store, gate=gate, neighbors=neighbors, agent=agent)
    assert merges == 1
    survivor = store.card_store.cards["mem-a"]
    assert survivor.keywords == ["union"]


@pytest.mark.asyncio
async def test_consolidation_threads_union_explanation_summary_onto_survivor() -> None:
    # The union card's explanation_summary must reach the survivor so the merged
    # card keeps feeding A-MEM's explanation_summary Chroma channel.
    store, gate, neighbors, agent = _stack(
        _card("mem-a", "g1", ["p1"]), _card("mem-b", "g1", ["p2"])
    )
    merges = await consolidate(store=store, gate=gate, neighbors=neighbors, agent=agent)
    assert merges == 1
    survivor = store.card_store.cards["mem-a"]
    assert survivor.explanation_summary == "g1: union why"


@pytest.mark.asyncio
async def test_consolidation_preserves_partner_task_fields_when_survivor_lacks_them() -> (
    None
):
    # The survivor (kept card) may carry empty task provenance while the absorbed
    # partner has it populated; the fold must not drop the partner's task fields.
    survivor = _card("mem-a", "g1", ["p1"])  # task fields default to ""
    partner = _card("mem-b", "g1", ["p2"]).model_copy(
        update={
            "task_description": "maximize area under constraints",
            "task_description_summary": "max area",
        }
    )
    store, gate, neighbors, agent = _stack(survivor, partner)
    merges = await consolidate(store=store, gate=gate, neighbors=neighbors, agent=agent)
    assert merges == 1
    kept = store.card_store.cards["mem-a"]
    assert kept.task_description == "maximize area under constraints"
    assert kept.task_description_summary == "max area"


@pytest.mark.asyncio
async def test_harmful_union_does_not_delete_the_partner() -> None:
    store = _FakeStore()
    for c in (_card("mem-a", "g1", ["p1"]), _card("mem-b", "g1", ["p2"])):
        store.card_store.cards[c.id] = c
    gate, neighbors, agent = _HarmGate(store), _FakeNeighbors(store), _FakeMergeAgent()
    merges = await consolidate(store=store, gate=gate, neighbors=neighbors, agent=agent)
    assert merges == 0
    assert "mem-b" in store.card_store.cards
    assert "mem-a" not in store.card_store.cards


@pytest.mark.asyncio
async def test_single_card_bank_is_a_no_op() -> None:
    store, gate, neighbors, agent = _stack(_card("mem-a", "g1"))
    merges = await consolidate(store=store, gate=gate, neighbors=neighbors, agent=agent)
    assert merges == 0
    assert agent.calls == 0
    assert "mem-a" in store.card_store.cards


@pytest.mark.asyncio
async def test_distinct_cards_within_eps_guard_are_not_merged() -> None:
    store, gate, neighbors, agent = _stack(_card("mem-a", "g1"), _card("mem-b", "g2"))
    merges = await consolidate(store=store, gate=gate, neighbors=neighbors, agent=agent)
    assert merges == 0
    assert agent.calls == 0
    assert set(store.card_store.cards) == {"mem-a", "mem-b"}


@pytest.mark.asyncio
async def test_agent_abstain_keeps_both_candidates() -> None:
    # Two cards drift within eps but the agent rules they name distinct levers
    # (merge=False). The eps gate only recalls candidates; the agent is the
    # precision arbiter, so a loosened candidate eps can never force-fold them.
    store, gate, neighbors, _ = _stack(
        _card("mem-a", "g1", ["p1"]), _card("mem-b", "g1", ["p2"])
    )
    agent = _AbstainAgent()
    merges = await consolidate(store=store, gate=gate, neighbors=neighbors, agent=agent)
    assert merges == 0
    assert agent.calls == 1
    assert set(store.card_store.cards) == {"mem-a", "mem-b"}


@pytest.mark.asyncio
async def test_closest_candidate_abstain_folds_next_genuine_duplicate() -> None:
    # mem-b is a distinct lever that drifted into eps proximity of the g1 cluster.
    # When folding mem-a, the closest candidate may be mem-b; the agent abstains,
    # and the pass must move on to fold the genuine duplicate mem-c rather than
    # giving up on mem-a after one declined candidate.
    store = _FakeStore()
    for c in (
        _card("mem-a", "g1", ["p1"]),
        _card("mem-b", "g1", ["p2"]),
        _card("mem-c", "g1", ["p3"]),
    ):
        store.card_store.cards[c.id] = c
    gate, neighbors = _FakeGate(store), _FakeNeighbors(store)
    agent = _AbstainOnIdAgent("mem-b")
    merges = await consolidate(store=store, gate=gate, neighbors=neighbors, agent=agent)
    assert merges == 1
    assert "mem-c" not in store.card_store.cards
    assert set(store.card_store.cards) == {"mem-a", "mem-b"}


@pytest.mark.asyncio
async def test_second_run_on_deduped_bank_returns_zero() -> None:
    store, gate, neighbors, agent = _stack(
        _card("mem-a", "g1", ["p1"]), _card("mem-b", "g1", ["p2"])
    )
    first = await consolidate(store=store, gate=gate, neighbors=neighbors, agent=agent)
    second = await consolidate(store=store, gate=gate, neighbors=neighbors, agent=agent)
    assert first == 1
    assert second == 0
    assert set(store.card_store.cards) == {"mem-a"}


@pytest.mark.asyncio
async def test_midpass_failure_still_deletes_committed_partners() -> None:
    """A merge-LLM failure on a later pair must not orphan partners already
    folded into survivors earlier in the pass. The absorbed partner's evidence
    is already on the survivor; leaving the partner in the bank double-counts it.
    """
    store = _FakeStore()
    for c in (
        _card("mem-a", "g1", ["p1"]),
        _card("mem-b", "g1", ["p2"]),
        _card("mem-c", "g2", ["p3"]),
        _card("mem-d", "g2", ["p4"]),
    ):
        store.card_store.cards[c.id] = c
    gate, neighbors = _FakeGate(store), _FakeNeighbors(store)
    agent = _FailOnNthMergeAgent(fail_on=2)

    with pytest.raises(RuntimeError):
        await consolidate(store=store, gate=gate, neighbors=neighbors, agent=agent)

    assert "mem-b" not in store.card_store.cards


@pytest.mark.asyncio
async def test_harm_path_delete_raise_does_not_orphan_the_partner() -> None:
    """gate.merge() raises only from the harmful-union eviction ``delete``, before
    any fold (``apply_merges`` swallows persist failures and never raises). No
    partner evidence reached a survivor, so the partner must survive — queuing it
    for deletion before the merge commits would orphan it on this raise.
    """
    store = _FakeStore()
    for c in (_card("mem-a", "g1", ["p1"]), _card("mem-b", "g1", ["p2"])):
        store.card_store.cards[c.id] = c
    gate, neighbors, agent = (
        _RaiseOnDeleteGate(store),
        _FakeNeighbors(store),
        _FakeMergeAgent(),
    )

    with pytest.raises(RuntimeError):
        await consolidate(store=store, gate=gate, neighbors=neighbors, agent=agent)

    assert "mem-b" in store.card_store.cards


@pytest.mark.asyncio
async def test_consumed_neighbors_do_not_starve_a_valid_partner_in_one_pass() -> None:
    """A fixed top-k crowded by self + already-consumed cards must not hide a
    valid same-lever partner at rank k+1. After a+b merge, c's top-k can be
    [self, a(consumed), b(consumed)] with its true partner d beyond the cutoff;
    a single pass must still fold c+d, not defer it to a later (maybe absent) run.
    """
    store = _FakeStore()
    for c in (
        _card("mem-a", "g1", ["p1"]),
        _card("mem-b", "g1", ["p2"]),
        _card("mem-c", "g1", ["p3"]),
        _card("mem-d", "g1", ["p4"]),
    ):
        store.card_store.cards[c.id] = c
    gate, neighbors, agent = _FakeGate(store), _FakeNeighbors(store), _FakeMergeAgent()
    merges = await consolidate(
        store=store, gate=gate, neighbors=neighbors, agent=agent, k=2
    )
    assert merges == 2
    assert len(store.card_store.cards) == 2


@pytest.mark.asyncio
async def test_three_mutual_near_dups_collapse_to_one() -> None:
    store, gate, neighbors, agent = _stack(
        _card("mem-a", "g1", ["p1"]),
        _card("mem-b", "g1", ["p2"]),
        _card("mem-c", "g1", ["p3"]),
    )
    first = await consolidate(store=store, gate=gate, neighbors=neighbors, agent=agent)
    # a single pass folds pairwise; idempotent re-runs converge to one card.
    while await consolidate(store=store, gate=gate, neighbors=neighbors, agent=agent):
        pass
    assert first >= 1
    assert len(store.card_store.cards) == 1
