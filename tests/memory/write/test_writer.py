"""MemoryWriter orchestration over a pre-built (faked) write stack."""

from __future__ import annotations

import asyncio
import threading

import pytest

from gigaevo.evolution.mutation.constants import (
    MUTATION_MEMORY_BASE_ID_METADATA_KEY,
    MUTATION_MEMORY_BASE_METRICS_METADATA_KEY,
    MUTATION_MEMORY_BASE_SELECTED_IDS_METADATA_KEY,
    MUTATION_OUTPUT_METADATA_KEY,
)
from gigaevo.llm.agents.program_author import ProgramAuthorResponse
from gigaevo.memory.cards import Card, CardKind
from gigaevo.memory.write.admission import (
    CardAdmissionGate,
    WriteOutcome,
    WriteResult,
)
from gigaevo.memory.write.eviction import NullEvictor
from gigaevo.memory.write.librarian import code_sha256
from gigaevo.memory.write.merge import ProgramExemplarPolicy
from gigaevo.memory.write.writer import MemoryWriter
from gigaevo.programs.metrics.context import VALIDITY_KEY, MetricsContext, MetricSpec
from gigaevo.programs.program import EXCLUDE_STAGE_RESULTS


class FakeLibrarian:
    def __init__(self, store) -> None:
        self._store = store
        self.ingest_calls: list[dict] = []
        self.authored: list[str] = []
        self.ingest_delay_s = 0.0

    async def ingest_idea(self, **kwargs) -> list[WriteResult]:
        if self.ingest_delay_s:
            await asyncio.sleep(self.ingest_delay_s)
        self.ingest_calls.append(kwargs)
        return [
            WriteResult(
                outcome=WriteOutcome.ADDED,
                card_id=f"card-for-{kwargs['child_id']}",
            )
        ]

    async def author_program(
        self, *, program_id: str, code: str, fitness: float | None
    ) -> ProgramAuthorResponse:
        self.authored.append(program_id)
        return ProgramAuthorResponse(description=f"exemplar {program_id}")

    def admit_program(self, card: Card, *, higher_is_better: bool, **kwargs) -> str:
        return self._store.save(card)


class FakeProgramStorage:
    def __init__(self, programs) -> None:
        self._programs = programs
        self.get_all_calls: list[dict] = []

    async def get_all(self, exclude=None):
        self.get_all_calls.append({"exclude": exclude})
        return list(self._programs)


def make_writer(store, metrics_context, tmp_path, **overrides) -> MemoryWriter:
    params = {
        "llm": object(),
        "evictor": NullEvictor(),
        "store": store,
        "checkpoint_dir": tmp_path,
        "metrics_context": metrics_context,
        "task_description": "task",
        "best_programs_percent": 100.0,
    }
    params.update(overrides)
    writer = MemoryWriter(**params)
    stack = writer._stack
    stack._gate = CardAdmissionGate(store=store, evictor=NullEvictor())
    stack._neighbors = store
    stack._librarian = FakeLibrarian(store)
    stack._consolidation_agent = object()
    stack._summary = "task-summary"
    return writer


async def test_run_increment_ingests_and_authors_exemplars(
    store, make_program, metrics_context, tmp_path
):
    parent = make_program(parents=[])
    child = make_program(
        fitness=0.7,
        parents=[parent.id],
        metadata={
            MUTATION_OUTPUT_METADATA_KEY: {
                "changes": [{"description": "swapped solver", "explanation": ""}]
            }
        },
    )
    writer = make_writer(store, metrics_context, tmp_path)
    librarian = writer._stack.librarian

    await writer.run_increment([parent, child])

    assert len(librarian.ingest_calls) == 1
    call = librarian.ingest_calls[0]
    assert call["child_id"] == child.id
    assert call["base_parent_code"] == parent.code
    assert call["note"] == "swapped solver"

    assert set(librarian.authored) == {parent.id, child.id}
    exemplar = store.get(f"program-{child.id}")
    assert exemplar is not None
    assert exemplar.kind is CardKind.PROGRAM
    assert exemplar.description == f"exemplar {child.id}"
    assert exemplar.fitness == 0.7
    assert exemplar.task_description_summary == "task-summary"

    await writer.run_increment([parent, child])
    assert len(librarian.ingest_calls) == 1


async def test_run_increment_folds_same_batch_duplicate_ideas(
    store, make_program, metrics_context, tmp_path
):
    # Two co-batch children whose ideas are the same lever: the librarian banks a
    # real card for each, and the inline intra-batch consolidation must fold them
    # to one before the increment returns — no waiting for the every_n cadence.
    from types import SimpleNamespace

    from gigaevo.llm.agents.reconcile import LibrarianCard
    from gigaevo.memory.storage.base import ScoredCard

    class DupLibrarian:
        def __init__(self, store) -> None:
            self._store = store

        async def ingest_idea(self, *, child_id, **kwargs) -> list[WriteResult]:
            cid = f"card-{child_id}"
            self._store.save(
                Card(
                    id=cid,
                    kind=CardKind.INSIGHT,
                    description="same lever",
                    explanation_summary="why",
                )
            )
            return [WriteResult(outcome=WriteOutcome.ADDED, card_id=cid)]

        async def author_program(self, *, program_id, code, fitness):
            return ProgramAuthorResponse(description=f"exemplar {program_id}")

        def admit_program(self, card, *, higher_is_better, **kwargs):
            return self._store.save(card)

    class MergingAgent:
        async def arun(self, *, card_a, card_b):
            return SimpleNamespace(
                merge=True,
                card=LibrarianCard(
                    description="union", explanation_summary="u", keywords=["u"]
                ),
            )

    def note(text: str) -> dict:
        return {MUTATION_OUTPUT_METADATA_KEY: {"changes": [{"description": text}]}}

    p1 = make_program(fitness=0.7, parents=["p"], metadata=note("lever A"))
    p2 = make_program(fitness=0.6, parents=["p"], metadata=note("lever A"))
    writer = make_writer(
        store,
        metrics_context,
        tmp_path,
        consolidation_every_n=2,
        best_programs_percent=0.0,
    )
    writer._stack._librarian = DupLibrarian(store)
    writer._stack._consolidation_agent = MergingAgent()

    def nearest(text, k, kind=None):
        insights = [c for c in store.snapshot() if c.kind is CardKind.INSIGHT]
        return [ScoredCard(card=c, distance=0.1) for c in insights][:k]

    store.nearest = nearest

    await writer.run_increment([p1, p2])
    await writer._consolidation.drain()

    insights = [c for c in store.snapshot() if c.kind is CardKind.INSIGHT]
    assert len(insights) == 1


async def test_consolidation_inline_subset_is_freshly_added_only(
    store, make_program, metrics_context, tmp_path, monkeypatch
):
    # A MERGED target was already arbitrated by the reconcile agent this
    # increment and a rejected card never landed — feeding either back into the
    # inline consolidation pass re-pays the arbiter for cards that need no
    # fold. Only genuinely fresh ADDED cards can be the unseen half of a
    # same-batch duplicate pair. Both landed writes still count toward the
    # whole-bank cadence.
    class MixedLibrarian:
        async def ingest_idea(self, *, child_id, **kwargs) -> list[WriteResult]:
            return [
                WriteResult(outcome=WriteOutcome.ADDED, card_id=f"new-{child_id}"),
                WriteResult(outcome=WriteOutcome.MERGED, card_id="old-target"),
                WriteResult(outcome=WriteOutcome.REJECTED_NOVELTY),
            ]

    child = make_program(fitness=0.7, parents=["p"])
    writer = make_writer(store, metrics_context, tmp_path, best_programs_percent=0.0)
    writer._stack._librarian = MixedLibrarian()
    subsets: list[set] = []
    notes: list[int] = []

    async def spy(ids, *, timeout=None):
        subsets.append(set(ids))

    monkeypatch.setattr(writer._consolidation, "consolidate_written", spy)
    monkeypatch.setattr(writer._consolidation, "note_writes", notes.append)

    await writer.run_increment([child])

    assert subsets == [{f"new-{child.id}"}]
    assert notes == [2]


async def test_ingest_timeout_forgets_record_for_retry(
    store, make_program, metrics_context, tmp_path
):
    child = make_program(fitness=0.7, parents=["p"])
    writer = make_writer(
        store,
        metrics_context,
        tmp_path,
        ingest_call_timeout_s=0.01,
        best_programs_percent=0.0,
    )
    librarian = writer._stack.librarian
    librarian.ingest_delay_s = 5.0

    await writer.run_increment([child])

    assert librarian.ingest_calls == []
    assert child.id not in writer._extractor.seen_ids

    librarian.ingest_delay_s = 0.0
    writer._ingest_call_timeout_s = 30.0
    await writer.run_increment([child])
    assert [c["child_id"] for c in librarian.ingest_calls] == [child.id]


@pytest.mark.parametrize(
    ("outcomes", "expected_notes"),
    [
        # All-MERGED: both landed on existing targets → cadence counts 2,
        # but nothing is freshly ADDED so the inline subset stays empty.
        (
            [
                WriteResult(outcome=WriteOutcome.MERGED, card_id="t1"),
                WriteResult(outcome=WriteOutcome.MERGED, card_id="t2"),
            ],
            2,
        ),
        # All-rejected: nothing landed → cadence must not advance.
        (
            [
                WriteResult(outcome=WriteOutcome.REJECTED_HARM),
                WriteResult(outcome=WriteOutcome.REJECTED_NOVELTY),
            ],
            0,
        ),
    ],
)
async def test_consolidation_accounting_counts_landed_not_returned(
    store,
    make_program,
    metrics_context,
    tmp_path,
    monkeypatch,
    outcomes,
    expected_notes,
):
    class OutcomeLibrarian:
        async def ingest_idea(self, **kwargs) -> list[WriteResult]:
            return list(outcomes)

    child = make_program(fitness=0.7, parents=["p"])
    writer = make_writer(store, metrics_context, tmp_path, best_programs_percent=0.0)
    writer._stack._librarian = OutcomeLibrarian()
    subsets: list[set] = []
    notes: list[int] = []

    async def spy(ids, *, timeout=None):
        subsets.append(set(ids))

    monkeypatch.setattr(writer._consolidation, "consolidate_written", spy)
    monkeypatch.setattr(writer._consolidation, "note_writes", notes.append)

    await writer.run_increment([child])

    assert subsets == [set()]
    assert notes == [expected_notes]


async def test_ingest_timeout_still_counts_partially_banked_cards(
    store, make_program, metrics_context, tmp_path, monkeypatch
):
    # wait_for cancels a hung ingest AFTER the librarian may have already
    # routed cards through the gate: the return value dies with the coroutine
    # but the banked cards are in the store. They must still feed the inline
    # consolidation subset and the cadence counter, else both silently
    # under-count exactly on the slowest (most timeout-prone) calls.
    class PartialLibrarian:
        async def ingest_idea(self, *, sink=None, **kwargs) -> list[WriteResult]:
            sink.append(WriteResult(outcome=WriteOutcome.ADDED, card_id="banked-1"))
            await asyncio.sleep(60)
            return []

    child = make_program(fitness=0.7, parents=["p"])
    writer = make_writer(
        store,
        metrics_context,
        tmp_path,
        ingest_call_timeout_s=0.05,
        best_programs_percent=0.0,
    )
    writer._stack._librarian = PartialLibrarian()
    subsets: list[set] = []
    notes: list[int] = []

    async def spy(ids, *, timeout=None):
        subsets.append(set(ids))

    monkeypatch.setattr(writer._consolidation, "consolidate_written", spy)
    monkeypatch.setattr(writer._consolidation, "note_writes", notes.append)

    await writer.run_increment([child])

    assert subsets == [{"banked-1"}]
    assert notes == [1]
    assert child.id not in writer._extractor.seen_ids


async def test_cancelled_stats_restamp_holds_writer_lock_until_thread_finishes(
    store, make_program, metrics_context, tmp_path, monkeypatch
):
    writer = make_writer(store, metrics_context, tmp_path, best_programs_percent=0.0)
    first_child = make_program(fitness=0.7, parents=["p"])
    second_child = make_program(fitness=0.8, parents=["p"])
    started = threading.Event()
    release = threading.Event()
    calls: list[str] = []

    def blocking_update(*args, **kwargs):
        calls.append("start")
        started.set()
        release.wait(timeout=5.0)
        calls.append("finish")

    monkeypatch.setattr(writer._stats, "update", blocking_update)

    first = asyncio.create_task(writer.run_increment([first_child]))
    assert await asyncio.to_thread(started.wait, 2.0)
    first.cancel()
    await asyncio.sleep(0.05)
    assert not first.done()

    second = asyncio.create_task(writer.run_increment([second_child]))
    await asyncio.sleep(0.05)
    assert calls == ["start"]

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await first
    await second
    assert calls == ["start", "finish", "start", "finish"]


async def test_tombstoned_exemplar_never_repays_author_llm(
    store, make_card, make_program, metrics_context, tmp_path
):
    class SetEvictor:
        def __init__(self, harmful: set[str]) -> None:
            self._harmful = harmful

        def should_evict(self, card) -> bool:
            return card.id in self._harmful

        def sweep(self, cards) -> list[str]:
            return [c.id for c in cards if self.should_evict(c)]

    good = make_program(fitness=0.9, parents=[])
    bad = make_program(fitness=0.7, parents=[])
    writer = make_writer(store, metrics_context, tmp_path)
    harmful = {f"program-{bad.id}"}
    gate = CardAdmissionGate(store=store, evictor=SetEvictor(harmful))
    writer._stack._gate = gate
    store.save(
        make_card(
            id=f"program-{bad.id}",
            kind=CardKind.PROGRAM,
            program_id=bad.id,
            code="x = 1",
            fitness=0.2,
        )
    )
    assert gate.sweep() == [f"program-{bad.id}"]
    harmful.clear()

    await writer.run_increment([good, bad])

    librarian = writer._stack.librarian
    assert good.id in librarian.authored
    assert bad.id not in librarian.authored
    assert store.get(f"program-{bad.id}") is None


async def test_program_exemplar_policy_caps_authored_per_refresh(
    store, make_program, metrics_context, tmp_path
):
    programs = [make_program(fitness=0.5 + i * 0.01, parents=[]) for i in range(5)]
    writer = make_writer(
        store,
        metrics_context,
        tmp_path,
        program_exemplars=ProgramExemplarPolicy(top_k_per_refresh=2, max_cards=10),
    )
    librarian = writer._stack.librarian

    await writer.run_increment(programs)

    expected = [
        prog.id
        for prog, _ in metrics_context.top_valid_programs(
            programs, key="fitness", percent=100.0
        )[:2]
    ]
    assert librarian.authored == expected
    banked = [c.program_id for c in store.snapshot() if c.kind is CardKind.PROGRAM]
    assert sorted(banked) == sorted(expected)


async def test_program_exemplar_policy_stores_hash_not_code_by_default(
    store, make_program, metrics_context, tmp_path
):
    program = make_program(fitness=0.8, parents=[], code="def solve():\n    return 1\n")
    writer = make_writer(
        store,
        metrics_context,
        tmp_path,
        program_exemplars=ProgramExemplarPolicy(top_k_per_refresh=1),
    )

    await writer.run_increment([program])

    card = store.get(f"program-{program.id}")
    assert card is not None
    assert card.code == ""
    assert card.code_sha256 == code_sha256(program.code)


def test_program_exemplar_policy_prunes_to_max_cards(
    store, make_card, metrics_context, tmp_path
):
    cards = [
        make_card(
            id="program-low",
            kind=CardKind.PROGRAM,
            program_id="low",
            fitness=0.1,
            code_sha256="low",
        ),
        make_card(
            id="program-mid",
            kind=CardKind.PROGRAM,
            program_id="mid",
            fitness=0.5,
            code_sha256="mid",
        ),
        make_card(
            id="program-high",
            kind=CardKind.PROGRAM,
            program_id="high",
            fitness=0.9,
            code_sha256="high",
        ),
    ]
    for card in cards:
        store.save(card)
    writer = make_writer(
        store,
        metrics_context,
        tmp_path,
        program_exemplars=ProgramExemplarPolicy(max_cards=2),
    )

    writer._prune_program_exemplars()

    assert store.get("program-low") is None
    assert store.get("program-mid") is not None
    assert store.get("program-high") is not None


async def test_gain_events_restamp_from_posterior_pool(
    store, make_card, make_program, metrics_context, tmp_path
):
    credited = make_card(id="card-a")
    store.save(credited)
    parent = make_program(parents=[])
    no_card_child = make_program(
        fitness=0.5,
        parents=[parent.id],
        metadata={
            MUTATION_MEMORY_BASE_METRICS_METADATA_KEY: {
                VALIDITY_KEY: 1.0,
                "fitness": 0.5,
            },
            MUTATION_MEMORY_BASE_ID_METADATA_KEY: parent.id,
        },
    )
    child = make_program(
        fitness=0.7,
        parents=[parent.id],
        metadata={
            MUTATION_MEMORY_BASE_SELECTED_IDS_METADATA_KEY: ["card-a"],
            MUTATION_MEMORY_BASE_METRICS_METADATA_KEY: {
                VALIDITY_KEY: 1.0,
                "fitness": 0.5,
            },
            MUTATION_MEMORY_BASE_ID_METADATA_KEY: parent.id,
            MUTATION_OUTPUT_METADATA_KEY: {"card_ids_used": ["card-a"]},
        },
    )
    writer = make_writer(store, metrics_context, tmp_path, best_programs_percent=0.0)

    await writer.run_increment(
        [child], posterior_programs=[parent, no_card_child, child]
    )

    events = store.get("card-a").gain_events
    assert len(events) == 1
    assert events[0].gain == pytest.approx(0.2)
    assert events[0].context.parent_id == parent.id


async def test_on_run_complete_empty_storage_skips_build(
    store, metrics_context, tmp_path
):
    writer = MemoryWriter(
        llm=object(),
        evictor=NullEvictor(),
        store=store,
        checkpoint_dir=tmp_path,
        metrics_context=metrics_context,
    )
    storage = FakeProgramStorage([])
    await writer.on_run_complete(storage)
    assert storage.get_all_calls == [{"exclude": EXCLUDE_STAGE_RESULTS}]
    assert writer._stack.librarian is None


async def test_on_run_complete_runs_final_sweep(
    store, make_program, metrics_context, tmp_path
):
    child = make_program(fitness=0.7, parents=["p"])
    writer = make_writer(store, metrics_context, tmp_path, best_programs_percent=0.0)
    storage = FakeProgramStorage([child])
    await writer.on_run_complete(storage)
    librarian = writer._stack.librarian
    assert [c["child_id"] for c in librarian.ingest_calls] == [child.id]


def test_unknown_fitness_key_fails_fast(store, metrics_context, tmp_path):
    with pytest.raises(KeyError):
        MemoryWriter(
            llm=object(),
            evictor=NullEvictor(),
            store=store,
            checkpoint_dir=tmp_path,
            metrics_context=metrics_context,
            fitness_key="not-a-metric",
        )


def test_direction_derives_from_metrics_context(store, tmp_path):
    minimize = MetricsContext(
        specs={
            "loss": MetricSpec(
                description="loss", higher_is_better=False, is_primary=True
            )
        }
    )
    writer = MemoryWriter(
        llm=object(),
        evictor=NullEvictor(),
        store=store,
        checkpoint_dir=tmp_path,
        metrics_context=minimize,
        fitness_key="loss",
    )
    assert writer._higher_is_better is False


def test_fitness_key_defaults_to_primary_metric(store, tmp_path):
    # An omitted fitness_key must resolve to the task's primary metric, not a
    # literal "fitness" — else gain attribution silently zeroes on any task whose
    # primary key differs.
    ctx = MetricsContext(
        specs={
            "r2": MetricSpec(description="r2", higher_is_better=True, is_primary=True)
        }
    )
    writer = MemoryWriter(
        llm=object(),
        evictor=NullEvictor(),
        store=store,
        checkpoint_dir=tmp_path,
        metrics_context=ctx,
    )
    assert writer._fitness_key == "r2"
