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
from gigaevo.memory.cards import Card, CardKind
from gigaevo.memory.write.admission import (
    CardAdmissionGate,
    WriteOutcome,
    WriteResult,
)
from gigaevo.memory.write.eviction import NullEvictor
from gigaevo.memory.write.policies import ProgramExemplarPolicy
from gigaevo.memory.write.writer import MemoryWriter
from gigaevo.programs.metrics.context import VALIDITY_KEY, MetricsContext, MetricSpec
from gigaevo.programs.program import EXCLUDE_STAGE_RESULTS


class FakeLibrarian:
    def __init__(
        self,
        store,
        *,
        task_key: str = "",
        task_description: str = "task",
        task_description_summary: str = "task-summary",
    ) -> None:
        self._store = store
        self._task_key = task_key
        self._task_description = task_description
        self._task_description_summary = task_description_summary
        self.ingest_calls: list[dict] = []
        self.authored: list[str] = []
        self.ingest_delay_s = 0.0
        self.ingest_error: Exception | None = None

    async def ingest_idea(self, **kwargs) -> list[WriteResult]:
        if self.ingest_delay_s:
            await asyncio.sleep(self.ingest_delay_s)
        if self.ingest_error is not None:
            raise self.ingest_error
        self.ingest_calls.append(kwargs)
        return [
            WriteResult(
                outcome=WriteOutcome.ADDED,
                card_id=f"card-for-{kwargs['child_id']}",
            )
        ]

    async def ingest_program(
        self,
        *,
        program_id: str,
        code: str,
        fitness: float | None,
        archive_rank: int,
        higher_is_better: bool,
        min_fitness_gap: float,
        store_code: bool,
    ) -> WriteResult:
        del archive_rank, higher_is_better, min_fitness_gap
        self.authored.append(program_id)
        card = Card(
            kind=CardKind.PROGRAM,
            id=f"program-{program_id}",
            task_key=self._task_key,
            program_id=program_id,
            programs=(program_id,),
            task_description=self._task_description,
            task_description_summary=self._task_description_summary,
            description=f"exemplar {program_id}",
            explanation_summary="test strategy",
            fitness=fitness,
            code=code if store_code else "",
        )
        return WriteResult(outcome=WriteOutcome.ADDED, card_id=self._store.save(card))


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
    stack._gate = CardAdmissionGate(
        store=store,
        evictor=NullEvictor(),
        task_key=params.get("task_key", ""),
        min_effective_events=params.get("min_effective_events", 0.0),
    )
    stack._neighbors = store
    stack._summary = "task-summary"
    stack._librarian = FakeLibrarian(
        store,
        task_key=params.get("task_key", ""),
        task_description=params["task_description"],
        task_description_summary=stack._summary,
    )
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
    assert "Change: swapped solver" in call["mutation_report"]

    assert set(librarian.authored) == {parent.id, child.id}
    exemplar = store.get(f"program-{child.id}")
    assert exemplar is not None
    assert exemplar.kind is CardKind.PROGRAM
    assert exemplar.description == f"exemplar {child.id}"
    assert exemplar.fitness == 0.7
    assert exemplar.task_description_summary == "task-summary"
    assert exemplar.task_key == ""

    await writer.run_increment([parent, child])
    assert len(librarian.ingest_calls) == 1


async def test_writer_stamps_task_key_on_founding_event_and_program_exemplar(
    store, make_program, metrics_context, tmp_path
):
    parent = make_program(fitness=0.5, parents=[])
    child = make_program(
        fitness=0.7,
        parents=[parent.id],
        metadata={
            MUTATION_MEMORY_BASE_ID_METADATA_KEY: parent.id,
            MUTATION_MEMORY_BASE_METRICS_METADATA_KEY: {
                VALIDITY_KEY: 1.0,
                "fitness": 0.5,
            },
            MUTATION_OUTPUT_METADATA_KEY: {
                "changes": [{"description": "swapped solver", "explanation": ""}]
            },
        },
    )
    writer = make_writer(store, metrics_context, tmp_path, task_key="heilbronn")

    await writer.run_increment([parent, child])

    call = writer._stack.librarian.ingest_calls[0]
    assert call["founding_gain"].context.task_key == "heilbronn"
    assert store.get(f"program-{child.id}").task_key == "heilbronn"


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


async def test_ingest_failure_forgets_record_and_continues_for_retry(
    store, make_program, metrics_context, tmp_path
):
    child = make_program(fitness=0.7, parents=["p"])
    writer = make_writer(
        store,
        metrics_context,
        tmp_path,
        best_programs_percent=0.0,
    )
    librarian = writer._stack.librarian
    librarian.ingest_error = RuntimeError("temporary author failure")

    await writer.run_increment([child])

    assert librarian.ingest_calls == []
    assert child.id not in writer._extractor.seen_ids

    librarian.ingest_error = None
    await writer.run_increment([child])
    assert [c["child_id"] for c in librarian.ingest_calls] == [child.id]


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

        def eviction_reason(self, card) -> str:
            return "test evictor"

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


async def test_program_exemplar_policy_omits_code_by_default(
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


def test_program_exemplar_policy_prunes_to_max_cards(
    store, make_card, metrics_context, tmp_path
):
    cards = [
        make_card(
            id="program-low",
            kind=CardKind.PROGRAM,
            program_id="low",
            fitness=0.1,
        ),
        make_card(
            id="program-mid",
            kind=CardKind.PROGRAM,
            program_id="mid",
            fitness=0.5,
        ),
        make_card(
            id="program-high",
            kind=CardKind.PROGRAM,
            program_id="high",
            fitness=0.9,
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


def test_program_exemplar_prune_keeps_foreign_helpful_and_retires_cold(
    store, make_card, make_event, metrics_context, tmp_path
):
    helpful = make_card(
        id="program-helpful",
        task_key="own-task",
        kind=CardKind.PROGRAM,
        program_id="helpful",
        fitness=0.1,
        gain_events=(
            make_event(0.2, task_key="foreign-task"),
            make_event(0.1, task_key="foreign-task"),
            make_event(-0.1, task_key="foreign-task"),
        ),
    )
    cold = make_card(
        id="program-cold",
        task_key="own-task",
        kind=CardKind.PROGRAM,
        program_id="cold",
        fitness=0.2,
    )
    best = make_card(
        id="program-best",
        task_key="own-task",
        kind=CardKind.PROGRAM,
        program_id="best",
        fitness=0.9,
    )
    for card in (helpful, cold, best):
        store.save(card)
    writer = make_writer(
        store,
        metrics_context,
        tmp_path,
        task_key="own-task",
        min_effective_events=3,
        program_exemplars=ProgramExemplarPolicy(max_cards=1),
    )

    writer._prune_program_exemplars()

    assert store.get(helpful.id) == helpful
    assert store.get(cold.id) is None
    assert store.get(best.id) == best


def test_program_exemplar_cap_ignores_foreign_task_cards(
    store, make_card, metrics_context, tmp_path
):
    foreign = [
        make_card(
            id=f"foreign-{i}",
            task_key="foreign-task",
            kind=CardKind.PROGRAM,
            program_id=f"foreign-{i}",
            fitness=100.0 + i,
        )
        for i in range(3)
    ]
    own = make_card(
        id="own-only",
        task_key="own-task",
        kind=CardKind.PROGRAM,
        program_id="own-only",
        fitness=0.5,
    )
    for card in [*foreign, own]:
        store.save(card)
    writer = make_writer(
        store,
        metrics_context,
        tmp_path,
        task_key="own-task",
        program_exemplars=ProgramExemplarPolicy(max_cards=2),
    )

    writer._prune_program_exemplars()

    assert store.snapshot() == tuple(sorted([*foreign, own], key=lambda c: c.id))


def test_program_exemplar_cap_prunes_only_own_task_cards(
    store, make_card, metrics_context, tmp_path
):
    foreign = make_card(
        id="foreign-best-looking",
        task_key="foreign-task",
        kind=CardKind.PROGRAM,
        program_id="foreign-best-looking",
        fitness=-100.0,
    )
    own = [
        make_card(
            id=f"own-{label}",
            task_key="own-task",
            kind=CardKind.PROGRAM,
            program_id=f"own-{label}",
            fitness=fitness,
        )
        for label, fitness in [("low", 0.1), ("mid", 0.5), ("high", 0.9)]
    ]
    for card in [foreign, *own]:
        store.save(card)
    writer = make_writer(
        store,
        metrics_context,
        tmp_path,
        task_key="own-task",
        program_exemplars=ProgramExemplarPolicy(max_cards=2),
    )

    writer._prune_program_exemplars()

    assert store.get(foreign.id) == foreign
    assert store.get("own-low") is None
    assert store.get("own-mid") is not None
    assert store.get("own-high") is not None


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
