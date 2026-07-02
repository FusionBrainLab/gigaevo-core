"""MemoryWriter orchestration over a pre-built (faked) write stack."""

from __future__ import annotations

import asyncio

import pytest

from gigaevo.evolution.mutation.constants import (
    MUTATION_MEMORY_BASE_ID_METADATA_KEY,
    MUTATION_MEMORY_BASE_METRICS_METADATA_KEY,
    MUTATION_MEMORY_BASE_SELECTED_IDS_METADATA_KEY,
    MUTATION_OUTPUT_METADATA_KEY,
)
from gigaevo.llm.agents.program_author import ProgramAuthorResponse
from gigaevo.memory.cards import Card, CardKind
from gigaevo.memory.write.admission import CardAdmissionGate
from gigaevo.memory.write.eviction import NullEvictor
from gigaevo.memory.write.writer import MemoryWriter
from gigaevo.programs.metrics.context import VALIDITY_KEY, MetricsContext, MetricSpec
from gigaevo.programs.program import EXCLUDE_STAGE_RESULTS


class FakeLibrarian:
    def __init__(self, store) -> None:
        self._store = store
        self.ingest_calls: list[dict] = []
        self.authored: list[str] = []
        self.ingest_delay_s = 0.0

    async def ingest_idea(self, **kwargs) -> list[str]:
        if self.ingest_delay_s:
            await asyncio.sleep(self.ingest_delay_s)
        self.ingest_calls.append(kwargs)
        return [f"card-for-{kwargs['child_id']}"]

    async def author_program(
        self, *, program_id: str, code: str, fitness: float | None
    ) -> ProgramAuthorResponse:
        self.authored.append(program_id)
        return ProgramAuthorResponse(description=f"exemplar {program_id}")

    def admit_program(self, card: Card, *, higher_is_better: bool) -> str:
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
        "checkpoint_dir": tmp_path,
        "metrics_context": metrics_context,
        "task_description": "task",
        "best_programs_percent": 100.0,
    }
    params.update(overrides)
    writer = MemoryWriter(**params)
    stack = writer._stack
    stack._store = store
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
        metadata={MUTATION_OUTPUT_METADATA_KEY: {"changes": ["swapped solver"]}},
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


async def test_gain_events_restamp_from_posterior_pool(
    store, make_card, make_program, metrics_context, tmp_path
):
    credited = make_card(id="card-a")
    store.save(credited)
    parent = make_program(parents=[])
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

    await writer.run_increment([child], posterior_programs=[parent, child])

    events = store.get("card-a").gain_events
    assert len(events) == 1
    assert events[0].gain == pytest.approx(0.2)
    assert events[0].context.parent_id == parent.id


async def test_on_run_complete_empty_storage_skips_build(metrics_context, tmp_path):
    writer = MemoryWriter(
        llm=object(),
        evictor=NullEvictor(),
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


def test_unknown_fitness_key_fails_fast(metrics_context, tmp_path):
    with pytest.raises(KeyError):
        MemoryWriter(
            llm=object(),
            evictor=NullEvictor(),
            checkpoint_dir=tmp_path,
            metrics_context=metrics_context,
            fitness_key="not-a-metric",
        )


def test_direction_derives_from_metrics_context(tmp_path):
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
        checkpoint_dir=tmp_path,
        metrics_context=minimize,
        fitness_key="loss",
    )
    assert writer._higher_is_better is False
