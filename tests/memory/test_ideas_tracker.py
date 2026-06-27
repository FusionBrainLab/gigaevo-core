"""Behaviour of the librarian-driven IdeaTracker write path.

The tracker filters a run's programs to eligible mutation records, hands each
one to the librarian, authors program cards for the top-fitness exemplars, then
restamps gain events and runs one harm-eviction pass. These tests drive the
public ``run_increment`` with a fake librarian / gate / store so the contract is
exercised without LLM calls or disk I/O.
"""

from __future__ import annotations

import asyncio
from typing import Any
import uuid

import pytest

from gigaevo.llm.agents.program_author import ProgramAuthorResponse
from gigaevo.llm.agents.task_summary import TaskSummaryResponse
from gigaevo.memory.ideas_tracker.ideas_tracker import IdeaTracker
from gigaevo.programs.program import Lineage, Program
from gigaevo.programs.program_state import ProgramState

_NS = uuid.NAMESPACE_DNS


def _uid(name: str) -> str:
    return str(uuid.uuid5(_NS, name))


def _make_program(
    *,
    code: str = "def solve(): return 42",
    fitness: float = 0.75,
    is_valid: float = 1.0,
    generation: int = 3,
    parents: list[str] | None = None,
    changes: list[str] | None = None,
    program_id: str | None = None,
) -> Program:
    metadata: dict[str, Any] = {}
    if changes is not None:
        metadata["mutation_output"] = {"archetype": "exploitation", "changes": changes}
    parent_list = (
        parents if parents is not None else (["parent-1"] if generation > 1 else [])
    )
    lineage = Lineage(
        parents=[_uid(p) for p in parent_list], generation=max(generation, 1)
    )
    prog = Program(
        code=code,
        state=ProgramState.DONE,
        metrics={"fitness": fitness, "is_valid": is_valid},
        metadata=metadata,
        lineage=lineage,
    )
    if program_id is not None:
        object.__setattr__(prog, "id", _uid(program_id))
    return prog


class _FakeLibrarian:
    def __init__(self, *, ingest_error: BaseException | None = None) -> None:
        self.ingested: list[dict[str, str]] = []
        self.authored: list[str] = []
        self._ingest_error = ingest_error

    async def ingest_idea(
        self,
        *,
        base_parent_id: str,
        base_parent_code: str,
        child_id: str,
        child_code: str,
        note: str,
    ) -> list[str]:
        if self._ingest_error is not None:
            raise self._ingest_error
        self.ingested.append(
            {
                "base_parent_id": base_parent_id,
                "base_parent_code": base_parent_code,
                "child_id": child_id,
                "child_code": child_code,
                "note": note,
            }
        )
        return [f"card-{child_id}"]

    async def author_program(
        self, *, program_id: str, code: str, fitness: float
    ) -> ProgramAuthorResponse:
        self.authored.append(program_id)
        return ProgramAuthorResponse(
            description=f"desc-{program_id}", keywords=[f"kw-{program_id}"]
        )


class _HangingLibrarian(_FakeLibrarian):
    """Hangs (sleeps past any test timeout) on a designated ingest child or
    authored program; every other call behaves like the plain fake."""

    def __init__(
        self,
        *,
        hang_ingest_child_id: str | None = None,
        hang_author_program_id: str | None = None,
    ) -> None:
        super().__init__()
        self._hang_ingest_child_id = hang_ingest_child_id
        self._hang_author_program_id = hang_author_program_id

    async def ingest_idea(self, *, child_id: str, **kw: Any) -> list[str]:
        if child_id == self._hang_ingest_child_id:
            await asyncio.sleep(3600)
        return await super().ingest_idea(child_id=child_id, **kw)

    async def author_program(
        self, *, program_id: str, **kw: Any
    ) -> ProgramAuthorResponse:
        if program_id == self._hang_author_program_id:
            await asyncio.sleep(3600)
        return await super().author_program(program_id=program_id, **kw)


class _FakeCardStore:
    def __init__(self) -> None:
        self.cards: dict[str, Any] = {}


class _FakeStore:
    def __init__(self) -> None:
        self.card_store = _FakeCardStore()
        self.saved: list[Any] = []

    def save_card_direct(self, card: Any) -> Any:
        self.saved.append(card)
        self.card_store.cards[card.id] = card
        return card


class _FakeGate:
    def __init__(self, store: _FakeStore) -> None:
        self._store = store
        self.admitted: list[Any] = []
        self.sweeps = 0

    def admit(self, card: Any) -> None:
        self.admitted.append(card)
        self._store.card_store.cards[card.id] = card

    def sweep(self) -> None:
        self.sweeps += 1


def _make_tracker(
    librarian: _FakeLibrarian, *, ingest_call_timeout_s: float = 120.0
) -> tuple[IdeaTracker, _FakeGate, _FakeStore]:
    tracker = IdeaTracker(
        llm=object(),
        backend=object(),
        memory_write_enabled=True,
        memory_write_best_programs_percent=5.0,
        fitness_higher_is_better=True,
        task_description="Maximise the minimum triangle area.",
        ingest_call_timeout_s=ingest_call_timeout_s,
    )
    store = _FakeStore()
    gate = _FakeGate(store)
    tracker._store = store
    tracker._gate = gate
    tracker._librarian = librarian
    tracker._task_description_summary = "Maximise min area (summary)."
    return tracker, gate, store


def _run(tracker: IdeaTracker, programs: list[Program], **kw: Any) -> None:
    asyncio.run(tracker.run_increment(programs, **kw))


class TestEligibility:
    def test_one_ingest_per_evolved_program(self) -> None:
        lib = _FakeLibrarian()
        tracker, _, _ = _make_tracker(lib)
        evolved = [_make_program(program_id="a"), _make_program(program_id="b")]
        _run(tracker, evolved)
        assert {row["child_id"] for row in lib.ingested} == {_uid("a"), _uid("b")}

    def test_root_program_is_not_ingested(self) -> None:
        lib = _FakeLibrarian()
        tracker, _, _ = _make_tracker(lib)
        root = _make_program(parents=[], generation=1, program_id="root")
        _run(tracker, [root])
        assert lib.ingested == []

    def test_invalid_program_is_not_ingested(self) -> None:
        lib = _FakeLibrarian()
        tracker, _, _ = _make_tracker(lib)
        invalid = _make_program(is_valid=0.0, program_id="bad")
        _run(tracker, [invalid])
        assert lib.ingested == []

    def test_already_seen_program_not_reingested(self) -> None:
        lib = _FakeLibrarian()
        tracker, _, _ = _make_tracker(lib)
        prog = _make_program(program_id="a")
        _run(tracker, [prog])
        _run(tracker, [prog])
        assert len(lib.ingested) == 1


class TestNote:
    def test_note_joins_mutation_changes(self) -> None:
        lib = _FakeLibrarian()
        tracker, _, _ = _make_tracker(lib)
        prog = _make_program(program_id="a", changes=["Vectorise H2", "Drop max clamp"])
        _run(tracker, [prog])
        assert lib.ingested[0]["note"] == "Vectorise H2; Drop max clamp"

    def test_note_falls_back_when_no_changes(self) -> None:
        lib = _FakeLibrarian()
        tracker, _, _ = _make_tracker(lib)
        prog = _make_program(program_id="a", changes=None)
        _run(tracker, [prog])
        assert lib.ingested[0]["note"] == "Unspecified change"


class TestParentResolution:
    def test_parent_code_resolves_from_posterior_pool(self) -> None:
        lib = _FakeLibrarian()
        tracker, _, _ = _make_tracker(lib)
        parent = _make_program(
            program_id="seed", code="PARENT_CODE", parents=["root"], generation=2
        )
        child = _make_program(
            program_id="child", code="CHILD_CODE", parents=["seed"], generation=3
        )
        _run(tracker, [child], posterior_programs=[parent, child])
        row = lib.ingested[0]
        assert row["base_parent_id"] == _uid("seed")
        assert row["base_parent_code"] == "PARENT_CODE"
        assert row["child_code"] == "CHILD_CODE"


class TestExemplars:
    def test_top_fitness_exemplar_is_authored_and_admitted(self) -> None:
        lib = _FakeLibrarian()
        tracker, gate, _ = _make_tracker(lib)
        low = _make_program(program_id="low", fitness=0.1)
        high = _make_program(program_id="high", fitness=9.0)
        _run(tracker, [low, high])
        assert lib.authored == [high.id]
        admitted = [c for c in gate.admitted if c.id == f"program-{high.id}"]
        assert len(admitted) == 1
        assert admitted[0].description == f"desc-{high.id}"
        assert admitted[0].keywords == [f"kw-{high.id}"]
        assert admitted[0].fitness == 9.0


class _SummaryStructured:
    def __init__(self, summary: str) -> None:
        self._summary = summary
        self.calls: list = []

    async def ainvoke(self, messages: Any) -> TaskSummaryResponse:
        self.calls.append(messages)
        return TaskSummaryResponse(summary=self._summary)


class _SummaryLLM:
    def __init__(self, summary: str) -> None:
        self._structured = _SummaryStructured(summary)

    def with_structured_output(self, schema: Any, **kw: Any) -> _SummaryStructured:
        return self._structured


class _BoomStructured:
    async def ainvoke(self, messages: Any) -> TaskSummaryResponse:
        raise RuntimeError("summary llm down")


class _BoomLLM:
    def with_structured_output(self, schema: Any, **kw: Any) -> _BoomStructured:
        return _BoomStructured()


class TestTaskSummary:
    """The one-line task summary stamped onto every card is a genuine LLM
    condensation, with a hard fallback to the full task text so a memory-LLM
    failure never blocks the write path."""

    def test_summary_is_llm_condensation(self) -> None:
        tracker = IdeaTracker(
            llm=_SummaryLLM("max min triangle area"),
            memory_write_enabled=False,
            task_description="Place N points to maximise the minimum triangle area.",
        )
        asyncio.run(tracker._ensure_task_summary())
        assert tracker._task_description_summary == "max min triangle area"

    def test_empty_task_yields_empty_summary(self) -> None:
        tracker = IdeaTracker(
            llm=_SummaryLLM("unused"),
            memory_write_enabled=False,
            task_description="placeholder",
        )
        tracker._task_description = ""
        asyncio.run(tracker._ensure_task_summary())
        assert tracker._task_description_summary == ""

    def test_llm_failure_falls_back_to_full_task_text(self) -> None:
        full = "Place N points to maximise the minimum triangle area."
        tracker = IdeaTracker(
            llm=_BoomLLM(),
            memory_write_enabled=False,
            task_description=full,
        )
        asyncio.run(tracker._ensure_task_summary())
        assert tracker._task_description_summary == full

    def test_summary_is_memoised(self) -> None:
        tracker = IdeaTracker(
            llm=_BoomLLM(),
            memory_write_enabled=False,
            task_description="some task",
        )
        tracker._task_description_summary = "already condensed"
        asyncio.run(tracker._ensure_task_summary())
        assert tracker._task_description_summary == "already condensed"


class TestSweep:
    def test_harm_sweep_runs_once_per_increment(self) -> None:
        lib = _FakeLibrarian()
        tracker, gate, _ = _make_tracker(lib)
        _run(tracker, [_make_program(program_id="a")])
        assert gate.sweeps == 1


class TestRollback:
    def test_cancelled_ingest_unsees_record_for_retry(self) -> None:
        lib = _FakeLibrarian(ingest_error=asyncio.CancelledError())
        tracker, _, _ = _make_tracker(lib)
        prog = _make_program(program_id="a")
        with pytest.raises(asyncio.CancelledError):
            _run(tracker, [prog])
        assert prog.id not in tracker._seen_ids

    def test_writer_disabled_skips_ingest(self) -> None:
        tracker = IdeaTracker(memory_write_enabled=False)
        lib = _FakeLibrarian()
        tracker._librarian = lib
        _run(tracker, [_make_program(program_id="a")])
        assert lib.ingested == []


class TestStalledCallIsolation:
    """A single stalled memory-LLM call must not freeze the whole sweep: the
    timed-out record is skipped (and unseen for retry) while the rest of the
    increment — sibling ingests, exemplar authoring, harm sweep — completes."""

    def test_hung_ingest_is_skipped_and_sweep_completes(self) -> None:
        good = _make_program(program_id="good")
        hung = _make_program(program_id="hung")
        lib = _HangingLibrarian(hang_ingest_child_id=hung.id)
        tracker, gate, _ = _make_tracker(lib, ingest_call_timeout_s=0.05)

        _run(tracker, [good, hung])

        ingested = {row["child_id"] for row in lib.ingested}
        assert good.id in ingested
        assert hung.id not in ingested
        assert hung.id not in tracker._seen_ids
        assert good.id in tracker._seen_ids
        assert lib.authored  # the increment reached exemplar authoring
        assert gate.sweeps == 1  # ...and ran the harm-eviction pass

    def test_hung_author_is_skipped_and_sweep_completes(self) -> None:
        prog = _make_program(program_id="solo")
        lib = _HangingLibrarian(hang_author_program_id=prog.id)
        tracker, gate, _ = _make_tracker(lib, ingest_call_timeout_s=0.05)

        _run(tracker, [prog])

        assert prog.id in {row["child_id"] for row in lib.ingested}
        assert not any(c.id == f"program-{prog.id}" for c in gate.admitted)
        assert gate.sweeps == 1
