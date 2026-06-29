"""Behaviour of the librarian-driven IdeaTracker write path.

The tracker filters a run's programs to eligible mutation records, hands each
one to the librarian, authors program cards for the top-fitness exemplars, then
restamps gain events and runs one harm-eviction pass. These tests drive the
public ``run_increment`` with a fake librarian / gate / store so the contract is
exercised without LLM calls or disk I/O.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
import uuid

import pytest

from gigaevo.llm.agents.program_author import ProgramAuthorResponse
from gigaevo.llm.agents.task_summary import TaskSummaryResponse
from gigaevo.memory.context import ContextualGain, DecisionContext
from gigaevo.memory.ideas_tracker.card_stats import CardStatsUpdater
from gigaevo.memory.ideas_tracker.dedup_policy import DedupPolicy
from gigaevo.memory.ideas_tracker.ideas_tracker import IdeaTracker
from gigaevo.memory.ideas_tracker.write_stack import LibrarianWriteStack
from gigaevo.memory.shared_memory.models import MemoryCard
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
    base_parent: int | None = None,
    program_id: str | None = None,
) -> Program:
    metadata: dict[str, Any] = {}
    if changes is not None or base_parent is not None:
        mutation_output: dict[str, Any] = {"archetype": "exploitation"}
        if changes is not None:
            mutation_output["changes"] = changes
        if base_parent is not None:
            mutation_output["base_parent"] = base_parent
        metadata["mutation_output"] = mutation_output
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
        self._gate: Any = None

    async def ingest_idea(
        self,
        *,
        base_parent_code: str,
        child_id: str,
        child_code: str,
        note: str,
    ) -> list[str]:
        if self._ingest_error is not None:
            raise self._ingest_error
        self.ingested.append(
            {
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
            description=f"desc-{program_id}",
            keywords=[f"kw-{program_id}"],
            explanation_summary=f"why-{program_id}",
        )

    def admit_program(self, card: Any, *, higher_is_better: bool) -> Any:
        return self._gate.admit(card)


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


class _FailingAuthorLibrarian(_FakeLibrarian):
    """Raises a non-timeout error when authoring a designated exemplar (an
    API/schema failure, not a stall); every other call behaves like the plain
    fake."""

    def __init__(self, *, fail_author_program_id: str) -> None:
        super().__init__()
        self._fail_author_program_id = fail_author_program_id

    async def author_program(
        self, *, program_id: str, **kw: Any
    ) -> ProgramAuthorResponse:
        if program_id == self._fail_author_program_id:
            raise RuntimeError("program-author LLM down")
        return await super().author_program(program_id=program_id, **kw)


class _FakeCardStore:
    def __init__(self) -> None:
        self.cards: dict[str, Any] = {}


class _FakeStore:
    def __init__(self) -> None:
        self.card_store = _FakeCardStore()
        self.saved: list[Any] = []

    def get_card(self, card_id: str) -> Any:
        return self.card_store.cards.get(card_id)

    def all_cards_snapshot(self) -> dict[str, Any]:
        return dict(self.card_store.cards)

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


class _FakeStack:
    """A built write stack: ``ensure`` is a no-op and every component the
    orchestrator reads is pre-seeded, so ``run_increment`` exercises the write
    contract without an LLM or a backend."""

    def __init__(
        self,
        *,
        librarian: Any,
        store: _FakeStore,
        gate: Any,
        summary: str,
    ) -> None:
        self._librarian = librarian
        self._store = store
        self._gate = gate
        self._summary = summary

    async def ensure(self) -> None:
        return None

    @property
    def librarian(self) -> Any:
        return self._librarian

    def require_librarian(self) -> Any:
        return self._librarian

    @property
    def store(self) -> _FakeStore:
        return self._store

    @property
    def gate(self) -> Any:
        return self._gate

    @property
    def neighbors(self) -> Any:
        return None

    @property
    def consolidation_agent(self) -> Any:
        return None

    @property
    def task_description_summary(self) -> str:
        return self._summary


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
    librarian._gate = gate
    stack = _FakeStack(
        librarian=librarian,
        store=store,
        gate=gate,
        summary="Maximise min area (summary).",
    )
    tracker._stack = stack
    tracker._consolidation._stack = stack
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
        assert row["base_parent_code"] == "PARENT_CODE"
        assert row["child_code"] == "CHILD_CODE"

    def test_base_parent_honours_mutator_named_base(self) -> None:
        lib = _FakeLibrarian()
        tracker, _, _ = _make_tracker(lib)
        donor = _make_program(
            program_id="donor", code="DONOR_CODE", parents=["root"], generation=2
        )
        base = _make_program(
            program_id="base", code="BASE_CODE", parents=["root"], generation=2
        )
        child = _make_program(
            program_id="child",
            code="CHILD_CODE",
            parents=["donor", "base"],
            generation=3,
            changes=["blend donor mechanism onto base"],
            base_parent=2,
        )
        _run(tracker, [child], posterior_programs=[donor, base, child])
        row = lib.ingested[0]
        assert row["base_parent_code"] == "BASE_CODE"


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
        assert admitted[0].explanation_summary == f"why-{high.id}"
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


class TestRestamp:
    """Gain events are a pure function of the current pool, so each sweep is
    authoritative: credited cards get this sweep's events; cards no longer
    credited have stale events cleared. Only changed cards are rewritten."""

    @staticmethod
    def _gain(value: float) -> ContextualGain:
        return ContextualGain(
            context=DecisionContext(parent_metrics={"f": value}), gain=value
        )

    @staticmethod
    def _updater() -> CardStatsUpdater:
        return CardStatsUpdater(fitness_key="fitness", higher_is_better=True)

    def test_uncredited_card_has_stale_gain_events_cleared(self) -> None:
        store = _FakeStore()
        gate = _FakeGate(store)
        store.card_store.cards["mem-stale"] = MemoryCard(
            id="mem-stale", description="x", gain_events=[self._gain(0.1)]
        )
        self._updater().restamp_and_sweep({}, store=store, gate=gate)
        assert store.card_store.cards["mem-stale"].gain_events is None
        assert gate.sweeps == 1

    def test_credited_card_gets_events_and_uncredited_is_cleared(self) -> None:
        store = _FakeStore()
        gate = _FakeGate(store)
        store.card_store.cards["mem-c"] = MemoryCard(id="mem-c", description="c")
        store.card_store.cards["mem-s"] = MemoryCard(
            id="mem-s", description="s", gain_events=[self._gain(0.9)]
        )
        self._updater().restamp_and_sweep(
            {"mem-c": [self._gain(0.5)]}, store=store, gate=gate
        )
        assert store.card_store.cards["mem-c"].gain_events == [self._gain(0.5)]
        assert store.card_store.cards["mem-s"].gain_events is None

    def test_unchanged_card_is_not_rewritten(self) -> None:
        store = _FakeStore()
        gate = _FakeGate(store)
        store.card_store.cards["mem-clean"] = MemoryCard(
            id="mem-clean", description="x"
        )
        self._updater().restamp_and_sweep({}, store=store, gate=gate)
        assert store.saved == []


class TestTaskSummary:
    """The one-line task summary stamped onto every card is a genuine LLM
    condensation, with a hard fallback to the full task text so a memory-LLM
    failure never blocks the write path."""

    def test_summary_is_llm_condensation(self) -> None:
        stack = LibrarianWriteStack(
            backend=None,
            llm=_SummaryLLM("max min triangle area"),
            task_description="Place N points to maximise the minimum triangle area.",
        )
        assert asyncio.run(stack.ensure_summary()) == "max min triangle area"
        assert stack.task_description_summary == "max min triangle area"

    def test_empty_task_yields_empty_summary(self) -> None:
        stack = LibrarianWriteStack(
            backend=None,
            llm=_SummaryLLM("unused"),
            task_description="",
        )
        assert asyncio.run(stack.ensure_summary()) == ""
        assert stack.task_description_summary == ""

    def test_llm_failure_falls_back_to_full_task_text(self) -> None:
        full = "Place N points to maximise the minimum triangle area."
        stack = LibrarianWriteStack(
            backend=None,
            llm=_BoomLLM(),
            task_description=full,
        )
        assert asyncio.run(stack.ensure_summary()) == full
        assert stack.task_description_summary == full

    def test_summary_is_memoised(self) -> None:
        stack = LibrarianWriteStack(
            backend=None,
            llm=_BoomLLM(),
            task_description="some task",
        )
        stack._summary = "already condensed"
        assert asyncio.run(stack.ensure_summary()) == "already condensed"
        assert stack.task_description_summary == "already condensed"


class TestDedupPolicy:
    """The five dedup thresholds are one config object, threaded from the
    tracker into the two surfaces that apply them: the online pre-gate
    thresholds reach the Librarian, the batch thresholds reach the scheduler."""

    def test_consolidation_thresholds_thread_to_scheduler(self) -> None:
        tracker = IdeaTracker(
            llm=object(),
            backend=object(),
            memory_write_enabled=True,
            task_description="t",
            consolidation_every_n=64,
            dedup_policy=DedupPolicy(consolidation_eps=0.2, consolidation_k=9),
        )
        assert tracker._consolidation._eps == 0.2
        assert tracker._consolidation._k == 9

    def test_write_stack_uses_backend_as_neighbor_source(
        self, tmp_path, monkeypatch
    ) -> None:
        """The neighbor source IS the backend: the writer's nearest-card primitive
        is a backend contract method, not a write-path object reaching through
        ``store.memory_system``. So a new backend works with the writer unchanged."""
        import gigaevo.memory.ideas_tracker.write_stack as ws

        monkeypatch.setattr(ws, "WriteLedger", lambda path: None)
        store = SimpleNamespace(checkpoint_path=tmp_path)
        stack = LibrarianWriteStack(
            backend=lambda **kw: store,
            llm=_SummaryLLM("s"),
            task_description="t",
            dedup_policy=DedupPolicy(
                online_eps=0.2, online_top_k=9, max_cards_per_diff=1
            ),
        )
        stack._build("summary")
        assert stack.neighbors is store

    def test_online_thresholds_thread_to_librarian(self, tmp_path, monkeypatch) -> None:
        import gigaevo.memory.ideas_tracker.write_stack as ws

        monkeypatch.setattr(ws, "WriteLedger", lambda path: None)
        store = SimpleNamespace(checkpoint_path=tmp_path)
        stack = LibrarianWriteStack(
            backend=lambda **kw: store,
            llm=_SummaryLLM("s"),
            task_description="t",
            dedup_policy=DedupPolicy(
                online_eps=0.2, online_top_k=9, max_cards_per_diff=1
            ),
        )
        stack._build("summary")
        lib = stack.librarian
        assert lib._eps == 0.2
        assert lib._top_k == 9
        assert lib._max_cards == 1


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
        assert prog.id not in tracker._extractor.seen_ids

    def test_writer_disabled_skips_ingest(self) -> None:
        lib = _FakeLibrarian()
        tracker, _, _ = _make_tracker(lib)
        tracker._memory_write_enabled = False
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
        assert hung.id not in tracker._extractor.seen_ids
        assert good.id in tracker._extractor.seen_ids
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

    def test_failed_author_is_skipped_and_sweep_completes(self) -> None:
        # A non-timeout authoring error (API/schema failure, not a stall) must
        # degrade per-exemplar like the idea path — not escape and abort the
        # increment before the stats restamp and harm sweep run.
        prog = _make_program(program_id="solo")
        lib = _FailingAuthorLibrarian(fail_author_program_id=prog.id)
        tracker, gate, _ = _make_tracker(lib)

        _run(tracker, [prog])

        assert prog.id in {row["child_id"] for row in lib.ingested}
        assert not any(c.id == f"program-{prog.id}" for c in gate.admitted)
        assert gate.sweeps == 1
