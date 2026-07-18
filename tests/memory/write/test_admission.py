"""CardAdmissionGate verdicts and the WriteLedger audit trail."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import multiprocessing

from loguru import logger

from gigaevo.memory.cards import Card, CardKind
import gigaevo.memory.prior_evidence as prior_evidence_module
from gigaevo.memory.selection_leases import InFlightSelectionRegistry
from gigaevo.memory.write.admission import (
    CardAdmissionGate,
    WriteLedger,
    WriteLedgerRecord,
    WriteOutcome,
    WriteResult,
)
from gigaevo.memory.write.eviction import NullEvictor


class MarkingEvictor:
    """Evicts exactly the ids in ``harmful``."""

    def __init__(self, harmful: set[str]) -> None:
        self._harmful = harmful
        self.judged_cards: list[Card] = []

    def should_evict(self, card: Card) -> bool:
        self.judged_cards.append(card)
        return card.id in self._harmful

    def eviction_reason(self, card: Card) -> str:
        return "test evictor"

    def sweep(self, cards) -> list[str]:
        return [card.id for card in cards if self.should_evict(card)]


def read_rows(path) -> list[WriteLedgerRecord]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [WriteLedgerRecord.model_validate(json.loads(line)) for line in f]


def _record_ledger_rows(path, worker, count, start=None, ready=None):
    if ready is not None:
        ready.put(True)
    if start is not None:
        assert start.wait(timeout=30)
    ledger = WriteLedger(path)
    for index in range(count):
        ledger.record(
            incoming_id=f"{worker}-{index}",
            final_id=f"{worker}-{index}",
            outcome=WriteOutcome.ADDED,
            reason=f"{worker}-{index}-" + "x" * 8192,
        )


def make_gate(store, tmp_path, evictor=None):
    ledger = WriteLedger(tmp_path / "write_ledger.jsonl")
    gate = CardAdmissionGate(
        store=store, evictor=evictor or NullEvictor(), ledger=ledger
    )
    return gate, ledger


def test_admit_empty_id_mints_and_records_added(store, make_card, tmp_path):
    gate, ledger = make_gate(store, tmp_path)
    result = gate.admit(make_card(id=""))
    assert result.outcome is WriteOutcome.ADDED
    assert result.landed
    assert result.card_id.startswith("minted-")
    assert store.get(result.card_id) is not None
    rows = read_rows(ledger.path)
    assert len(rows) == 1
    assert rows[0].outcome is WriteOutcome.ADDED
    assert rows[0].final_id == result.card_id
    assert rows[0].incoming_description


def test_admit_known_id_preserves_payload_and_records_updated(
    store, make_card, tmp_path
):
    gate, ledger = make_gate(store, tmp_path)
    card = make_card(description="stable")
    store.save(card)
    result = gate.admit(card.model_copy(update={"description": "revised"}))
    assert result.outcome is WriteOutcome.UPDATED
    assert result.card_id == card.id
    assert store.get(card.id).description == "stable"
    assert read_rows(ledger.path)[-1].outcome is WriteOutcome.UPDATED


def test_task_card_cap_rejects_only_new_cards_in_same_task(store, make_card, tmp_path):
    first = make_card(id="first", task_key="task")
    second = make_card(id="second", task_key="task")
    foreign = make_card(id="foreign", task_key="other-task")
    store.save(first)
    store.save(foreign)
    ledger = WriteLedger(tmp_path / "write_ledger.jsonl")
    gate = CardAdmissionGate(
        store=store,
        evictor=NullEvictor(),
        ledger=ledger,
        task_key="task",
        max_task_cards=1,
    )

    rejected = gate.admit(second)
    updated = gate.admit(first.model_copy(update={"description": "updated"}))

    assert rejected.outcome is WriteOutcome.REJECTED_CAPACITY
    assert not rejected.landed
    assert store.get(second.id) is None
    assert updated.outcome is WriteOutcome.UPDATED
    assert store.get(first.id).description == first.description
    row = read_rows(ledger.path)[0]
    assert row.outcome is WriteOutcome.REJECTED_CAPACITY
    assert row.reason == "active task card cap reached (1)"


def test_task_card_cap_must_be_positive(store):
    try:
        CardAdmissionGate(
            store=store,
            evictor=NullEvictor(),
            max_task_cards=0,
        )
    except ValueError as exc:
        assert str(exc) == "max_task_cards must be positive when configured"
    else:
        raise AssertionError("expected invalid card cap to be rejected")


def test_sweep_does_not_delete_card_selected_by_in_flight_mutation(
    store, make_card, tmp_path
):
    card = make_card()
    store.save(card)
    registry = InFlightSelectionRegistry()
    lease = registry.open_attempt("attempt-1", "parent-1")
    selected_ids = lease.attach_cards((card.id,))
    gate = CardAdmissionGate(
        store=store,
        evictor=MarkingEvictor({card.id}),
        selection_leases=registry,
    )

    gate.sweep()

    assert selected_ids == (card.id,)
    assert store.get(card.id) is not None
    lease.release()
    assert gate.sweep() == [card.id]


def test_sweep_protects_a_leased_historical_alias(store, make_card):
    card = make_card(absorbed_ids=("historical-id",))
    store.save(card)
    registry = InFlightSelectionRegistry()
    lease = registry.open_attempt("attempt-1", "parent-1")
    lease.attach_cards(("historical-id",))
    gate = CardAdmissionGate(
        store=store,
        evictor=MarkingEvictor({card.id}),
        selection_leases=registry,
    )

    assert gate.sweep() == []
    assert store.get(card.id) == card
    lease.release()
    assert gate.sweep() == [card.id]


def test_retire_exemplar_without_task_scope_keeps_single_task_behavior(
    store, make_card, make_event
):
    card = make_card(
        kind=CardKind.PROGRAM,
        program_id="p1",
        code="x = 1",
        gain_events=tuple(make_event(0.2, task_key="foreign-task") for _ in range(3)),
    )
    store.save(card)
    gate = CardAdmissionGate(store=store, evictor=NullEvictor())

    result = gate.retire_exemplar(card, reason="single-task prune")

    assert result.card_id == ""
    assert result.outcome is WriteOutcome.RETIRED
    assert store.get(card.id) is None


def test_gate_without_selection_registry_still_sweeps(store, make_card):
    card = make_card()
    store.save(card)
    gate = CardAdmissionGate(store=store, evictor=MarkingEvictor({card.id}))

    assert gate.sweep() == [card.id]
    assert store.get(card.id) is None


def test_tombstoned_program_id_does_not_block_bare_insight_id(
    store, make_card, tmp_path
):
    # Exemplar cache ids live in the "program-<id>" namespace; retiring
    # one must not tombstone the bare insight id it embeds.
    exemplar = make_card(
        id="program-p1", kind=CardKind.PROGRAM, program_id="p1", code="x = 1"
    )
    store.save(exemplar)
    harmful = {exemplar.id}
    gate, _ = make_gate(store, tmp_path, MarkingEvictor(harmful))
    assert gate.sweep() == [exemplar.id]
    harmful.clear()
    result = gate.admit(make_card(id="p1"))
    assert result.outcome is WriteOutcome.ADDED
    assert store.get("p1") is not None


def test_equivalent_insight_pools_provenance_and_evidence_without_rewriting(
    store, make_card, make_event, tmp_path
):
    target_event = make_event(0.1)
    incoming_event = make_event(0.2)
    target = make_card(
        task_key="task",
        description="stable action",
        programs=("parent",),
        gain_events=(target_event,),
    )
    store.save(target)
    gate, ledger = make_gate(store, tmp_path)
    incoming = make_card(
        id="",
        task_key="task",
        description="different wording",
        programs=("child",),
        gain_events=(incoming_event,),
    )

    result = gate.update_equivalent(target.id, incoming)
    updated = store.get(target.id)

    assert result.outcome is WriteOutcome.UPDATED
    assert updated.description == "stable action"
    assert updated.programs == ("parent", "child")
    assert updated.gain_events == (target_event, incoming_event)
    row = read_rows(ledger.path)[-1]
    assert row.duplicate_of == target.id
    assert row.incoming_description == "different wording"


def test_equivalent_update_requires_same_kind_and_task(store, make_card, tmp_path):
    program = make_card(
        task_key="task",
        kind=CardKind.PROGRAM,
        program_id="p1",
        code="x = 1",
        fitness=0.5,
    )
    store.save(program)
    gate, _ = make_gate(store, tmp_path)
    insight = make_card(task_key="task", programs=("child",))
    foreign = program.model_copy(
        update={"id": "program-p2", "task_key": "foreign", "program_id": "p2"}
    )

    assert gate.update_equivalent("absent", insight).outcome is WriteOutcome.DISCARDED
    assert gate.update_equivalent(program.id, insight).outcome is WriteOutcome.DISCARDED
    assert gate.update_equivalent(program.id, foreign).outcome is WriteOutcome.DISCARDED


def test_sweep_deletes_evicted_and_records(store, make_card, tmp_path):
    bad = make_card()
    good = make_card()
    store.save(bad)
    store.save(good)
    gate, ledger = make_gate(store, tmp_path, MarkingEvictor({bad.id}))
    assert gate.sweep() == [bad.id]
    assert store.get(bad.id) is None
    assert store.get(good.id) is not None
    row = read_rows(ledger.path)[-1]
    assert row.outcome is WriteOutcome.EVICTED
    assert row.incoming_id == bad.id


def test_sweep_honors_foreign_task_help_veto(store, make_card, make_event, tmp_path):
    card = make_card(
        task_key="current",
        gain_events=tuple(make_event(0.2, task_key="foreign") for _ in range(3)),
    )
    store.save(card)
    gate = CardAdmissionGate(
        store=store,
        evictor=MarkingEvictor({card.id}),
        task_key="current",
        min_effective_events=2.0,
    )

    assert gate.sweep() == []
    assert store.get(card.id) == card


def test_sweep_captures_harm_evicted_card_evidence(
    store, make_card, make_event, tmp_path
):
    from gigaevo.memory.prior_evidence import JsonlEvictedEvidence

    harmful = make_card(gain_events=(make_event(-1.0),))
    store.save(harmful)
    evidence = JsonlEvictedEvidence(tmp_path / "prior_evidence.jsonl")
    gate = CardAdmissionGate(
        store=store,
        evictor=MarkingEvictor({harmful.id}),
        evicted_evidence_sink=evidence,
    )

    assert gate.sweep() == [harmful.id]
    assert store.get(harmful.id) is None
    assert evidence.cards() == (harmful,)


def test_sweep_default_none_keeps_behavior_and_creates_no_evidence_file(
    store, make_card, tmp_path
):
    harmful = make_card()
    survivor = make_card()
    store.save(harmful)
    store.save(survivor)
    evidence_path = tmp_path / "prior_evidence.jsonl"
    gate = CardAdmissionGate(
        store=store,
        evictor=MarkingEvictor({harmful.id}),
    )

    assert gate.sweep() == [harmful.id]
    assert store.get(harmful.id) is None
    assert store.get(survivor.id) == survivor
    assert not evidence_path.exists()


def test_sweep_rescues_card_restamped_after_snapshot(
    store, make_card, make_event, tmp_path
):
    card = make_card()
    rescue_event = make_event(0.5)
    store.save(card)

    class RestampingEvictor:
        def should_evict(self, candidate: Card) -> bool:
            return not candidate.gain_events

        def eviction_reason(self, candidate: Card) -> str:
            del candidate
            return "no gain evidence"

        def sweep(self, cards) -> list[str]:
            stale = [
                candidate.id for candidate in cards if self.should_evict(candidate)
            ]
            store.save(card.model_copy(update={"gain_events": (rescue_event,)}))
            return stale

    gate, ledger = make_gate(store, tmp_path, RestampingEvictor())
    records: list = []
    handler = logger.add(records.append)
    try:
        evicted = gate.sweep()
    finally:
        logger.remove(handler)

    assert evicted == []
    assert store.get(card.id).gain_events == (rescue_event,)
    assert not gate.is_tombstoned(card.id)
    assert read_rows(ledger.path) == []
    assert any(card.id in str(record) and "rescu" in str(record) for record in records)


def test_sweep_tombstones_id_against_readmission(store, make_card, tmp_path):
    card = make_card()
    store.save(card)
    harmful = {card.id}
    gate, ledger = make_gate(store, tmp_path, MarkingEvictor(harmful))
    assert gate.sweep() == [card.id]
    assert gate.is_tombstoned(card.id)

    # A re-authored twin carries no harm evidence, so the evictor alone would
    # wave it straight back in — the tombstone must be what blocks it.
    harmful.clear()
    result = gate.admit(card.model_copy(update={"description": "re-authored twin"}))
    assert result.rejected_retired
    assert store.get(card.id) is None
    rows = read_rows(ledger.path)
    assert [r.outcome for r in rows] == [
        WriteOutcome.EVICTED,
        WriteOutcome.REJECTED_RETIRED,
    ]

    exact_reauthor = gate.admit(card.model_copy(update={"id": ""}))
    assert exact_reauthor.outcome is WriteOutcome.REJECTED_RETIRED

    fresh = gate.admit(make_card(id=""))
    assert fresh.outcome is WriteOutcome.ADDED


def test_reject_novelty_records_row_and_does_not_bank(store, make_card, tmp_path):
    gate, ledger = make_gate(store, tmp_path)
    result = gate.reject_novelty(make_card(id=""), "prior-known staple")
    assert result.outcome is WriteOutcome.REJECTED_NOVELTY
    assert not result.landed
    assert not result.rejected_retired
    assert not result.benign_noop
    assert store.snapshot() == ()
    row = read_rows(ledger.path)[-1]
    assert row.outcome is WriteOutcome.REJECTED_NOVELTY
    assert row.reason == "prior-known staple"
    assert row.final_id == ""


def test_write_result_benign_noop_is_discarded_only():
    # Only DISCARDED is safe to re-author as fresh. Every other non-landed
    # verdict is a harm-driven deletion the librarian must drop — EVICTED must
    # never read as a benign no-op or a swept-harmful card resurrects as NEW.
    landed = {WriteOutcome.ADDED, WriteOutcome.UPDATED}
    for outcome in WriteOutcome:
        card_id = "x" if outcome in landed else ""
        result = WriteResult(outcome=outcome, card_id=card_id)
        assert result.benign_noop is (outcome is WriteOutcome.DISCARDED)
        assert result.rejected_retired is (outcome is WriteOutcome.REJECTED_RETIRED)
        assert result.landed is (outcome in landed)
    assert not WriteResult(outcome=WriteOutcome.EVICTED).benign_noop


def test_ledger_record_never_raises_on_unwritable_path(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    ledger = WriteLedger(blocker / "ledger.jsonl")
    ledger.record(incoming_id="a", final_id="a", outcome=WriteOutcome.ADDED)


def test_ledger_concurrent_threads_and_processes_write_complete_rows(tmp_path):
    path = tmp_path / "write_ledger.jsonl"
    context = multiprocessing.get_context("fork")
    start = context.Event()
    ready = context.Queue()
    process_count = 3
    thread_count = 8
    rows_per_worker = 8
    processes = [
        context.Process(
            target=_record_ledger_rows,
            args=(path, f"process-{index}", rows_per_worker, start, ready),
        )
        for index in range(process_count)
    ]
    for process in processes:
        process.start()
    for _ in processes:
        assert ready.get(timeout=30) is True

    with ThreadPoolExecutor(max_workers=thread_count) as executor:
        futures = [
            executor.submit(
                _record_ledger_rows,
                path,
                f"thread-{index}",
                rows_per_worker,
                start,
            )
            for index in range(thread_count)
        ]
        start.set()
        for future in futures:
            future.result(timeout=30)
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0

    rows = read_rows(path)
    expected_count = (process_count + thread_count) * rows_per_worker
    assert len(rows) == expected_count
    assert len({row.incoming_id for row in rows}) == expected_count


def test_ledger_record_logs_and_swallows_lock_failure(tmp_path, monkeypatch):
    def fail_lock(*_args):
        raise OSError("lock unavailable")

    monkeypatch.setattr(prior_evidence_module.fcntl, "flock", fail_lock)
    records: list = []
    handler = logger.add(records.append)
    try:
        WriteLedger(tmp_path / "write_ledger.jsonl").record(
            incoming_id="a", final_id="a", outcome=WriteOutcome.ADDED
        )
    finally:
        logger.remove(handler)

    assert any(
        "failed to record" in str(record) and "lock unavailable" in str(record)
        for record in records
    )
