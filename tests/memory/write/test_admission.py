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


def test_admit_known_id_records_updated(store, make_card, tmp_path):
    gate, ledger = make_gate(store, tmp_path)
    card = make_card()
    store.save(card)
    result = gate.admit(card.model_copy(update={"description": "revised"}))
    assert result.outcome is WriteOutcome.UPDATED
    assert result.card_id == card.id
    assert store.get(card.id).description == "revised"
    assert read_rows(ledger.path)[-1].outcome is WriteOutcome.UPDATED


def test_admit_harmful_known_card_revalidates_union_and_deletes(
    store, make_card, make_event, tmp_path
):
    banked_event = make_event(0.2, parent_id="banked")
    submitted_event = make_event(-0.8, parent_id="submitted")
    card = make_card(gain_events=(banked_event,))
    store.save(card)
    submitted = card.model_copy(update={"gain_events": (submitted_event,)})
    evictor = MarkingEvictor({card.id})
    gate, ledger = make_gate(store, tmp_path, evictor)

    result = gate.admit(submitted)

    assert result.rejected_harm
    assert result.card_id == ""
    assert not result.landed
    assert store.get(card.id) is None
    assert evictor.judged_cards[0].gain_events == (banked_event, submitted_event)
    row = read_rows(ledger.path)[-1]
    assert row.outcome is WriteOutcome.REJECTED_HARM
    assert row.final_id == ""


def test_admit_harmful_unknown_card_does_not_delete(store, make_card, tmp_path):
    card = make_card()
    gate, _ = make_gate(store, tmp_path, MarkingEvictor({card.id}))
    assert gate.admit(card).rejected_harm
    assert store.deleted_ids == []


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


def test_admit_does_not_harm_delete_leased_known_card(store, make_card):
    card = make_card()
    store.save(card)
    registry = InFlightSelectionRegistry()
    lease = registry.open_attempt("attempt-1", "parent-1")
    lease.attach_cards((card.id,))
    gate = CardAdmissionGate(
        store=store,
        evictor=MarkingEvictor({card.id}),
        selection_leases=registry,
    )

    result = gate.admit(card)

    assert result.rejected_harm
    assert store.get(card.id) == card
    assert not gate.is_tombstoned(card.id)


def test_merge_missing_target_is_benign_noop(store, make_card, tmp_path):
    gate, ledger = make_gate(store, tmp_path)
    result = gate.merge("absent", make_card())
    assert result.outcome is WriteOutcome.DISCARDED
    assert not result.landed
    assert not result.rejected_harm
    assert read_rows(ledger.path) == []


def test_merge_program_target_is_benign_noop(store, make_card, tmp_path):
    target = make_card(
        kind=CardKind.PROGRAM, program_id="p1", code="x = 1", fitness=0.5
    )
    store.save(target)
    gate, _ = make_gate(store, tmp_path)
    result = gate.merge(target.id, make_card())
    assert result.outcome is WriteOutcome.DISCARDED
    assert not result.rejected_harm


def test_merge_success_records_submitted_card_id(store, make_card, tmp_path):
    target = make_card()
    store.save(target)
    incoming = make_card()
    gate, ledger = make_gate(store, tmp_path)
    result = gate.merge(target.id, incoming)
    assert result.outcome is WriteOutcome.MERGED
    assert result.card_id == target.id
    row = read_rows(ledger.path)[-1]
    assert row.outcome is WriteOutcome.MERGED
    assert row.incoming_id == incoming.id
    assert row.final_id == target.id
    assert row.merge_targets == (target.id,)
    assert store.get(target.id).description == incoming.description


def test_merge_atomically_retires_banked_unleased_partner(store, make_card, tmp_path):
    target = make_card()
    partner = make_card()
    store.save(target)
    store.save(partner)
    gate, _ = make_gate(store, tmp_path)

    result = gate.merge(target.id, partner)

    assert result.outcome is WriteOutcome.MERGED
    assert store.get(target.id) is not None
    assert store.get(partner.id) is None


def test_merge_skips_banked_leased_partner_before_fold(store, make_card):
    target = make_card()
    partner = make_card()
    store.save(target)
    store.save(partner)
    registry = InFlightSelectionRegistry()
    lease = registry.open_attempt("attempt-1", "parent-1")
    lease.attach_cards((partner.id,))
    gate = CardAdmissionGate(
        store=store, evictor=NullEvictor(), selection_leases=registry
    )

    result = gate.merge(target.id, partner)

    assert result.outcome is WriteOutcome.DISCARDED
    assert store.get(target.id) == target
    assert store.get(partner.id) == partner


def test_merge_harmful_union_deletes_target(store, make_card, tmp_path):
    target = make_card()
    store.save(target)
    incoming = make_card()
    gate, ledger = make_gate(store, tmp_path, MarkingEvictor({target.id}))
    result = gate.merge(target.id, incoming)
    assert result.rejected_harm
    assert result.card_id == ""
    assert store.get(target.id) is None
    row = read_rows(ledger.path)[-1]
    assert row.outcome is WriteOutcome.REJECTED_HARM
    assert row.incoming_id == incoming.id


def test_merge_does_not_harm_delete_leased_target(store, make_card):
    target = make_card()
    incoming = make_card()
    store.save(target)
    registry = InFlightSelectionRegistry()
    lease = registry.open_attempt("attempt-1", "parent-1")
    lease.attach_cards((target.id,))
    gate = CardAdmissionGate(
        store=store,
        evictor=MarkingEvictor({target.id}),
        selection_leases=registry,
    )

    result = gate.merge(target.id, incoming)

    assert result.outcome is WriteOutcome.DISCARDED
    assert store.get(target.id) == target
    assert not gate.is_tombstoned(target.id)


def test_twin_retirement_skips_leased_program_card(store, make_card):
    twin = make_card(
        task_key="own-task",
        kind=CardKind.PROGRAM,
        program_id="old-program",
        code="x = 1",
        fitness=0.5,
    )
    store.save(twin)
    registry = InFlightSelectionRegistry()
    lease = registry.open_attempt("attempt-1", "parent-1")
    lease.attach_cards((twin.id,))
    gate = CardAdmissionGate(
        store=store, evictor=NullEvictor(), selection_leases=registry
    )

    gate.retire_twin(twin, successor_id="program-new")

    assert store.get(twin.id) == twin


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
    assert result.outcome is WriteOutcome.UPDATED
    assert store.get(card.id) is None


def test_gate_without_registry_keeps_legacy_eviction_behavior(store, make_card):
    card = make_card()
    store.save(card)
    gate = CardAdmissionGate(store=store, evictor=MarkingEvictor({card.id}))

    assert gate.sweep() == [card.id]
    assert store.get(card.id) is None


def test_merge_harmful_union_deletes_banked_incoming(store, make_card, tmp_path):
    target = make_card()
    incoming = make_card()
    store.save(target)
    store.save(incoming)
    gate, ledger = make_gate(store, tmp_path, MarkingEvictor({target.id}))

    result = gate.merge(target.id, incoming)

    assert result.rejected_harm
    assert store.get(target.id) is None
    assert store.get(incoming.id) is None
    assert gate.is_tombstoned(target.id)
    assert gate.is_tombstoned(incoming.id)
    rows = read_rows(ledger.path)
    assert [row.outcome for row in rows] == [
        WriteOutcome.EVICTED,
        WriteOutcome.REJECTED_HARM,
    ]
    assert rows[-1].incoming_id == incoming.id


def test_merge_harmful_union_ledgers_target_eviction(store, make_card, tmp_path):
    # The harm verdict destroys TWO cards: the banked target (deleted) and the
    # submitted partner (rejected). Each needs its own ledger row or the
    # target's disappearance is unexplained and the rejected row points at the
    # wrong card's fate.
    target = make_card()
    store.save(target)
    incoming = make_card()
    gate, ledger = make_gate(store, tmp_path, MarkingEvictor({target.id}))
    result = gate.merge(target.id, incoming)
    assert result.rejected_harm
    rows = read_rows(ledger.path)
    assert [r.outcome for r in rows] == [
        WriteOutcome.EVICTED,
        WriteOutcome.REJECTED_HARM,
    ]
    evicted, rejected = rows
    assert evicted.incoming_id == target.id
    assert evicted.final_id == ""
    assert rejected.incoming_id == incoming.id
    assert rejected.merge_targets == (target.id,)


def test_merge_onto_tombstoned_target_is_benign_noop(store, make_card, tmp_path):
    # Gate-level pin: a tombstoned id is deleted from the store, so a MERGE
    # onto it degrades to the missing-target no-op and writes no row. No live
    # path reaches this — reconcile only offers targets from the store's
    # neighbor context, which never contains a tombstoned (deleted) id.
    card = make_card()
    store.save(card)
    harmful = {card.id}
    gate, ledger = make_gate(store, tmp_path, MarkingEvictor(harmful))
    gate.sweep()
    harmful.clear()
    rows_before = len(read_rows(ledger.path))
    result = gate.merge(card.id, make_card())
    assert result.outcome is WriteOutcome.DISCARDED
    assert len(read_rows(ledger.path)) == rows_before


def test_tombstoned_program_id_does_not_block_bare_insight_id(
    store, make_card, tmp_path
):
    # Exemplar cache ids live in the "program-<id>" namespace; harm-evicting
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


def test_merge_store_failure_is_benign_noop(store, make_card, tmp_path):
    target = make_card()
    store.save(target)
    store.fail_merges = True
    gate, ledger = make_gate(store, tmp_path)
    result = gate.merge(target.id, make_card())
    assert result.outcome is WriteOutcome.DISCARDED
    assert not result.landed
    assert not result.rejected_harm
    assert read_rows(ledger.path) == []


def test_bump_provenance_appends_child_once(store, make_card, tmp_path):
    target = make_card()
    store.save(target)
    gate, ledger = make_gate(store, tmp_path)

    assert gate.bump_provenance(target.id, "child-1").card_id == target.id
    assert store.get(target.id).programs == ("child-1",)

    saves_before = len(store.saved_ids)
    assert gate.bump_provenance(target.id, "child-1").card_id == target.id
    assert len(store.saved_ids) == saves_before
    assert store.get(target.id).programs == ("child-1",)

    rows = read_rows(ledger.path)
    assert [r.outcome for r in rows] == [WriteOutcome.UPDATED, WriteOutcome.UPDATED]


def test_bump_provenance_missing_or_program_target_is_benign_noop(
    store, make_card, tmp_path
):
    program = make_card(
        kind=CardKind.PROGRAM, program_id="p1", code="x = 1", fitness=0.5
    )
    store.save(program)
    gate, _ = make_gate(store, tmp_path)
    assert gate.bump_provenance("absent", "child-1").outcome is WriteOutcome.DISCARDED
    assert gate.bump_provenance(program.id, "child-1").outcome is WriteOutcome.DISCARDED


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
    assert result.rejected_harm
    assert store.get(card.id) is None
    rows = read_rows(ledger.path)
    assert [r.outcome for r in rows] == [
        WriteOutcome.EVICTED,
        WriteOutcome.REJECTED_HARM,
    ]

    fresh = gate.admit(make_card(id=""))
    assert fresh.outcome is WriteOutcome.ADDED


def test_admit_harm_rejection_tombstones_id(store, make_card, tmp_path):
    card = make_card()
    store.save(card)
    harmful = {card.id}
    gate, _ = make_gate(store, tmp_path, MarkingEvictor(harmful))
    assert gate.admit(card).rejected_harm
    harmful.clear()
    assert gate.admit(card).rejected_harm
    assert store.get(card.id) is None


def test_merge_harmful_union_tombstones_target(store, make_card, tmp_path):
    target = make_card()
    store.save(target)
    harmful = {target.id}
    gate, _ = make_gate(store, tmp_path, MarkingEvictor(harmful))
    assert gate.merge(target.id, make_card()).rejected_harm
    harmful.clear()
    assert gate.admit(target).rejected_harm
    assert store.get(target.id) is None
    assert not gate.is_tombstoned("unrelated-id")


def test_reject_novelty_records_row_and_does_not_bank(store, make_card, tmp_path):
    gate, ledger = make_gate(store, tmp_path)
    result = gate.reject_novelty(make_card(id=""), "prior-known staple")
    assert result.outcome is WriteOutcome.REJECTED_NOVELTY
    assert not result.landed
    assert not result.rejected_harm
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
    landed = {WriteOutcome.ADDED, WriteOutcome.UPDATED, WriteOutcome.MERGED}
    for outcome in WriteOutcome:
        card_id = "x" if outcome in landed else ""
        result = WriteResult(outcome=outcome, card_id=card_id)
        assert result.benign_noop is (outcome is WriteOutcome.DISCARDED)
        assert result.rejected_harm is (outcome is WriteOutcome.REJECTED_HARM)
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
    assert len({row.record_id for row in rows}) == expected_count


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
