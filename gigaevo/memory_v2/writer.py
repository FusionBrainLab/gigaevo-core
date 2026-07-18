"""Content-only writer feedback for the causal memory-v2 action bank."""

from __future__ import annotations

from gigaevo.memory.events import emit_memory_event
from gigaevo.memory.selection_leases import InFlightSelectionRegistry
from gigaevo.memory.storage.base import MemoryStore
from gigaevo.memory.write.admission import CardAdmissionGate
from gigaevo.memory_v2.events import MemoryV2WriterSync
from gigaevo.memory_v2.ledger import SqliteCausalLedger
from gigaevo.programs.program import Program


class CausalV2ContentOnlyUpdater:
    """Release completed leases without creating a second efficacy system."""

    def __init__(
        self,
        *,
        ledger: SqliteCausalLedger,
        selection_leases: InFlightSelectionRegistry,
    ) -> None:
        self.ledger = ledger
        self.selection_leases = selection_leases

    def update(
        self,
        pool: list[Program],
        *,
        store: MemoryStore,
        gate: CardAdmissionGate,
    ) -> None:
        snapshot = self.ledger.snapshot()
        terminal_child_ids = {
            terminal.child_id
            for terminal in self.ledger.terminals()
            if not terminal.child_id.startswith("no-child:")
        }
        child_ids = {program.id for program in pool} & terminal_child_ids
        for child_id in child_ids:
            self.selection_leases.release_child(child_id)
        retired_card_ids = tuple(gate.sweep())
        emit_memory_event(
            MemoryV2WriterSync(
                evidence_version=snapshot.version,
                model_evidence_version=snapshot.model_version,
                evidence_count=len(snapshot.observations),
                pending_count=sum(snapshot.pending_by_treatment.values()),
                bank_size=len(store.snapshot()),
                released_child_count=len(child_ids),
                retired_card_ids=retired_card_ids,
            )
        )
