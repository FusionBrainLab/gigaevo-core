"""Content-only writer feedback for the causal memory-v2 action bank."""

from __future__ import annotations

from gigaevo.memory.cards import CardUseTrial, union_use_trials
from gigaevo.memory.events import emit_memory_event
from gigaevo.memory.selection_leases import InFlightSelectionRegistry
from gigaevo.memory.storage.base import MemoryStore
from gigaevo.memory.write.admission import CardAdmissionGate
from gigaevo.memory_v2.events import MemoryV2WriterSync
from gigaevo.memory_v2.ledger import SqliteCausalLedger
from gigaevo.memory_v2.models import CausalObservation
from gigaevo.programs.program import Program


class CausalV2ContentOnlyUpdater:
    """Release completed leases without creating a second efficacy system."""

    def __init__(
        self,
        *,
        ledger: SqliteCausalLedger,
        selection_leases: InFlightSelectionRegistry,
        record_use_trials: bool = False,
    ) -> None:
        self.ledger = ledger
        self.selection_leases = selection_leases
        self.record_use_trials = record_use_trials

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
        if self.record_use_trials:
            self._record_use_trials(snapshot.observations, store=store)
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

    @staticmethod
    def _record_use_trials(
        observations: tuple[CausalObservation, ...], *, store: MemoryStore
    ) -> None:
        cards = store.snapshot()
        owner_by_lineage_id = {
            lineage_id: card.id
            for card in cards
            for lineage_id in (card.id, *card.absorbed_ids)
        }
        trials_by_owner: dict[str, list[CardUseTrial]] = {}
        for observation in observations:
            owner = next(
                (
                    owner_by_lineage_id[lineage_id]
                    for lineage_id in observation.card.bank_lineage_ids
                    if lineage_id in owner_by_lineage_id
                ),
                None,
            )
            if owner is None:
                continue
            successful = (
                not observation.invalid
                and observation.measurement is not None
                and observation.measurement.value > 0.0
            )
            trials_by_owner.setdefault(owner, []).append(
                CardUseTrial(
                    decision_id=observation.decision_id,
                    run_id=observation.context.run_id,
                    task_key=observation.context.environment.task_key,
                    treatment=observation.treatment,
                    success=successful,
                )
            )

        for card_id, trials in trials_by_owner.items():
            incoming = tuple(trials)

            def fold(fresh, *, incoming=incoming):
                return fresh.model_copy(
                    update={"use_trials": union_use_trials(fresh.use_trials, incoming)}
                )

            store.update(card_id, fold)
