"""Engine-facing memory provider adapters, injected via Hydra.

The provider is the read-side strategy object the DAG pipeline consumes
(``MemoryContextStage``). The ``memory={none,v2}`` config arms swap the
``_target_``:

- ``NullMemoryProvider`` — no-op, returns empty selection (read side off)
- ``LeasedMemoryProvider`` — wraps a provider so only cards still in the bank
  acquire attempt leases
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

from loguru import logger

from gigaevo.exceptions import MemoryStorageError
from gigaevo.memory.events import memory_event_buffer
from gigaevo.memory.read.reader import MemorySelection
from gigaevo.memory.selection_leases import InFlightSelectionRegistry
from gigaevo.memory.storage.base import MemoryStore
from gigaevo.programs.program import Program


class MemoryProvider(ABC):
    """Abstract memory provider injected via Hydra."""

    @property
    def consumes_pending_counts(self) -> bool:
        return False

    @abstractmethod
    async def select_cards(
        self,
        program: Program,
        *,
        task_description: str,
        metrics_description: str,
        parent_context: str | None = None,
        pending_counts: Mapping[str, int] | None = None,
    ) -> MemorySelection:
        """Select memory cards relevant to this program."""


class NullMemoryProvider(MemoryProvider):
    """No-op provider. Returns empty selection. Default when ``memory=none``."""

    async def select_cards(
        self,
        program: Program,
        *,
        task_description: str,
        metrics_description: str,
        parent_context: str | None = None,
        pending_counts: Mapping[str, int] | None = None,
    ) -> MemorySelection:
        return MemorySelection()


class LeasedMemoryProvider(MemoryProvider):
    """Acquire attempt leases only for selected cards still in the bank."""

    def __init__(
        self,
        *,
        provider: MemoryProvider,
        store: MemoryStore,
        registry: InFlightSelectionRegistry,
    ) -> None:
        self._provider = provider
        self._store = store
        self._registry = registry
        self._registry.bind_store(store)

    async def select_cards(
        self,
        program: Program,
        *,
        task_description: str,
        metrics_description: str,
        parent_context: str | None = None,
        pending_counts: Mapping[str, int] | None = None,
        selection_attempt_id: str | None = None,
    ) -> MemorySelection:
        del pending_counts
        attempt_id = selection_attempt_id or self._registry.attempt_for_parent(
            program.id
        )
        attempts = self._registry.attempts_for_parent(program.id)
        if attempt_id is not None and attempt_id not in attempts:
            raise MemoryStorageError(
                f"selection attempt {attempt_id!r} does not own parent {program.id!r}"
            )
        if attempt_id is None and attempts:
            raise MemoryStorageError(
                f"no active selection attempt for parent {program.id!r}"
            )
        consumes_pending_counts = self._provider.consumes_pending_counts
        while True:
            snapshot = self._registry.selection_snapshot()
            with memory_event_buffer() as events:
                selection = await self._provider.select_cards(
                    program,
                    task_description=task_description,
                    metrics_description=metrics_description,
                    parent_context=parent_context,
                    pending_counts=snapshot.counts,
                )
                if attempt_id is None:
                    with self._registry.eviction_guard():
                        kept_ids = tuple(
                            card_id
                            for card_id in selection.card_ids
                            if self._store.get(card_id) is not None
                        )
                    committed = True
                else:
                    reservation = self._registry.reserve_selection(
                        attempt_id,
                        selection.card_ids,
                        expected_version=(
                            snapshot.version if consumes_pending_counts else None
                        ),
                        card_lookup=self._store.get,
                    )
                    committed = reservation.committed
                    kept_ids = reservation.card_ids
                if committed:
                    if kept_ids != selection.card_ids:
                        corrected_assignment = events.rewrite_terminal_selection(
                            decision_id=selection.decision_id,
                            selected_ids=kept_ids,
                            assignment=selection.assignment,
                        )
                        selection = selection.model_copy(
                            update={"assignment": corrected_assignment}
                        )
                    events.commit()
                    break
            logger.info(
                "[Memory][Leases] retrying stale pending snapshot for attempt {}",
                attempt_id,
            )
        kept_set = set(kept_ids)
        kept = [
            (card, card_id)
            for card, card_id in zip(selection.cards, selection.card_ids)
            if card_id in kept_set
        ]
        if len(kept) == len(selection.card_ids):
            return selection
        vanished = set(selection.card_ids) - {card_id for _, card_id in kept}
        logger.info(
            "[Memory][Leases] dropped {} vanished selected card(s) for {}: {}",
            len(vanished),
            program.id[:8],
            sorted(vanished),
        )
        return selection.model_copy(
            update={
                "cards": tuple(card for card, _ in kept),
                "card_ids": tuple(card_id for _, card_id in kept),
                "decision_id": selection.decision_id,
                "assignment": selection.assignment,
            }
        )
