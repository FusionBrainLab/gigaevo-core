"""CardStatsUpdater: use-attributed gain events restamped onto the bank.

After each write sweep the updater recomputes, from the full program pool, the
base-relative gain each card earned, restamps every changed card, and runs one
harm-eviction pass. Gain events are a pure function of the pool, so each sweep is
authoritative.
"""

from __future__ import annotations

from typing import Any

from gigaevo.memory.context import ContextualGain
from gigaevo.memory.core.events import emit_memory_event
from gigaevo.memory.efficacy.stamping import CardStatsStamper
from gigaevo.memory.ideas_tracker.fitness import (
    base_fitness,
    base_id,
    base_metrics,
    base_selected_ids,
    card_ids_used,
    evaluated_invalid,
    valid_fitness,
)
from gigaevo.memory.shared_memory.injection_posterior import (
    InjectionOutcome,
    compute_contextual_gains,
)
from gigaevo.programs.metrics.context import MetricsContext
from gigaevo.programs.program import Program


def card_gain_events_from_programs(
    programs: list[Program],
    *,
    fitness_key: str,
    higher_is_better: bool,
    metrics_context: MetricsContext | None = None,
) -> dict[str, list[ContextualGain]]:
    """Use-attributed base-relative gain events per card id, from live programs.

    Credits a card only when it was both selected for the mutator's named base
    parent (``memory_base_selected_idea_ids``) and declared used
    (``mutation_output.card_ids_used``); the base fitness baseline is resolved
    here from the frozen base metrics under ``fitness_key``.
    """
    rows = []
    for prog in programs:
        bm = base_metrics(prog)
        rows.append(
            InjectionOutcome(
                id=prog.id,
                parents=prog.lineage.parents,
                fitness=valid_fitness(prog, fitness_key, metrics_context),
                invalid=evaluated_invalid(prog, fitness_key, metrics_context),
                base_selected_ids=base_selected_ids(prog),
                base_metrics=bm,
                base_id=base_id(prog),
                base_fitness=base_fitness(bm, fitness_key, metrics_context),
                created_at=prog.created_at,
                card_ids_used=card_ids_used(prog),
            )
        )
    return compute_contextual_gains(rows, higher_is_better=higher_is_better)


class CardStatsUpdater:
    """Recomputes and restamps use-attributed gain events, then sweeps for harm.

    Pure with respect to construction (fitness key, direction, metrics context);
    the bank ``store`` and admission ``gate`` are passed per call so one updater
    serves both the read-side and write-side backends built off one checkpoint.
    """

    def __init__(
        self,
        *,
        fitness_key: str,
        higher_is_better: bool,
        metrics_context: MetricsContext | None = None,
    ) -> None:
        self._fitness_key = fitness_key
        self._higher_is_better = higher_is_better
        self._metrics_context = metrics_context

    def update(self, pool: list[Program], *, store: Any, gate: Any) -> None:
        """Attribute gain across the pool, emit the posterior event, restamp the
        credited cards, and run one harm-eviction sweep. Blocking I/O (per-card
        store writes); the orchestrator runs it off the event loop."""
        events = card_gain_events_from_programs(
            pool,
            fitness_key=self._fitness_key,
            higher_is_better=self._higher_is_better,
            metrics_context=self._metrics_context,
        )
        emit_memory_event(
            component="ideas_tracker",
            event_type="injection_posterior.compute",
            payload={
                "card_count": len(events),
                "event_count_by_card_id": {
                    cid: len(evs) for cid, evs in events.items()
                },
            },
        )
        self.restamp_and_sweep(events, store=store, gate=gate)

    def restamp_and_sweep(
        self,
        card_gain_events: dict[str, list[ContextualGain]],
        *,
        store: Any,
        gate: Any,
    ) -> None:
        """Attach this sweep's use-attributed gain events onto credited cards,
        then run one harm-eviction pass. Credited cards get this sweep's events;
        cards no longer credited have stale events cleared. Only cards whose
        events changed are rewritten."""
        stamper = CardStatsStamper()
        for card in list(store.all_cards_snapshot().values()):
            stamped = stamper.stamp_gain_events(card, card_gain_events)
            if stamped.gain_events != card.gain_events:
                store.save_card_direct(stamped)
        gate.sweep()
