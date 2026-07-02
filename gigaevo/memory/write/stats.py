"""Use-attributed, base-relative gain events restamped onto the bank.

A card is credited for a child only when it was both selected for the
mutator's named base parent (``base_selected_ids``) and declared applied by
the mutator (``card_ids_used``) — the intersection. Each credit is one
``ContextualGain``: the child's base-relative fitness delta, tagged with the
base parent's metrics as its decision context. The card stores only the raw
events; ``read/reputation.py`` computes every per-card statistic from them at
read time. After each write sweep ``CardStatsUpdater`` recomputes the events
from the full program pool, restamps every changed card, and runs one
harm-eviction pass — gain events are a pure function of the pool, so each
sweep is authoritative. Validity/sentinel semantics come from
``MetricsContext.strict_fitness`` / ``is_evaluated_invalid``.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from gigaevo.evolution.mutation.constants import (
    MUTATION_MEMORY_BASE_ID_METADATA_KEY,
    MUTATION_MEMORY_BASE_METRICS_METADATA_KEY,
    MUTATION_MEMORY_BASE_SELECTED_IDS_METADATA_KEY,
    MUTATION_OUTPUT_METADATA_KEY,
)
from gigaevo.memory.cards import Card, CardKind, ContextualGain, DecisionContext
from gigaevo.memory.events import MemoryGainRestamp, emit_memory_event
from gigaevo.memory.storage.base import MemoryStore
from gigaevo.memory.write.admission import CardAdmissionGate
from gigaevo.programs.metrics.context import MetricsContext
from gigaevo.programs.program import Program


def card_ids_used(prog: Program) -> list[str]:
    """Card ids the mutator declared it applied, from the stamped structured output."""
    out = prog.get_metadata(MUTATION_OUTPUT_METADATA_KEY)
    if isinstance(out, dict):
        return list(out.get("card_ids_used", []) or [])
    return []


def base_selected_ids(prog: Program) -> list[str]:
    """Cards selected for the mutator's named base parent, frozen at birth."""
    ids = prog.get_metadata(MUTATION_MEMORY_BASE_SELECTED_IDS_METADATA_KEY)
    return list(ids) if isinstance(ids, list) else []


def base_metrics(prog: Program) -> dict[str, float]:
    """The base parent's metric dict, frozen at birth."""
    metrics = prog.get_metadata(MUTATION_MEMORY_BASE_METRICS_METADATA_KEY)
    return dict(metrics) if isinstance(metrics, dict) else {}


def base_id(prog: Program) -> str:
    """The base parent's program id, frozen at birth ("" for legacy programs)."""
    pid = prog.get_metadata(MUTATION_MEMORY_BASE_ID_METADATA_KEY)
    return pid if isinstance(pid, str) else ""


class InjectionOutcome(BaseModel):
    """One program's injection-relevant facts, extracted at the writer seam."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(default="", description="Program id.")
    fitness: float | None = Field(
        default=None,
        description="Validated fitness signal; None for invalid or missing.",
    )
    invalid: bool = Field(
        default=False,
        description="Evaluated and judged invalid: one forced harm event per "
        "credited card, never part of the baseline cohort.",
    )
    base_selected_ids: tuple[str, ...] = Field(
        default=(),
        description="Cards selected for the mutator's named base parent, frozen "
        "onto this child at birth. Use-attribution credits only these.",
    )
    base_metrics: dict[str, float] = Field(
        default_factory=dict,
        description="The base parent's metric dict, frozen at birth — the decision "
        "context and the reward baseline source.",
    )
    base_id: str = Field(
        default="",
        description="The base parent's program id, frozen at birth — the decision "
        "context's parent identity.",
    )
    base_fitness: float | None = Field(
        default=None,
        description="Base parent's fitness (base_metrics[fitness_key], resolved at "
        "the write seam); None means no base baseline, so no gain events.",
    )
    created_at: datetime | None = Field(
        default=None,
        description="This child's creation time (UTC) — the decision-outcome "
        "timestamp stamped onto its gain events.",
    )
    card_ids_used: tuple[str, ...] = Field(
        default=(),
        description="Card ids the mutator declared it actually applied.",
    )


def compute_contextual_gains(
    programs: Sequence[InjectionOutcome],
    *,
    higher_is_better: bool = True,
) -> dict[str, list[ContextualGain]]:
    """Map each used-and-base-selected card id to its base-relative gain events.

    A card is credited for a child only when it was both selected for the
    mutator's named base parent (``base_selected_ids``) and declared applied by
    the mutator (``card_ids_used``) — the intersection. Donor cards (used but
    selected for the other parent) and hallucinated ids (used but selected for
    neither) earn nothing. Reward is the base-relative fitness delta and context
    is the base parent's id and metrics plus the child's creation time. An
    invalid child emits one forced-harm event (gain 0.0, invalid) per credited
    card. Children with no frozen base baseline (``base_fitness is None`` or
    empty ``base_selected_ids``) contribute nothing.
    """
    events: dict[str, list[ContextualGain]] = {}
    for p in programs:
        if p.base_fitness is None or not p.base_selected_ids:
            continue
        credited = {c for c in p.base_selected_ids if c} & {
            c for c in p.card_ids_used if c
        }
        if not credited:
            continue
        context = DecisionContext(
            parent_metrics=dict(p.base_metrics),
            parent_id=p.base_id,
            timestamp=p.created_at,
        )
        if p.invalid:
            gain_event = ContextualGain(context=context, gain=0.0, invalid=True)
        elif p.fitness is None:
            continue
        else:
            delta = (
                p.fitness - p.base_fitness
                if higher_is_better
                else p.base_fitness - p.fitness
            )
            gain_event = ContextualGain(context=context, gain=delta)
        for card_id in credited:
            events.setdefault(card_id, []).append(gain_event)
    return events


def card_gain_events_from_programs(
    programs: list[Program],
    *,
    fitness_key: str,
    higher_is_better: bool,
    metrics_context: MetricsContext,
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
                fitness=metrics_context.strict_fitness(prog.metrics, fitness_key),
                invalid=metrics_context.is_evaluated_invalid(prog.metrics, fitness_key),
                base_selected_ids=tuple(base_selected_ids(prog)),
                base_metrics=bm,
                base_id=base_id(prog),
                base_fitness=metrics_context.strict_fitness(bm, fitness_key),
                created_at=prog.created_at,
                card_ids_used=tuple(card_ids_used(prog)),
            )
        )
    return compute_contextual_gains(rows, higher_is_better=higher_is_better)


class CardStatsStamper(BaseModel):
    """Single writer of card-side efficacy evidence: attaches the use-attributed
    gain events a card earned this sweep."""

    model_config = ConfigDict(frozen=True)

    def stamp_gain_events(
        self, card: Card, gain_events: dict[str, list[ContextualGain]]
    ) -> Card:
        """Card with the current sweep's authoritative gain events attached.

        The full pool is authoritative each sweep: a credited card carries this
        sweep's events; an uncredited card has any stale events cleared. A
        merge/consolidation survivor also folds in the events the pool still
        attributes to its ``absorbed_ids`` — children frozen with a since-merged
        card id credit that id, which no longer exists in the bank, so without the
        re-alias their attribution would orphan on the deleted id.

        Multiplicity is the trial count the harm gate reads (intro_events): every
        invalid child of one base parent emits a value-identical forced-harm event,
        and those are distinct trials that must all survive — on the own-id list and
        on a folded absorbed-id list alike. The absorbed fold still has to drop the
        ONE event a single child contributes when it credited both the survivor and
        an absorbed id, but that is the same event *object*: ``card_gain_events_from_programs``
        binds one ``ContextualGain`` per child and appends it to every credited id's
        list, so identity (not value) is the trial discriminator. Dedup by object
        identity keeps distinct value-equal trials and drops only the shared one.
        This relies on receiving one sweep's freshly built pool, which is the sole
        caller's contract (CardStatsUpdater.update).
        """
        folded: list[ContextualGain] = list(gain_events.get(card.id.strip()) or [])
        if card.kind is CardKind.INSIGHT and card.absorbed_ids:
            seen = {id(event) for event in folded}
            for aid in card.absorbed_ids:
                for event in gain_events.get(aid.strip()) or []:
                    if id(event) not in seen:
                        seen.add(id(event))
                        folded.append(event)
        return card.model_copy(update={"gain_events": tuple(folded)})


class CardStatsUpdater:
    """Recomputes and restamps use-attributed gain events, then sweeps for harm.

    Pure with respect to construction (fitness key, direction, metrics context);
    the bank ``store`` and admission ``gate`` are passed per call so one updater
    serves any store built off one checkpoint.
    """

    def __init__(
        self,
        *,
        fitness_key: str,
        higher_is_better: bool,
        metrics_context: MetricsContext,
    ) -> None:
        self._fitness_key = fitness_key
        self._higher_is_better = higher_is_better
        self._metrics_context = metrics_context

    def update(
        self, pool: list[Program], *, store: MemoryStore, gate: CardAdmissionGate
    ) -> None:
        """Attribute gain across the pool, emit the restamp event, restamp the
        credited cards, and run one harm-eviction sweep. Blocking I/O (per-card
        store writes); the orchestrator runs it off the event loop."""
        events = card_gain_events_from_programs(
            pool,
            fitness_key=self._fitness_key,
            higher_is_better=self._higher_is_better,
            metrics_context=self._metrics_context,
        )
        emit_memory_event(
            MemoryGainRestamp(
                credited_card_count=len(events),
                event_count_by_card_id={cid: len(evs) for cid, evs in events.items()},
            )
        )
        self.restamp_and_sweep(events, store=store, gate=gate)

    def restamp_and_sweep(
        self,
        card_gain_events: dict[str, list[ContextualGain]],
        *,
        store: MemoryStore,
        gate: CardAdmissionGate,
    ) -> None:
        """Attach this sweep's use-attributed gain events onto credited cards,
        then run one harm-eviction pass. Credited cards get this sweep's events;
        cards no longer credited have stale events cleared. Only cards whose
        events changed are rewritten."""
        stamper = CardStatsStamper()
        for card in store.snapshot():
            stamped = stamper.stamp_gain_events(card, card_gain_events)
            if stamped.gain_events != card.gain_events:
                store.save(stamped)
        gate.sweep()
