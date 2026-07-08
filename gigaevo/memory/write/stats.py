"""Selected-card, base-relative outcome events restamped onto the bank.

Every card selected into the prompt receives an outcome. Selected-and-used cards
split the child's base-relative fitness delta after subtracting the fitted
no-card baseline; selected but unused cards receive an explicit ``unused``
exposure event. The card stores only event rows; ``read/reputation.py`` computes
every per-card statistic from them at read time. After each write sweep
``CardStatsUpdater`` recomputes the events from the full program pool, restamps
every changed card, and runs one configured eviction pass — gain events are a
pure function of the pool, so each sweep is authoritative. Validity/sentinel semantics
come from
``MetricsContext.strict_fitness`` / ``is_evaluated_invalid``.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
import statistics
from typing import Protocol, runtime_checkable

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from gigaevo.evolution.mutation.constants import (
    MUTATION_MEMORY_BASE_ID_METADATA_KEY,
    MUTATION_MEMORY_BASE_METRICS_METADATA_KEY,
    MUTATION_MEMORY_BASE_SELECTED_IDS_METADATA_KEY,
    MUTATION_MEMORY_INJECTED_IDS_METADATA_KEY,
    MUTATION_MEMORY_NO_CARD_CONTROL_METADATA_KEY,
    MUTATION_OUTPUT_METADATA_KEY,
)
from gigaevo.memory.cards import (
    Card,
    CardKind,
    CausalStrength,
    ContextualGain,
    DecisionContext,
    EvidenceAttribution,
    EvidenceSource,
)
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


def injected_ids(prog: Program) -> list[str]:
    """Full prompt slate frozen at birth: cards selected from all parents."""
    ids = prog.get_metadata(MUTATION_MEMORY_INJECTED_IDS_METADATA_KEY)
    return list(ids) if isinstance(ids, list) else []


def creditable_selected_ids(prog: Program) -> list[str]:
    """Cards eligible for credit/exposure events.

    New children carry the full injected slate, which makes crossover donor-card
    citations creditable. Legacy children fall back to the old base-parent stamp.
    """
    return injected_ids(prog) or base_selected_ids(prog)


def base_metrics(prog: Program) -> dict[str, float]:
    """The base parent's metric dict, frozen at birth."""
    metrics = prog.get_metadata(MUTATION_MEMORY_BASE_METRICS_METADATA_KEY)
    return dict(metrics) if isinstance(metrics, dict) else {}


def base_id(prog: Program) -> str:
    """The base parent's program id, frozen at birth ("" for legacy programs)."""
    pid = prog.get_metadata(MUTATION_MEMORY_BASE_ID_METADATA_KEY)
    return pid if isinstance(pid, str) else ""


def no_card_control(prog: Program) -> bool:
    """Whether this child was born from a randomized memory-withheld control."""
    return bool(prog.get_metadata(MUTATION_MEMORY_NO_CARD_CONTROL_METADATA_KEY))


def founding_gain_event(
    program: Program,
    *,
    fitness_key: str,
    higher_is_better: bool,
    metrics_context: MetricsContext,
) -> ContextualGain | None:
    """The ``founding`` gain event for a freshly-authored card: the true signed
    delta of the child it was distilled from against that child's base parent.

    Baseline and context are resolved identically to use-attribution
    (``base_metrics`` / ``base_id`` / ``strict_fitness``) so a founding event and
    the later use events of the same card are the same kind of evidence — the
    only distinguisher is ``founding=True``, which carries it across the
    from-scratch restamp that recomputes all use events from the pool. Returns
    None when the child predates the memory path (no frozen base snapshot) or
    either fitness is missing/sentinel — there is then no honest founding delta.
    """
    bm = base_metrics(program)
    base_fit = metrics_context.strict_fitness(bm, fitness_key)
    child_fit = metrics_context.strict_fitness(program.metrics, fitness_key)
    if base_fit is None or child_fit is None:
        return None
    delta = child_fit - base_fit if higher_is_better else base_fit - child_fit
    return ContextualGain(
        context=DecisionContext(
            parent_metrics=dict(bm),
            parent_id=base_id(program),
            timestamp=program.created_at,
        ),
        gain=delta,
        founding=True,
        attribution=EvidenceAttribution(
            source=EvidenceSource.FOUNDING,
            causal_strength=CausalStrength.ORIGIN,
            source_child_id=program.id,
            credit_weight=0.0,
        ),
    )


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
        "selected-and-used card, never part of the baseline cohort.",
    )
    base_selected_ids: tuple[str, ...] = Field(
        default=(),
        description="Creditable selected card ids frozen onto this child at birth. "
        "For current runs this is the full injected slate across all parents; "
        "legacy rows may contain only the mutator's named base parent.",
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
    no_card_control: bool = Field(
        default=False,
        description="True when selected cards were deliberately withheld by the "
        "randomized no-card control arm.",
    )


@runtime_checkable
class FittedNoCardBaseline(Protocol):
    """Resolved no-card baseline for one stats sweep."""

    has_evidence: bool

    def baseline_for(self, outcome: InjectionOutcome) -> float: ...


@runtime_checkable
class NoCardBaselineEstimator(Protocol):
    """Pluggable context model for expected no-card child-parent progress."""

    def fit_no_card_baseline(
        self,
        outcomes: Sequence[InjectionOutcome],
        *,
        higher_is_better: bool,
    ) -> FittedNoCardBaseline: ...


def compute_contextual_gains(
    programs: Sequence[InjectionOutcome],
    *,
    higher_is_better: bool = True,
    baseline_estimator: NoCardBaselineEstimator | None = None,
) -> dict[str, list[ContextualGain]]:
    """Map each prompt-selected card id to baseline-adjusted outcome events.

    First learn the run-local no-card baseline from children with a frozen base
    baseline and an empty selected slate, preferring rows created by the randomized
    no-card control arm when any exist. The estimator is injectable: the default
    is a global median, while contextual reputations can fit per-cell/per-regime
    baselines behind the same method. Used cards receive ``child-parent -
    no_card_baseline`` split across cited cards. Selected but unused cards receive
    a zero-gain ``unused=True`` exposure failure only once a no-card baseline has
    evidence (invalid children still emit forced-harm / unused events).
    """
    events: dict[str, list[ContextualGain]] = {}
    baseline = _fit_no_card_baseline(
        baseline_estimator, programs, higher_is_better=higher_is_better
    )
    has_baseline_evidence = bool(getattr(baseline, "has_evidence", True))
    for p in programs:
        if p.base_fitness is None or not p.base_selected_ids:
            continue
        selected = _clean_ids(p.base_selected_ids)
        used = selected & _clean_ids(p.card_ids_used)
        unused = selected - used
        if not selected:
            continue
        context = DecisionContext(
            parent_metrics=dict(p.base_metrics),
            parent_id=p.base_id,
            timestamp=p.created_at,
        )
        if used and p.invalid:
            gain_event = ContextualGain(
                context=context,
                gain=0.0,
                invalid=True,
                attribution=EvidenceAttribution(
                    source=EvidenceSource.INVALID,
                    causal_strength=CausalStrength.INVALID,
                    source_child_id=p.id,
                    used_card_count=len(used),
                    credit_weight=1.0 / len(used),
                ),
            )
            for card_id in used:
                events.setdefault(card_id, []).append(gain_event)
        elif used and p.fitness is not None and has_baseline_evidence:
            delta = _oriented_delta(
                p.fitness, p.base_fitness, higher_is_better
            ) - baseline.baseline_for(p)
            gain_event = ContextualGain(
                context=context,
                gain=delta / len(used),
                attribution=EvidenceAttribution(
                    source=EvidenceSource.DIRECT,
                    causal_strength=(
                        CausalStrength.DIRECT_ISOLATED
                        if len(used) == 1
                        else CausalStrength.DIRECT_BUNDLED
                    ),
                    source_child_id=p.id,
                    used_card_count=len(used),
                    credit_weight=1.0 / len(used),
                ),
            )
            for card_id in used:
                events.setdefault(card_id, []).append(gain_event)
        if unused and (has_baseline_evidence or p.invalid):
            unused_event = ContextualGain(
                context=context,
                gain=0.0,
                unused=True,
                attribution=EvidenceAttribution(
                    source=EvidenceSource.UNUSED,
                    causal_strength=CausalStrength.EXPOSURE,
                    source_child_id=p.id,
                    used_card_count=len(used),
                    credit_weight=1.0,
                ),
            )
            for card_id in unused:
                events.setdefault(card_id, []).append(unused_event)
    return events


def _clean_ids(ids: Sequence[str]) -> set[str]:
    return {c.strip() for c in ids if c.strip()}


def _oriented_delta(
    child_fitness: float, base_fitness: float, higher_is_better: bool
) -> float:
    return (
        child_fitness - base_fitness
        if higher_is_better
        else base_fitness - child_fitness
    )


class GlobalNoCardBaseline(BaseModel):
    """Default no-card estimator: median delta over all no-card children."""

    model_config = ConfigDict(frozen=True)

    def fit_no_card_baseline(
        self,
        outcomes: Sequence[InjectionOutcome],
        *,
        higher_is_better: bool,
    ) -> FittedNoCardBaseline:
        del self
        deltas = _no_card_deltas(outcomes, higher_is_better)
        return _ConstantNoCardBaseline(
            baseline=_median(deltas),
            has_evidence=bool(deltas),
        )


class _ConstantNoCardBaseline(BaseModel):
    model_config = ConfigDict(frozen=True)

    baseline: float = 0.0
    has_evidence: bool = False

    def baseline_for(self, outcome: InjectionOutcome) -> float:
        del outcome
        return self.baseline


def _fit_no_card_baseline(
    estimator: NoCardBaselineEstimator | None,
    outcomes: Sequence[InjectionOutcome],
    *,
    higher_is_better: bool,
) -> FittedNoCardBaseline:
    if estimator is None or not hasattr(estimator, "fit_no_card_baseline"):
        estimator = GlobalNoCardBaseline()
    try:
        return estimator.fit_no_card_baseline(
            outcomes, higher_is_better=higher_is_better
        )
    except Exception:
        logger.opt(exception=True).warning(
            "[Memory][Stats] no-card baseline estimator failed; using global median"
        )
        return GlobalNoCardBaseline().fit_no_card_baseline(
            outcomes, higher_is_better=higher_is_better
        )


def _no_card_deltas(
    outcomes: Sequence[InjectionOutcome], higher_is_better: bool
) -> list[float]:
    no_card = [
        p
        for p in outcomes
        if p.base_fitness is not None and not _clean_ids(p.base_selected_ids)
    ]
    controls = [p for p in no_card if p.no_card_control]
    cohort = controls if controls else no_card
    deltas: list[float] = []
    for p in cohort:
        if p.base_fitness is None:
            continue
        if p.invalid:
            deltas.append(0.0)
            continue
        if p.fitness is None:
            continue
        deltas.append(_oriented_delta(p.fitness, p.base_fitness, higher_is_better))
    return deltas


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def card_gain_events_from_programs(
    programs: list[Program],
    *,
    fitness_key: str,
    higher_is_better: bool,
    metrics_context: MetricsContext,
    baseline_estimator: NoCardBaselineEstimator | None = None,
) -> dict[str, list[ContextualGain]]:
    """Selected-card base-relative outcome events per card id, from live programs.

    Used cards split the child delta; selected-but-unused cards receive an
    explicit unused exposure. The base fitness baseline is resolved here from
    the frozen base metrics under ``fitness_key``.
    """
    rows = []
    for prog in programs:
        bm = base_metrics(prog)
        rows.append(
            InjectionOutcome(
                id=prog.id,
                fitness=metrics_context.strict_fitness(prog.metrics, fitness_key),
                invalid=metrics_context.is_evaluated_invalid(prog.metrics, fitness_key),
                base_selected_ids=tuple(creditable_selected_ids(prog)),
                base_metrics=bm,
                base_id=base_id(prog),
                base_fitness=metrics_context.strict_fitness(bm, fitness_key),
                created_at=prog.created_at,
                card_ids_used=tuple(card_ids_used(prog)),
                no_card_control=no_card_control(prog),
            )
        )
    return compute_contextual_gains(
        rows, higher_is_better=higher_is_better, baseline_estimator=baseline_estimator
    )


class CardStatsStamper(BaseModel):
    """Single writer of card-side efficacy evidence: attaches the selected-card
    outcome events a card earned this sweep."""

    model_config = ConfigDict(frozen=True)

    def stamp_gain_events(
        self,
        card: Card,
        gain_events: dict[str, list[ContextualGain]],
        *,
        preserve_events_outside_child_ids: set[str] | None = None,
    ) -> Card:
        """Card with the current sweep's authoritative gain events attached.

        The full pool is authoritative each sweep: a selected card carries this
        sweep's events; an unselected card has any stale events cleared. A
        merge/consolidation survivor also folds in the events the pool still
        attributes to its ``absorbed_ids`` — children frozen with a since-merged
        card id credit that id, which no longer exists in the bank, so without the
        re-alias their attribution would orphan on the deleted id.

        Multiplicity is the trial count the harm gate reads (intro_events): every
        invalid child of one base parent emits a value-identical forced-harm event,
        and those are distinct trials that must all survive — on the own-id list and
        on a folded absorbed-id list alike. The absorbed fold still has to drop the
        ONE event a single child contributes when it selected both the survivor and
        an absorbed id, but that is the same event *object*: ``card_gain_events_from_programs``
        binds one ``ContextualGain`` per outcome class and appends it to every selected id's
        list, so identity (not value) is the trial discriminator. Dedup by object
        identity keeps distinct value-equal trials and drops only the shared one.
        This relies on receiving one sweep's freshly built pool, which is the sole
        caller's contract (CardStatsUpdater.update).

        Founding events (the delta a card was distilled from) are the one class of
        event the pool cannot recompute — the founding child predates the card, so
        outcome attribution never re-credits it. They live only on the card, are
        carried onto a merge survivor by ``merge_cards`` unioning gain events, and
        are preserved here across the recompute. They never double-count: the pool
        only ever holds use events, so the absorbed-id fold below cannot re-add one.

        In a shared bank, two runs may restamp the same card from disjoint program
        pools. When ``preserve_events_outside_child_ids`` is supplied, this
        restamp replaces only events whose source child belongs to the current
        pool and keeps already-stamped evidence from other runs.
        """
        founding = [
            event
            for event in card.gain_events
            if event.founding
            or self._preserve_external_event(event, preserve_events_outside_child_ids)
        ]
        folded: list[ContextualGain] = founding + list(
            gain_events.get(card.id.strip()) or []
        )
        if card.kind is CardKind.INSIGHT and card.absorbed_ids:
            seen = {id(event) for event in folded}
            for aid in card.absorbed_ids:
                for event in gain_events.get(aid.strip()) or []:
                    if id(event) not in seen:
                        seen.add(id(event))
                        folded.append(event)
        return card.model_copy(update={"gain_events": tuple(folded)})

    @staticmethod
    def _preserve_external_event(
        event: ContextualGain, preserve_events_outside_child_ids: set[str] | None
    ) -> bool:
        if preserve_events_outside_child_ids is None or event.founding:
            return False
        source_child_id = (
            event.attribution.source_child_id if event.attribution is not None else ""
        )
        return bool(
            source_child_id and source_child_id not in preserve_events_outside_child_ids
        )


class CardStatsUpdater:
    """Recomputes and restamps selected-card outcome events, then sweeps for harm.

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
        baseline_estimator: NoCardBaselineEstimator | None = None,
    ) -> None:
        self._fitness_key = fitness_key
        self._higher_is_better = higher_is_better
        self._metrics_context = metrics_context
        self._baseline_estimator = baseline_estimator
        self._logged_orphans: set[str] = set()

    def update(
        self, pool: list[Program], *, store: MemoryStore, gate: CardAdmissionGate
    ) -> None:
        """Attribute outcomes across the pool, emit the restamp event, restamp the
        selected cards, and run one configured eviction sweep. Events for a card id
        nothing in the bank can receive — neither a banked id nor an absorbed
        alias, e.g. a card evicted after the child froze its selection — is
        dropped before the restamp event, so the telemetry never reports events
        the stamper is about to discard. Blocking I/O (per-card store writes);
        the orchestrator runs it off the event loop."""
        events = card_gain_events_from_programs(
            pool,
            fitness_key=self._fitness_key,
            higher_is_better=self._higher_is_better,
            metrics_context=self._metrics_context,
            baseline_estimator=self._baseline_estimator,
        )
        bank = store.snapshot()
        known = {card.id.strip() for card in bank}
        known.update(aid.strip() for card in bank for aid in card.absorbed_ids)
        orphaned = set(events) - known
        if orphaned:
            fresh = orphaned - self._logged_orphans
            if fresh:
                logger.debug(
                    "[Memory][Stats] dropping outcome events for {} card id(s) not "
                    "resolvable in the bank (evicted or never landed): {}",
                    len(fresh),
                    sorted(fresh),
                )
                self._logged_orphans.update(fresh)
            events = {cid: evs for cid, evs in events.items() if cid in known}
        emit_memory_event(
            MemoryGainRestamp(
                credited_card_count=len(events),
                event_count_by_card_id={cid: len(evs) for cid, evs in events.items()},
            )
        )
        self.restamp_and_sweep(
            events,
            store=store,
            gate=gate,
            preserve_events_outside_child_ids={prog.id for prog in pool},
        )

    def restamp_and_sweep(
        self,
        card_gain_events: dict[str, list[ContextualGain]],
        *,
        store: MemoryStore,
        gate: CardAdmissionGate,
        preserve_events_outside_child_ids: set[str] | None = None,
    ) -> None:
        """Attach this sweep's outcome events onto selected cards, then run one
        configured eviction pass. Selected cards get this sweep's events; cards no
        longer selected have stale events cleared. Only cards whose
        events changed are rewritten."""
        stamper = CardStatsStamper()
        for card in store.snapshot():
            stamped = stamper.stamp_gain_events(
                card,
                card_gain_events,
                preserve_events_outside_child_ids=preserve_events_outside_child_ids,
            )
            if stamped.gain_events != card.gain_events:
                store.save(stamped)
        gate.sweep()
