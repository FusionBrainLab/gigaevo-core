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

from collections.abc import Callable, Sequence
from datetime import datetime
import math
from typing import Protocol, runtime_checkable

from loguru import logger
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from gigaevo.evolution.mutation.constants import (
    MUTATION_MEMORY_BASE_ID_METADATA_KEY,
    MUTATION_MEMORY_BASE_METRICS_METADATA_KEY,
    MUTATION_MEMORY_BASE_SCORE_SIGNATURE_METADATA_KEY,
    MUTATION_MEMORY_BASE_SCORES_METADATA_KEY,
    MUTATION_MEMORY_BASE_SELECTED_IDS_METADATA_KEY,
    MUTATION_MEMORY_CARD_PROVENANCE_METADATA_KEY,
    MUTATION_MEMORY_INJECTED_IDS_METADATA_KEY,
    MUTATION_MEMORY_NO_CARD_CONTROL_METADATA_KEY,
    MUTATION_OUTPUT_METADATA_KEY,
)
from gigaevo.memory.cards import (
    Card,
    CardAssignmentSource,
    CardKind,
    CausalStrength,
    ContextualGain,
    DecisionContext,
    EvidenceAttribution,
    EvidenceSource,
)
from gigaevo.memory.context.evidence import clean_ids, median, oriented_delta
from gigaevo.memory.context.no_card import NoCardEvidenceRecorder
from gigaevo.memory.events import MemoryGainRestamp, emit_memory_event
from gigaevo.memory.selection_leases import InFlightSelectionRegistry
from gigaevo.memory.storage.base import MemoryStore
from gigaevo.memory.uncertainty import outcome_uncertainty
from gigaevo.memory.write.admission import CardAdmissionGate
from gigaevo.memory.write.crediting import EffectEstimator, PointEffectEstimator
from gigaevo.programs.metrics.context import MetricsContext
from gigaevo.programs.metrics.paired import (
    PER_SAMPLE_SCORES_KEY,
    PER_SAMPLE_SIGNATURE_KEY,
)
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


def base_scores(prog: Program) -> tuple[float, ...] | None:
    """The base parent's per-sample score vector, frozen at birth (None when absent)."""
    return _score_vector(prog.get_metadata(MUTATION_MEMORY_BASE_SCORES_METADATA_KEY))


def child_scores(prog: Program) -> tuple[float, ...] | None:
    """This program's own live per-sample score vector (None when absent)."""
    return _score_vector(prog.get_metadata(PER_SAMPLE_SCORES_KEY))


def base_score_signature(prog: Program) -> str:
    raw = prog.get_metadata(MUTATION_MEMORY_BASE_SCORE_SIGNATURE_METADATA_KEY)
    return raw if isinstance(raw, str) else ""


def child_score_signature(prog: Program) -> str:
    raw = prog.get_metadata(PER_SAMPLE_SIGNATURE_KEY)
    return raw if isinstance(raw, str) else ""


def _score_vector(raw: object) -> tuple[float, ...] | None:
    """Tolerant scalar-sequence coercion; None on anything else. Finiteness and
    coherence are the effect estimator's checks — this only shields extraction
    from raising on malformed metadata."""
    if not isinstance(raw, (list, tuple)) or not raw:
        return None
    try:
        return tuple(float(x) for x in raw)
    except (TypeError, ValueError):
        return None


def no_card_control(prog: Program) -> bool:
    """Whether this child was born from a randomized memory-withheld control."""
    return bool(prog.get_metadata(MUTATION_MEMORY_NO_CARD_CONTROL_METADATA_KEY))


def card_assignment_sources(prog: Program) -> dict[str, CardAssignmentSource]:
    """Per-card parent contexts frozen at child birth; empty for legacy rows."""
    raw = prog.get_metadata(MUTATION_MEMORY_CARD_PROVENANCE_METADATA_KEY)
    if not isinstance(raw, dict):
        return {}
    sources: dict[str, CardAssignmentSource] = {}
    for raw_card_id, value in raw.items():
        card_id = str(raw_card_id).strip()
        if not card_id or not isinstance(value, dict):
            continue
        try:
            sources[card_id] = CardAssignmentSource.model_validate(value)
        except Exception:
            logger.warning(
                "[Memory][Stats] ignoring malformed card provenance for {} on {}",
                card_id,
                prog.short_id,
            )
    return sources


def founding_gain_event(
    program: Program,
    *,
    fitness_key: str,
    higher_is_better: bool,
    metrics_context: MetricsContext,
    task_key: str = "",
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
    gain_se, _, _, _ = outcome_uncertainty(
        program,
        metric_key=fitness_key,
        child_fitness=child_fit,
        base_fitness=base_fit,
        higher_is_better=higher_is_better,
    )
    return ContextualGain(
        context=DecisionContext(
            task_key=task_key,
            parent_metrics=dict(bm),
            parent_id=base_id(program),
            timestamp=program.created_at,
        ),
        gain=delta,
        gain_se=gain_se,
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
    base_scores: tuple[float, ...] | None = Field(
        default=None,
        description="Base parent's per-sample score vector, frozen at birth "
        "alongside base_metrics; None when the eval is scalar-only.",
    )
    child_scores: tuple[float, ...] | None = Field(
        default=None,
        description="This child's own per-sample score vector, read live at the "
        "write seam so it coheres with the live fitness; None when absent.",
    )
    base_score_signature: str = ""
    child_score_signature: str = ""
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
    card_sources: dict[str, CardAssignmentSource] = Field(default_factory=dict)
    card_base_fitness: dict[str, float | None] = Field(default_factory=dict)


def _card_source_outcome(
    outcome: InjectionOutcome, card_id: str, *, task_key: str
) -> tuple[InjectionOutcome, DecisionContext, tuple[str, str]]:
    source = outcome.card_sources.get(card_id)
    if source is None:
        context = DecisionContext(
            task_key=task_key,
            parent_metrics=dict(outcome.base_metrics),
            parent_id=outcome.base_id,
            timestamp=outcome.created_at,
        )
        return outcome, context, (outcome.base_id, "")
    source_context = source.source_context
    context = source_context.model_copy(
        update={
            "task_key": source_context.task_key or task_key,
            "timestamp": outcome.created_at,
        }
    )
    sourced = outcome.model_copy(
        update={
            "base_metrics": dict(source.parent_metrics),
            "base_id": source.parent_id,
            "base_fitness": outcome.card_base_fitness.get(card_id),
            "base_scores": source.parent_scores,
            "base_score_signature": source.parent_score_signature,
        }
    )
    return sourced, context, (source.parent_id, source.decision_id)


@runtime_checkable
class FittedNoCardBaseline(Protocol):
    """Resolved no-card baseline for one stats sweep."""

    has_evidence: bool

    def baseline_for(self, outcome: InjectionOutcome) -> float: ...

    def baseline_se_for(self, outcome: InjectionOutcome) -> float | None:
        """Sampling se of the fitted location, or None when not modeled."""
        ...


@runtime_checkable
class NoCardBaselineEstimator(Protocol):
    """Pluggable context model for expected no-card child-parent progress."""

    def fit_no_card_baseline(
        self,
        outcomes: Sequence[InjectionOutcome],
        *,
        higher_is_better: bool,
    ) -> FittedNoCardBaseline: ...


def _combined_se(measured_se: float | None, baseline_se: float | None) -> float | None:
    """Combine measured and fitted-baseline uncertainty without inflating exacts."""
    if measured_se is None:
        return None
    measured = float(measured_se)
    if measured == 0.0:
        return 0.0
    if not math.isfinite(measured):
        return None
    fitted = 0.0 if baseline_se is None else float(baseline_se)
    if not math.isfinite(fitted) or fitted < 0.0:
        fitted = 0.0
    return float(np.hypot(measured, fitted))


def compute_contextual_gains(
    programs: Sequence[InjectionOutcome],
    *,
    higher_is_better: bool = True,
    baseline_estimator: NoCardBaselineEstimator | None = None,
    effect_estimator: EffectEstimator | None = None,
    task_key: str = "",
) -> dict[str, list[ContextualGain]]:
    """Map each prompt-selected card id to baseline-adjusted outcome events.

    First learn the run-local no-card baseline from children with a frozen base
    baseline and an empty selected slate, preferring rows created by the randomized
    no-card control arm when any exist (invalid controls are excluded — a crash
    has no honest progress magnitude). The estimator is injectable: the default
    is a global median, while contextual reputations can fit per-cell/per-regime
    baselines behind the same method. Used cards share one ``child-parent -
    no_card_baseline`` event carrying the full delta with ``credit_weight=1/k``
    (splitting the delta too would price a bundled win at ``1/k**2``). Selected
    but unused cards receive a zero-gain ``unused=True`` exposure failure at
    ``credit_weight=1/len(selected)`` — an ignore in a crowded prompt is weak
    evidence — and only once a no-card baseline has evidence (invalid children
    still emit forced-harm / unused events).

    ``effect_estimator`` owns how the child-vs-base effect becomes a
    ``(value, se)`` measurement (default: exact point delta, ``se=0``). Exact
    measurements stay exact; positive measured se folds the fitted baseline's
    location se in quadrature, while a degraded ``None`` se stays unknown.
    Invalid/unused events are exact binary observations and never carry an se.
    """
    events: dict[str, list[ContextualGain]] = {}
    estimator = (
        effect_estimator if effect_estimator is not None else PointEffectEstimator()
    )
    baseline = _fit_no_card_baseline(
        baseline_estimator, programs, higher_is_better=higher_is_better
    )
    has_baseline_evidence = bool(baseline.has_evidence)
    for p in programs:
        if not p.base_selected_ids:
            continue
        selected = clean_ids(p.base_selected_ids)
        used = selected & clean_ids(p.card_ids_used)
        unused = selected - used
        if not selected:
            continue
        gain_cache: dict[tuple[str, str], ContextualGain] = {}
        if used and p.invalid:
            for card_id in used:
                sourced, context, source_key = _card_source_outcome(
                    p, card_id, task_key=task_key
                )
                if sourced.base_fitness is None:
                    continue
                gain_event = gain_cache.get(source_key)
                if gain_event is None:
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
                    gain_cache[source_key] = gain_event
                events.setdefault(card_id, []).append(gain_event)
        elif used and p.fitness is not None and has_baseline_evidence:
            for card_id in used:
                sourced, context, source_key = _card_source_outcome(
                    p, card_id, task_key=task_key
                )
                if sourced.base_fitness is None:
                    continue
                gain_event = gain_cache.get(source_key)
                if gain_event is None:
                    measured = estimator.estimate(
                        sourced, higher_is_better=higher_is_better
                    )
                    delta = measured.value - baseline.baseline_for(sourced)
                    gain_se = _combined_se(
                        measured.se, baseline.baseline_se_for(sourced)
                    )
                    gain_event = ContextualGain(
                        context=context,
                        gain=delta,
                        gain_se=gain_se,
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
                    gain_cache[source_key] = gain_event
                events.setdefault(card_id, []).append(gain_event)
        if unused and (has_baseline_evidence or p.invalid):
            for card_id in unused:
                sourced, context, _ = _card_source_outcome(
                    p, card_id, task_key=task_key
                )
                if sourced.base_fitness is None:
                    continue
                unused_event = ContextualGain(
                    context=context,
                    gain=0.0,
                    unused=True,
                    attribution=EvidenceAttribution(
                        source=EvidenceSource.UNUSED,
                        causal_strength=CausalStrength.EXPOSURE,
                        source_child_id=p.id,
                        used_card_count=len(used),
                        credit_weight=1.0 / len(selected),
                    ),
                )
                events.setdefault(card_id, []).append(unused_event)
    return events


class GlobalNoCardBaseline(BaseModel):
    """Default no-card estimator: median delta over all no-card children."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    def fit_no_card_baseline(
        self,
        outcomes: Sequence[InjectionOutcome],
        *,
        higher_is_better: bool,
    ) -> FittedNoCardBaseline:
        del self
        deltas = _no_card_deltas(outcomes, higher_is_better)
        baseline_se: float | None = None
        if len(deltas) >= 2:
            candidate = float(np.std(deltas, ddof=1) / np.sqrt(len(deltas)))
            if math.isfinite(candidate):
                baseline_se = candidate
        return _ConstantNoCardBaseline(
            baseline=median(deltas),
            baseline_se=baseline_se,
            has_evidence=bool(deltas),
        )


class _ConstantNoCardBaseline(BaseModel):
    model_config = ConfigDict(frozen=True)

    baseline: float = 0.0
    baseline_se: float | None = None
    has_evidence: bool = False

    def baseline_for(self, outcome: InjectionOutcome) -> float:
        del outcome
        return self.baseline

    def baseline_se_for(self, outcome: InjectionOutcome) -> float | None:
        del outcome
        return self.baseline_se


def _fit_no_card_baseline(
    estimator: NoCardBaselineEstimator | None,
    outcomes: Sequence[InjectionOutcome],
    *,
    higher_is_better: bool,
) -> FittedNoCardBaseline:
    if estimator is None:
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
        if p.base_fitness is not None and not clean_ids(p.base_selected_ids)
    ]
    controls = [p for p in no_card if p.no_card_control]
    cohort = controls if controls else no_card
    deltas: list[float] = []
    for p in cohort:
        if p.base_fitness is None or p.invalid or p.fitness is None:
            continue
        delta = oriented_delta(p.fitness, p.base_fitness, higher_is_better)
        if delta is not None and math.isfinite(delta):
            deltas.append(delta)
    return deltas


def card_gain_events_from_programs(
    programs: list[Program],
    *,
    fitness_key: str,
    higher_is_better: bool,
    metrics_context: MetricsContext,
    baseline_estimator: NoCardBaselineEstimator | None = None,
    effect_estimator: EffectEstimator | None = None,
    task_key: str = "",
) -> dict[str, list[ContextualGain]]:
    """Selected-card base-relative outcome events per card id, from live programs.

    Used cards split the child delta; selected-but-unused cards receive an
    explicit unused exposure. The base fitness baseline is resolved here from
    the frozen base metrics under ``fitness_key``.
    """
    rows = injection_outcomes_from_programs(
        programs, fitness_key=fitness_key, metrics_context=metrics_context
    )
    return compute_contextual_gains(
        rows,
        higher_is_better=higher_is_better,
        baseline_estimator=baseline_estimator,
        effect_estimator=effect_estimator,
        task_key=task_key,
    )


def injection_outcomes_from_programs(
    programs: Sequence[Program],
    *,
    fitness_key: str,
    metrics_context: MetricsContext,
) -> list[InjectionOutcome]:
    rows = []
    for prog in programs:
        bm = base_metrics(prog)
        sources = card_assignment_sources(prog)
        rows.append(
            InjectionOutcome(
                id=prog.id,
                fitness=metrics_context.strict_fitness(prog.metrics, fitness_key),
                invalid=metrics_context.is_evaluated_invalid(prog.metrics, fitness_key),
                base_selected_ids=tuple(creditable_selected_ids(prog)),
                base_metrics=bm,
                base_id=base_id(prog),
                base_fitness=metrics_context.strict_fitness(bm, fitness_key),
                base_scores=base_scores(prog),
                child_scores=child_scores(prog),
                base_score_signature=base_score_signature(prog),
                child_score_signature=child_score_signature(prog),
                created_at=prog.created_at,
                card_ids_used=tuple(card_ids_used(prog)),
                no_card_control=no_card_control(prog),
                card_sources=sources,
                card_base_fitness={
                    card_id: metrics_context.strict_fitness(
                        source.parent_metrics, fitness_key
                    )
                    for card_id, source in sources.items()
                },
            )
        )
    return rows


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
        A card carrying historical ``absorbed_ids`` also folds in events the pool
        still attributes to those ids, so frozen child attribution does not
        orphan on a retired alias.

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
        carried on the card and preserved here across the recompute. They never
        double-count: the pool only ever holds use events, so the absorbed-id fold
        below cannot re-add one.

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
        if event.attribution is None:
            return True
        source_child_id = event.attribution.source_child_id
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
        effect_estimator: EffectEstimator | None = None,
        no_card_recorder: NoCardEvidenceRecorder | None = None,
        task_key: str = "",
        selection_leases: InFlightSelectionRegistry | None = None,
    ) -> None:
        self._fitness_key = fitness_key
        self._higher_is_better = higher_is_better
        self._metrics_context = metrics_context
        self._baseline_estimator = baseline_estimator
        self._effect_estimator = effect_estimator
        self._no_card_recorder = no_card_recorder
        self._task_key = task_key
        self._selection_leases = selection_leases
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
        rows = injection_outcomes_from_programs(
            pool, fitness_key=self._fitness_key, metrics_context=self._metrics_context
        )
        if self._no_card_recorder is not None:
            try:
                self._no_card_recorder.record_outcomes(
                    rows,
                    higher_is_better=self._higher_is_better,
                    task_key=self._task_key,
                )
            except Exception as exc:
                logger.warning(
                    "[Memory][Stats] failed to persist no-card evidence: {}; "
                    "continuing card restamp",
                    exc,
                )
        events = compute_contextual_gains(
            rows,
            higher_is_better=self._higher_is_better,
            baseline_estimator=self._baseline_estimator,
            effect_estimator=self._effect_estimator,
            task_key=self._task_key,
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
        bank = store.snapshot()
        redirected = 0
        dropped = 0
        for card in bank:
            scheduled = stamper.stamp_gain_events(
                card,
                card_gain_events,
                preserve_events_outside_child_ids=preserve_events_outside_child_ids,
            )
            if scheduled.gain_events == card.gain_events:
                continue

            def restamp(fresh: Card) -> Card:
                return stamper.stamp_gain_events(
                    fresh,
                    card_gain_events,
                    preserve_events_outside_child_ids=preserve_events_outside_child_ids,
                )

            if store.update(card.id, restamp) is None:
                if self._reconcile_vanished_restamp(
                    card.id,
                    restamp,
                    store=store,
                    hop_budget=max(1, len(bank)),
                ):
                    redirected += 1
                else:
                    dropped += 1
        if redirected or dropped:
            logger.debug(
                "[Memory][Stats] restamp not-found reconciliation redirected={} "
                "dropped={}",
                redirected,
                dropped,
            )
        gate.sweep()
        if (
            self._selection_leases is not None
            and preserve_events_outside_child_ids is not None
        ):
            for child_id in preserve_events_outside_child_ids:
                self._selection_leases.release_child(child_id)

    @staticmethod
    def _reconcile_vanished_restamp(
        card_id: str,
        restamp: Callable[[Card], Card | None],
        *,
        store: MemoryStore,
        hop_budget: int,
    ) -> bool:
        alias_id = card_id
        seen = {card_id}
        for hop in range(hop_budget):
            absorber = next(
                (card for card in store.snapshot() if alias_id in card.absorbed_ids),
                None,
            )
            if absorber is None:
                logger.debug(
                    "[Memory][Stats] dropping restamp for evicted card {} "
                    "with no absorbing survivor",
                    card_id,
                )
                return False
            if absorber.id in seen:
                logger.warning(
                    "[Memory][Stats] dropping restamp for card {} after cyclic "
                    "absorbed-id chain at {}",
                    card_id,
                    absorber.id,
                )
                return False
            seen.add(absorber.id)
            if hop:
                logger.warning(
                    "[Memory][Stats] following multi-hop restamp alias for card {} "
                    "through {}",
                    card_id,
                    absorber.id,
                )
            if store.update(absorber.id, restamp) is not None:
                return True
            alias_id = absorber.id
        logger.warning(
            "[Memory][Stats] dropping restamp for card {} after exhausting {} "
            "absorbed-id hop(s)",
            card_id,
            hop_budget,
        )
        return False
