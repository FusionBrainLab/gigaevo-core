"""Per-program-card injection-efficacy posterior (Fix B bridge).

The Thompson auction (``ThompsonAuctioneer``) draws each candidate's downside
Beta-Binomial posterior, but its candidates are ``program-<uuid>`` cards. Their
posterior must therefore be keyed in that id-space and derived from how a card
performed *when injected into a mutation prompt*. Card selection is stamped on
the PARENT (``selected_ids``) and the child is the outcome: the cards a child's
prompt actually contained are the union of its parents' ``selected_ids``, so
each such card receives the child's parent-relative improvement as one event.
(A child's own ``selected_ids`` feed its future children's prompts and credit
nothing at its own birth — crediting them would measure selection bias one
generation off.)

The gain -> posterior math (parent-local counterfactual, noise band, downside
Beta-Binomial) lives in ``gigaevo.memory.efficacy.EfficacyScorer``;
``BetaBinomialReputation`` (gigaevo/memory/core/reputation.py) is the
injectable façade that binds its configured thresholds to that scorer.
"""

from __future__ import annotations

from collections.abc import Sequence

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from gigaevo.memory.context import ContextualGain, DecisionContext
from gigaevo.memory.core.events import emit_memory_event
from gigaevo.memory.efficacy import EfficacyScorer, GainObservation
from gigaevo.memory.shared_memory.models import CardStatsBlock


class InjectionOutcome(BaseModel):
    """One program's injection-relevant facts, extracted at the tracker seam.

    ``selected_ids`` are the cards stamped on THIS program at mutation time —
    they feed its future children's prompts. ``injected_ids`` is the child's
    birth-time frozen slate (union of parents' prompt-time selections); when
    present it overrides the parents' CURRENT ``selected_ids``, which a
    NO_CACHE requeue may have rewritten since the child was born. ``fitness``
    must already be the validated signal (``None`` for invalid or missing).
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(default="", description="Program id.")
    parents: list[str] = Field(default_factory=list, description="Parent program ids.")
    fitness: float | None = Field(
        default=None,
        description="Validated fitness signal; None for invalid or missing.",
    )
    selected_ids: list[str] = Field(
        default_factory=list,
        description="Card ids stamped on this program at mutation time.",
    )
    injected_ids: list[str] | None = Field(
        default=None,
        description="Birth-time frozen card slate from the child's own metadata; "
        "None for legacy programs without the stamp (fall back to parents' "
        "current selected_ids).",
    )
    invalid: bool = Field(
        default=False,
        description="Evaluated and judged invalid: one forced harm event per "
        "injected card, never part of the baseline cohort.",
    )
    base_selected_ids: list[str] = Field(
        default_factory=list,
        description="Cards selected for the mutator's named base parent, frozen "
        "onto this child at birth. Use-attribution credits only these.",
    )
    base_metrics: dict[str, float] = Field(
        default_factory=dict,
        description="The base parent's metric dict, frozen at birth — the decision "
        "context and the reward baseline source.",
    )
    base_fitness: float | None = Field(
        default=None,
        description="Base parent's fitness (base_metrics[fitness_key], resolved at "
        "the write seam); None means no base baseline, so no gain events.",
    )
    card_ids_used: list[str] = Field(
        default_factory=list,
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
    is the base parent's metrics. An invalid child emits one forced-harm event
    (gain 0.0, invalid) per credited card. Children with no frozen base baseline
    (``base_fitness is None`` or empty ``base_selected_ids``) contribute nothing.
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
        context = DecisionContext(parent_metrics=dict(p.base_metrics))
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


def compute_injection_posterior(
    programs: Sequence[InjectionOutcome],
    *,
    higher_is_better: bool = True,
    scorer: EfficacyScorer | None = None,
) -> dict[str, CardStatsBlock]:
    """Map each injected card id to its injection posterior.

    For each child program, every card in its birth-time frozen slate
    (``injected_ids``; legacy fallback: union of its resolvable parents'
    current ``selected_ids``) receives one intro event. A valid child's gain
    is the parent-relative improvement *minus* the parent-fitness-local
    leave-one-out counterfactual; it counts as harm only if it falls below the
    population's noise band. An evaluated-and-judged-invalid child is one
    forced harm event per injected card and never enters the baseline cohort.
    Cards never injected into a child with a valid parent baseline are absent
    from the result, which the auction treats as COLD Beta(1, 1).
    """
    by_id = {p.id: p for p in programs if p.id}
    cohort: list[GainObservation] = []
    events: dict[str, list[GainObservation]] = {}
    for p in programs:
        legacy_union: set[str] = set()
        parent_fits: list[float] = []
        for par_id in p.parents:
            parent = by_id.get(par_id)
            if parent is None:
                continue
            legacy_union |= {c for c in parent.selected_ids if c}
            if parent.fitness is not None:
                parent_fits.append(parent.fitness)
        if not parent_fits:
            continue
        card_ids = (
            {c for c in p.injected_ids if c}
            if p.injected_ids is not None
            else legacy_union
        )
        ref = max(parent_fits) if higher_is_better else min(parent_fits)
        if p.invalid:
            if card_ids:
                observation = GainObservation(
                    child_id=p.id, parent_fitness=ref, gain=0.0, invalid=True
                )
                for card_id in card_ids:
                    events.setdefault(card_id, []).append(observation)
            continue
        if p.fitness is None:
            continue
        gain = p.fitness - ref if higher_is_better else ref - p.fitness
        observation = GainObservation(child_id=p.id, parent_fitness=ref, gain=gain)
        cohort.append(observation)
        for card_id in card_ids:
            events.setdefault(card_id, []).append(observation)

    if not events:
        return {}

    fitted = (scorer if scorer is not None else EfficacyScorer()).fit(cohort)
    posteriors = {card_id: fitted.posterior(evs) for card_id, evs in events.items()}
    confident_count = sum(
        1 for block in posteriors.values() if block.efficacy_confident
    )
    emit_memory_event(
        component="InjectionPosterior",
        event_type="injection_posterior.compute",
        payload={
            "card_count": len(posteriors),
            "scorable_child_count": len(cohort),
            "epsilon": fitted.epsilon,
            "confident_count": confident_count,
            "event_count_by_card_id": {
                card_id: len(card_events) for card_id, card_events in events.items()
            },
        },
    )
    logger.debug(
        "[Memory][InjectionPosterior] {} card(s) credited from {} scorable child(ren); "
        "noise band epsilon={:.4g}, confident={}",
        len(posteriors),
        len(cohort),
        fitted.epsilon,
        confident_count,
    )
    return posteriors
