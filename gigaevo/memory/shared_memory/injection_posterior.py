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
    logger.debug(
        "[Memory][InjectionPosterior] {} card(s) credited from {} scorable child(ren); "
        "noise band epsilon={:.4g}, confident={}",
        len(posteriors),
        len(cohort),
        fitted.epsilon,
        sum(1 for block in posteriors.values() if block.efficacy_confident),
    )
    return posteriors
