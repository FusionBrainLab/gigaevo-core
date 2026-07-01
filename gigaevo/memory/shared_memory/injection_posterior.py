"""Use-attributed, base-relative card gain events at the tracker write seam.

A memory card is credited for a child only when it was both selected for the
mutator's named base parent (``base_selected_ids``) and declared applied by the
mutator (``card_ids_used``) — the intersection. Each such credit is one
``ContextualGain``: the child's base-relative fitness delta, tagged with the
base parent's metrics as its decision context. Reputation
(gigaevo/memory/core/reputation.py) computes every per-card statistic from these
stored events at read time; this module only produces them.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from gigaevo.memory.context import ContextualGain, DecisionContext


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
    is the base parent's id and metrics plus the child's creation time. An
    invalid child emits one forced-harm event
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
