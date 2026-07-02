"""Pure card-union helper plus the dedup thresholds of the write path.

Folding one idea card into another must never drop the survivor's accumulated
evidence. ``merge_cards`` unions provenance and gain events; lets task fields
fall back to whichever side carries them; and keeps the target's id and
category. ``replace_description`` selects whose prose wins — and keywords follow
the prose: a MERGE replaces them with the author's curated union set (so the
author's de-bloated keyword choice is not re-inflated by re-unioning the old
list), while a provenance bump unions them (no author curated that incoming card,
so the target's accumulated keywords must be preserved). No I/O, no LLM — the
gate and the consolidation pass both route their unions through here.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from gigaevo.memory.cards import Card, ContextualGain


class DedupPolicy(BaseModel):
    """The dedup knobs of the librarian write path.

    One frozen value object groups every dedup knob — the reconcile agent's
    neighbor-context width per mutation diff, the exemplar twin threshold, and
    the batch consolidation candidate width — so they live in one
    Hydra-instantiable node instead of scattered scalar defaults on the
    Librarian, the consolidation pass, and the scheduler. Idea-card dedup is
    LLM-arbitrated end to end: recall is top-k by rank, with no embedding
    distance threshold anywhere (same-lever cards average only ~0.78 cosine
    similarity on this geometry, so any useful cosine gate would either be
    inert or force-merge distinct levers).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    online_top_k: int = Field(
        default=5,
        description="Neighbours fetched per diff as the reconcile agent's "
        "context — the agent is the only dedup arbiter on the idea path.",
    )
    max_cards_per_diff: int = Field(
        default=3,
        description="Upper bound on cards authored from a single mutation diff.",
    )
    program_twin_eps: float = Field(
        default=0.05,
        description="Cosine distance below which a banked program exemplar with "
        "a different id counts as a same-strategy twin of an incoming exemplar; "
        "the higher-fitness one is kept. Exemplar dedup has no LLM arbiter, so "
        "this is the only distance threshold in the write path.",
    )
    consolidation_k: int = Field(
        default=5,
        description="Neighbours fetched per card during a consolidation pass; "
        "every one is offered to the consolidate agent as a merge candidate "
        "(pure top-k, no distance cut).",
    )


def merge_cards(target: Card, incoming: Card, *, replace_description: bool) -> Card:
    return target.model_copy(
        update={
            "description": incoming.description
            if replace_description
            else target.description,
            "explanation_summary": incoming.explanation_summary
            if replace_description
            else target.explanation_summary,
            "keywords": incoming.keywords
            if replace_description
            else _union(target.keywords, incoming.keywords),
            "programs": _union(target.programs, incoming.programs),
            "absorbed_ids": _absorbed_ids(target, incoming),
            "gain_events": _union_events(target.gain_events, incoming.gain_events),
            "task_description": target.task_description or incoming.task_description,
            "task_description_summary": target.task_description_summary
            or incoming.task_description_summary,
        }
    )


def _union(a: Sequence[str], b: Sequence[str]) -> tuple[str, ...]:
    out: list[str] = []
    for item in [*a, *b]:
        if item not in out:
            out.append(item)
    return tuple(out)


def _absorbed_ids(target: Card, incoming: Card) -> tuple[str, ...]:
    # The survivor keeps target's id; the folded-away incoming id (and any ids the
    # incoming card had itself absorbed) become aliases of the survivor. Skip blank
    # ids (the librarian folds freshly-authored id="" cards) and never alias the
    # survivor onto itself.
    out = list(target.absorbed_ids)
    for aid in [*incoming.absorbed_ids, incoming.id]:
        if aid and aid != target.id and aid not in out:
            out.append(aid)
    return tuple(out)


def _union_events(
    a: Sequence[ContextualGain], b: Sequence[ContextualGain]
) -> tuple[ContextualGain, ...]:
    out: list[ContextualGain] = []
    for event in [*a, *b]:
        if event not in out:
            out.append(event)
    return tuple(out)
