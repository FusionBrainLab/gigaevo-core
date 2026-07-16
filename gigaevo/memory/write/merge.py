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
    neighbor-context width per mutation diff and the batch consolidation
    candidate width — so they live in one Hydra-instantiable node instead of
    scattered scalar defaults on the Librarian, the consolidation pass, and the
    scheduler. Idea-card dedup is LLM-arbitrated end to end: recall is top-k by
    rank, with no embedding distance threshold anywhere (same-lever cards average
    only ~0.78 cosine similarity on this geometry, so any useful cosine gate
    would either be inert or force-merge distinct levers). Program-exemplar
    controls live in ``ProgramExemplarPolicy`` because their constraints are
    different: bounded top-k authoring, a hard archive cap, and code-hash
    identity.
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
    consolidation_k: int = Field(
        default=5,
        description="Neighbours fetched per card during a consolidation pass; "
        "every one is offered to the consolidate agent as a merge candidate "
        "(pure top-k, no distance cut).",
    )
    preserve_survivor_payload: bool = Field(
        default=False,
        description="Keep the banked card's mutator-facing text when an incoming "
        "card is ruled equivalent. This lets causal evidence remain attached to "
        "one stable intervention while provenance is merged.",
    )


class ProgramExemplarPolicy(BaseModel):
    """Bounded write policy for program exemplar cards.

    Program cards are not durable ideas; they are a small reference shelf of
    high-fitness concrete exemplars. The policy therefore caps both per-refresh
    authoring and total bank residency, and stores only a source hash by default
    so cards do not consume the bank with large program bodies.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = Field(
        default=True,
        description="Whether the writer authors and maintains program exemplars.",
    )
    top_k_per_refresh: int | None = Field(
        default=4,
        ge=0,
        description="Maximum number of top programs authored per write refresh; "
        "None keeps all programs selected by best_programs_percent.",
    )
    max_cards: int = Field(
        default=12,
        ge=0,
        description="Per-task hard cap on program exemplar cards kept in the bank.",
    )
    min_fitness_gap: float = Field(
        default=0.0,
        ge=0.0,
        description="Absolute improvement required to replace an existing "
        "same-code exemplar. Default is strict improvement with no task-scale "
        "epsilon.",
    )
    store_code: bool = Field(
        default=False,
        description="When false, new program cards keep only code_sha256, not "
        "the full source body.",
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
            else union_strings(target.keywords, incoming.keywords),
            "programs": union_strings(target.programs, incoming.programs),
            "absorbed_ids": _absorbed_ids(target, incoming),
            "gain_events": union_events(target.gain_events, incoming.gain_events),
            "task_key": target.task_key,
            "task_description": target.task_description or incoming.task_description,
            "task_description_summary": target.task_description_summary
            or incoming.task_description_summary,
        }
    )


def union_strings(a: Sequence[str], b: Sequence[str]) -> tuple[str, ...]:
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


def union_events(
    a: Sequence[ContextualGain], b: Sequence[ContextualGain]
) -> tuple[ContextualGain, ...]:
    out: list[ContextualGain] = []
    for event in [*a, *b]:
        if event not in out:
            out.append(event)
    return tuple(out)
