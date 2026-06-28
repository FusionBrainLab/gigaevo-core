"""Pure card-union helper for the memory write path.

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

from gigaevo.memory.context import ContextualGain
from gigaevo.memory.shared_memory.models import MemoryCard


def merge_cards(
    target: MemoryCard, incoming: MemoryCard, *, replace_description: bool
) -> MemoryCard:
    return target.model_copy(
        update={
            "description": incoming.description
            if replace_description
            else target.description,
            "explanation_summary": incoming.explanation_summary
            if replace_description
            else target.explanation_summary,
            "keywords": list(incoming.keywords)
            if replace_description
            else _union(target.keywords, incoming.keywords),
            "programs": _union(target.programs, incoming.programs),
            "gain_events": _union_events(target.gain_events, incoming.gain_events),
            "task_description": target.task_description or incoming.task_description,
            "task_description_summary": target.task_description_summary
            or incoming.task_description_summary,
        }
    )


def _union(a: list[str], b: list[str]) -> list[str]:
    out: list[str] = []
    for item in [*a, *b]:
        if item not in out:
            out.append(item)
    return out


def _union_events(
    a: list[ContextualGain] | None, b: list[ContextualGain] | None
) -> list[ContextualGain] | None:
    out: list[ContextualGain] = []
    for event in [*(a or []), *(b or [])]:
        if event not in out:
            out.append(event)
    return out or None
