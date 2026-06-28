"""Behavior tests for ``merge_cards``: the pure card-union helper.

A merge folds an ``incoming`` card into a ``target`` card without losing the
target's accumulated evidence. Provenance, keywords, and gain events are
unioned; task fields fall back to whichever side carries them; the survivor
keeps the target's id and category. ``replace_description`` chooses whose prose
wins (MERGE replaces with synthesized union prose; provenance bumps keep it).
"""

from __future__ import annotations

from gigaevo.memory.context import ContextualGain, DecisionContext
from gigaevo.memory.shared_memory.card_merge import merge_cards
from gigaevo.memory.shared_memory.models import MemoryCard


def _gain(value: float) -> ContextualGain:
    return ContextualGain(
        context=DecisionContext(parent_metrics={"f": value}), gain=value
    )


def test_programs_are_unioned_order_preserving_without_dups() -> None:
    target = MemoryCard(id="mem-A", description="t", programs=["p1", "p2"])
    incoming = MemoryCard(id="", description="i", programs=["p2", "p3"])
    merged = merge_cards(target, incoming, replace_description=True)
    assert merged.programs == ["p1", "p2", "p3"]


def test_keywords_replaced_with_incoming_on_merge() -> None:
    # A MERGE (replace_description=True) carries an agent-authored union card
    # whose keywords are the curated few-most-distinctive set across both cards.
    # Re-unioning them with the target's old list would undo that curation and
    # re-bloat the survivor, so the curated set wins outright.
    target = MemoryCard(id="mem-A", description="t", keywords=["a", "b"])
    incoming = MemoryCard(id="", description="i", keywords=["b", "c"])
    merged = merge_cards(target, incoming, replace_description=True)
    assert merged.keywords == ["b", "c"]


def test_keywords_unioned_on_provenance_bump() -> None:
    # A provenance bump (replace_description=False) is not an authored union — no
    # agent curated its keywords — so the target's accumulated keywords must be
    # preserved by unioning, never dropped.
    target = MemoryCard(id="mem-A", description="t", keywords=["a", "b"])
    incoming = MemoryCard(id="", description="i", keywords=["b", "c"])
    merged = merge_cards(target, incoming, replace_description=False)
    assert merged.keywords == ["a", "b", "c"]


def test_gain_events_are_unioned_and_dedup_by_value() -> None:
    shared = _gain(0.1)
    target = MemoryCard(id="mem-A", description="t", gain_events=[shared, _gain(0.2)])
    incoming = MemoryCard(id="", description="i", gain_events=[shared, _gain(0.3)])
    merged = merge_cards(target, incoming, replace_description=True)
    assert merged.gain_events == [_gain(0.1), _gain(0.2), _gain(0.3)]


def test_gain_events_none_when_both_empty() -> None:
    target = MemoryCard(id="mem-A", description="t")
    incoming = MemoryCard(id="", description="i")
    merged = merge_cards(target, incoming, replace_description=True)
    assert merged.gain_events is None


def test_replace_description_true_takes_incoming_prose() -> None:
    target = MemoryCard(id="mem-A", description="old")
    incoming = MemoryCard(id="", description="synthesized union")
    merged = merge_cards(target, incoming, replace_description=True)
    assert merged.description == "synthesized union"


def test_replace_description_false_keeps_target_prose() -> None:
    target = MemoryCard(id="mem-A", description="canonical")
    incoming = MemoryCard(id="", description="ignored")
    merged = merge_cards(target, incoming, replace_description=False)
    assert merged.description == "canonical"


def test_explanation_summary_replaced_with_incoming_when_replacing_description() -> (
    None
):
    # explanation_summary is the description's "why" twin: a MERGE that replaces
    # the prose must replace its explanation_summary too, not keep a stale one.
    target = MemoryCard(id="mem-A", description="old", explanation_summary="old why")
    incoming = MemoryCard(id="", description="union", explanation_summary="union why")
    merged = merge_cards(target, incoming, replace_description=True)
    assert merged.explanation_summary == "union why"


def test_explanation_summary_kept_from_target_on_provenance_bump() -> None:
    target = MemoryCard(
        id="mem-A", description="t", explanation_summary="canonical why"
    )
    incoming = MemoryCard(id="", description="i", explanation_summary="ignored why")
    merged = merge_cards(target, incoming, replace_description=False)
    assert merged.explanation_summary == "canonical why"


def test_survivor_keeps_target_id_and_category() -> None:
    target = MemoryCard(id="mem-A", category="general", description="t")
    incoming = MemoryCard(id="mem-B", category="other", description="i")
    merged = merge_cards(target, incoming, replace_description=True)
    assert merged.id == "mem-A"
    assert merged.category == "general"


def test_task_fields_fall_back_to_incoming_when_target_empty() -> None:
    target = MemoryCard(id="mem-A", description="t")
    incoming = MemoryCard(
        id="",
        description="i",
        task_description="full task",
        task_description_summary="short",
    )
    merged = merge_cards(target, incoming, replace_description=True)
    assert merged.task_description == "full task"
    assert merged.task_description_summary == "short"


def test_task_fields_keep_target_when_present() -> None:
    target = MemoryCard(
        id="mem-A",
        description="t",
        task_description="target task",
        task_description_summary="target short",
    )
    incoming = MemoryCard(
        id="",
        description="i",
        task_description="incoming task",
        task_description_summary="incoming short",
    )
    merged = merge_cards(target, incoming, replace_description=True)
    assert merged.task_description == "target task"
    assert merged.task_description_summary == "target short"
