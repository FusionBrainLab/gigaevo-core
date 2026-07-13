"""merge_cards union semantics and DedupPolicy invariants."""

from __future__ import annotations

from pydantic import ValidationError
import pytest

from gigaevo.memory.write.merge import (
    DedupPolicy,
    ProgramExemplarPolicy,
    merge_cards,
)


def test_replace_description_takes_incoming_prose_and_keywords(make_card):
    target = make_card(keywords=("old", "stale"))
    incoming = make_card(
        description="union prose",
        explanation_summary="union why",
        keywords=("curated",),
    )
    merged = merge_cards(target, incoming, replace_description=True)
    assert merged.description == "union prose"
    assert merged.explanation_summary == "union why"
    assert merged.keywords == ("curated",)


def test_keep_description_unions_keywords(make_card):
    target = make_card(keywords=("a", "b"))
    incoming = make_card(keywords=("b", "c"))
    merged = merge_cards(target, incoming, replace_description=False)
    assert merged.description == target.description
    assert merged.explanation_summary == target.explanation_summary
    assert merged.keywords == ("a", "b", "c")


def test_survivor_keeps_target_identity(make_card):
    target = make_card(category="general")
    incoming = make_card(id="other-id")
    merged = merge_cards(target, incoming, replace_description=True)
    assert merged.id == target.id
    assert merged.category == target.category


def test_survivor_keeps_target_task_key(make_card):
    target = make_card(task_key="authoring-task")
    incoming = make_card(task_key="consolidating-task")

    merged = merge_cards(target, incoming, replace_description=True)

    assert merged.task_key == "authoring-task"


def test_programs_union_preserves_order_and_dedups(make_card):
    target = make_card(programs=("p1", "p2"))
    incoming = make_card(programs=("p2", "p3"))
    merged = merge_cards(target, incoming, replace_description=True)
    assert merged.programs == ("p1", "p2", "p3")


def test_absorbed_ids_fold_incoming_chain_and_id(make_card):
    target = make_card(absorbed_ids=("dead-1",))
    incoming = make_card(id="dead-3", absorbed_ids=("dead-2", "dead-1"))
    merged = merge_cards(target, incoming, replace_description=True)
    assert merged.absorbed_ids == ("dead-1", "dead-2", "dead-3")


def test_absorbed_ids_skip_blank_and_survivor(make_card):
    target = make_card()
    incoming = make_card(id="", absorbed_ids=(target.id,))
    merged = merge_cards(target, incoming, replace_description=True)
    assert merged.absorbed_ids == ()


def test_gain_events_union_dedups_by_value(make_card, make_event):
    shared = make_event(0.1)
    only_target = make_event(0.2)
    only_incoming = make_event(0.3)
    target = make_card(gain_events=(only_target, shared))
    incoming = make_card(gain_events=(shared, only_incoming))
    merged = merge_cards(target, incoming, replace_description=True)
    assert merged.gain_events == (only_target, shared, only_incoming)


def test_gain_events_empty_union_is_empty_tuple(make_card):
    merged = merge_cards(make_card(), make_card(), replace_description=True)
    assert merged.gain_events == ()


def test_task_fields_fall_back_to_incoming(make_card):
    incoming = make_card(task_description="task", task_description_summary="short")
    merged = merge_cards(make_card(), incoming, replace_description=False)
    assert merged.task_description == "task"
    assert merged.task_description_summary == "short"


def test_task_fields_prefer_target(make_card):
    target = make_card(task_description="mine", task_description_summary="mine-s")
    incoming = make_card(task_description="theirs", task_description_summary="theirs-s")
    merged = merge_cards(target, incoming, replace_description=True)
    assert merged.task_description == "mine"
    assert merged.task_description_summary == "mine-s"


def test_dedup_policy_defaults_and_frozen():
    policy = DedupPolicy()
    assert policy.online_top_k == 5
    assert policy.max_cards_per_diff == 3
    assert policy.consolidation_k == 5
    with pytest.raises(ValidationError):
        policy.consolidation_k = 7


def test_program_exemplar_policy_defaults_and_frozen():
    policy = ProgramExemplarPolicy()
    assert policy.enabled is True
    assert policy.top_k_per_refresh == 4
    assert policy.max_cards == 12
    assert policy.min_fitness_gap == 0.0
    assert policy.store_code is False
    with pytest.raises(ValidationError):
        policy.max_cards = 7


def test_program_exemplar_policy_rejects_negative_caps():
    with pytest.raises(ValidationError):
        ProgramExemplarPolicy(top_k_per_refresh=-1)
    with pytest.raises(ValidationError):
        ProgramExemplarPolicy(max_cards=-1)
