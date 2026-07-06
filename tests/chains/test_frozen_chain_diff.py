"""Tests for FrozenTopologyChainChanges: prompt-only diffs on a frozen DAG."""

from __future__ import annotations

import json

from pydantic import ValidationError
import pytest

from gigaevo.exceptions import MutationError
from gigaevo.llm.schema_compat import nonportable_keys
from problems.chains.chain_validation import validate_chain_spec
from problems.chains.frozen_chain_diff import FrozenTopologyChainChanges
from problems.chains.hover.full7.config import FULL_CHAIN_CONFIG, load_baseline

EVIDENCE = {"archetype": "Guided Innovation", "justification": "polish prompts"}
LLM_POSITIONS = (2, 3, 5, 6)  # baseline: tool, llm, llm, tool, llm, llm, tool


def _keep_all(ns: str = "a") -> dict:
    return {f"slot_{p}": {"id": f"{ns}{i + 1}"} for i, p in enumerate(LLM_POSITIONS)}


@pytest.fixture
def hover_parent() -> dict[str, str]:
    return {"A": json.dumps(load_baseline())}


@pytest.fixture
def changes() -> FrozenTopologyChainChanges:
    return FrozenTopologyChainChanges()


def test_schema_has_slots_only_for_llm_positions(changes, hover_parent):
    schema = changes.build_schema(hover_parent).json_schema
    slots = {k for k in schema["properties"] if k.startswith("slot_")}
    assert slots == {f"slot_{p}" for p in LLM_POSITIONS}
    assert all(f"slot_{p}" in schema["required"] for p in LLM_POSITIONS)


def test_schema_exposes_no_topology_freedom(changes, hover_parent):
    schema = changes.build_schema(hover_parent).json_schema
    dumped = json.dumps(schema)
    for forbidden in ("new_llm", "new_tool", "tool_name", "dependencies"):
        assert forbidden not in dumped
    slot = schema["properties"]["slot_2"]
    assert set(slot["properties"]) == {"id", "edits"}
    assert slot["properties"]["id"]["enum"] == ["a1"]


def test_emitted_schema_is_portable(changes, hover_parent):
    assert nonportable_keys(changes.build_schema(hover_parent).json_schema) == set()


def test_keep_all_reproduces_parent_chain(changes, hover_parent):
    schema = changes.build_schema(hover_parent)
    diff = schema.validate({**EVIDENCE, "base_parent": "A", **_keep_all()})
    child = json.loads(changes.apply(diff, parents=hover_parent))
    parent = load_baseline()
    assert [s["step_type"] for s in child["steps"]] == [
        s["step_type"] for s in parent["steps"]
    ]
    assert [s["dependencies"] for s in child["steps"]] == [
        s.get("dependencies", []) for s in parent["steps"]
    ]
    child_tools = [s["step_config"] for s in child["steps"] if s["step_type"] == "tool"]
    parent_tools = [
        s["step_config"] for s in parent["steps"] if s["step_type"] == "tool"
    ]
    assert child_tools == parent_tools
    validate_chain_spec(child, mode="full_chain", full_chain_config=FULL_CHAIN_CONFIG)


def test_edits_override_prompt_fields_but_not_wiring(changes, hover_parent):
    schema = changes.build_schema(hover_parent)
    payload = {**EVIDENCE, "base_parent": "A", **_keep_all()}
    payload["slot_3"] = {
        "id": "a2",
        "edits": {"aim": "sharper second-hop gap analysis"},
    }
    diff = schema.validate(payload)
    child = json.loads(changes.apply(diff, parents=hover_parent))
    parent = load_baseline()
    edited = next(s for s in child["steps"] if s["number"] == 3)
    original = next(s for s in parent["steps"] if s["number"] == 3)
    assert edited["aim"] == "sharper second-hop gap analysis"
    assert edited["stage_action"] == original["stage_action"]
    assert edited["dependencies"] == original["dependencies"]


def test_slot_at_tool_position_is_rejected(changes, hover_parent):
    schema = changes.build_schema(hover_parent)
    payload = {**EVIDENCE, "base_parent": "A", **_keep_all()}
    payload["slot_1"] = {"id": "a1"}
    with pytest.raises(ValidationError):
        schema.validate(payload)


def test_missing_slot_is_rejected(changes, hover_parent):
    schema = changes.build_schema(hover_parent)
    payload = {**EVIDENCE, "base_parent": "A", **_keep_all()}
    del payload["slot_6"]
    with pytest.raises(ValidationError):
        schema.validate(payload)


def test_empty_edit_field_is_rejected(changes, hover_parent):
    schema = changes.build_schema(hover_parent)
    payload = {**EVIDENCE, "base_parent": "A", **_keep_all()}
    payload["slot_2"] = {"id": "a1", "edits": {"aim": ""}}
    with pytest.raises(ValidationError):
        schema.validate(payload)


def test_two_parents_offer_both_ids_per_slot(changes, hover_parent):
    parents = {**hover_parent, "B": hover_parent["A"]}
    schema = changes.build_schema(parents).json_schema
    assert set(schema["properties"]["slot_2"]["properties"]["id"]["enum"]) == {
        "a1",
        "b1",
    }


def test_cross_parent_keep_takes_other_parents_text(changes, hover_parent):
    other = load_baseline()
    step = next(s for s in other["steps"] if s["number"] == 2)
    step["aim"] = "parent B's distinctive aim"
    parents = {**hover_parent, "B": json.dumps(other)}
    schema = changes.build_schema(parents)
    payload = {**EVIDENCE, "base_parent": "A", **_keep_all()}
    payload["slot_2"] = {"id": "b1"}
    diff = schema.validate(payload)
    child = json.loads(changes.apply(diff, parents=parents))
    assert (
        next(s for s in child["steps"] if s["number"] == 2)["aim"]
        == "parent B's distinctive aim"
    )


def test_mismatched_parent_topologies_are_rejected(changes, hover_parent):
    other = load_baseline()
    other["steps"] = other["steps"][:-1]
    parents = {**hover_parent, "B": json.dumps(other)}
    with pytest.raises(MutationError, match="frozen_topology_mismatch"):
        changes.build_schema(parents)


def test_mismatched_parent_step_numbers_are_rejected(changes, hover_parent):
    other = load_baseline()
    other["steps"][1]["number"] = 20
    parents = {**hover_parent, "B": json.dumps(other)}
    with pytest.raises(MutationError, match="frozen_topology_mismatch"):
        changes.build_schema(parents)


def test_parse_rejects_unparseable_parent(changes):
    with pytest.raises(MutationError, match="carl_validation_error"):
        changes.build_schema({"A": "{not json"})


def test_parse_rejects_empty_parents(changes):
    with pytest.raises(MutationError, match="no parents"):
        changes.build_schema({})
