"""Tests for AllowedToolChainChanges: schema-valid diffs -> valid RawChainSpec chains."""

from __future__ import annotations

import json

from pydantic import ValidationError
import pytest

from gigaevo.exceptions import MutationError
from gigaevo.llm.schema_compat import nonportable_keys
from problems.chains.chain_runner import _resolve_reference
from problems.chains.chain_validation import validate_chain_spec
from problems.chains.hover.full7.config import FULL_CHAIN_CONFIG, load_baseline
from problems.chains.tool_chain_diff import AllowedToolChainChanges

TOOLS = FULL_CHAIN_CONFIG["available_tools"]
EVIDENCE = {"archetype": "Guided Innovation", "justification": "act on insight 1"}


@pytest.fixture
def hover_parent() -> dict[str, str]:
    return {"A": json.dumps(load_baseline())}


@pytest.fixture
def changes() -> AllowedToolChainChanges:
    return AllowedToolChainChanges(max_steps=7, available_tools=TOOLS)


def _object_branches(prop: dict) -> list[dict]:
    return [b for b in prop.get("anyOf", [prop]) if b.get("type") == "object"]


# --- parse / render ---------------------------------------------------------


def test_parse_accepts_tool_steps(changes, hover_parent):
    specs = changes._parse(hover_parent)
    assert len(specs["A"].steps) == 7


def test_render_lists_llm_ids_and_tool_sources(changes, hover_parent):
    rendered = changes.render_parents(hover_parent)
    assert "a1" in rendered  # first llm step gets a keepable id
    assert "retrieve" in rendered
    assert "query=outer_context" in rendered


def test_parse_rejects_unparseable_parent(changes):
    with pytest.raises(MutationError, match="carl_validation_error"):
        changes._parse({"A": "{not json"})


def test_parse_rejects_empty_parents(changes):
    with pytest.raises(MutationError, match="no parents"):
        changes._parse({})


# --- schema -----------------------------------------------------------------


def test_emitted_schema_is_portable(changes, hover_parent):
    assert nonportable_keys(changes.build_schema(hover_parent).json_schema) == set()


def test_slot_has_three_forms_and_narrowed_enums(changes, hover_parent):
    schema = changes.build_schema(hover_parent).json_schema
    for k in range(1, changes.max_steps + 1):
        branches = _object_branches(schema["properties"][f"slot_{k}"])
        kinds = {b["properties"]["kind"]["enum"][0] for b in branches}
        assert kinds == {"keep", "new_llm", "new_tool"}
        tool = next(
            b for b in branches if b["properties"]["kind"]["enum"][0] == "new_tool"
        )
        assert set(tool["properties"]["tool_name"]["enum"]) == set(TOOLS)
        qsrc = set(tool["properties"]["query_source"]["enum"])
        assert qsrc == {"outer_context", *[f"slot_{j}" for j in range(1, k)]}


def test_describe_mentions_tool_wiring(changes):
    text = changes.describe().lower()
    assert "new_tool" in text
    assert "query_source" in text
    assert "retrieve" in text  # available_tools listed


def test_schema_validates_a_tool_slot(changes, hover_parent):
    schema = changes.build_schema(hover_parent)
    schema.validate(
        {
            **EVIDENCE,
            "base_parent": "A",
            "slot_1": {
                "kind": "new_tool",
                "tool_name": "retrieve",
                "query_source": "outer_context",
            },
        }
    )


def test_single_parent_keep_id_enum_spans_llm_steps(changes, hover_parent):
    schema = changes.build_schema(hover_parent).json_schema
    keep = next(
        b
        for b in _object_branches(schema["properties"]["slot_2"])
        if b["properties"]["kind"]["enum"][0] == "keep"
    )
    # baseline has 4 llm steps -> a1..a4
    assert set(keep["properties"]["id"]["enum"]) == {"a1", "a2", "a3", "a4"}


# --- apply / transcribe -----------------------------------------------------


def _rebuild_baseline_diff() -> dict:
    # Reconstruct the 7-step baseline through the diff grammar: 3 new_tool +
    # 4 keeps (the baseline's LLM steps are #2,#3,#5,#6 -> ids a1..a4).
    return {
        **EVIDENCE,
        "base_parent": "A",
        "slot_1": {
            "kind": "new_tool",
            "tool_name": "retrieve",
            "query_source": "outer_context",
        },
        "slot_2": {"kind": "keep", "id": "a1", "dependencies": ["slot_1"]},
        "slot_3": {"kind": "keep", "id": "a2", "dependencies": ["slot_2"]},
        "slot_4": {
            "kind": "new_tool",
            "tool_name": "retrieve",
            "query_source": "slot_3",
        },
        "slot_5": {"kind": "keep", "id": "a3", "dependencies": ["slot_2", "slot_4"]},
        "slot_6": {"kind": "keep", "id": "a4", "dependencies": ["slot_5"]},
        "slot_7": {
            "kind": "new_tool",
            "tool_name": "retrieve_deep",
            "query_source": "slot_6",
        },
    }


def test_apply_on_baseline_round_trips_full_chain(changes, hover_parent):
    schema = changes.build_schema(hover_parent)
    diff = schema.validate(_rebuild_baseline_diff())
    child = json.loads(changes.apply(diff, parents=hover_parent))
    spec = validate_chain_spec(
        child, mode="full_chain", full_chain_config=FULL_CHAIN_CONFIG
    )
    assert len(spec.steps) == 7
    assert sum(1 for s in child["steps"] if s["step_type"] == "tool") == 3


def test_tool_query_wiring_is_absolute_and_reorder_safe(changes, hover_parent):
    schema = changes.build_schema(hover_parent)
    diff = schema.validate(
        {
            **EVIDENCE,
            "base_parent": "A",
            "slot_1": {"kind": "keep", "id": "a1"},
            "slot_2": {
                "kind": "new_tool",
                "tool_name": "retrieve",
                "query_source": "slot_1",
            },
        }
    )
    child = json.loads(changes.apply(diff, parents=hover_parent))
    tool = child["steps"][1]
    assert tool["step_config"]["input_mapping"]["query"] == "$history[0]"
    assert tool["dependencies"] == [1]
    # the absolute ref resolves to slot_1's output regardless of ordering
    assert (
        _resolve_reference("$history[0]", "ctx", ["out-of-step-1"]) == "out-of-step-1"
    )


def test_new_tool_first_slot_reads_outer_context(changes, hover_parent):
    schema = changes.build_schema(hover_parent)
    diff = schema.validate(
        {
            **EVIDENCE,
            "base_parent": "A",
            "slot_1": {
                "kind": "new_tool",
                "tool_name": "retrieve",
                "query_source": "outer_context",
            },
        }
    )
    child = json.loads(changes.apply(diff, parents=hover_parent))
    assert (
        child["steps"][0]["step_config"]["input_mapping"]["query"] == "$outer_context"
    )
    assert child["steps"][0]["dependencies"] == []


def test_keep_edit_overrides_llm_field(changes, hover_parent):
    schema = changes.build_schema(hover_parent)
    diff = schema.validate(
        {
            **EVIDENCE,
            "base_parent": "A",
            "slot_1": {"kind": "keep", "id": "a1", "edits": {"aim": "crisp new aim"}},
        }
    )
    child = json.loads(changes.apply(diff, parents=hover_parent))
    assert child["steps"][0]["aim"] == "crisp new aim"


def test_new_llm_slot_transcribes(changes, hover_parent):
    schema = changes.build_schema(hover_parent)
    diff = schema.validate(
        {
            **EVIDENCE,
            "base_parent": "A",
            "slot_1": {
                "kind": "new_llm",
                "title": "Answer",
                "aim": "state a verdict",
                "stage_action": "decide supported or not",
            },
        }
    )
    child = json.loads(changes.apply(diff, parents=hover_parent))
    step = child["steps"][0]
    assert step["step_type"] == "llm"
    assert step["stage_action"] == "decide supported or not"


def test_forward_and_self_refs_unrepresentable(changes, hover_parent):
    schema = changes.build_schema(hover_parent)
    with pytest.raises(ValidationError):
        schema.validate(
            {
                **EVIDENCE,
                "base_parent": "A",
                "slot_1": {
                    "kind": "new_tool",
                    "tool_name": "retrieve",
                    "query_source": "slot_1",
                },
            }
        )


def test_unknown_tool_unrepresentable(changes, hover_parent):
    schema = changes.build_schema(hover_parent)
    with pytest.raises(ValidationError):
        schema.validate(
            {
                **EVIDENCE,
                "base_parent": "A",
                "slot_1": {
                    "kind": "new_tool",
                    "tool_name": "web_search",
                    "query_source": "outer_context",
                },
            }
        )


def test_gap_in_slots_repaired(changes, hover_parent):
    # A gap (slot_2 empty, slot_3 filled) is deterministically compacted left,
    # not rejected: the same steps slide into contiguous slots and transcribe
    # as a 2-step chain.
    schema = changes.build_schema(hover_parent)
    diff = schema.validate(
        {
            **EVIDENCE,
            "base_parent": "A",
            "slot_1": {"kind": "keep", "id": "a1"},
            "slot_3": {"kind": "keep", "id": "a2", "dependencies": ["slot_1"]},
        }
    )
    assert diff.slot_1 is not None and diff.slot_2 is not None
    assert diff.slot_3 is None
    assert diff.slot_2.id == "a2"
    assert diff.slot_2.dependencies == ["slot_1"]
    child = json.loads(changes.apply(diff, parents=hover_parent))
    assert [s["number"] for s in child["steps"]] == [1, 2]
    assert child["steps"][1]["dependencies"] == [1]


def test_gap_repair_remaps_tool_query_source(changes, hover_parent):
    # slot_4 (a tool reading slot_1) compacts to slot_2; its query_source is
    # remapped so the absolute $history wiring still points at the same step.
    schema = changes.build_schema(hover_parent)
    diff = schema.validate(
        {
            **EVIDENCE,
            "base_parent": "A",
            "slot_1": {"kind": "keep", "id": "a1"},
            "slot_4": {
                "kind": "new_tool",
                "tool_name": "retrieve",
                "query_source": "slot_1",
            },
        }
    )
    assert diff.slot_2 is not None and diff.slot_3 is None
    child = json.loads(changes.apply(diff, parents=hover_parent))
    assert child["steps"][1]["step_config"]["input_mapping"]["query"] == "$history[0]"
    assert child["steps"][1]["dependencies"] == [1]


def test_multi_gap_repair_compacts_all_holes(changes, hover_parent):
    # Two holes (slot_2, slot_4 empty) close in one pass: slot_3 -> slot_2,
    # slot_5 -> slot_3, with every dependency renumbered to the new positions.
    schema = changes.build_schema(hover_parent)
    diff = schema.validate(
        {
            **EVIDENCE,
            "base_parent": "A",
            "slot_1": {"kind": "keep", "id": "a1"},
            "slot_3": {"kind": "keep", "id": "a2", "dependencies": ["slot_1"]},
            "slot_5": {"kind": "keep", "id": "a3", "dependencies": ["slot_3"]},
        }
    )
    assert diff.slot_2 is not None and diff.slot_3 is not None
    assert diff.slot_4 is None and diff.slot_5 is None
    assert diff.slot_2.id == "a2" and diff.slot_2.dependencies == ["slot_1"]
    assert diff.slot_3.id == "a3" and diff.slot_3.dependencies == ["slot_2"]
    child = json.loads(changes.apply(diff, parents=hover_parent))
    assert [s["number"] for s in child["steps"]] == [1, 2, 3]
    assert child["steps"][2]["dependencies"] == [2]


def test_leading_gap_slot1_empty_repaired(changes, hover_parent):
    # The hole can be slot_1 itself: a lone slot_2 slides down to slot_1 and
    # the diff transcribes as a single-step chain.
    schema = changes.build_schema(hover_parent)
    diff = schema.validate(
        {
            **EVIDENCE,
            "base_parent": "A",
            "slot_2": {"kind": "keep", "id": "a1"},
        }
    )
    assert diff.slot_1 is not None and diff.slot_1.id == "a1"
    assert diff.slot_2 is None
    child = json.loads(changes.apply(diff, parents=hover_parent))
    assert [s["number"] for s in child["steps"]] == [1]


def test_gap_with_dangling_ref_still_rejected(changes, hover_parent):
    # slot_3 depends on slot_2, but slot_2 is the hole -> not cleanly repairable,
    # so the original contiguity error stands.
    schema = changes.build_schema(hover_parent)
    with pytest.raises(ValidationError):
        schema.validate(
            {
                **EVIDENCE,
                "base_parent": "A",
                "slot_1": {"kind": "keep", "id": "a1"},
                "slot_3": {"kind": "keep", "id": "a2", "dependencies": ["slot_2"]},
            }
        )


def test_alias_slot_key_aborts_repair(changes, hover_parent):
    # "slot_01" parses to index 1 and would silently collide with slot_1 under
    # compaction; non-canonical keys abort the repair.
    schema = changes.build_schema(hover_parent)
    with pytest.raises(ValidationError):
        schema.validate(
            {
                **EVIDENCE,
                "base_parent": "A",
                "slot_1": {"kind": "keep", "id": "a1"},
                "slot_01": {"kind": "keep", "id": "a2"},
                "slot_3": {"kind": "keep", "id": "a2", "dependencies": ["slot_1"]},
            }
        )


def test_alias_query_source_aborts_repair(changes, hover_parent):
    # A non-canonical "slot_01" query_source must not be laundered into a valid
    # ref by the repair's remapping.
    schema = changes.build_schema(hover_parent)
    with pytest.raises(ValidationError):
        schema.validate(
            {
                **EVIDENCE,
                "base_parent": "A",
                "slot_1": {"kind": "keep", "id": "a1"},
                "slot_4": {
                    "kind": "new_tool",
                    "tool_name": "retrieve",
                    "query_source": "slot_01",
                },
            }
        )


def test_min_steps_pins_slot_one_required(hover_parent):
    changes = AllowedToolChainChanges(max_steps=7, available_tools=TOOLS)
    schema = changes.build_schema(hover_parent)
    with pytest.raises(ValidationError):
        schema.validate({**EVIDENCE, "base_parent": "A"})


def test_invalid_bounds_rejected():
    with pytest.raises(ValueError):
        AllowedToolChainChanges(min_steps=3, max_steps=2, available_tools=TOOLS)
    with pytest.raises(ValueError):
        AllowedToolChainChanges(max_steps=5, available_tools=[])
