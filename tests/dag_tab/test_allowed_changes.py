from __future__ import annotations

import json
from pathlib import Path

import pytest

from gigaevo.llm.schema_compat import nonportable_keys
from problems.dag_tab.allowed_changes import AllowedDagTabChanges
from problems.dag_tab.graph import FeatureGraph


ROOT = Path(__file__).parents[2]
PARENT = (ROOT / "problems/dag_tab/initial_programs/baseline.json").read_text()


def _evidence(**updates):
    payload = {
        "archetype": "Guided Innovation",
        "justification": "Add a stable interaction informed by the parent score.",
        "insights_used": [],
        "insight_ids_used": [],
        "base_parent": "A",
        "card_ids_used": [],
        "changes": [],
    }
    payload.update(updates)
    return payload


def test_schema_is_portable_and_parent_specific():
    changes = AllowedDagTabChanges(max_nodes=3)
    schema = changes.build_schema({"A": PARENT})

    assert nonportable_keys(schema.json_schema) == set()
    text = json.dumps(schema.json_schema)
    assert "a1" in text
    assert "slot_1" in text and "slot_3" in text


def test_keep_parent_node_round_trip():
    changes = AllowedDagTabChanges(max_nodes=2)
    schema = changes.build_schema({"A": PARENT})
    diff = schema.validate(
        _evidence(
            slot_1={"kind": "keep", "id": "a1", "edits": {}},
            slot_2=None,
        )
    )

    child = FeatureGraph.from_json(changes.apply(diff, {"A": PARENT}))

    assert child.nodes[0].id == "income_per_age"
    assert child.nodes[0].is_output is True


def test_new_multi_node_chain_can_omit_and_rewire_parent():
    changes = AllowedDagTabChanges(max_nodes=3)
    schema = changes.build_schema({"A": PARENT})
    diff = schema.validate(
        _evidence(
            slot_1={
                "kind": "new",
                "id": "rooms_per_bedroom",
                "input_cols": ["x2", "x3"],
                "output_cols": ["fe_rooms_per_bedroom"],
                "code": "df['fe_rooms_per_bedroom'] = df['x2'] / (df['x3'].abs() + 1e-6)\nreturn df",
                "rationale": "Measure room composition.",
                "is_output": False,
            },
            slot_2={
                "kind": "new",
                "id": "income_room_interaction",
                "input_cols": ["x0", "fe_rooms_per_bedroom"],
                "output_cols": ["fe_income_room_interaction"],
                "code": "df['fe_income_room_interaction'] = df['x0'] * df['fe_rooms_per_bedroom']\nreturn df",
                "rationale": "Combine income with room composition.",
                "is_output": True,
                "dependencies": ["slot_1"],
            },
            slot_3=None,
        )
    )

    child = FeatureGraph.from_json(changes.apply(diff, {"A": PARENT}))

    assert [node.id for node in child.nodes] == [
        "rooms_per_bedroom",
        "income_room_interaction",
    ]
    assert child.nodes[1].dependencies == ["rooms_per_bedroom"]
    assert "income_per_age" not in {node.id for node in child.nodes}


def test_schema_rejects_forward_dependency():
    changes = AllowedDagTabChanges(max_nodes=2)
    schema = changes.build_schema({"A": PARENT})
    payload = _evidence(
        slot_1={"kind": "keep", "id": "a1", "edits": {}},
        slot_2=None,
    )
    payload["slot_1"]["dependencies"] = ["slot_2"]

    with pytest.raises(Exception):
        schema.validate(payload)


def test_schema_repairs_slot_gap_and_remaps_dependencies():
    changes = AllowedDagTabChanges(max_nodes=3)
    schema = changes.build_schema({"A": PARENT})
    diff = schema.validate(
        _evidence(
            slot_1={"kind": "keep", "id": "a1", "edits": {"is_output": False}},
            slot_2=None,
            slot_3={
                "kind": "new",
                "id": "scaled_income",
                "input_cols": ["fe_income_per_age"],
                "output_cols": ["fe_scaled_income"],
                "code": "df['fe_scaled_income'] = df['fe_income_per_age'] * 2\nreturn df",
                "rationale": "Expose a rescaled form of the parent signal.",
                "is_output": True,
                "dependencies": ["slot_1"],
            },
        )
    )

    child = FeatureGraph.from_json(changes.apply(diff, {"A": PARENT}))

    assert [node.id for node in child.nodes] == ["income_per_age", "scaled_income"]
    assert child.nodes[1].dependencies == ["income_per_age"]


def test_slot_gap_repair_rejects_dependency_on_empty_slot():
    changes = AllowedDagTabChanges(max_nodes=3)
    schema = changes.build_schema({"A": PARENT})
    payload = _evidence(
        slot_1={"kind": "keep", "id": "a1", "edits": {"is_output": False}},
        slot_2=None,
        slot_3={
            "kind": "new",
            "id": "broken_dependency",
            "input_cols": ["x0"],
            "output_cols": ["fe_broken_dependency"],
            "code": "df['fe_broken_dependency'] = df['x0']\nreturn df",
            "rationale": "Exercise safe gap repair rejection.",
            "is_output": True,
            "dependencies": ["slot_2"],
        },
    )

    with pytest.raises(Exception):
        schema.validate(payload)


def test_apply_appends_missing_final_return_df():
    changes = AllowedDagTabChanges(max_nodes=1)
    schema = changes.build_schema({"A": PARENT})
    diff = schema.validate(
        _evidence(
            slot_1={
                "kind": "new",
                "id": "normalized",
                "input_cols": ["x0"],
                "output_cols": ["fe_normalized"],
                "code": "df['fe_normalized'] = df['x0']",
                "rationale": "Exercise deterministic code normalization.",
                "is_output": True,
            }
        )
    )

    child = FeatureGraph.from_json(changes.apply(diff, {"A": PARENT}))

    assert child.nodes[0].code.endswith("\nreturn df")


def test_schema_keeps_code_string_unconstrained_by_regex():
    changes = AllowedDagTabChanges(max_nodes=1)
    schema = changes.build_schema({"A": PARENT}).json_schema

    def code_fields(value):
        if isinstance(value, dict):
            if value.get("title") == "Code":
                yield value
            for nested in value.values():
                yield from code_fields(nested)
        elif isinstance(value, list):
            for nested in value:
                yield from code_fields(nested)

    fields = list(code_fields(schema))
    assert fields
    assert all("return" in field.get("description", "") for field in fields)
    assert all("pattern" not in field for field in fields)


def test_apply_rejects_code_without_declared_output():
    changes = AllowedDagTabChanges(max_nodes=1)
    schema = changes.build_schema({"A": PARENT})
    diff = schema.validate(
        _evidence(
            slot_1={
                "kind": "new",
                "id": "broken",
                "input_cols": ["x0"],
                "output_cols": ["fe_broken"],
                "code": "value = df['x0']\nreturn df",
                "rationale": "Intentionally invalid test node.",
                "is_output": True,
            }
        )
    )

    with pytest.raises(Exception, match="diff_apply_assertion"):
        changes.apply(diff, {"A": PARENT})
