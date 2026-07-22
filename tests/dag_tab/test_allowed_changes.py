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
        "structural_intent": "local_edit",
        "card_ids_used": [],
        "changes": [],
    }
    payload.update(updates)
    return payload


def test_schema_is_portable_and_parent_specific():
    changes = AllowedDagTabChanges(max_nodes=3)
    parents = {"A": PARENT, "B": PARENT}
    schema = changes.build_schema(parents)

    assert nonportable_keys(schema.json_schema) == set()
    text = json.dumps(schema.json_schema)
    assert "A_income_per_age" in text and "B_income_per_age" in text
    rendered = changes.render_parents(parents)
    assert "id=A_income_per_age" in rendered
    assert "id=B_income_per_age" in rendered
    nodes = schema.json_schema["properties"]["nodes"]
    assert "maxItems" not in nodes
    assert "at most 3" in nodes["description"]


def test_neutral_parent_schema_allows_first_new_node_without_keep_literal():
    neutral = FeatureGraph(
        dataset="adult",
        raw_columns=["x0", "x1"],
        nodes=[],
    ).to_json()
    changes = AllowedDagTabChanges(max_nodes=2)
    schema = changes.build_schema({"A": neutral})
    diff = schema.validate(
        _evidence(
            nodes=[
                {
                    "kind": "new_rowwise",
                    "id": "first_feature",
                    "input_cols": ["x0"],
                    "output_cols": ["fe_first"],
                    "code": "df['fe_first'] = df['x0']\nreturn df",
                    "rationale": "Expose a stable first-order transform.",
                    "is_output": True,
                }
            ],
        )
    )

    child = FeatureGraph.from_json(changes.apply(diff, {"A": neutral}))

    assert [node.id for node in child.nodes] == ["first_feature"]
    nodes_schema = json.dumps(schema.json_schema["properties"]["nodes"])
    assert "KeepFeatureNode" not in nodes_schema


def test_keep_parent_node_round_trip():
    changes = AllowedDagTabChanges(max_nodes=2)
    schema = changes.build_schema({"A": PARENT})
    diff = schema.validate(
        _evidence(nodes=[{"kind": "keep", "id": "income_per_age", "edits": {}}])
    )

    child = FeatureGraph.from_json(changes.apply(diff, {"A": PARENT}))

    assert child.nodes[0].id == "income_per_age"
    assert child.nodes[0].is_output is True


def test_keep_edit_repairs_stale_output_types_after_output_rename():
    changes = AllowedDagTabChanges(max_nodes=1)
    schema = changes.build_schema({"A": PARENT})
    diff = schema.validate(
        _evidence(
            nodes=[
                {
                    "kind": "keep",
                    "id": "income_per_age",
                    "edits": {
                        "output_cols": ["fe_income_age_interaction"],
                        "code": (
                            "df['fe_income_age_interaction'] = df['x0'] * df['x1']\n"
                            "return df"
                        ),
                    },
                }
            ]
        )
    )

    child = FeatureGraph.from_json(changes.apply(diff, {"A": PARENT}))

    assert child.nodes[0].output_types == {"fe_income_age_interaction": "numerical"}


def test_keep_edit_preserves_retained_types_and_defaults_new_outputs():
    typed_parent = FeatureGraph.from_json(PARENT)
    typed_parent.nodes[0].output_types = {"fe_income_per_age": "ordinal"}
    parent_json = typed_parent.to_json()
    changes = AllowedDagTabChanges(max_nodes=1)
    schema = changes.build_schema({"A": parent_json})
    diff = schema.validate(
        _evidence(
            nodes=[
                {
                    "kind": "keep",
                    "id": "income_per_age",
                    "edits": {
                        "output_cols": ["fe_income_per_age", "fe_income_squared"],
                        "code": (
                            "df['fe_income_per_age'] = df['x0'] / (df['x1'].abs() + 1.0)\n"
                            "df['fe_income_squared'] = df['x0'] ** 2\nreturn df"
                        ),
                    },
                }
            ]
        )
    )

    child = FeatureGraph.from_json(changes.apply(diff, {"A": parent_json}))

    assert child.nodes[0].output_types == {
        "fe_income_per_age": "ordinal",
        "fe_income_squared": "numerical",
    }


def test_keep_edit_honors_explicit_output_types_with_changed_outputs():
    changes = AllowedDagTabChanges(max_nodes=1)
    schema = changes.build_schema({"A": PARENT})
    diff = schema.validate(
        _evidence(
            nodes=[
                {
                    "kind": "keep",
                    "id": "income_per_age",
                    "edits": {
                        "output_cols": ["fe_income_band"],
                        "output_types": {"fe_income_band": "categorical"},
                        "code": (
                            "df['fe_income_band'] = "
                            "pd.cut(df['x0'], bins=3, labels=['low', 'mid', 'high'])\n"
                            "return df"
                        ),
                    },
                }
            ]
        )
    )

    child = FeatureGraph.from_json(changes.apply(diff, {"A": PARENT}))

    assert child.nodes[0].output_types == {"fe_income_band": "categorical"}


def test_diff_normalizes_complete_target_transform_functions():
    changes = AllowedDagTabChanges(max_nodes=1)
    schema = changes.build_schema({"A": PARENT})
    diff = schema.validate(
        _evidence(
            target_change={
                "kind": "set",
                "code": "import numpy as np\ndef transform(y_fit, y):\n    return np.log1p(y)",
                "inverse_code": (
                    "import numpy as np\ndef inverse(y_fit, predictions):\n"
                    "    return np.expm1(predictions)"
                ),
            },
            nodes=[{"kind": "keep", "id": "income_per_age", "edits": {}}],
        )
    )

    child = FeatureGraph.from_json(changes.apply(diff, {"A": PARENT}))

    assert child.target is not None
    assert child.target.code == "import numpy as np\nreturn np.log1p(y)"
    assert (
        child.target.inverse_code == "import numpy as np\nreturn np.expm1(predictions)"
    )


def test_diff_round_trip_edits_graph_level_selection_and_target():
    changes = AllowedDagTabChanges(max_nodes=1)
    schema = changes.build_schema({"A": PARENT})
    diff = schema.validate(
        _evidence(
            dropped_raw_columns=["x7"],
            target_change={
                "kind": "set",
                "code": "return np.log1p(y)",
                "inverse_code": "return np.expm1(predictions)",
            },
            nodes=[{"kind": "keep", "id": "income_per_age", "edits": {}}],
        )
    )

    child = FeatureGraph.from_json(changes.apply(diff, {"A": PARENT}))

    assert child.dropped_raw_columns == ["x7"]
    assert child.target is not None
    assert child.target.code == "return np.log1p(y)"


def test_diff_can_drop_existing_target_and_preserve_selection_by_default():
    parent = FeatureGraph.from_json(PARENT)
    parent.dropped_raw_columns = ["x7"]
    parent_data = parent.model_dump()
    parent_data["target"] = {
        "code": "return np.log1p(y)",
        "inverse_code": "return np.expm1(predictions)",
    }
    parent_json = FeatureGraph.model_validate(parent_data).to_json()
    changes = AllowedDagTabChanges(max_nodes=1)
    schema = changes.build_schema({"A": parent_json})
    diff = schema.validate(
        _evidence(
            target_change={"kind": "drop"},
            nodes=[{"kind": "keep", "id": "income_per_age", "edits": {}}],
        )
    )

    child = FeatureGraph.from_json(changes.apply(diff, {"A": parent_json}))

    assert child.dropped_raw_columns == ["x7"]
    assert child.target is None


def test_new_multi_node_chain_can_omit_and_rewire_parent():
    changes = AllowedDagTabChanges(max_nodes=3)
    schema = changes.build_schema({"A": PARENT})
    diff = schema.validate(
        _evidence(
            nodes=[
                {
                    "kind": "new_rowwise",
                    "id": "rooms_per_bedroom",
                    "input_cols": ["x2", "x3"],
                    "output_cols": ["fe_rooms_per_bedroom"],
                    "code": "df['fe_rooms_per_bedroom'] = df['x2'] / (df['x3'].abs() + 1e-6)\nreturn df",
                    "rationale": "Measure room composition.",
                    "is_output": False,
                },
                {
                    "kind": "new_rowwise",
                    "id": "income_room_interaction",
                    "input_cols": ["x0", "fe_rooms_per_bedroom"],
                    "output_cols": ["fe_income_room_interaction"],
                    "code": "df['fe_income_room_interaction'] = df['x0'] * df['fe_rooms_per_bedroom']\nreturn df",
                    "rationale": "Combine income with room composition.",
                    "is_output": True,
                    "dependencies": ["rooms_per_bedroom"],
                },
            ],
        )
    )

    child = FeatureGraph.from_json(changes.apply(diff, {"A": PARENT}))

    assert [node.id for node in child.nodes] == [
        "rooms_per_bedroom",
        "income_room_interaction",
    ]
    assert child.nodes[1].dependencies == ["rooms_per_bedroom"]
    assert "income_per_age" not in {node.id for node in child.nodes}


def test_default_schema_bounds_nodes_array_at_twelve():
    schema = AllowedDagTabChanges().build_schema({"A": PARENT})

    nodes = schema.json_schema["properties"]["nodes"]
    assert nodes["minItems"] == 1
    assert "maxItems" not in nodes
    assert "at most 12" in nodes["description"]


def test_keep_preserves_existing_dependency_when_field_is_omitted():
    parent = FeatureGraph.from_json(PARENT)
    parent.nodes.append(
        parent.nodes[0].model_copy(
            update={
                "id": "income_per_age_log",
                "input_cols": ["fe_income_per_age"],
                "output_cols": ["fe_income_per_age_log"],
                "output_types": {"fe_income_per_age_log": "numerical"},
                "code": (
                    "df['fe_income_per_age_log'] = "
                    "np.log1p(df['fe_income_per_age'].clip(lower=0))\nreturn df"
                ),
                "rationale": "Compose a bounded transform from the earlier ratio.",
                "dependencies": ["income_per_age"],
                "is_output": True,
            }
        )
    )
    parent.nodes[0].is_output = False
    parent_json = parent.to_json()
    changes = AllowedDagTabChanges(max_nodes=2)
    diff = changes.build_schema({"A": parent_json}).validate(
        _evidence(
            nodes=[
                {"kind": "keep", "id": "income_per_age", "edits": {}},
                {"kind": "keep", "id": "income_per_age_log", "edits": {}},
            ],
        )
    )

    child = FeatureGraph.from_json(changes.apply(diff, {"A": parent_json}))

    assert child.nodes[1].dependencies == ["income_per_age"]
    assert child.depth == 2


def test_keep_can_explicitly_rewire_existing_dependency():
    parent = FeatureGraph.from_json(PARENT)
    parent.nodes.append(
        parent.nodes[0].model_copy(
            update={
                "id": "income_copy",
                "input_cols": ["x0"],
                "output_cols": ["fe_income_copy"],
                "output_types": {"fe_income_copy": "numerical"},
                "code": "df['fe_income_copy'] = df['x0']\nreturn df",
                "rationale": "Expose raw income under a generated feature name.",
                "dependencies": ["income_per_age"],
                "is_output": True,
            }
        )
    )
    parent.nodes[0].is_output = False
    parent_json = parent.to_json()
    changes = AllowedDagTabChanges(max_nodes=2)
    diff = changes.build_schema({"A": parent_json}).validate(
        _evidence(
            nodes=[
                {"kind": "keep", "id": "income_per_age", "edits": {}},
                {
                    "kind": "keep",
                    "id": "income_copy",
                    "edits": {},
                    "dependencies": [],
                },
            ],
        )
    )

    child = FeatureGraph.from_json(changes.apply(diff, {"A": parent_json}))

    assert child.nodes[1].dependencies == []
    assert child.depth == 1


def test_compose_chain_intent_builds_depth_four():
    neutral = FeatureGraph(
        dataset="adult",
        raw_columns=["x0", "x1"],
        nodes=[],
    ).to_json()
    changes = AllowedDagTabChanges(max_nodes=4)
    diff = changes.build_schema({"A": neutral}).validate(
        _evidence(
            structural_intent="compose_chain",
            minimum_child_depth=4,
            nodes=[
                {
                    "kind": "new_rowwise",
                    "id": "ratio",
                    "input_cols": ["x0", "x1"],
                    "output_cols": ["fe_ratio"],
                    "code": "df['fe_ratio'] = df['x0'] / (df['x1'].abs() + 1)\nreturn df",
                    "rationale": "Create a stable reusable ratio.",
                    "is_output": False,
                },
                {
                    "kind": "new_rowwise",
                    "id": "bounded_ratio",
                    "input_cols": ["fe_ratio"],
                    "output_cols": ["fe_bounded_ratio"],
                    "code": "df['fe_bounded_ratio'] = np.tanh(df['fe_ratio'])\nreturn df",
                    "rationale": "Bound the generated ratio.",
                    "dependencies": ["ratio"],
                    "is_output": False,
                },
                {
                    "kind": "new_rowwise",
                    "id": "ratio_interaction",
                    "input_cols": ["fe_bounded_ratio", "x0"],
                    "output_cols": ["fe_ratio_interaction"],
                    "code": "df['fe_ratio_interaction'] = df['fe_bounded_ratio'] * df['x0']\nreturn df",
                    "rationale": "Interact the bounded intermediate with raw scale.",
                    "dependencies": ["bounded_ratio"],
                    "is_output": False,
                },
                {
                    "kind": "new_rowwise",
                    "id": "final_composition",
                    "input_cols": ["fe_ratio_interaction"],
                    "output_cols": ["fe_final"],
                    "code": "df['fe_final'] = np.sign(df['fe_ratio_interaction']) * np.log1p(df['fe_ratio_interaction'].abs())\nreturn df",
                    "rationale": "Export a robust final composition.",
                    "dependencies": ["ratio_interaction"],
                    "is_output": True,
                },
            ],
        )
    )

    child = FeatureGraph.from_json(changes.apply(diff, {"A": neutral}))

    assert child.depth == 4
    assert [node.dependencies for node in child.nodes] == [
        [],
        ["ratio"],
        ["bounded_ratio"],
        ["ratio_interaction"],
    ]


def test_extend_chain_intent_rejects_flat_child():
    changes = AllowedDagTabChanges(max_nodes=1)
    diff = changes.build_schema({"A": PARENT}).validate(
        _evidence(
            structural_intent="extend_chain",
            nodes=[{"kind": "keep", "id": "income_per_age", "edits": {}}],
        )
    )

    with pytest.raises(Exception, match="requires child depth >= 2, got 1"):
        changes.apply(diff, {"A": PARENT})


def test_minimum_child_depth_rejects_shallow_composition_claim():
    changes = AllowedDagTabChanges(max_nodes=2)
    diff = changes.build_schema({"A": PARENT}).validate(
        _evidence(
            structural_intent="compose_chain",
            minimum_child_depth=3,
            nodes=[
                {"kind": "keep", "id": "income_per_age", "edits": {}},
                {
                    "kind": "new_rowwise",
                    "id": "independent",
                    "input_cols": ["x2"],
                    "output_cols": ["fe_independent"],
                    "code": "df['fe_independent'] = df['x2'] ** 2\nreturn df",
                    "rationale": "An independent feature cannot satisfy a depth claim.",
                    "is_output": True,
                },
            ],
        )
    )

    with pytest.raises(Exception, match="requires child depth >= 3, got 1"):
        changes.apply(diff, {"A": PARENT})


def test_apply_repairs_missing_dependency_for_generated_input():
    changes = AllowedDagTabChanges(max_nodes=2)
    schema = changes.build_schema({"A": PARENT})
    diff = schema.validate(
        _evidence(
            nodes=[
                {
                    "kind": "keep",
                    "id": "income_per_age",
                    "edits": {"is_output": False},
                },
                {
                    "kind": "new_rowwise",
                    "id": "income_interaction",
                    "input_cols": ["x0", "fe_income_per_age"],
                    "output_cols": ["fe_income_interaction"],
                    "code": (
                        "df['fe_income_interaction'] = "
                        "df['x0'] * df['fe_income_per_age']\nreturn df"
                    ),
                    "rationale": "Compose raw income with the earlier generated feature.",
                    "dependencies": [],
                    "is_output": True,
                },
            ],
        )
    )

    child = FeatureGraph.from_json(changes.apply(diff, {"A": PARENT}))

    assert child.nodes[1].dependencies == ["income_per_age"]


def test_apply_repairs_input_cols_and_dependency_from_literal_code_reads():
    changes = AllowedDagTabChanges(max_nodes=2)
    schema = changes.build_schema({"A": PARENT})
    diff = schema.validate(
        _evidence(
            nodes=[
                {
                    "kind": "keep",
                    "id": "income_per_age",
                    "edits": {"is_output": False},
                },
                {
                    "kind": "new_rowwise",
                    "id": "income_interaction",
                    "input_cols": ["x0"],
                    "output_cols": ["fe_income_interaction"],
                    "code": (
                        "df['fe_income_interaction'] = "
                        "df['x0'] * df['fe_income_per_age'] + df['x2']\nreturn df"
                    ),
                    "rationale": "Exercise literal-read contract synchronization.",
                    "dependencies": [],
                    "is_output": True,
                },
            ],
        )
    )

    child = FeatureGraph.from_json(changes.apply(diff, {"A": PARENT}))

    assert child.nodes[1].input_cols == ["x0", "fe_income_per_age", "x2"]
    assert child.nodes[1].dependencies == ["income_per_age"]


def test_apply_normalizes_complete_transform_and_copy_alias_reads():
    changes = AllowedDagTabChanges(max_nodes=1)
    schema = changes.build_schema({"A": PARENT})
    diff = schema.validate(
        _evidence(
            nodes=[
                {
                    "kind": "new_rowwise",
                    "id": "complete_transform",
                    "input_cols": ["x0"],
                    "output_cols": ["fe_result"],
                    "code": (
                        "OFFSET = 1\n"
                        "def helper(values):\n"
                        "    return values + OFFSET\n"
                        "def transform(df):\n"
                        "    result = df.copy()\n"
                        "    result['fe_result'] = helper(result['x2']) + result['x0']\n"
                        "    return result"
                    ),
                    "rationale": "Exercise complete-module normalization.",
                    "is_output": True,
                }
            ]
        )
    )

    child = FeatureGraph.from_json(changes.apply(diff, {"A": PARENT}))

    assert child.nodes[0].code.startswith("OFFSET = 1")
    assert "def helper(values):" in child.nodes[0].code
    assert child.nodes[0].code.endswith("return result")
    assert child.nodes[0].input_cols == ["x0", "x2"]


def test_apply_repairs_literal_loc_column_reads():
    changes = AllowedDagTabChanges(max_nodes=1)
    schema = changes.build_schema({"A": PARENT})
    diff = schema.validate(
        _evidence(
            nodes=[
                {
                    "kind": "new_rowwise",
                    "id": "loc_reader",
                    "input_cols": ["x0"],
                    "output_cols": ["fe_result"],
                    "code": "df['fe_result'] = df.loc[:, 'x2'] + df['x0']\nreturn df",
                    "rationale": "Exercise literal loc synchronization.",
                    "is_output": True,
                }
            ]
        )
    )

    child = FeatureGraph.from_json(changes.apply(diff, {"A": PARENT}))

    assert child.nodes[0].input_cols == ["x0", "x2"]


@pytest.mark.parametrize(
    "code, expected_input_cols",
    [
        (
            "alias = df\ndf['fe_result'] = alias['x1'] + df['x0']\nreturn df",
            ["x0", "x1"],
        ),
        (
            "df['fe_result'] = df.iloc[:, 0] + df['x0']\nreturn df",
            ["x0"],
        ),
    ],
)
def test_apply_uses_best_effort_inference_for_opaque_frame_reads(
    code, expected_input_cols
):
    changes = AllowedDagTabChanges(max_nodes=1)
    schema = changes.build_schema({"A": PARENT})
    diff = schema.validate(
        _evidence(
            nodes=[
                {
                    "kind": "new_rowwise",
                    "id": "unsafe_reader",
                    "input_cols": ["x0"],
                    "output_cols": ["fe_result"],
                    "code": code,
                    "rationale": "Exercise best-effort inference for opaque frame reads.",
                    "is_output": True,
                }
            ]
        )
    )

    child = FeatureGraph.from_json(changes.apply(diff, {"A": PARENT}))

    assert child.nodes[0].input_cols == expected_input_cols


def test_apply_leaves_unplaceable_literal_reads_to_execution():
    changes = AllowedDagTabChanges(max_nodes=1)
    schema = changes.build_schema({"A": PARENT})
    diff = schema.validate(
        _evidence(
            nodes=[
                {
                    "kind": "new_rowwise",
                    "id": "unknown_reader",
                    "input_cols": ["x0"],
                    "output_cols": ["fe_result"],
                    "code": "df['fe_result'] = df['fe_future'] + df['x0']\nreturn df",
                    "rationale": "Exercise safe repair rejection.",
                    "is_output": True,
                }
            ]
        )
    )

    child = FeatureGraph.from_json(changes.apply(diff, {"A": PARENT}))

    assert child.nodes[0].input_cols == ["x0"]


def test_apply_keeps_explicit_dependencies_before_repaired_ones():
    changes = AllowedDagTabChanges(max_nodes=3)
    schema = changes.build_schema({"A": PARENT})
    diff = schema.validate(
        _evidence(
            nodes=[
                {
                    "kind": "keep",
                    "id": "income_per_age",
                    "edits": {"is_output": False},
                },
                {
                    "kind": "new_rowwise",
                    "id": "raw_double",
                    "input_cols": ["x0"],
                    "output_cols": ["fe_raw_double"],
                    "code": "df['fe_raw_double'] = df['x0'] * 2\nreturn df",
                    "rationale": "Create an explicit dependency test feature.",
                    "is_output": False,
                },
                {
                    "kind": "new_rowwise",
                    "id": "composed",
                    "input_cols": ["fe_income_per_age", "fe_raw_double"],
                    "output_cols": ["fe_composed"],
                    "code": (
                        "df['fe_composed'] = "
                        "df['fe_income_per_age'] + df['fe_raw_double']\nreturn df"
                    ),
                    "rationale": "Exercise deterministic dependency union ordering.",
                    "dependencies": ["raw_double"],
                    "is_output": True,
                },
            ],
        )
    )

    child = FeatureGraph.from_json(changes.apply(diff, {"A": PARENT}))

    assert child.nodes[2].dependencies == ["raw_double", "income_per_age"]


def test_apply_drops_forward_dependency():
    changes = AllowedDagTabChanges(max_nodes=2)
    schema = changes.build_schema({"A": PARENT})
    diff = schema.validate(
        _evidence(
            nodes=[
                {
                    "kind": "keep",
                    "id": "income_per_age",
                    "edits": {},
                    "dependencies": ["later"],
                },
                {
                    "kind": "new_rowwise",
                    "id": "later",
                    "input_cols": ["x0"],
                    "output_cols": ["fe_later"],
                    "code": "df['fe_later'] = df['x0']\nreturn df",
                    "rationale": "Exercise forward dependency dropping.",
                    "is_output": True,
                },
            ],
        )
    )

    child = FeatureGraph.from_json(changes.apply(diff, {"A": PARENT}))

    assert child.nodes[0].dependencies == []


def test_apply_rejects_unknown_generated_input_after_dependency_repair():
    changes = AllowedDagTabChanges(max_nodes=1)
    schema = changes.build_schema({"A": PARENT})
    diff = schema.validate(
        _evidence(
            nodes=[
                {
                    "kind": "new_rowwise",
                    "id": "unknown_reader",
                    "input_cols": ["fe_missing"],
                    "output_cols": ["fe_result"],
                    "code": "df['fe_result'] = df['fe_missing']\nreturn df",
                    "rationale": "Exercise rejection of an unavailable generated input.",
                    "is_output": True,
                }
            ]
        )
    )

    with pytest.raises(Exception, match="unavailable inputs"):
        changes.apply(diff, {"A": PARENT})


def test_schema_truncates_overlong_node_rationale():
    changes = AllowedDagTabChanges(max_nodes=1)
    schema = changes.build_schema({"A": PARENT})
    diff = schema.validate(
        _evidence(
            nodes=[
                {
                    "kind": "new_rowwise",
                    "id": "compact_rationale",
                    "input_cols": ["x0"],
                    "output_cols": ["fe_compact"],
                    "code": "df['fe_compact'] = df['x0']\nreturn df",
                    "rationale": "r" * 900,
                    "is_output": True,
                }
            ]
        )
    )

    assert len(diff.nodes[0].rationale) == 500
    child = FeatureGraph.from_json(changes.apply(diff, {"A": PARENT}))
    assert len(child.nodes[0].rationale) == 500


def test_apply_appends_missing_final_return_df():
    changes = AllowedDagTabChanges(max_nodes=1)
    schema = changes.build_schema({"A": PARENT})
    diff = schema.validate(
        _evidence(
            nodes=[
                {
                    "kind": "new_rowwise",
                    "id": "normalized",
                    "input_cols": ["x0"],
                    "output_cols": ["fe_normalized"],
                    "code": "df['fe_normalized'] = df['x0']",
                    "rationale": "Exercise deterministic code normalization.",
                    "is_output": True,
                }
            ]
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


def test_dependencies_field_explains_feature_composition():
    changes = AllowedDagTabChanges(max_nodes=3)
    schema = changes.build_schema({"A": PARENT}).json_schema

    def dependency_fields(value):
        if isinstance(value, dict):
            props = value.get("properties")
            if isinstance(props, dict) and "dependencies" in props:
                yield props["dependencies"]
            for nested in value.values():
                yield from dependency_fields(nested)
        elif isinstance(value, list):
            for nested in value:
                yield from dependency_fields(nested)

    fields = list(dependency_fields(schema))
    assert fields
    assert all("output_cols" in field.get("description", "") for field in fields)


def test_diff_round_trip_adds_aggregate_node_with_output_type():
    changes = AllowedDagTabChanges(max_nodes=1)
    schema = changes.build_schema({"A": PARENT})
    diff = schema.validate(
        _evidence(
            nodes=[
                {
                    "id": "fitted_income",
                    "kind": "new_aggregate",
                    "input_cols": ["x0"],
                    "output_cols": ["fe_income_band"],
                    "output_types": {"fe_income_band": "categorical"},
                    "code": (
                        "threshold = df_fit['x0'].median()\n"
                        "df['fe_income_band'] = np.where(df['x0'] >= threshold, 'high', 'low')\n"
                        "return df"
                    ),
                    "rationale": "Fit an income split on training rows only.",
                    "is_output": True,
                }
            ]
        )
    )

    child = FeatureGraph.from_json(changes.apply(diff, {"A": PARENT}))

    assert child.nodes[0].kind == "aggregate"
    assert child.nodes[0].output_types == {"fe_income_band": "categorical"}
