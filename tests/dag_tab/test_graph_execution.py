from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import problems.dag_tab.execution as graph_execution
from problems.dag_tab.execution import (
    FeatureExecutionError,
    assert_split_invariant,
    assert_target_round_trip,
    execute_graph,
    execute_graph_triplet,
    inverse_target,
    literal_frame_reads,
    normalize_node_code,
    normalize_target_code,
    transform_target,
)
from problems.dag_tab.graph import FeatureGraph, FeatureNode

ROOT = Path(__file__).parents[2]
SEED = ROOT / "problems/dag_tab/initial_programs/baseline.json"


def _node(**updates) -> FeatureNode:
    data = {
        "id": "first",
        "input_cols": ["x0"],
        "output_cols": ["fe_first"],
        "code": "df['fe_first'] = df['x0'] * 2\nreturn df",
        "rationale": "Double the first raw feature.",
        "dependencies": [],
        "is_output": True,
    }
    data.update(updates)
    return FeatureNode(**data)


def test_module_form_is_preserved_and_module_state_never_crosses_row_blocks():
    code = """calls = 0
def add_call_number(values, call_number):
    return values + call_number

def transform(df):
    global calls
    calls += 1
    result = df.copy()
    result['fe_first'] = add_call_number(df['x0'], calls)
    return result
"""

    normalized = normalize_node_code(code)

    assert normalized == code.strip()
    assert "def add_call_number" in normalized
    assert normalized.endswith("return result")
    graph = FeatureGraph(
        dataset="california",
        raw_columns=["x0"],
        nodes=[_node(code=normalized)],
    )
    result = execute_graph_triplet(
        graph,
        pd.DataFrame({"x0": [1.0]}),
        pd.DataFrame({"x0": [2.0]}),
        pd.DataFrame({"x0": [3.0]}),
    )
    assert result.fit["fe_first"].tolist() == [2.0]
    assert result.validation["fe_first"].tolist() == [3.0]
    assert result.query["fe_first"].tolist() == [4.0]


def test_normalize_target_code_extracts_function_and_preserves_imports():
    code = """import numpy as np
def transform(y_fit, y):
    return np.log1p(y)
"""
    inverse_code = """import numpy as np
def inverse(y_fit, predictions):
    return np.expm1(predictions)
"""

    assert normalize_target_code(code) == "import numpy as np\nreturn np.log1p(y)"
    assert normalize_target_code(inverse_code, inverse=True) == (
        "import numpy as np\nreturn np.expm1(predictions)"
    )


def test_literal_frame_reads_supports_copy_alias_and_assign():
    copy_code = normalize_node_code(
        """def transform(df):
    result = df.copy()
    result['fe_first'] = result['x0'] * 2
    return result
"""
    )
    assign_code = "return df.assign(fe_first=df['x0'] * 2)"

    assert literal_frame_reads(copy_code) == {"x0"}
    assert literal_frame_reads(assign_code) == {"x0"}


def test_literal_frame_reads_allows_declared_whole_frame_reductions():
    rowwise = "return df.sum(axis=1).to_frame('total_count')"
    aggregate = "return df_fit.groupby('x10')['x0'].mean()"

    assert literal_frame_reads(rowwise) == set()
    assert literal_frame_reads(aggregate) == set()


def test_seed_round_trips_and_executes():
    graph = FeatureGraph.model_validate(json.loads(SEED.read_text()))
    frame = pd.DataFrame(np.ones((3, 8)), columns=graph.raw_columns)

    result = execute_graph(graph, frame)

    assert result.columns.tolist() == [*graph.raw_columns, "fe_income_per_age"]
    assert result["fe_income_per_age"].tolist() == [0.5, 0.5, 0.5]
    assert FeatureGraph.from_json(graph.to_json()) == graph


def test_graph_accepts_backward_dependency_chain():
    graph = FeatureGraph(
        dataset="california",
        raw_columns=["x0"],
        nodes=[
            _node(is_output=False),
            _node(
                id="second",
                input_cols=["fe_first"],
                output_cols=["fe_second"],
                code="df['fe_second'] = df['fe_first'] + 1\nreturn df",
                dependencies=["first"],
            ),
        ],
    )

    assert graph.depth == 2
    assert graph.output_columns == ["fe_second"]


def test_graph_rejects_forward_dependency():
    with pytest.raises(ValueError, match="earlier nodes"):
        FeatureGraph(
            dataset="california",
            raw_columns=["x0"],
            nodes=[_node(dependencies=["later"])],
        )


def test_graph_rejects_generated_input_without_dependency():
    with pytest.raises(ValueError, match="declared dependencies"):
        FeatureGraph(
            dataset="california",
            raw_columns=["x0"],
            nodes=[
                _node(is_output=False),
                _node(
                    id="second",
                    input_cols=["fe_first"],
                    output_cols=["fe_second"],
                    code="df['fe_second'] = df['fe_first'] + 1\nreturn df",
                    dependencies=[],
                ),
            ],
        )


def test_execution_rejects_missing_declared_assignment():
    graph = FeatureGraph(
        dataset="california",
        raw_columns=["x0"],
        nodes=[_node(code="value = df['x0'] * 2\nreturn df")],
    )
    with pytest.raises(FeatureExecutionError, match="missing declared outputs"):
        execute_graph(graph, pd.DataFrame({"x0": [1.0]}))


def test_execution_rejects_string_output_at_execution_boundary():
    graph = FeatureGraph(
        dataset="california",
        raw_columns=["x0"],
        nodes=[_node(code="df['fe_first'] = 'not numeric'\nreturn df")],
    )

    with pytest.raises(FeatureExecutionError, match="non-numeric dtype"):
        execute_graph(graph, pd.DataFrame({"x0": [1.0]}))


def test_execution_allows_nan_but_rejects_inf():
    nan_graph = FeatureGraph(
        dataset="california",
        raw_columns=["x0"],
        nodes=[_node(code="df['fe_first'] = np.nan\nreturn df")],
    )
    result = execute_graph(nan_graph, pd.DataFrame({"x0": [1.0]}))
    assert result["fe_first"].isna().all()

    inf_graph = FeatureGraph(
        dataset="california",
        raw_columns=["x0"],
        nodes=[_node(code="df['fe_first'] = np.inf\nreturn df")],
    )
    with pytest.raises(FeatureExecutionError, match="contains inf"):
        execute_graph(inf_graph, pd.DataFrame({"x0": [1.0]}))


def test_execution_rejects_undeclared_assignment_target():
    graph = FeatureGraph(
        dataset="california",
        raw_columns=["x0"],
        nodes=[
            _node(
                code=(
                    "df['fe_first'] = df['x0'] * 2\ndf['extra'] = df['x0']\nreturn df"
                )
            )
        ],
    )

    with pytest.raises(FeatureExecutionError, match="created undeclared columns"):
        execute_graph(graph, pd.DataFrame({"x0": [1.0]}))


def test_execution_rejects_undeclared_read():
    graph = FeatureGraph(
        dataset="california",
        raw_columns=["x0", "x1"],
        nodes=[_node(code="df['fe_first'] = df['x1'] * 2\nreturn df")],
    )

    with pytest.raises(FeatureExecutionError, match="unavailable input column"):
        execute_graph(graph, pd.DataFrame({"x0": [1.0], "x1": [2.0]}))


def test_execution_allows_reading_own_output():
    graph = FeatureGraph(
        dataset="california",
        raw_columns=["x0"],
        nodes=[
            _node(
                output_cols=["fe_a", "fe_b"],
                code=(
                    "df['fe_a'] = df['x0'] + 1\ndf['fe_b'] = df['fe_a'] * 2\nreturn df"
                ),
            )
        ],
    )

    result = execute_graph(graph, pd.DataFrame({"x0": [1.0, 2.0]}))

    assert result.columns.tolist() == ["x0", "fe_a", "fe_b"]
    assert result["fe_a"].tolist() == [2.0, 3.0]
    assert result["fe_b"].tolist() == [4.0, 6.0]


@pytest.mark.parametrize(
    "code",
    [
        "df['fe_first'] = df.x1 * 2\nreturn df",
        "df['fe_first'] = df.loc[:, 'x1']\nreturn df",
    ],
)
def test_execution_restricts_frame_to_declared_inputs(code):
    graph = FeatureGraph(
        dataset="california",
        raw_columns=["x0", "x1"],
        nodes=[_node(code=code)],
    )

    with pytest.raises(FeatureExecutionError):
        execute_graph(graph, pd.DataFrame({"x0": [1.0], "x1": [2.0]}))


def test_static_validation_rejects_rowwise_fit_abi_names():
    graph = FeatureGraph(
        dataset="california",
        raw_columns=["x0"],
        nodes=[
            _node(
                code=("df['fe_first'] = df['x0'] + np.asarray(y_fit).mean()\nreturn df")
            )
        ],
    )

    with pytest.raises(
        FeatureExecutionError, match="rowwise code cannot reference y_fit"
    ):
        execute_graph(graph, pd.DataFrame({"x0": [1.0]}))


def test_static_validation_rejects_pandas_api_on_numpy_y_fit():
    graph = FeatureGraph(
        dataset="california",
        raw_columns=["x0"],
        nodes=[
            _node(
                kind="aggregate",
                code=(
                    "means = y_fit.groupby(df_fit['x0']).mean()\n"
                    "df['fe_first'] = df['x0'].map(means)\nreturn df"
                ),
            )
        ],
    )

    with pytest.raises(FeatureExecutionError, match="numpy.ndarray.*groupby"):
        execute_graph_triplet(
            graph,
            pd.DataFrame({"x0": [1.0, 2.0]}),
            pd.DataFrame({"x0": []}),
            pd.DataFrame({"x0": []}),
            y_fit=np.array([1.0, 2.0]),
        )


def test_execution_rejects_literal_undeclared_frame_read():
    graph = FeatureGraph(
        dataset="california",
        raw_columns=["x0", "x1"],
        nodes=[_node(code="df['fe_first'] = df['x1'] * 2\nreturn df")],
    )

    with pytest.raises(FeatureExecutionError, match="unavailable input column.*x1"):
        execute_graph(graph, pd.DataFrame({"x0": [1.0], "x1": [2.0]}))


def test_execution_allows_full_python_imports():
    graph = FeatureGraph(
        dataset="california",
        raw_columns=["x0"],
        nodes=[
            _node(
                code="import numpy as imported_np\ndf['fe_first'] = imported_np.sqrt(df['x0'])\nreturn df"
            )
        ],
    )

    result = execute_graph(graph, pd.DataFrame({"x0": [4.0]}))
    assert result["fe_first"].tolist() == [2.0]


def test_execution_rejects_overwriting_pre_existing_column():
    graph = FeatureGraph(
        dataset="california",
        raw_columns=["x0"],
        nodes=[
            _node(
                code=("df.loc[:, 'x0'] = 0\ndf['fe_first'] = df['x0'] * 2\nreturn df")
            )
        ],
    )

    with pytest.raises(FeatureExecutionError, match="modified pre-existing columns"):
        execute_graph(graph, pd.DataFrame({"x0": [1.0]}))


def test_aggregate_node_fits_on_fit_rows_and_applies_to_every_role():
    graph = FeatureGraph(
        dataset="california",
        raw_columns=["x0"],
        nodes=[
            _node(
                kind="aggregate",
                code=(
                    "mean = df_fit['x0'].mean()\n"
                    "df['fe_first'] = df['x0'] - mean\nreturn df"
                ),
            )
        ],
    )

    result = execute_graph_triplet(
        graph,
        pd.DataFrame({"x0": [1.0, 3.0]}),
        pd.DataFrame({"x0": [10.0]}),
        pd.DataFrame({"x0": [100.0]}),
    )

    assert result.fit["fe_first"].tolist() == [-1.0, 1.0]
    assert result.validation["fe_first"].tolist() == [8.0]
    assert result.query["fe_first"].tolist() == [98.0]


def test_execution_allows_row_wise_reduction():
    graph = FeatureGraph(
        dataset="california",
        raw_columns=["x0", "x1"],
        nodes=[
            _node(
                input_cols=["x0", "x1"],
                code=("df['fe_first'] = df[['x0', 'x1']].sum(axis=1)\nreturn df"),
            )
        ],
    )

    result = execute_graph(
        graph,
        pd.DataFrame({"x0": [1.0, 2.0], "x1": [3.0, 4.0]}),
    )

    assert result["fe_first"].tolist() == [4.0, 6.0]


def test_batch_purity_probe_rejects_transform_batch_rank():
    graph = FeatureGraph(
        dataset="california",
        raw_columns=["x0"],
        nodes=[
            _node(
                kind="aggregate",
                code="df['fe_first'] = df['x0'].rank(pct=True)\nreturn df",
            )
        ],
    )

    with pytest.raises(FeatureExecutionError, match="use an aggregate node and fit"):
        assert_split_invariant(graph, pd.DataFrame({"x0": [1.0, 2.0, 3.0, 4.0]}))


def test_split_invariance_probe_rejects_position_dependent_output():
    graph = FeatureGraph(
        dataset="california",
        raw_columns=["x0"],
        nodes=[_node(code="df['fe_first'] = np.arange(len(df)) * 1.0\nreturn df")],
    )

    with pytest.raises(FeatureExecutionError, match="split-dependent behavior"):
        assert_split_invariant(graph, pd.DataFrame({"x0": [1.0, 2.0, 3.0, 4.0]}))


def test_split_invariance_probe_rejects_last_row_broadcast():
    graph = FeatureGraph(
        dataset="california",
        raw_columns=["x0"],
        nodes=[_node(code="df['fe_first'] = df['x0'].iloc[-1]\nreturn df")],
    )

    with pytest.raises(FeatureExecutionError, match="split-dependent behavior"):
        assert_split_invariant(graph, pd.DataFrame({"x0": [1.0, 2.0, 3.0, 4.0]}))


def test_split_invariance_probe_rejects_first_row_broadcast():
    graph = FeatureGraph(
        dataset="california",
        raw_columns=["x0"],
        nodes=[_node(code="df['fe_first'] = df['x0'].iloc[0]\nreturn df")],
    )

    with pytest.raises(FeatureExecutionError, match="split-dependent behavior"):
        assert_split_invariant(graph, pd.DataFrame({"x0": [1.0, 2.0, 3.0, 4.0]}))


def test_split_invariance_probe_allows_row_wise_output():
    graph = FeatureGraph(
        dataset="california",
        raw_columns=["x0"],
        nodes=[_node()],
    )

    assert_split_invariant(graph, pd.DataFrame({"x0": [1.0, 2.0, 3.0, 4.0]}))


def test_determinism_probe_rejects_unseeded_random_output():
    graph = FeatureGraph(
        dataset="california",
        raw_columns=["x0"],
        nodes=[
            _node(
                code="df['fe_first'] = np.random.default_rng().random(len(df))\nreturn df"
            )
        ],
    )

    with pytest.raises(FeatureExecutionError, match="non-deterministic behavior"):
        assert_split_invariant(graph, pd.DataFrame({"x0": [1.0, 2.0, 3.0, 4.0]}))


def test_generated_categorical_output_is_explicitly_typed():
    graph = FeatureGraph(
        dataset="california",
        raw_columns=["x0"],
        nodes=[
            _node(
                output_types={"fe_first": "categorical"},
                code="df['fe_first'] = np.where(df['x0'] > 0, 'positive', 'other')\nreturn df",
            )
        ],
    )

    result = execute_graph(graph, pd.DataFrame({"x0": [-1.0, 1.0]}))
    assert result["fe_first"].tolist() == ["other", "positive"]


def test_supervised_aggregate_receives_y_fit():
    graph = FeatureGraph(
        dataset="california",
        raw_columns=["x0"],
        nodes=[
            _node(
                kind="aggregate",
                code=("df['fe_first'] = float(np.mean(y_fit))\nreturn df"),
            )
        ],
    )

    result = execute_graph_triplet(
        graph,
        pd.DataFrame({"x0": [1.0, 2.0]}),
        pd.DataFrame({"x0": [3.0]}),
        pd.DataFrame({"x0": [4.0]}),
        y_fit=np.array([10.0, 20.0]),
    )

    assert result.fit["fe_first"].tolist() == [15.0, 15.0]
    assert result.query["fe_first"].tolist() == [15.0]


def test_own_target_probe_rejects_naive_target_encoding():
    graph = FeatureGraph(
        dataset="california",
        raw_columns=["x0"],
        nodes=[
            _node(
                kind="aggregate",
                code=(
                    "means = pd.Series(y_fit).groupby(df_fit['x0']).mean()\n"
                    "df['fe_first'] = df['x0'].map(means).fillna(np.mean(y_fit))\n"
                    "return df"
                ),
            )
        ],
    )

    with pytest.raises(FeatureExecutionError, match="own-target leakage"):
        assert_split_invariant(
            graph,
            pd.DataFrame({"x0": [0, 0, 1, 1]}),
            np.array([1.0, 3.0, 10.0, 14.0]),
        )


def test_own_target_probe_allows_leave_one_out_encoding():
    graph = FeatureGraph(
        dataset="california",
        raw_columns=["x0"],
        nodes=[
            _node(
                kind="aggregate",
                code=(
                    "keys = df_fit['x0'].to_numpy()\n"
                    "targets = np.asarray(y_fit, dtype=float)\n"
                    "global_mean = targets.mean()\n"
                    "def encode(row):\n"
                    "    matches = np.flatnonzero(keys == row['x0'])\n"
                    "    if row.name < len(df_fit) and row.name in matches:\n"
                    "        matches = matches[matches != row.name]\n"
                    "    return targets[matches].mean() if len(matches) else global_mean\n"
                    "df['fe_first'] = df.apply(encode, axis=1)\n"
                    "return df"
                ),
            )
        ],
    )

    assert_split_invariant(
        graph,
        pd.DataFrame({"x0": [0, 0, 1, 1]}),
        np.array([1.0, 3.0, 10.0, 14.0]),
    )


def test_own_target_probe_bounds_and_spreads_checked_rows(monkeypatch):
    graph = FeatureGraph(
        dataset="california",
        raw_columns=["x0"],
        nodes=[
            _node(
                kind="aggregate",
                code="df['fe_first'] = df['x0'] * 2\nreturn df",
            )
        ],
    )
    checked: list[int] = []
    real_perturb = graph_execution._perturbed_target

    def record_perturb(target, index, scale):
        checked.append(index)
        return real_perturb(target, index, scale)

    monkeypatch.setattr(graph_execution, "_perturbed_target", record_perturb)
    assert_split_invariant(
        graph,
        pd.DataFrame({"x0": np.arange(128, dtype=float)}),
        np.arange(128, dtype=float),
    )

    assert len(checked) == 64
    assert checked[0] == 0
    assert checked[-1] == 127
    assert checked == sorted(set(checked))


def test_dropped_raw_columns_remain_available_to_nodes_but_not_output():
    graph = FeatureGraph(
        dataset="california",
        raw_columns=["x0", "x1"],
        dropped_raw_columns=["x1"],
        nodes=[
            _node(
                input_cols=["x1"],
                code="df['fe_first'] = df['x1'] * 2\nreturn df",
            )
        ],
    )

    result = execute_graph(graph, pd.DataFrame({"x0": [1.0], "x1": [3.0]}))

    assert result.columns.tolist() == ["x0", "fe_first"]
    assert result["fe_first"].tolist() == [6.0]


def test_graph_rejects_unknown_dropped_raw_column():
    with pytest.raises(ValueError, match="unknown raw columns"):
        FeatureGraph(
            dataset="california",
            raw_columns=["x0"],
            dropped_raw_columns=["x1"],
            nodes=[_node()],
        )


def test_target_log_transform_round_trips():
    graph = FeatureGraph(
        dataset="california",
        raw_columns=["x0"],
        target={
            "code": "return np.log1p(y)",
            "inverse_code": "return np.expm1(predictions)",
        },
        nodes=[_node()],
    )
    y_fit = np.array([0.0, 1.0, 3.0])

    assert_target_round_trip(graph.target, y_fit)
    transformed = transform_target(graph.target, y_fit, np.array([1.0, 3.0]))
    restored = inverse_target(graph.target, y_fit, transformed)

    np.testing.assert_allclose(restored, [1.0, 3.0])


def test_target_round_trip_rejects_non_inverse_pair():
    graph = FeatureGraph(
        dataset="california",
        raw_columns=["x0"],
        target={
            "code": "return y * 2",
            "inverse_code": "return predictions / 3",
        },
        nodes=[_node()],
    )

    with pytest.raises(FeatureExecutionError, match="do not round-trip"):
        assert_target_round_trip(graph.target, np.array([1.0, 2.0, 3.0]))


def test_declared_but_unread_dependencies_do_not_inflate_topology():
    phantom = FeatureGraph(
        dataset="california",
        raw_columns=["x0"],
        nodes=[
            _node(),
            _node(
                id="second",
                input_cols=["x0"],
                output_cols=["fe_second"],
                code="df['fe_second'] = df['x0'] + 1\nreturn df",
                dependencies=["first"],
            ),
        ],
    )

    assert phantom.depth == 1
