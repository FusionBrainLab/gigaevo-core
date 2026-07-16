from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from problems.dag_tab.execution import (
    FeatureExecutionError,
    assert_split_invariant,
    execute_graph,
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
    with pytest.raises(FeatureExecutionError, match="explicitly assign"):
        execute_graph(graph, pd.DataFrame({"x0": [1.0]}))


def test_execution_rejects_non_finite_output():
    graph = FeatureGraph(
        dataset="california",
        raw_columns=["x0"],
        nodes=[_node(code="df['fe_first'] = np.inf\nreturn df")],
    )
    with pytest.raises(FeatureExecutionError, match="NaN or inf"):
        execute_graph(graph, pd.DataFrame({"x0": [1.0]}))


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

    with pytest.raises(FeatureExecutionError, match="assigns undeclared columns"):
        execute_graph(graph, pd.DataFrame({"x0": [1.0]}))


def test_execution_rejects_undeclared_read():
    graph = FeatureGraph(
        dataset="california",
        raw_columns=["x0", "x1"],
        nodes=[_node(code="df['fe_first'] = df['x1'] * 2\nreturn df")],
    )

    with pytest.raises(FeatureExecutionError, match="reads undeclared input columns"):
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


def test_execution_rejects_string_expression_api():
    graph = FeatureGraph(
        dataset="california",
        raw_columns=["x0"],
        nodes=[_node(code="df['fe_first'] = df.eval('x0 * 2')\nreturn df")],
    )

    with pytest.raises(FeatureExecutionError, match="string-expression API"):
        execute_graph(graph, pd.DataFrame({"x0": [1.0]}))


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


def test_execution_rejects_split_dependent_statistics():
    graph = FeatureGraph(
        dataset="california",
        raw_columns=["x0"],
        nodes=[_node(code="df['fe_first'] = df['x0'].rank()\nreturn df")],
    )

    with pytest.raises(FeatureExecutionError, match="split-dependent operation"):
        execute_graph(graph, pd.DataFrame({"x0": [1.0, 2.0]}))


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


def test_execution_rejects_reduction_without_axis():
    graph = FeatureGraph(
        dataset="california",
        raw_columns=["x0", "x1"],
        nodes=[
            _node(
                input_cols=["x0", "x1"],
                code="df['fe_first'] = df[['x0', 'x1']].sum()\nreturn df",
            )
        ],
    )

    with pytest.raises(FeatureExecutionError, match="split-dependent operation"):
        execute_graph(graph, pd.DataFrame({"x0": [1.0], "x1": [2.0]}))


def test_split_invariance_probe_rejects_position_dependent_output():
    graph = FeatureGraph(
        dataset="california",
        raw_columns=["x0"],
        nodes=[_node(code="df['fe_first'] = np.arange(len(df)) * 1.0\nreturn df")],
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
