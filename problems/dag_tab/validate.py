"""GigaEvo validator for FeatureGraph JSON genomes."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

_SOURCE_PATH = globals().get("__file__")
_PROBLEM_DIR = (
    Path(_SOURCE_PATH).resolve().parent if _SOURCE_PATH else Path(sys.path[0]).resolve()
)
_TABULAR_COMMON = _PROBLEM_DIR.parent / "tabular" / "_common"
if str(_TABULAR_COMMON) not in sys.path:
    sys.path.insert(0, str(_TABULAR_COMMON))

import tabular_data  # noqa: E402
from tabular_problem import build  # noqa: E402

from problems.dag_tab.execution import execute_graph  # noqa: E402
from problems.dag_tab.graph import FeatureGraph  # noqa: E402

_INVALID = {
    "fitness": -1.0,
    "is_valid": 0.0,
    "cv_score_std": 2.0,
    "graph_node_count": 0.0,
    "graph_max_depth": 0.0,
    "generated_feature_count": 0.0,
    "local_lipschitz_p95": 4.0,
    "ood_delta_slope": 2.0,
}


def _frame(values: np.ndarray, columns: list[str]) -> pd.DataFrame:
    array = np.asarray(values)
    if array.ndim != 2 or array.shape[1] != len(columns):
        raise ValueError(
            f"dataset matrix shape {array.shape} does not match {len(columns)} raw columns"
        )
    return pd.DataFrame(array, columns=columns)


class FeatureGraphModel:
    """Fixed estimator whose evolvable component is the feature graph."""

    def __init__(self, graph: FeatureGraph):
        self.graph = graph
        self.task_type = tabular_data.load_dataset(graph.dataset).task_type

    def fit_predict(self, X_train, y_train, X_val, y_val, X_query):
        train = execute_graph(self.graph, _frame(X_train, self.graph.raw_columns))
        val = execute_graph(self.graph, _frame(X_val, self.graph.raw_columns))
        query = execute_graph(self.graph, _frame(X_query, self.graph.raw_columns))
        fit_x = np.concatenate([train.to_numpy(dtype=float), val.to_numpy(dtype=float)])
        fit_y = np.concatenate([np.asarray(y_train), np.asarray(y_val)])

        if self.task_type == tabular_data.REGRESSION:
            model = HistGradientBoostingRegressor(
                learning_rate=0.08,
                max_iter=180,
                max_leaf_nodes=31,
                l2_regularization=0.1,
                random_state=0,
            )
            model.fit(fit_x, fit_y)
            return model.predict(query.to_numpy(dtype=float))

        model = HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=180,
            max_leaf_nodes=31,
            l2_regularization=0.1,
            random_state=0,
        )
        model.fit(fit_x, fit_y.astype(int))
        return model.predict_proba(query.to_numpy(dtype=float))


def _factory(graph: FeatureGraph):
    return lambda: FeatureGraphModel(graph)


def validate(payload):
    """Validate and score a decoded FeatureGraph JSON document."""
    try:
        graph = FeatureGraph.model_validate(payload)
        dataset = tabular_data.load_dataset(graph.dataset)
        expected = [f"x{i}" for i in range(dataset.X_train.shape[1])]
        if graph.raw_columns != expected:
            raise ValueError(
                f"raw_columns must exactly match dataset columns {expected}; "
                f"got {graph.raw_columns}"
            )
        metrics = build(graph.dataset).validate(_factory(graph))
        metrics.update(
            {
                "graph_node_count": float(len(graph.nodes)),
                "graph_max_depth": float(graph.depth),
                "generated_feature_count": float(len(graph.output_columns)),
            }
        )
        return metrics, {
            "dataset": graph.dataset,
            "output_columns": graph.output_columns,
            "graph_node_count": len(graph.nodes),
        }
    except Exception as exc:
        return dict(_INVALID), {"error": f"{type(exc).__name__}: {exc}"}


def score_on_test(payload):
    graph = FeatureGraph.model_validate(payload)
    return build(graph.dataset).score_on_test(_factory(graph))
