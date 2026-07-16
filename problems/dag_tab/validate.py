"""GigaEvo validator for FeatureGraph JSON genomes."""

from __future__ import annotations

from pathlib import Path
import sys

from catboost import CatBoostClassifier, CatBoostRegressor
import numpy as np
import pandas as pd

_SOURCE_PATH = globals().get("__file__")
_PROBLEM_DIR = (
    Path(_SOURCE_PATH).resolve().parent if _SOURCE_PATH else Path(sys.path[0]).resolve()
)
_TABULAR_COMMON = _PROBLEM_DIR.parent / "tabular" / "_common"
if str(_TABULAR_COMMON) not in sys.path:
    sys.path.insert(0, str(_TABULAR_COMMON))

import tabular_data  # noqa: E402
from tabular_problem import build  # noqa: E402

from problems.dag_tab.execution import (  # noqa: E402
    assert_split_invariant,
    execute_graph,
)
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
        dataset = tabular_data.load_dataset(graph.dataset)
        self.task_type = dataset.task_type
        self.n_classes = dataset.n_classes

    def fit_predict(self, X_train, y_train, X_val, y_val, X_query):
        train = execute_graph(self.graph, _frame(X_train, self.graph.raw_columns))
        val = execute_graph(self.graph, _frame(X_val, self.graph.raw_columns))
        query = execute_graph(self.graph, _frame(X_query, self.graph.raw_columns))
        train_x = train.to_numpy(dtype=float)
        val_x = val.to_numpy(dtype=float)
        query_x = query.to_numpy(dtype=float)
        train_y = np.asarray(y_train)
        val_y = np.asarray(y_val)
        fit_x = np.concatenate([train_x, val_x])
        fit_y = np.concatenate([train_y, val_y])
        params = {
            "learning_rate": 0.05,
            "depth": 6,
            "random_seed": 0,
            "thread_count": 4,
            "logging_level": "Silent",
            "allow_writing_files": False,
        }

        if self.task_type == tabular_data.REGRESSION:
            search = CatBoostRegressor(
                iterations=2000,
                early_stopping_rounds=50,
                **params,
            )
            search.fit(train_x, train_y, eval_set=(val_x, val_y))
            best_iterations = max(1, search.get_best_iteration() + 1)
            model = CatBoostRegressor(iterations=best_iterations, **params)
            model.fit(fit_x, fit_y)
            return model.predict(query_x)

        train_y = train_y.astype(int)
        val_y = val_y.astype(int)
        fit_y = fit_y.astype(int)
        if self.n_classes is None or self.n_classes < 2:
            raise ValueError("classification dataset must declare n_classes >= 2")
        n_classes = int(self.n_classes)
        observed_classes = np.unique(fit_y)
        if np.any(observed_classes < 0) or np.any(observed_classes >= n_classes):
            raise ValueError(
                f"classification labels {observed_classes.tolist()} fall outside "
                f"declared class universe [0, {n_classes})"
            )
        classification_params = dict(params)
        if n_classes > 2:
            classification_params.update(
                {"loss_function": "MultiClass", "classes_count": n_classes}
            )
        search = CatBoostClassifier(
            iterations=2000,
            early_stopping_rounds=50,
            **classification_params,
        )
        search.fit(train_x, train_y, eval_set=(val_x, val_y))
        best_iterations = max(1, search.get_best_iteration() + 1)
        model = CatBoostClassifier(iterations=best_iterations, **classification_params)
        model.fit(fit_x, fit_y)
        probabilities = np.zeros((query_x.shape[0], n_classes))
        probabilities[:, model.classes_.astype(int)] = model.predict_proba(query_x)
        return probabilities


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
        sample_size = min(1024, len(dataset.X_train))
        assert_split_invariant(
            graph,
            _frame(dataset.X_train[:sample_size], graph.raw_columns),
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
