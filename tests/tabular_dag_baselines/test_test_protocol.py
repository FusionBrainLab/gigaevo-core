from __future__ import annotations

import numpy as np

from problems.dag_tab.graph import FeatureGraph
from problems.tabular_dag_baselines import validation


def test_test_scoring_delegates_to_canonical_tabular_problem(monkeypatch):
    payload = FeatureGraph(
        dataset="california", raw_columns=["x0", "x1"], nodes=[]
    ).model_dump(mode="json")
    calls = {}

    class _Problem:
        def score_on_test(self, factory):
            calls["instance"] = factory()
            return {"test_rmse": 0.4, "test_r2": 0.8}

    def build(dataset):
        calls["dataset"] = dataset
        return _Problem()

    monkeypatch.setattr(validation, "build", build)
    sentinel = object()

    result = validation.score_payload_on_test(
        payload,
        model_builder=lambda _graph, device: (sentinel, device),
    )

    assert calls["dataset"] == "california"
    assert calls["instance"] == (sentinel, None)
    assert result == {"test_rmse": 0.4, "test_r2": 0.8}


def test_canonical_test_problem_passes_train_val_and_untouched_test(monkeypatch):
    from problems.tabular._common import tabular_problem

    arrays = {
        "X_train": np.arange(12).reshape(6, 2),
        "y_train": np.arange(6.0),
        "X_val": np.arange(12, 20).reshape(4, 2),
        "y_val": np.arange(6.0, 10.0),
        "X_test": np.arange(20, 26).reshape(3, 2),
        "y_test": np.arange(10.0, 13.0),
    }

    class _Dataset:
        task_type = "regression"
        n_classes = None

        def __init__(self):
            for name, value in arrays.items():
                setattr(self, name, value)

    class _Spy:
        def fit_predict(self, X_train, y_train, X_val, y_val, X_query):
            np.testing.assert_array_equal(X_train, arrays["X_train"])
            np.testing.assert_array_equal(y_train, arrays["y_train"])
            np.testing.assert_array_equal(X_val, arrays["X_val"])
            np.testing.assert_array_equal(y_val, arrays["y_val"])
            np.testing.assert_array_equal(X_query, arrays["X_test"])
            return arrays["y_test"].copy()

    monkeypatch.setattr(
        tabular_problem.tabular_data, "load_dataset", lambda _name: _Dataset()
    )
    result = tabular_problem.build("california").score_on_test(_Spy)

    assert result["test_rmse"] == 0.0
    assert result["test_r2"] == 1.0
