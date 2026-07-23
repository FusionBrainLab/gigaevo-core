from __future__ import annotations

import gc
import weakref

import numpy as np
import pytest

from problems.dag_tab.graph import FeatureGraph
from problems.tabular_dag_baselines import validation


def test_failed_evaluation_drops_traceback_resources_before_cleanup(
    monkeypatch,
):
    payload = FeatureGraph(
        dataset="california", raw_columns=["x0", "x1"], nodes=[]
    ).model_dump(mode="json")
    state = {}
    cache = {}

    class _Dataset:
        task_type = "regression"
        X_train = np.zeros((4, 2))
        y_train = np.zeros(4)

    class _Resource:
        pass

    class _Problem:
        def validate(self, factory):
            resource = factory()
            state["reference"] = weakref.ref(resource)
            raise RuntimeError("model failure")

    def build(_dataset):
        return _Problem()

    def model_builder(_graph, _device):
        resource = _Resource()
        cache["resource"] = resource
        return resource

    def cleanup():
        cache.clear()
        gc.collect()
        state["collected_before_release"] = state["reference"]() is None

    monkeypatch.setattr(
        validation.tabular_data, "load_dataset", lambda _name: _Dataset()
    )
    monkeypatch.setattr(validation, "build", build)

    metrics, artifact = validation.validate_payload(
        payload,
        estimator_name="resource-test",
        model_builder=model_builder,
        config={},
        resource_cleanup=cleanup,
    )

    assert metrics["is_valid"] == 0.0
    assert artifact["validation_failure_stage"] == "model_fit"
    assert state["collected_before_release"] is True


def test_failed_test_score_drops_chained_traceback_resources_before_cleanup(
    monkeypatch,
):
    payload = FeatureGraph(
        dataset="california", raw_columns=["x0", "x1"], nodes=[]
    ).model_dump(mode="json")
    state = {}
    cache = {}

    class _Resource:
        def fail(self):
            raise ValueError("inner failure")

    class _Problem:
        def score_on_test(self, factory):
            resource = factory()
            state["reference"] = weakref.ref(resource)
            try:
                resource.fail()
            except ValueError as exc:
                raise RuntimeError("outer failure") from exc

    def model_builder(_graph, _device):
        resource = _Resource()
        cache["resource"] = resource
        return resource

    def cleanup():
        cache.clear()
        gc.collect()
        state["collected_before_release"] = state["reference"]() is None

    monkeypatch.setattr(validation, "build", lambda _dataset: _Problem())

    with pytest.raises(RuntimeError, match="RuntimeError: outer failure"):
        validation.score_payload_on_test(
            payload,
            model_builder=model_builder,
            resource_cleanup=cleanup,
        )

    assert state["collected_before_release"] is True


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
