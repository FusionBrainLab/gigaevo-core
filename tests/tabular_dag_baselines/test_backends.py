from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from problems.dag_tab.execution import GraphTriplet
from problems.dag_tab.graph import FeatureGraph
from problems.tabular._common.tabular_data import ColumnSpec
from problems.tabular_dag_baselines.foundation_backend import (
    InContextFeatureGraphModel,
)
from problems.tabular_dag_baselines.lightgbm.backend import (
    LightGBMConfig,
    LightGBMFeatureGraphModel,
)
from problems.tabular_dag_baselines.realmlp.backend import (
    RealMLPConfig,
    RealMLPFeatureGraphModel,
)
from problems.tabular_dag_baselines.tabfm.backend import (
    TabFMConfig,
    TabFMFeatureGraphModel,
    ensure_tabfm_ready,
)
from problems.tabular_dag_baselines.tabicl.backend import (
    TabICLConfig,
    TabICLFeatureGraphModel,
)
from problems.tabular_dag_baselines.tabpfn.backend import (
    TabPFNConfig,
    TabPFNFeatureGraphModel,
    ensure_tabpfn_ready,
)
from problems.tabular_dag_baselines.tabpfn.validate import validate as validate_tabpfn
from problems.tabular_dag_baselines.xgboost.backend import (
    XGBoostConfig,
    XGBoostFeatureGraphModel,
)


class _RegressionDataset:
    task_type = "regression"
    n_classes = None
    columns = (
        ColumnSpec(0, "numeric", None, None),
        ColumnSpec(1, "categorical", 3, ("a", "b", "c")),
    )


class _MulticlassDataset:
    task_type = "multiclass"
    n_classes = 3
    columns = _RegressionDataset.columns


@pytest.fixture
def graph(monkeypatch):
    monkeypatch.setattr(
        "problems.dag_tab.validate.tabular_data.load_dataset",
        lambda _name: _RegressionDataset(),
    )
    return FeatureGraph(dataset="california", raw_columns=["x0", "x1"], nodes=[])


def _regression_arrays():
    rng = np.random.default_rng(0)
    values = np.column_stack([rng.normal(size=96), rng.integers(0, 3, size=96)])
    target = 1.5 * values[:, 0] + 0.2 * values[:, 1]
    return values, target


@pytest.mark.parametrize(
    ("dependency", "model_class", "config"),
    [
        (
            "lightgbm",
            LightGBMFeatureGraphModel,
            LightGBMConfig(
                max_estimators=20,
                early_stopping_rounds=3,
                n_jobs=1,
            ),
        ),
        (
            "xgboost",
            XGBoostFeatureGraphModel,
            XGBoostConfig(
                max_estimators=20,
                early_stopping_rounds=3,
                n_jobs=1,
            ),
        ),
    ],
)
def test_boosting_backends_fit_installed_versions(
    graph, dependency, model_class, config
):
    pytest.importorskip(dependency)
    values, target = _regression_arrays()
    model = model_class(graph, config=config)

    prediction = model.fit_predict(
        values[:64], target[:64], values[64:80], target[64:80], values[80:]
    )

    assert prediction.shape == (16,)
    assert np.isfinite(prediction).all()
    assert 1 <= model.last_fit_summary["best_iterations"] <= 20


@pytest.mark.parametrize(
    ("dependency", "model_class", "config"),
    [
        (
            "lightgbm",
            LightGBMFeatureGraphModel,
            LightGBMConfig(
                max_estimators=20,
                early_stopping_rounds=3,
                n_jobs=1,
            ),
        ),
        (
            "xgboost",
            XGBoostFeatureGraphModel,
            XGBoostConfig(
                max_estimators=20,
                early_stopping_rounds=3,
                n_jobs=1,
            ),
        ),
    ],
)
def test_boosting_backends_return_full_multiclass_probabilities(
    monkeypatch, dependency, model_class, config
):
    pytest.importorskip(dependency)
    monkeypatch.setattr(
        "problems.dag_tab.validate.tabular_data.load_dataset",
        lambda _name: _MulticlassDataset(),
    )
    graph = FeatureGraph(dataset="otto", raw_columns=["x0", "x1"], nodes=[])
    rng = np.random.default_rng(4)
    target = np.tile(np.arange(3), 40)
    values = np.column_stack([target + rng.normal(scale=0.4, size=len(target)), target])
    model = model_class(graph, config=config)

    prediction = model.fit_predict(
        values[:84], target[:84], values[84:102], target[84:102], values[102:]
    )

    assert prediction.shape == (18, 3)
    assert np.isfinite(prediction).all()
    np.testing.assert_allclose(prediction.sum(axis=1), 1.0, atol=1e-6)


class _DummyContextEstimator:
    def __init__(self):
        self.fit_X = None
        self.fit_y = None

    def fit(self, X, y):
        self.fit_X = X.copy()
        self.fit_y = np.asarray(y).copy()
        return self

    def predict(self, X):
        return np.full(len(X), float(np.mean(self.fit_y)))


class _DummyFoundationModel(InContextFeatureGraphModel):
    estimator_name = "dummy-foundation"

    def __init__(self, graph):
        super().__init__(graph, device="cpu", config={})
        self.estimator = _DummyContextEstimator()

    def _make_estimator(self, categorical_indices):
        self.categorical_indices = categorical_indices
        return self.estimator


def test_foundation_model_uses_train_plus_validation_as_frozen_context(graph):
    values, target = _regression_arrays()
    model = _DummyFoundationModel(graph)

    prediction = model.fit_predict(
        values[:50], target[:50], values[50:65], target[50:65], values[65:70]
    )

    assert len(model.estimator.fit_X) == 65
    np.testing.assert_allclose(model.estimator.fit_y, target[:65])
    assert model.last_fit_summary["context_rows"] == 65
    assert model.categorical_indices == [1]
    assert prediction.shape == (5,)


def test_shared_categorical_vocabulary_is_fit_local(graph):
    model = _DummyFoundationModel(graph)
    prepared = model._prepare_model_triplet(
        GraphTriplet(
            pd.DataFrame({"x0": [0.0, 1.0], "x1": [0, 1]}),
            pd.DataFrame({"x0": [2.0], "x1": [2]}),
            pd.DataFrame({"x0": [3.0], "x1": [None]}),
        )
    )

    assert str(prepared.fit["x1"].dtype) == "category"
    assert prepared.validation["x1"].iloc[0] == "__UNKNOWN__"
    assert prepared.query["x1"].iloc[0] == "__MISSING__"


class _FakeRealMLP:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.fit_params_ = {"stop_epoch": {"rmse": 4}}
        self.fit_rows = None
        self.val_rows = None
        _FakeRealMLP.instances.append(self)

    def fit(self, X, y, *, X_val=None, y_val=None, val_idxs=None, **_kwargs):
        self.fit_X = X.copy()
        self.val_X = None if X_val is None else X_val.copy()
        self.fit_rows = len(X)
        self.val_rows = None if X_val is None else len(X_val)
        self.val_idxs = val_idxs
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


def test_realmlp_discards_search_model_and_refits_selected_epoch(graph, monkeypatch):
    _FakeRealMLP.instances.clear()
    monkeypatch.setattr(
        RealMLPFeatureGraphModel, "_model_class", lambda _self: _FakeRealMLP
    )
    model = RealMLPFeatureGraphModel(
        graph,
        device="cpu",
        config=replace(RealMLPConfig(), max_epochs=8, n_threads=1),
    )
    values, target = _regression_arrays()
    values[:50, 1] = np.arange(50) % 2
    values[50:65, 1] = 2

    model.fit_predict(
        values[:50], target[:50], values[50:65], target[50:65], values[65:70]
    )

    search, final = _FakeRealMLP.instances
    assert search.fit_rows == 50
    assert search.val_rows == 15
    assert set(search.val_X["x1"].astype(object)) <= set(
        search.fit_X["x1"].astype(object)
    )
    assert final.fit_rows == 65
    assert len(set(final.fit_X["x1"].astype(object))) == 3
    assert final.kwargs["val_fraction"] == 0.0
    assert final.kwargs["stop_epoch"] == {"rmse": 4}
    assert final.val_idxs.size == 0
    assert model.last_fit_summary["best_epochs"] == 4


def test_tabicl_constructor_is_pinned_without_downloading_checkpoint(graph):
    pytest.importorskip("tabicl")
    model = TabICLFeatureGraphModel(
        graph, device="cpu", config=TabICLConfig(n_estimators=3, batch_size=2)
    )

    estimator = model._make_estimator([1])
    params = estimator.get_params()

    assert params["checkpoint_version"] == "tabicl-regressor-v2-20260212.ckpt"
    assert params["n_estimators"] == 3
    assert params["batch_size"] == 2
    assert params["device"] == "cpu"


def test_tabfm_constructor_uses_pinned_recipe_and_shared_model(
    graph, monkeypatch, tmp_path
):
    pytest.importorskip("tabfm")
    checkpoint = tmp_path / "regression"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text('{"is_classifier": false}')
    (checkpoint / "model.safetensors").touch()
    loaded_model = object()
    calls = []

    def fake_load(**kwargs):
        calls.append(kwargs)
        return loaded_model

    monkeypatch.setattr(
        "problems.tabular_dag_baselines.tabfm.backend._load_tabfm_model",
        fake_load,
    )
    model_cache = {}
    config = TabFMConfig(
        n_estimators=3,
        batch_size=2,
        max_num_features=123,
        max_num_rows=456,
        model_path=str(tmp_path),
    )
    model = TabFMFeatureGraphModel(
        graph,
        device="cpu",
        config=config,
        model_cache=model_cache,
    )

    first = model._make_estimator([1])
    second = TabFMFeatureGraphModel(
        graph,
        device="cpu",
        config=config,
        model_cache=model_cache,
    )._make_estimator([1])
    params = first.get_params()

    assert len(calls) == 1
    assert second.model is loaded_model
    assert params["model"] is loaded_model
    assert params["n_estimators"] == 3
    assert params["batch_size"] == 2
    assert params["max_num_features"] == 123
    assert params["max_num_rows"] == 456
    assert params["random_state"] == 0
    with pytest.raises(RuntimeError, match="task does not match"):
        ensure_tabfm_ready(
            TabFMConfig(model_path=str(checkpoint)),
            model_type="classification",
        )


def test_tabpfn_constructor_is_pinned_to_v3_without_loading_weights(graph):
    pytest.importorskip("tabpfn")
    model = TabPFNFeatureGraphModel(
        graph, device="cpu", config=TabPFNConfig(n_estimators=3)
    )

    estimator = model._make_estimator([1])
    params = estimator.get_params()

    assert params["n_estimators"] == 3
    assert params["auto_scale_n_estimators"] is False
    assert params["categorical_features_indices"] == [1]
    assert params["device"] == "cpu"
    assert "v3" in str(params["model_path"]).lower()
    assert "mediumdata" in str(params["model_path"]).lower()


def test_tabpfn_preflight_accepts_explicit_checkpoint_and_rejects_missing(tmp_path):
    checkpoint = tmp_path / "tabpfn-v3.ckpt"
    checkpoint.touch()

    assert (
        ensure_tabpfn_ready(TabPFNConfig(model_path=str(checkpoint)), which="regressor")
        == checkpoint
    )
    with pytest.raises(RuntimeError, match="is not a file"):
        ensure_tabpfn_ready(
            TabPFNConfig(model_path=str(tmp_path / "missing.ckpt")),
            which="regressor",
        )
    wrong_version = tmp_path / "tabpfn-v2.ckpt"
    wrong_version.touch()
    with pytest.raises(RuntimeError, match="v3 checkpoint filename"):
        ensure_tabpfn_ready(
            TabPFNConfig(model_path=str(wrong_version)), which="regressor"
        )


def test_tabpfn_preflight_is_task_specific(monkeypatch, tmp_path):
    pytest.importorskip("tabpfn")
    import huggingface_hub
    from tabpfn import model_loading

    config = TabPFNConfig()
    classifier = tmp_path / config.classifier_checkpoint
    classifier.touch()
    monkeypatch.setattr(model_loading, "get_cache_dir", lambda: tmp_path)

    def unavailable(*_args, **_kwargs):
        raise RuntimeError("offline")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", unavailable)
    monkeypatch.setattr(model_loading, "download_model", unavailable)

    assert ensure_tabpfn_ready(config, which="classifier") == classifier
    with pytest.raises(RuntimeError, match="Could not prepare.*regressor"):
        ensure_tabpfn_ready(config, which="regressor")


def test_tabpfn_preflight_downloads_the_selected_checkpoint(monkeypatch, tmp_path):
    pytest.importorskip("tabpfn")
    import huggingface_hub
    from tabpfn import model_loading

    config = TabPFNConfig()
    calls = []
    monkeypatch.setattr(model_loading, "get_cache_dir", lambda: tmp_path)

    def fake_download(**kwargs):
        calls.append(kwargs)
        path = tmp_path / kwargs["filename"]
        path.touch()
        return str(path)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    expected = tmp_path / config.regressor_checkpoint

    assert ensure_tabpfn_ready(config, which="regressor") == expected
    assert calls[0]["repo_id"] == "Prior-Labs/tabpfn_3"
    assert calls[0]["filename"] == config.regressor_checkpoint
    assert calls[0]["local_dir"] == tmp_path


def test_tabpfn_malformed_graph_uses_canonical_invalid_result():
    metrics, artifact = validate_tabpfn({})

    assert metrics["is_valid"] == 0.0
    assert artifact["validation_failure_stage"] == "schema"
