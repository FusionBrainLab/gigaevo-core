from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from problems.dag_tab.execution import GraphTriplet
from problems.dag_tab.graph import FeatureGraph
from problems.tabular._common.tabular_data import ColumnSpec
from problems.tabular_dag_baselines.tabm.tabm_backend import (
    TabMConfig,
    TabMFeatureGraphModel,
    _training_batches,
)


class _Dataset:
    task_type = "regression"
    n_classes = None
    columns = (
        ColumnSpec(0, "numeric", None, None),
        ColumnSpec(1, "categorical", 2, ("a", "b")),
    )


def _model(monkeypatch, *, config=None):
    monkeypatch.setattr(
        "problems.dag_tab.validate.tabular_data.load_dataset", lambda name: _Dataset()
    )
    graph = FeatureGraph(dataset="california", raw_columns=["x0", "x1"], nodes=[])
    return TabMFeatureGraphModel(graph, device="cpu", config=config)


def test_default_recipe_matches_official_tabm_example(monkeypatch):
    for name in tuple(os.environ):
        if name.startswith("GIGAEVO_TABM_"):
            monkeypatch.delenv(name)

    config = TabMConfig.from_env()

    assert config.arch_type == "tabm"
    assert config.k == 32
    assert config.n_blocks == 2
    assert config.d_block == 512
    assert config.dropout == 0.1
    assert config.learning_rate == 0.002
    assert config.weight_decay == 0.0003
    assert config.n_bins == 48
    assert config.d_embedding == 16
    assert config.batch_size == 256
    assert config.gradient_clipping_norm == 1.0
    assert config.share_training_batches is True


def test_feature_preprocessing_is_fit_local_and_has_unknown_category(monkeypatch):
    model = _model(monkeypatch)
    prepared = model._prepare_features(
        GraphTriplet(
            pd.DataFrame({"x0": [0.0, 1.0, 2.0], "x1": [0, 1, 0]}),
            pd.DataFrame({"x0": [3.0], "x1": [2]}),
            pd.DataFrame({"x0": [4.0], "x1": [None]}),
        )
    )

    assert prepared.x_num["fit"].shape == (3, 1)
    assert prepared.cat_cardinalities == [3]
    np.testing.assert_array_equal(prepared.x_cat["fit"].ravel(), [0, 1, 0])
    np.testing.assert_array_equal(prepared.x_cat["validation"].ravel(), [2])
    np.testing.assert_array_equal(prepared.x_cat["query"].ravel(), [2])


def test_small_cpu_fit_predict_smoke(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("rtdl_num_embeddings")
    pytest.importorskip("tabm")
    config = TabMConfig(
        k=2,
        n_blocks=1,
        d_block=16,
        dropout=0.0,
        learning_rate=1e-3,
        n_bins=4,
        d_embedding=4,
        batch_size=16,
        patience=0,
        max_epochs=2,
        amp=False,
        share_training_batches=True,
        refit=True,
    )
    model = _model(monkeypatch, config=config)
    rng = np.random.default_rng(0)
    values = np.column_stack([rng.normal(size=96), rng.integers(0, 2, size=96)])
    target = values[:, 0] + 0.25 * values[:, 1]

    prediction = model.fit_predict(
        values[:64], target[:64], values[64:80], target[64:80], values[80:]
    )

    assert prediction.shape == (16,)
    assert np.isfinite(prediction).all()
    assert model.last_fit_summary["best_epochs"] >= 1


def test_training_batch_order_uses_dedicated_rng():
    torch = pytest.importorskip("torch")
    config = TabMConfig(k=3, batch_size=4, share_training_batches=False)
    device = torch.device("cpu")

    def draw_after_global_rng_noise(noise_seed: int):
        torch.manual_seed(noise_seed)
        torch.rand(1000)
        generator = torch.Generator(device=device).manual_seed(config.seed)
        return torch.cat(
            list(_training_batches(torch, 11, config, device, generator)), dim=0
        )

    first = draw_after_global_rng_noise(1)
    second = draw_after_global_rng_noise(2)

    torch.testing.assert_close(first, second)
    for head in range(config.k):
        torch.testing.assert_close(first[:, head].sort().values, torch.arange(11))
