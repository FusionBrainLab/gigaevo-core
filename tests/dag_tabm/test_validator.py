from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import KFold

from problems.tabular._common.tabular_metrics import regression_fold_metrics
from problems.tabular_dag_baselines import validation as shared_validation
from problems.tabular_dag_baselines.tabm import validate as validator


@dataclass(frozen=True)
class _Dataset:
    task_type = "regression"
    n_classes = None
    columns = ()

    x = np.linspace(-2.0, 2.0, 15)
    X_train = np.column_stack([x, x**2])
    y_train = x + 0.2 * x**2
    xv = np.linspace(-1.8, 1.8, 6)
    X_val = np.column_stack([xv, xv**2])
    y_val = xv + 0.2 * xv**2
    xt = np.linspace(10.0, 11.0, 5)
    X_test = np.column_stack([xt, xt**2])
    y_test = xt + 0.2 * xt**2


def _payload():
    return {
        "schema_version": 1,
        "dataset": "california",
        "raw_columns": ["x0", "x1"],
        "nodes": [],
    }


def test_validator_reuses_shared_folds_std_bd_and_test_protocol(monkeypatch):
    dataset = _Dataset()
    query_calls: list[np.ndarray] = []
    factory_calls = []

    class PredictFirstColumn:
        def fit_predict(self, X_train, y_train, X_val, y_val, X_query):
            query_calls.append(np.asarray(X_query).copy())
            return np.asarray(X_query)[:, 0]

    def fake_factory(graph, device, config):
        factory_calls.append((graph.dataset, device, config.seed))
        return PredictFirstColumn()

    monkeypatch.setenv("GIGAEVO_TABM_DEVICE", "cpu")
    monkeypatch.setenv("GIGAEVO_TABULAR_CV_FOLDS", "3")
    monkeypatch.setattr(
        shared_validation.tabular_data, "load_dataset", lambda name: dataset
    )
    monkeypatch.setattr(
        validator,
        "_builder",
        lambda config: lambda graph, device: fake_factory(graph, device, config),
    )
    monkeypatch.setattr(validator, "effective_amp_dtype", lambda *args, **kwargs: None)

    metrics, artifact = validator.validate(_payload())

    X_dev = np.concatenate([dataset.X_train, dataset.X_val])
    y_dev = np.concatenate([dataset.y_train, dataset.y_val])
    splits = list(KFold(n_splits=3, shuffle=True, random_state=0).split(X_dev))
    scores = [
        regression_fold_metrics(y_dev[query_idx], X_dev[query_idx, 0])["score"]
        for _, query_idx in splits
    ]
    expected_std = float(np.std(scores, ddof=1))
    assert metrics["fitness"] == float(np.mean(scores))
    assert metrics["cv_score_std"] == expected_std
    assert metrics["graph_node_count"] == 0.0
    assert metrics["graph_max_depth"] == 0.0
    assert metrics["generated_feature_count"] == 0.0
    assert np.isfinite(metrics["local_lipschitz_p95"])
    assert np.isfinite(metrics["ood_delta_slope"])
    assert artifact["_evaluation_measurements"]["fitness"] == {
        "sample_sd": expected_std,
        "n": 3,
        "method": "cross_validation",
    }
    assert artifact["tabm_amp_dtype"] is None
    assert len(query_calls) == 4  # three CV folds plus one shared BD probe
    assert not any(np.array_equal(query, dataset.X_test) for query in query_calls)
    assert factory_calls and all(call[1] == "cpu" for call in factory_calls)

    test_metrics = validator.score_on_test(_payload())

    assert np.isfinite(test_metrics["test_rmse"])
    assert np.isfinite(test_metrics["test_r2"])
    assert np.array_equal(query_calls[-1], dataset.X_test)
