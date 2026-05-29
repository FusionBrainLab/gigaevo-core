from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tabular_problem
import yaml


class _RidgeReg:
    def fit_predict(self, X_train, y_train, X_val, y_val, X_query):
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        np.random.seed(0)
        pipe = Pipeline([("s", StandardScaler()), ("r", Ridge(alpha=1.0))])
        pipe.fit(X_train, y_train)
        return pipe.predict(X_query)


class _RFClf:
    def fit_predict(self, X_train, y_train, X_val, y_val, X_query):
        from sklearn.ensemble import RandomForestClassifier

        np.random.seed(0)
        n_classes = int(max(y_train.max(), y_val.max())) + 1
        clf = RandomForestClassifier(n_estimators=80, random_state=0, n_jobs=4)
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_query)
        full = np.zeros((X_query.shape[0], n_classes))
        full[:, clf.classes_.astype(int)] = proba
        return full


def _expected_keys(task_type):
    base = {
        "fitness",
        "is_valid",
        "cv_score_std",
        "local_lipschitz_p95",
        "ood_delta_slope",
    }
    if task_type == "regression":
        return base | {"rmse", "mae"}
    if task_type == "binclass":
        return base | {"roc_auc", "balanced_accuracy", "f1", "log_loss"}
    return base | {"macro_f1", "balanced_accuracy", "log_loss"}


def test_regression_validate_keys_and_fitness(data_root):
    out = tabular_problem.build("california").validate(lambda: _RidgeReg())
    assert set(out) == _expected_keys("regression")
    assert out["is_valid"] == 1
    assert -1.0 <= out["fitness"] <= 1.0
    assert out["fitness"] > 0.0  # Ridge beats the mean on california


def test_binclass_validate_keys(data_root):
    out = tabular_problem.build("churn").validate(lambda: _RFClf())
    assert set(out) == _expected_keys("binclass")
    assert 0.0 <= out["fitness"] <= 1.0


def test_multiclass_validate_keys(data_root):
    out = tabular_problem.build("otto").validate(lambda: _RFClf())
    assert set(out) == _expected_keys("multiclass")
    assert 0.0 <= out["fitness"] <= 1.0


def test_holdout_path_single_split(data_root, monkeypatch):
    # force black-friday (107k) down the holdout branch and confirm cv_score_std == 0
    out = tabular_problem.build("black-friday").validate(lambda: _RidgeReg())
    assert out["cv_score_std"] == 0.0


def test_score_on_test_regression(data_root):
    res = tabular_problem.build("california").score_on_test(lambda: _RidgeReg())
    assert set(res) == {"test_rmse", "test_r2"}
    assert res["test_rmse"] > 0.0


def test_score_on_test_binclass(data_root):
    res = tabular_problem.build("churn").score_on_test(lambda: _RFClf())
    assert set(res) == {"test_accuracy", "test_auc"}


def test_bad_factory_marks_invalid_via_exception(data_root):
    # a factory returning a non-model raises inside validate -> caller (framework)
    # turns it into sentinels; here we assert validate raises so the stage catches it
    with pytest.raises(Exception):
        tabular_problem.build("california").validate(lambda: object())


def test_k_folds_env_validation(monkeypatch):
    monkeypatch.setenv("GIGAEVO_TABULAR_CV_FOLDS", "4")
    with pytest.raises(ValueError):
        tabular_problem._k_folds()
    monkeypatch.setenv("GIGAEVO_TABULAR_CV_FOLDS", "5")
    assert tabular_problem._k_folds() == 5


def _yaml_keys(fname):
    p = Path(__file__).resolve().parents[3] / "problems" / "tabular" / "_common" / fname
    return set(yaml.safe_load(p.read_text())["specs"].keys())


@pytest.mark.parametrize(
    "fname,task_type",
    [
        ("metrics_regression.yaml", "regression"),
        ("metrics_binclass.yaml", "binclass"),
        ("metrics_multiclass.yaml", "multiclass"),
    ],
)
def test_metrics_yaml_keys_match_validate(fname, task_type):
    assert _yaml_keys(fname) == _expected_keys(task_type)
