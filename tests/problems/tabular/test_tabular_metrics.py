from __future__ import annotations

import numpy as np
import pytest
import tabular_metrics as tm


def test_r2_clamped_floor():
    y = np.array([0.0, 1.0, 2.0, 3.0])
    awful = np.array([100.0, -100.0, 100.0, -100.0])
    assert tm.r2_clamped(y, awful) == -1.0
    assert tm.r2_clamped(y, y) == pytest.approx(1.0)


def test_regression_fold_metrics_keys_and_values():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    pred = np.array([1.0, 2.0, 3.0, 4.0])
    m = tm.regression_fold_metrics(y, pred)
    assert set(m) == {"score", "rmse", "mae"}
    assert m["rmse"] == pytest.approx(0.0)
    assert m["score"] == pytest.approx(1.0)


def test_regression_rejects_nan():
    y = np.array([1.0, 2.0])
    with pytest.raises(ValueError):
        tm.regression_fold_metrics(y, np.array([1.0, np.nan]))


def test_to_proba_from_2d_normalises():
    p = np.array([[2.0, 2.0], [1.0, 3.0]])
    out = tm.to_proba(p, 2)
    assert out.shape == (2, 2)
    assert np.allclose(out.sum(axis=1), 1.0)


def test_to_proba_from_labels_onehot():
    out = tm.to_proba(np.array([0, 1, 2]), 3)
    assert out.shape == (3, 3)
    assert np.argmax(out, axis=1).tolist() == [0, 1, 2]


def test_to_labels_argmax_and_passthrough():
    assert tm.to_labels(np.array([[0.1, 0.9], [0.8, 0.2]]), 2).tolist() == [1, 0]
    assert tm.to_labels(np.array([1, 0, 1]), 2).tolist() == [1, 0, 1]


def test_classification_binclass_keys():
    y = np.array([0, 1, 0, 1])
    proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.1, 0.9]])
    m = tm.classification_fold_metrics(y, proba, 2, "binclass")
    assert set(m) == {"score", "roc_auc", "balanced_accuracy", "f1", "log_loss"}
    assert m["score"] == pytest.approx(1.0)
    assert m["roc_auc"] == pytest.approx(1.0)


def test_classification_multiclass_keys():
    y = np.array([0, 1, 2])
    proba = np.eye(3)
    m = tm.classification_fold_metrics(y, proba, 3, "multiclass")
    assert set(m) == {"score", "macro_f1", "balanced_accuracy", "log_loss"}
    assert m["score"] == pytest.approx(1.0)


def test_log_loss_capped_and_finite():
    y = np.array([0, 1])
    proba = np.array([[0.0, 1.0], [1.0, 0.0]])  # confidently wrong
    m = tm.classification_fold_metrics(y, proba, 2, "binclass")
    assert np.isfinite(m["log_loss"])
    assert m["log_loss"] <= tm.LOG_LOSS_CAP


def test_instantiate_checks_contract():
    class Good:
        def fit_predict(self, *a):
            return None

    assert isinstance(tm.instantiate(lambda: Good()), Good)
    with pytest.raises(ValueError):
        tm.instantiate(lambda: object())
