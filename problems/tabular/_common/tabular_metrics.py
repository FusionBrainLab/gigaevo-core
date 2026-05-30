from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

R2_FLOOR = -1.0
LOG_LOSS_CAP = 100.0
_EPS = 1e-12


def instantiate(model_factory):
    if not callable(model_factory):
        raise ValueError(
            f"entrypoint() must return a class or no-arg factory; got "
            f"{type(model_factory).__name__}"
        )
    instance = model_factory()
    if not hasattr(instance, "fit_predict"):
        raise ValueError(
            f"Model must implement .fit_predict(X_train, y_train, X_val, y_val, "
            f"X_query); got {type(instance).__name__}"
        )
    return instance


def r2_clamped(y_true, y_pred, floor: float = R2_FLOOR) -> float:
    return max(float(r2_score(y_true, y_pred)), floor)


def _check_1d_finite(y_pred, n: int) -> np.ndarray:
    y = np.asarray(y_pred, dtype=float)
    if y.ndim != 1 or y.shape[0] != n:
        raise ValueError(f"regression prediction shape {y.shape} != ({n},)")
    if not np.all(np.isfinite(y)):
        raise ValueError("predictions contain NaN or inf")
    return y


def regression_fold_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y = _check_1d_finite(y_pred, y_true.shape[0])
    return {
        "score": r2_clamped(y_true, y),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y))),
        "mae": float(mean_absolute_error(y_true, y)),
    }


def to_labels(pred, n_classes: int) -> np.ndarray:
    p = np.asarray(pred, dtype=float)
    if p.ndim == 2:
        return np.argmax(p, axis=1)
    return np.rint(p).astype(int)


def to_proba(pred, n_classes: int) -> np.ndarray:
    p = np.asarray(pred, dtype=float)
    if not np.all(np.isfinite(p)):
        raise ValueError("classification predictions contain NaN or inf")
    if p.ndim == 2:
        if p.shape[1] != n_classes:
            full = np.zeros((p.shape[0], n_classes))
            w = min(p.shape[1], n_classes)
            full[:, :w] = p[:, :w]
            p = full
        row = p.sum(axis=1, keepdims=True)
        row[row <= 0] = 1.0
        return np.clip(p / row, _EPS, 1.0)
    labels = np.rint(p).astype(int)
    onehot = np.zeros((labels.shape[0], n_classes))
    valid = (labels >= 0) & (labels < n_classes)
    onehot[np.flatnonzero(valid), labels[valid]] = 1.0
    return np.clip(onehot, _EPS, 1.0)


def _safe_log_loss(y_true, proba, n_classes: int) -> float:
    try:
        ll = float(log_loss(y_true, proba, labels=list(range(n_classes))))
    except ValueError:
        ll = LOG_LOSS_CAP
    return float(min(ll, LOG_LOSS_CAP))


def _safe_auc_binary(y_true, proba) -> float:
    try:
        return float(roc_auc_score(y_true, proba[:, 1]))
    except ValueError:
        return 0.5


def classification_fold_metrics(
    y_true, pred, n_classes: int, task_type: str
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    labels = to_labels(pred, n_classes)
    if labels.shape[0] != y_true.shape[0]:
        raise ValueError(
            f"prediction rows {labels.shape[0]} != labels {y_true.shape[0]}"
        )
    proba = to_proba(pred, n_classes)
    acc = float(accuracy_score(y_true, labels))
    bacc = float(balanced_accuracy_score(y_true, labels))
    ll = _safe_log_loss(y_true, proba, n_classes)
    if task_type == "binclass":
        return {
            "score": acc,
            "roc_auc": _safe_auc_binary(y_true, proba),
            "balanced_accuracy": bacc,
            "f1": float(f1_score(y_true, labels, zero_division=0)),
            "log_loss": ll,
        }
    return {
        "score": acc,
        "macro_f1": float(f1_score(y_true, labels, average="macro", zero_division=0)),
        "balanced_accuracy": bacc,
        "log_loss": ll,
    }
