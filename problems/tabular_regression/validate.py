import os
from pathlib import Path
import sys

import numpy as np
from sklearn.model_selection import KFold

try:
    DATA_DIR = Path(__file__).parent / "rtdl_split"
except NameError:
    DATA_DIR = Path(sys.path[0]) / "rtdl_split"

_DATA_CACHE: (
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None
) = None

_ALLOWED_FOLDS = {2, 3, 5, 10}
_DEFAULT_FOLDS = 5
_CV_SEED = 0


def _load_dataset() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    global _DATA_CACHE
    if _DATA_CACHE is None:
        X_train = np.load(DATA_DIR / "N_train.npy").astype(np.float64)
        y_train = np.load(DATA_DIR / "y_train.npy").astype(np.float64)
        X_val = np.load(DATA_DIR / "N_val.npy").astype(np.float64)
        y_val = np.load(DATA_DIR / "y_val.npy").astype(np.float64)
        X_test = np.load(DATA_DIR / "N_test.npy").astype(np.float64)
        y_test = np.load(DATA_DIR / "y_test.npy").astype(np.float64)
        _DATA_CACHE = (X_train, y_train, X_val, y_val, X_test, y_test)
    return _DATA_CACHE


def _instantiate(model_factory) -> object:
    if not callable(model_factory):
        raise ValueError(
            f"entrypoint() must return a class (or no-arg callable factory); "
            f"got {type(model_factory).__name__}"
        )
    instance = model_factory()
    if not hasattr(instance, "fit_predict"):
        raise ValueError(
            f"Model instance must implement "
            f".fit_predict(X_train, y_train, X_val, y_val, X_query); "
            f"got {type(instance).__name__} with attrs {dir(instance)}"
        )
    return instance


def _score(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    y_pred = np.asarray(y_pred, dtype=float)
    if y_pred.ndim != 1 or y_pred.shape[0] != y_true.shape[0]:
        raise ValueError(
            f"shape mismatch — y_pred {y_pred.shape} != expected ({y_true.shape[0]},)"
        )
    if not np.all(np.isfinite(y_pred)):
        raise ValueError("predictions contain NaN or inf")
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def _k_folds() -> int:
    raw = os.environ.get("GIGAEVO_TR_CV_FOLDS")
    if raw is None:
        return _DEFAULT_FOLDS
    try:
        k = int(raw)
    except ValueError as e:
        raise ValueError(
            f"GIGAEVO_TR_CV_FOLDS must be one of {sorted(_ALLOWED_FOLDS)}; got {raw!r}"
        ) from e
    if k not in _ALLOWED_FOLDS:
        raise ValueError(
            f"GIGAEVO_TR_CV_FOLDS must be one of {sorted(_ALLOWED_FOLDS)}; got {k}"
        )
    return k


def validate(model_factory) -> dict[str, float]:
    """Search-mode validator. Fitness = -mean(RMSE) across k-fold CV on train ∪ val.

    Test set is NEVER used here. End-of-evolution test scoring is done by
    `score_on_test(model_factory)`, which is invoked separately by the
    reporting tool, not by the evolutionary search loop.

    Each fold re-instantiates the model so internal state (e.g. a KMeans fit
    on fold 0) does not leak between folds. Fold seed is fixed so that
    re-eval of the same program produces the same fitness — keeps the
    fitness cache valid.
    """
    X_train, y_train, X_val, y_val, _X_test, _y_test = _load_dataset()
    X_pool = np.concatenate([X_train, X_val], axis=0)
    y_pool = np.concatenate([y_train, y_val], axis=0)

    k = _k_folds()
    kf = KFold(n_splits=k, shuffle=True, random_state=_CV_SEED)

    fold_rmses: list[float] = []
    for train_idx, val_idx in kf.split(X_pool):
        X_tr, y_tr = X_pool[train_idx], y_pool[train_idx]
        X_vl, y_vl = X_pool[val_idx], y_pool[val_idx]
        instance = _instantiate(model_factory)
        y_pred = instance.fit_predict(X_tr, y_tr, X_vl, y_vl, X_vl)
        fold_rmses.append(_score(y_pred, y_vl))

    cv_rmse_mean = float(np.mean(fold_rmses))
    cv_rmse_std = float(np.std(fold_rmses))

    return {
        "fitness": -cv_rmse_mean,
        "is_valid": 1,
        "cv_rmse_mean": cv_rmse_mean,
        "cv_rmse_std": cv_rmse_std,
    }


def score_on_test(model_factory) -> dict[str, float]:
    """Report-mode scorer. Returns held-out TEST RMSE for end-of-evolution reporting.

    MUST NOT be called from the evolutionary search loop. The framework's
    `validate()` entry never reads y_test; this function exists solely for
    post-evolution reporting on selected elites.
    """
    X_train, y_train, X_val, y_val, X_test, y_test = _load_dataset()

    instance = _instantiate(model_factory)
    y_pred_test = instance.fit_predict(X_train, y_train, X_val, y_val, X_test)
    test_rmse = _score(y_pred_test, y_test)

    return {"test_rmse": test_rmse}
