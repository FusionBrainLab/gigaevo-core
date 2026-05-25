"""1-fold validator used by OptunaOptimizationStage.

Single train→val split (no CV) so each Optuna trial costs one fit/predict.
The full k-fold CV lives in validate.py and remains the search-loop validator.
Test set is NEVER read here.
"""

from pathlib import Path
import sys

import numpy as np

try:
    DATA_DIR = Path(__file__).parent / "rtdl_split"
except NameError:
    DATA_DIR = Path(sys.path[0]) / "rtdl_split"

_DATA_CACHE: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None


def _load_split() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    global _DATA_CACHE
    if _DATA_CACHE is None:
        X_train = np.load(DATA_DIR / "N_train.npy").astype(np.float64)
        y_train = np.load(DATA_DIR / "y_train.npy").astype(np.float64)
        X_val = np.load(DATA_DIR / "N_val.npy").astype(np.float64)
        y_val = np.load(DATA_DIR / "y_val.npy").astype(np.float64)
        _DATA_CACHE = (X_train, y_train, X_val, y_val)
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


def validate(model_factory) -> dict[str, float]:
    """Single-fold validator. Fitness = -RMSE on the predetermined val set."""
    X_train, y_train, X_val, y_val = _load_split()
    instance = _instantiate(model_factory)
    y_pred = instance.fit_predict(X_train, y_train, X_val, y_val, X_val)
    val_rmse = _score(y_pred, y_val)
    return {
        "fitness": -val_rmse,
        "is_valid": 1,
        "val_rmse": val_rmse,
    }
