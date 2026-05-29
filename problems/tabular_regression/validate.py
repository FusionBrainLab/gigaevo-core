import os
from pathlib import Path
import sys

import numpy as np
from sklearn.model_selection import KFold

try:
    DATA_DIR = Path(__file__).parent / "rtdl_split"
except NameError:
    DATA_DIR = Path(sys.path[0]) / "rtdl_split"

_TRAINVAL_CACHE: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None

_ALLOWED_FOLDS = {2, 3, 5, 10}
_DEFAULT_FOLDS = 3
_CV_SEED = 0

_BD_SEED = 42
_BD_DELTA_FRAC = 0.005
_BD_N_PERT = 10
_BD_N_SUB = 32
_BD_SHIFT_SD = 1.5
_BD_EPS = 1e-12
_BD_LL_MAX = 4.0  # metrics.yaml upper bounds; sentinel corner if the BD probe fails
_BD_OOD_MAX = 2.0


def _load_train_val() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    global _TRAINVAL_CACHE
    if _TRAINVAL_CACHE is None:
        X_train = np.load(DATA_DIR / "N_train.npy").astype(np.float64)
        y_train = np.load(DATA_DIR / "y_train.npy").astype(np.float64)
        X_val = np.load(DATA_DIR / "N_val.npy").astype(np.float64)
        y_val = np.load(DATA_DIR / "y_val.npy").astype(np.float64)
        _TRAINVAL_CACHE = (X_train, y_train, X_val, y_val)
    return _TRAINVAL_CACHE


def _load_test_X() -> np.ndarray:
    return np.load(DATA_DIR / "N_test.npy").astype(np.float64)


def _load_test_y() -> np.ndarray:
    return np.load(DATA_DIR / "y_test.npy").astype(np.float64)


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


def _build_bd_super_query(
    X_train: np.ndarray, X_val: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, dict]:
    n_val = X_val.shape[0]
    feat_std = np.std(X_train, axis=0)
    feat_std_safe = feat_std + _BD_EPS

    X_shift = X_val + _BD_SHIFT_SD * feat_std[np.newaxis, :]

    n_anchor = min(_BD_N_SUB, n_val)
    anchor_idx = rng.choice(n_val, size=n_anchor, replace=False)
    anchors = X_val[anchor_idx]

    pert_rows = []
    eps_norms_z = []
    for _ in range(_BD_N_PERT):
        eps_dir = rng.normal(size=anchors.shape)
        eps_dir /= np.linalg.norm(eps_dir, axis=1, keepdims=True) + _BD_EPS
        eps = _BD_DELTA_FRAC * feat_std_safe[np.newaxis, :] * eps_dir
        pert_rows.append(anchors + eps)
        eps_z = eps / feat_std_safe[np.newaxis, :]
        eps_norms_z.append(np.linalg.norm(eps_z, axis=1))

    X_pert = np.vstack(pert_rows)
    eps_norms_z = np.concatenate(eps_norms_z)

    X_super = np.vstack([X_val, X_shift, X_pert])
    blocks = {
        "n_val": n_val,
        "n_shift": n_val,
        "n_anchor": n_anchor,
        "anchor_idx": anchor_idx,
        "eps_norms_z": eps_norms_z,
    }
    return X_super, blocks


def _compute_bd_axes(
    model_factory,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> dict[str, float]:
    """Single-fit_predict BD probe.

    Returns {local_lipschitz_p95, ood_delta_slope} computed from a single
    super-query [X_val, X_shift, X_pert] so both BDs are derived from the
    same model state (no RNG drift between BDs).
    """
    rng = np.random.default_rng(_BD_SEED)
    X_super, blocks = _build_bd_super_query(X_train, X_val, rng)

    instance = _instantiate(model_factory)
    y_super = np.asarray(
        instance.fit_predict(X_train, y_train, X_val, y_val, X_super),
        dtype=float,
    )
    if y_super.shape != (len(X_super),) or not np.all(np.isfinite(y_super)):
        raise ValueError(
            f"BD probe predictions non-finite or wrong shape: {y_super.shape}"
        )

    n_val = blocks["n_val"]
    n_shift = blocks["n_shift"]
    n_anchor = blocks["n_anchor"]
    y_pred = y_super[:n_val]
    y_shift = y_super[n_val : n_val + n_shift]
    y_pert = y_super[n_val + n_shift :]

    sd_y = float(np.std(y_train) + _BD_EPS)

    ood_delta_slope = float(np.log1p(np.median(np.abs(y_shift - y_pred)) / sd_y))

    y_anchor = y_pred[blocks["anchor_idx"]]
    diff_blocks = [
        np.abs(y_pert[j * n_anchor : (j + 1) * n_anchor] - y_anchor)
        for j in range(_BD_N_PERT)
    ]
    diffs = np.concatenate(diff_blocks)
    slopes = diffs / (sd_y * blocks["eps_norms_z"] + _BD_EPS)
    local_lipschitz_p95 = float(np.log1p(np.percentile(slopes, 95)))

    return {
        "local_lipschitz_p95": local_lipschitz_p95,
        "ood_delta_slope": ood_delta_slope,
    }


def validate(model_factory) -> dict[str, float]:
    """Search-mode validator. Fitness = -mean(RMSE) across k-fold CV on X_train.

    Canonical X_val fills the val slot in every fold (constant ES signal,
    never scored). Scoring rows come from X_train only, so no label passed
    to the model corresponds to any row in X_query.

    Test arrays are not loaded here. End-of-evolution test scoring is done
    by `score_on_test(model_factory)`.

    After CV, runs a single-call BD probe to populate `local_lipschitz_p95`
    and `ood_delta_slope` for the MAP-Elites 2D archive.
    """
    X_train, y_train, X_val, y_val = _load_train_val()

    k = _k_folds()
    kf = KFold(n_splits=k, shuffle=True, random_state=_CV_SEED)

    fold_rmses: list[float] = []
    for fit_idx, query_idx in kf.split(X_train):
        instance = _instantiate(model_factory)
        y_pred = instance.fit_predict(
            X_train[fit_idx],
            y_train[fit_idx],
            X_val.copy(),
            y_val.copy(),
            X_train[query_idx],
        )
        fold_rmses.append(_score(y_pred, y_train[query_idx]))

    cv_rmse_mean = float(np.mean(fold_rmses))
    cv_rmse_std = float(np.std(fold_rmses, ddof=1)) if k > 1 else 0.0

    # BD axes are archive coordinates, not fitness — a probe failure must not
    # discard a valid CV result, so fall back to the rough/OOD-sensitive corner.
    try:
        bds = _compute_bd_axes(model_factory, X_train, y_train, X_val, y_val)
    except Exception:
        bds = {"local_lipschitz_p95": _BD_LL_MAX, "ood_delta_slope": _BD_OOD_MAX}

    return {
        "fitness": -cv_rmse_mean,
        "is_valid": 1,
        "cv_rmse_mean": cv_rmse_mean,
        "cv_rmse_std": cv_rmse_std,
        **bds,
    }


def score_on_test(model_factory) -> dict[str, float]:
    """Report-mode scorer. Returns held-out TEST RMSE for end-of-evolution reporting.

    y_test is loaded only AFTER the model has returned predictions, so it
    is not present in process memory during fitting.
    """
    X_train, y_train, X_val, y_val = _load_train_val()
    X_test = _load_test_X()

    instance = _instantiate(model_factory)
    y_pred_test = instance.fit_predict(X_train, y_train, X_val, y_val, X_test)

    y_test = _load_test_y()
    test_rmse = _score(y_pred_test, y_test)

    return {"test_rmse": test_rmse}
