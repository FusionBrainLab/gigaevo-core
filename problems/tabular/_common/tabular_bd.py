from __future__ import annotations

import numpy as np
from tabular_metrics import instantiate

BD_SEED = 42
BD_DELTA_FRAC = 0.005
BD_N_PERT = 10
BD_N_SUB = 32
BD_SHIFT_SD = 1.5
BD_EPS = 1e-12
BD_LL_MAX = 4.0
BD_OOD_MAX = 2.0


def _subsample(X: np.ndarray, max_rows: int, rng: np.random.Generator) -> np.ndarray:
    if X.shape[0] <= max_rows:
        return X
    idx = rng.choice(X.shape[0], size=max_rows, replace=False)
    return X[idx]


def build_bd_super_query(
    X_train: np.ndarray, X_val: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, dict]:
    n_val = X_val.shape[0]
    feat_std = np.std(X_train, axis=0)
    feat_std_safe = feat_std + BD_EPS

    X_shift = X_val + BD_SHIFT_SD * feat_std[np.newaxis, :]

    n_anchor = min(BD_N_SUB, n_val)
    anchor_idx = rng.choice(n_val, size=n_anchor, replace=False)
    anchors = X_val[anchor_idx]

    pert_rows = []
    eps_norms_z = []
    for _ in range(BD_N_PERT):
        eps_dir = rng.normal(size=anchors.shape)
        eps_dir /= np.linalg.norm(eps_dir, axis=1, keepdims=True) + BD_EPS
        eps = BD_DELTA_FRAC * feat_std_safe[np.newaxis, :] * eps_dir
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


def output_diff(a, b, task_type: str, sd_y: float, is_label: bool) -> np.ndarray:
    a = np.asarray(a)
    b = np.asarray(b)
    if task_type == "regression":
        return np.abs(a.astype(float) - b.astype(float)) / sd_y
    if is_label:
        return (a != b).astype(float)
    return np.linalg.norm(a.astype(float) - b.astype(float), axis=1)


def compute_bd_axes(
    model_factory,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    task_type: str,
    sd_y: float,
    bd_max: int,
) -> dict[str, float]:
    rng = np.random.default_rng(BD_SEED)
    X_val_sub = _subsample(X_val, bd_max, rng)
    X_super, blocks = build_bd_super_query(X_train, X_val_sub, rng)

    instance = instantiate(model_factory)
    y_super = np.asarray(
        instance.fit_predict(X_train, y_train, X_val, y_val, X_super), dtype=float
    )

    n_val = blocks["n_val"]
    n_shift = blocks["n_shift"]
    n_anchor = blocks["n_anchor"]
    expected = n_val + n_shift + n_anchor * BD_N_PERT
    if y_super.shape[0] != expected or not np.all(np.isfinite(y_super)):
        raise ValueError(
            f"BD probe predictions non-finite or wrong rows: {y_super.shape}"
        )

    is_label = task_type != "regression" and y_super.ndim == 1
    y_pred = y_super[:n_val]
    y_shift = y_super[n_val : n_val + n_shift]
    y_pert = y_super[n_val + n_shift :]

    d_shift = output_diff(y_shift, y_pred, task_type, sd_y, is_label)
    agg_shift = float(np.mean(d_shift)) if is_label else float(np.median(d_shift))
    ood_delta_slope = float(np.log1p(agg_shift))

    y_anchor = y_pred[blocks["anchor_idx"]]
    diff_blocks = [
        output_diff(
            y_pert[j * n_anchor : (j + 1) * n_anchor],
            y_anchor,
            task_type,
            sd_y,
            is_label,
        )
        for j in range(BD_N_PERT)
    ]
    diffs = np.concatenate(diff_blocks)
    slopes = diffs / (blocks["eps_norms_z"] + BD_EPS)
    local_lipschitz_p95 = float(np.log1p(np.percentile(slopes, 95)))

    return {
        "local_lipschitz_p95": local_lipschitz_p95,
        "ood_delta_slope": ood_delta_slope,
    }
