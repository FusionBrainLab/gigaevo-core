from __future__ import annotations

import numpy as np
import tabular_bd as bd


def _xy(n=200, d=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = X[:, 0] * 2.0 + rng.normal(scale=0.1, size=n)
    return X, y


class _LinReg:
    """fit_predict that returns a smooth 1D regression output."""

    def fit_predict(self, X_train, y_train, X_val, y_val, X_query):
        w, *_ = np.linalg.lstsq(X_train, y_train, rcond=None)
        return X_query @ w


class _ProbaClf:
    """fit_predict returning (n, k) probabilities via a softmax of a linear score."""

    def __init__(self, k):
        self.k = k

    def fit_predict(self, X_train, y_train, X_val, y_val, X_query):
        rng = np.random.default_rng(0)
        W = rng.normal(size=(X_query.shape[1], self.k))
        z = X_query @ W
        z -= z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)


def test_super_query_block_sizes():
    X, _ = _xy()
    rng = np.random.default_rng(bd.BD_SEED)
    Xs, blocks = bd.build_bd_super_query(X[:120], X[:80], rng)
    assert (
        Xs.shape[0]
        == blocks["n_val"] + blocks["n_shift"] + blocks["n_anchor"] * bd.BD_N_PERT
    )
    assert blocks["n_val"] == 80
    assert blocks["n_anchor"] == min(bd.BD_N_SUB, 80)
    assert blocks["eps_norms_z"].shape[0] == blocks["n_anchor"] * bd.BD_N_PERT


def test_output_diff_branches():
    a1 = np.array([1.0, 5.0])
    b1 = np.array([1.0, 1.0])
    # regression: |a-b|/sd_y
    assert np.allclose(
        bd.output_diff(a1, b1, "regression", sd_y=2.0, is_label=False), [0.0, 2.0]
    )
    # probabilities: L2 row norm
    pa = np.array([[1.0, 0.0], [0.0, 1.0]])
    pb = np.array([[1.0, 0.0], [1.0, 0.0]])
    assert np.allclose(
        bd.output_diff(pa, pb, "binclass", sd_y=1.0, is_label=False), [0.0, np.sqrt(2)]
    )
    # hard labels: flip indicator
    la = np.array([0, 1, 2])
    lb = np.array([0, 0, 2])
    assert np.allclose(
        bd.output_diff(la, lb, "multiclass", sd_y=1.0, is_label=True), [0.0, 1.0, 0.0]
    )


def test_compute_bd_axes_regression_in_bounds():
    X, y = _xy()
    out = bd.compute_bd_axes(
        lambda: _LinReg(),
        X[:120],
        y[:120],
        X[120:],
        y[120:],
        task_type="regression",
        sd_y=float(np.std(y) + bd.BD_EPS),
        bd_max=2048,
    )
    assert 0.0 <= out["local_lipschitz_p95"] <= 4.0
    assert 0.0 <= out["ood_delta_slope"] <= 2.0


def test_compute_bd_axes_proba_in_bounds():
    X, y = _xy()
    yc = (X[:, 0] > 0).astype(int)
    out = bd.compute_bd_axes(
        lambda: _ProbaClf(2),
        X[:120],
        yc[:120],
        X[120:],
        yc[120:],
        task_type="binclass",
        sd_y=1.0,
        bd_max=2048,
    )
    assert 0.0 <= out["local_lipschitz_p95"] <= 4.0
    assert 0.0 <= out["ood_delta_slope"] <= 2.0


def test_compute_bd_axes_subsamples_val():
    X, y = _xy(n=6000)
    out = bd.compute_bd_axes(
        lambda: _LinReg(),
        X[:5000],
        y[:5000],
        X[5000:],
        y[5000:],
        task_type="regression",
        sd_y=float(np.std(y) + bd.BD_EPS),
        bd_max=256,
    )
    assert set(out) == {"local_lipschitz_p95", "ood_delta_slope"}
