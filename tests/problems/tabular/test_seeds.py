from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SEEDS = (
    Path(__file__).resolve().parents[3] / "problems" / "tabular" / "_common" / "seeds"
)


def _load(task, prog):
    path = _SEEDS / task / f"{prog}.py"
    spec = importlib.util.spec_from_file_location(f"{task}_{prog}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.entrypoint()


def _toy_reg(n=60, d=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = X[:, 0] + 0.1 * rng.normal(size=n)
    return X, y


def _toy_clf(n=90, d=4, k=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = (X[:, 0] > 0).astype(int) + (X[:, 1] > 0).astype(int)  # 0,1,2
    return X, np.clip(y, 0, k - 1)


@pytest.mark.parametrize("prog", ["prog1", "prog2", "prog3", "prog4", "prog5"])
def test_regression_seed_returns_1d(prog):
    X, y = _toy_reg()
    Model = _load("regression", prog)
    out = np.asarray(Model().fit_predict(X[:40], y[:40], X[40:50], y[40:50], X[50:]))
    assert out.shape == (10,)
    assert np.all(np.isfinite(out))


@pytest.mark.parametrize("task,k", [("binclass", 2), ("multiclass", 3)])
@pytest.mark.parametrize("prog", ["prog1", "prog2", "prog3", "prog4", "prog5"])
def test_classification_seed_returns_proba(task, k, prog):
    X, y = _toy_clf(k=k)
    Model = _load(task, prog)
    out = np.asarray(Model().fit_predict(X[:60], y[:60], X[60:75], y[60:75], X[75:]))
    assert out.ndim == 2
    assert out.shape[0] == 15
    assert out.shape[1] >= k
    assert np.all(np.isfinite(out))
