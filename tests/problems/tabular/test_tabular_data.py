from __future__ import annotations

import numpy as np
import pytest
import tabular_data


def test_california_regression_shapes(data_root):
    ds = tabular_data.load_dataset("california")
    assert ds.task_type == "regression"
    assert ds.n_classes is None
    assert ds.X_train.dtype == np.float64
    assert ds.X_train.shape[1] == 8
    assert ds.y_train.dtype == np.float64
    assert ds.X_train.shape[0] == ds.y_train.shape[0]
    assert all(c.kind == "numeric" for c in ds.columns)


def test_adult_string_categoricals_encoded(data_root):
    ds = tabular_data.load_dataset("adult")
    assert ds.task_type == "binclass"
    assert ds.n_classes == 2
    # assembled width = 6 num + 1 bin + 7 cat = 14
    assert ds.X_train.shape[1] == 14
    assert ds.X_train.dtype == np.float64
    assert np.all(np.isfinite(ds.X_train))
    cat_cols = [c for c in ds.columns if c.kind == "categorical"]
    assert len(cat_cols) == 7
    first_cat = cat_cols[0]
    assert first_cat.cardinality == len(first_cat.vocabulary) >= 2
    # codes in the assembled matrix are integer-valued floats in [0, cardinality)
    col = ds.X_train[:, first_cat.index]
    assert col.min() >= 0.0 and col.max() < first_cat.cardinality
    assert np.allclose(col, np.rint(col))
    assert ds.y_train.dtype == np.int64


def test_covtype_int_categoricals_and_order(data_root):
    ds = tabular_data.load_dataset("covtype2")
    assert ds.task_type == "multiclass"
    assert ds.n_classes == 7
    # column order is [num | bin | cat]
    kinds = [c.kind for c in ds.columns]
    assert kinds == ["numeric"] * 10 + ["binary"] * 4 + ["categorical"] * 1


def test_cache_returns_same_object(data_root):
    a = tabular_data.load_dataset("california")
    b = tabular_data.load_dataset("california")
    assert a is b


def test_missing_env_raises(monkeypatch):
    monkeypatch.delenv("GIGAEVO_TABULAR_DATA", raising=False)
    tabular_data._CACHE.clear()
    with pytest.raises(tabular_data.TabularDataError):
        tabular_data.load_dataset("nonexistent-xyz")


def test_describe_columns_lists_categorical_values(data_root):
    text = tabular_data.describe_columns("diamond")
    assert "categorical" in text
    # diamond cut categories include 'Ideal'
    assert "Ideal" in text


def test_describe_columns_named_path_keeps_vocab(data_root):
    # adult: numerics 0-5, binary 6, categoricals 7-13 (col 7 = workclass).
    names = {
        0: {"name": "age", "desc": "age in years"},
        7: {"name": "workclass", "desc": "employer / employment type"},
    }
    text = tabular_data.describe_columns("adult", names=names)
    # semantic name + desc surface for the numeric column
    assert "age in years" in text
    # a named categorical shows BOTH its semantic label/desc AND its decoded
    # vocabulary (a program needs the code->value mapping to encode it)
    assert "workclass" in text
    assert "employer / employment type" in text
    assert "Private" in text
