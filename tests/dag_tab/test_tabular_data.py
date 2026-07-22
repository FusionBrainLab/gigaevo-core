from __future__ import annotations

import numpy as np

from problems.tabular._common import tabular_data


def test_categorical_vocabulary_is_fit_on_train_only():
    train = np.array([["a"], ["b"]], dtype=object)
    validation = np.array([["c"], ["a"]], dtype=object)
    test = np.array([["d"]], dtype=object)

    encoded_train, encoded_validation, encoded_test, vocab = (
        tabular_data._encode_categoricals(train, validation, test)
    )

    assert vocab == [("a", "b")]
    np.testing.assert_array_equal(encoded_train[:, 0], [0.0, 1.0])
    np.testing.assert_array_equal(encoded_validation[:, 0], [-1.0, 0.0])
    np.testing.assert_array_equal(encoded_test[:, 0], [-1.0])


def test_categorical_encoding_separates_missing_from_unknown():
    train = np.array([[None], ["a"]], dtype=object)
    validation = np.array([[None], ["unseen"]], dtype=object)
    test = np.array([["a"]], dtype=object)

    _, encoded_validation, _, vocab = tabular_data._encode_categoricals(
        train, validation, test
    )

    assert "__MISSING__" in vocab[0]
    missing_code = float(vocab[0].index("__MISSING__"))
    np.testing.assert_array_equal(encoded_validation[:, 0], [missing_code, -1.0])


def test_integer_categories_keep_their_numeric_order():
    train = np.array([[2], [10], [1]], dtype=np.int64)
    validation = np.array([[10]], dtype=np.int64)
    test = np.array([[1]], dtype=np.int64)

    encoded_train, _, _, vocab = tabular_data._encode_categoricals(
        train, validation, test
    )

    assert vocab == [("1", "2", "10")]
    np.testing.assert_array_equal(encoded_train[:, 0], [1.0, 2.0, 0.0])
