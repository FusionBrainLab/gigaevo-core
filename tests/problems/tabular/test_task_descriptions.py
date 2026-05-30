from __future__ import annotations

from pathlib import Path

import pytest

_TABULAR = Path(__file__).resolve().parents[3] / "problems" / "tabular"
_ALL = [
    "california",
    "house",
    "diamond",
    "black-friday",
    "microsoft",
    "adult",
    "churn",
    "higgs-small",
    "otto",
    "covtype2",
]


@pytest.mark.parametrize("name", _ALL)
def test_task_description_exists_and_has_contract(name):
    p = _TABULAR / name / "task_description.txt"
    assert p.is_file(), f"missing {p}"
    text = p.read_text()
    assert "fit_predict" in text
    assert "COLUMNS" in text


def test_categorical_dataset_lists_values():
    text = (_TABULAR / "adult" / "task_description.txt").read_text()
    assert "categorical" in text
    # adult workclass vocabulary includes 'Private'
    assert "Private" in text
