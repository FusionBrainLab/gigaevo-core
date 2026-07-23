import argparse

import pytest

from problems.tabular_dag_baselines.compare import EVALUATOR_MODULES, SEED_ENV
from problems.tabular_dag_baselines.compare_matrix import (
    _graph_argument,
    _summarize,
)


def test_every_non_catboost_evaluator_has_a_seed_override():
    assert set(SEED_ENV) == set(EVALUATOR_MODULES) - {"catboost"}
    assert SEED_ENV["realmlp"] == "GIGAEVO_REALMLP_SEED"
    assert SEED_ENV["tabpfn"] == "GIGAEVO_TABPFN_SEED"


def test_graph_argument_is_named_and_resolved(tmp_path):
    name, path = _graph_argument(f"winner={tmp_path / 'graph.json'}")

    assert name == "winner"
    assert path == (tmp_path / "graph.json").resolve()

    with pytest.raises(argparse.ArgumentTypeError, match="NAME=PATH"):
        _graph_argument("graph.json")


def test_matrix_summary_reports_sample_std():
    rows = [
        {"test_metrics": {"test_rmse": 1.0, "test_r2": 0.5}},
        {"test_metrics": {"test_rmse": 3.0, "test_r2": 0.7}},
    ]

    summary = _summarize(rows, "test_metrics")

    assert summary["test_rmse"]["mean"] == 2.0
    assert summary["test_rmse"]["sample_std"] == pytest.approx(2**0.5)
    assert summary["test_rmse"]["n_seeds"] == 2
