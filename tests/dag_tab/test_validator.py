from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from problems.dag_tab import validate as validator


ROOT = Path(__file__).parents[2]
SEED = ROOT / "problems/dag_tab/initial_programs/baseline.json"


class _Dataset:
    task_type = "regression"
    X_train = np.ones((4, 8))


class _Problem:
    def validate(self, factory):
        model = factory()
        assert model.graph.dataset == "california"
        return {
            "fitness": 0.5,
            "is_valid": 1.0,
            "cv_score_std": 0.1,
            "local_lipschitz_p95": 0.2,
            "ood_delta_slope": 0.3,
        }

    def score_on_test(self, factory):
        assert factory().graph.dataset == "california"
        return {"test_r2": 0.4}


def test_validate_reuses_tabular_problem(monkeypatch):
    monkeypatch.setattr(validator.tabular_data, "load_dataset", lambda name: _Dataset())
    monkeypatch.setattr(validator, "build", lambda name: _Problem())
    payload = json.loads(SEED.read_text())

    metrics, artifact = validator.validate(payload)

    assert metrics["fitness"] == 0.5
    assert metrics["graph_node_count"] == 1.0
    assert metrics["graph_max_depth"] == 1.0
    assert metrics["generated_feature_count"] == 1.0
    assert artifact["output_columns"] == ["fe_income_per_age"]


def test_validate_returns_invalid_metrics_for_bad_payload():
    metrics, artifact = validator.validate({"not": "a graph"})

    assert metrics["is_valid"] == 0.0
    assert metrics["fitness"] == -1.0
    assert "error" in artifact


def test_validator_source_loads_without_dunder_file(monkeypatch):
    problem_dir = ROOT / "problems/dag_tab"
    source = (problem_dir / "validate.py").read_text()
    monkeypatch.setattr("sys.path", [str(problem_dir), *list(__import__("sys").path)[1:]])
    namespace = {"__name__": "dag_tab_exec_validator"}

    exec(compile(source, "user_code.py", "exec"), namespace)

    assert callable(namespace["validate"])
    assert callable(namespace["score_on_test"])
