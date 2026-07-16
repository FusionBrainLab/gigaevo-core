from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from problems.dag_tab import validate as validator

ROOT = Path(__file__).parents[2]
SEED = ROOT / "problems/dag_tab/initial_programs/baseline.json"


class _Dataset:
    task_type = "regression"
    n_classes = None
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


def test_validate_rejects_split_dependent_graph(monkeypatch):
    monkeypatch.setattr(validator.tabular_data, "load_dataset", lambda name: _Dataset())
    payload = json.loads(SEED.read_text())
    payload["nodes"][0]["code"] = (
        "df['fe_income_per_age'] = np.arange(len(df)) * 1.0\nreturn df"
    )

    metrics, artifact = validator.validate(payload)

    assert metrics["is_valid"] == 0.0
    assert "split-dependent behavior" in artifact["error"]


def test_regression_uses_catboost_early_stopping_then_refits(monkeypatch):
    payload = json.loads(SEED.read_text())
    graph = validator.FeatureGraph.model_validate(payload)
    instances = []

    class FakeRegressor:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.fit_calls = []
            instances.append(self)

        def fit(self, X, y, eval_set=None):
            self.fit_calls.append((np.asarray(X), np.asarray(y), eval_set))
            return self

        def get_best_iteration(self):
            return 6

        def predict(self, X):
            return np.full(len(X), 1.25)

    monkeypatch.setattr(validator, "CatBoostRegressor", FakeRegressor)
    monkeypatch.setattr(validator.tabular_data, "load_dataset", lambda name: _Dataset())
    model = validator.FeatureGraphModel(graph)
    X_train = np.arange(32, dtype=float).reshape(4, 8)
    X_val = np.arange(16, dtype=float).reshape(2, 8)
    X_query = np.arange(24, dtype=float).reshape(3, 8)

    predictions = model.fit_predict(
        X_train,
        np.arange(4, dtype=float),
        X_val,
        np.arange(2, dtype=float),
        X_query,
    )

    assert len(instances) == 2
    assert instances[0].kwargs["iterations"] == 2000
    assert instances[0].kwargs["early_stopping_rounds"] == 50
    assert instances[0].fit_calls[0][2] is not None
    assert instances[1].kwargs["iterations"] == 7
    assert instances[1].kwargs["allow_writing_files"] is False
    assert instances[1].fit_calls[0][0].shape == (6, 9)
    np.testing.assert_array_equal(predictions, np.full(3, 1.25))


def test_classifier_restores_full_probability_matrix(monkeypatch):
    payload = json.loads(SEED.read_text())
    graph = validator.FeatureGraph.model_validate(payload)

    class ClassificationDataset(_Dataset):
        task_type = "classification"
        n_classes = 4

    class FakeClassifier:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.classes_ = np.array([0, 2])

        def fit(self, X, y, eval_set=None):
            return self

        def get_best_iteration(self):
            return 2

        def predict_proba(self, X):
            return np.tile([0.25, 0.75], (len(X), 1))

    monkeypatch.setattr(validator, "CatBoostClassifier", FakeClassifier)
    monkeypatch.setattr(
        validator.tabular_data,
        "load_dataset",
        lambda name: ClassificationDataset(),
    )
    model = validator.FeatureGraphModel(graph)
    values = np.arange(48, dtype=float).reshape(6, 8)

    probabilities = model.fit_predict(
        values[:3],
        np.array([0, 2, 2]),
        values[3:5],
        np.array([0, 2]),
        values[5:],
    )

    assert probabilities.shape == (1, 4)
    np.testing.assert_array_equal(probabilities, [[0.25, 0.0, 0.75, 0.0]])


def test_tabular_problem_uses_stratified_classification_folds(monkeypatch):
    class ClassificationDataset(_Dataset):
        task_type = "multiclass"
        n_classes = 3
        X_train = np.ones((12, 8))
        y_train = np.array([0, 1, 2] * 4)

    monkeypatch.setenv("GIGAEVO_TABULAR_CV_FOLDS", "2")
    problem = validator.build("unused")
    splits = problem._splits(ClassificationDataset())

    for fit_idx, query_idx in splits:
        assert set(ClassificationDataset.y_train[fit_idx]) == {0, 1, 2}
        assert set(ClassificationDataset.y_train[query_idx]) == {0, 1, 2}


def test_tabular_problem_rejects_class_too_rare_for_stratification(monkeypatch):
    class RareClassDataset(_Dataset):
        task_type = "multiclass"
        n_classes = 3
        X_train = np.ones((7, 8))
        y_train = np.array([0, 0, 0, 1, 1, 1, 2])

    monkeypatch.setenv("GIGAEVO_TABULAR_CV_FOLDS", "3")
    problem = validator.build("unused")

    with np.testing.assert_raises_regex(ValueError, "every observed class"):
        problem._splits(RareClassDataset())


def test_validator_source_loads_without_dunder_file(monkeypatch):
    problem_dir = ROOT / "problems/dag_tab"
    source = (problem_dir / "validate.py").read_text()
    monkeypatch.setattr(
        "sys.path", [str(problem_dir), *list(__import__("sys").path)[1:]]
    )
    namespace = {"__name__": "dag_tab_exec_validator"}

    exec(compile(source, "user_code.py", "exec"), namespace)

    assert callable(namespace["validate"])
    assert callable(namespace["score_on_test"])


def test_fixed_estimator_predictions_match_california_prog5(monkeypatch):
    import importlib.util

    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(160, 8))
    y_train = X_train[:, 0] * 2.0 + rng.normal(scale=0.1, size=160)
    X_val = rng.normal(size=(40, 8))
    y_val = X_val[:, 0] * 2.0 + rng.normal(scale=0.1, size=40)
    X_query = rng.normal(size=(30, 8))

    monkeypatch.setattr(validator.tabular_data, "load_dataset", lambda name: _Dataset())
    monkeypatch.setattr(validator, "execute_graph", lambda graph, frame: frame)
    graph = validator.FeatureGraph.model_validate(json.loads(SEED.read_text()))
    ours = validator.FeatureGraphModel(graph).fit_predict(
        X_train, y_train, X_val, y_val, X_query
    )

    prog5_path = ROOT / "problems/tabular/california/initial_programs/prog5.py"
    spec = importlib.util.spec_from_file_location("california_prog5", prog5_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    theirs = module.entrypoint()().fit_predict(X_train, y_train, X_val, y_val, X_query)

    np.testing.assert_array_equal(ours, theirs)
