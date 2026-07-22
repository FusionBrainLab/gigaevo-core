from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from problems.dag_tab import validate as validator

ROOT = Path(__file__).parents[2]
SEED = ROOT / "problems/dag_tab/initial_programs/baseline.json"


class _Dataset:
    task_type = "regression"
    n_classes = None
    columns = ()
    X_train = np.ones((4, 8))
    y_train = np.arange(4, dtype=float)


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
    assert artifact["validation_failure_reason"] == "schema"
    assert artifact["validation_failure_stage"] == "schema"


@pytest.mark.parametrize(
    "stage, message, expected",
    [
        ("dataset_contract", "dataset values must be finite", "dataset_contract"),
        ("model_fit", "estimator contains inf and is not finite", "model_fit"),
    ],
)
def test_failure_reason_prefers_terminal_stage_over_incidental_message(
    stage, message, expected
):
    artifact = validator._failure_artifact(ValueError(message), stage)

    assert artifact["validation_failure_reason"] == expected
    assert artifact["validation_failure_stage"] == stage


def test_validate_rejects_split_dependent_graph(monkeypatch):
    monkeypatch.setattr(validator.tabular_data, "load_dataset", lambda name: _Dataset())
    payload = json.loads(SEED.read_text())
    payload["nodes"][0]["code"] = (
        "df['fe_income_per_age'] = np.arange(len(df)) * 1.0\nreturn df"
    )

    metrics, artifact = validator.validate(payload)

    assert metrics["is_valid"] == 0.0
    assert "split-dependent behavior" in artifact["error"]
    assert artifact["validation_failure_reason"] == "batch_purity"
    assert artifact["validation_failure_stage"] == "behavioral_probes"


def test_validate_reports_execution_node(monkeypatch):
    monkeypatch.setattr(validator.tabular_data, "load_dataset", lambda name: _Dataset())
    payload = json.loads(SEED.read_text())
    payload["nodes"][0]["code"] = "raise RuntimeError('broken transform')\nreturn df"

    metrics, artifact = validator.validate(payload)

    assert metrics["is_valid"] == 0.0
    assert artifact["validation_failure_reason"] == "execution"
    assert artifact["validation_failure_stage"] == "behavioral_probes"
    assert artifact["validation_failure_node"] == "income_per_age"


def test_validate_rejects_index_dependent_graph(monkeypatch):
    monkeypatch.setattr(validator.tabular_data, "load_dataset", lambda name: _Dataset())
    payload = json.loads(SEED.read_text())
    payload["nodes"][0]["code"] = (
        "df['fe_income_per_age'] = df.index.to_numpy() * 1.0\nreturn df"
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


def test_score_on_test_delegates_to_the_shared_tabular_scorer(monkeypatch):
    payload = json.loads(SEED.read_text())
    seen = []

    class RecordingProblem(_Problem):
        def score_on_test(self, factory):
            seen.append(factory().graph.dataset)
            return {"test_r2": 0.42, "test_rmse": 0.7}

    monkeypatch.setattr(validator.tabular_data, "load_dataset", lambda name: _Dataset())
    monkeypatch.setattr(validator, "build", lambda name: RecordingProblem())

    result = validator.score_on_test(payload)

    assert seen == ["california"]
    assert result == {"test_r2": 0.42, "test_rmse": 0.7}


def test_fixed_estimator_hyperparameters_are_pinned():
    source = (ROOT / "problems/dag_tab/validate.py").read_text()

    assert '"learning_rate": 0.05,' in source
    assert '"depth": 6,' in source
    assert '"random_seed": 0,' in source
    assert "iterations=2000," in source
    assert "early_stopping_rounds=50," in source


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


def test_catboost_receives_raw_and_generated_categorical_features(monkeypatch):
    payload = json.loads(SEED.read_text())
    payload["raw_columns"] = ["x0"]
    payload["nodes"] = [
        {
            "id": "bucket",
            "input_cols": ["x0"],
            "output_cols": ["fe_bucket"],
            "output_types": {"fe_bucket": "categorical"},
            "code": "df['fe_bucket'] = np.where(df['x0'] > 0, 'high', 'low')\nreturn df",
            "rationale": "Expose a categorical bucket.",
            "dependencies": [],
            "is_output": True,
        }
    ]
    graph = validator.FeatureGraph.model_validate(payload)
    seen = []

    class CategoricalDataset(_Dataset):
        columns = (validator.tabular_data.ColumnSpec(0, "categorical", 2, ("a", "b")),)

    class FakeRegressor:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            seen.append(self)

        def fit(self, X, y, eval_set=None):
            self.X = X
            return self

        def get_best_iteration(self):
            return 1

        def predict(self, X):
            return np.zeros(len(X))

    monkeypatch.setattr(validator, "CatBoostRegressor", FakeRegressor)
    monkeypatch.setattr(
        validator.tabular_data, "load_dataset", lambda name: CategoricalDataset()
    )
    model = validator.FeatureGraphModel(graph)
    model.fit_predict(
        np.array([[0.0], [1.0]]),
        np.array([0.0, 1.0]),
        np.array([[2.0]]),
        np.array([2.0]),
        np.array([[3.0]]),
    )

    assert seen[0].kwargs["cat_features"] == [0, 1]
    assert seen[0].X.iloc[:, 0].tolist() == ["0.0", "1.0"]
    assert seen[1].X.iloc[:, 0].tolist() == ["0.0", "1.0", "2.0"]


def test_aggregate_preprocessing_refits_on_train_plus_validation(monkeypatch):
    payload = json.loads(SEED.read_text())
    payload["raw_columns"] = ["x0"]
    payload["nodes"] = [
        {
            "id": "center",
            "kind": "aggregate",
            "input_cols": ["x0"],
            "output_cols": ["fe_centered"],
            "code": "mean = df_fit['x0'].mean()\ndf['fe_centered'] = df['x0'] - mean\nreturn df",
            "rationale": "Fit a center on the permitted fit rows.",
            "dependencies": [],
            "is_output": True,
        }
    ]
    graph = validator.FeatureGraph.model_validate(payload)
    fits = []

    class FakeRegressor:
        def __init__(self, **kwargs):
            pass

        def fit(self, X, y, eval_set=None):
            fits.append(X.copy())
            return self

        def get_best_iteration(self):
            return 1

        def predict(self, X):
            return np.zeros(len(X))

    monkeypatch.setattr(validator, "CatBoostRegressor", FakeRegressor)
    monkeypatch.setattr(validator.tabular_data, "load_dataset", lambda name: _Dataset())
    model = validator.FeatureGraphModel(graph)
    model.fit_predict(
        np.array([[0.0], [2.0]]),
        np.array([0.0, 1.0]),
        np.array([[10.0]]),
        np.array([2.0]),
        np.array([[20.0]]),
    )

    assert fits[0]["fe_centered"].tolist() == [-1.0, 1.0]
    assert fits[1]["fe_centered"].tolist() == [-4.0, -2.0, 6.0]


def test_sample_weight_is_passed_to_fit_and_removed_from_features(monkeypatch):
    payload = json.loads(SEED.read_text())
    payload["raw_columns"] = ["x0"]
    payload["nodes"] = [
        {
            "id": "weight_rows",
            "input_cols": ["x0"],
            "output_cols": ["sample_weight", "fe_weight_signal"],
            "code": (
                "df['sample_weight'] = np.where(df['x0'] > 0, 2.0, 0.5)\n"
                "df['fe_weight_signal'] = df['x0']\nreturn df"
            ),
            "rationale": "Weight positive rows more heavily.",
            "dependencies": [],
            "is_output": True,
        }
    ]
    graph = validator.FeatureGraph.model_validate(payload)
    fits = []

    class FakeRegressor:
        def __init__(self, **kwargs):
            pass

        def fit(self, X, y, **kwargs):
            fits.append((X.copy(), np.asarray(y), kwargs))
            return self

        def get_best_iteration(self):
            return 1

        def predict(self, X):
            return np.zeros(len(X))

    monkeypatch.setattr(validator, "CatBoostRegressor", FakeRegressor)
    monkeypatch.setattr(validator.tabular_data, "load_dataset", lambda name: _Dataset())
    model = validator.FeatureGraphModel(graph)
    model.fit_predict(
        np.array([[-1.0], [1.0]]),
        np.array([0.0, 1.0]),
        np.array([[2.0]]),
        np.array([2.0]),
        np.array([[3.0]]),
    )

    assert "sample_weight" not in fits[0][0].columns
    np.testing.assert_array_equal(fits[0][2]["sample_weight"], [0.5, 2.0])
    np.testing.assert_array_equal(fits[1][2]["sample_weight"], [0.5, 2.0, 2.0])


def test_invalid_sample_weight_is_rejected(monkeypatch):
    payload = json.loads(SEED.read_text())
    payload["raw_columns"] = ["x0"]
    payload["nodes"] = [
        {
            "id": "invalid_weights",
            "input_cols": ["x0"],
            "output_cols": ["sample_weight", "fe_signal"],
            "code": (
                "df['sample_weight'] = -1.0\ndf['fe_signal'] = df['x0']\nreturn df"
            ),
            "rationale": "Exercise weight validation.",
            "dependencies": [],
            "is_output": True,
        }
    ]
    graph = validator.FeatureGraph.model_validate(payload)
    monkeypatch.setattr(validator.tabular_data, "load_dataset", lambda name: _Dataset())
    model = validator.FeatureGraphModel(graph)

    with pytest.raises(ValueError, match="finite non-negative"):
        model.fit_predict(
            np.array([[0.0], [1.0]]),
            np.array([0.0, 1.0]),
            np.array([[2.0]]),
            np.array([2.0]),
            np.array([[3.0]]),
        )


def test_dropped_raw_column_is_not_passed_to_catboost(monkeypatch):
    payload = json.loads(SEED.read_text())
    payload["raw_columns"] = ["x0", "x1"]
    payload["dropped_raw_columns"] = ["x1"]
    payload["nodes"][0]["input_cols"] = ["x0", "x1"]
    graph = validator.FeatureGraph.model_validate(payload)
    fits = []

    class FakeRegressor:
        def __init__(self, **kwargs):
            pass

        def fit(self, X, y, **kwargs):
            fits.append(X.copy())
            return self

        def get_best_iteration(self):
            return 1

        def predict(self, X):
            return np.zeros(len(X))

    monkeypatch.setattr(validator, "CatBoostRegressor", FakeRegressor)
    monkeypatch.setattr(validator.tabular_data, "load_dataset", lambda name: _Dataset())
    model = validator.FeatureGraphModel(graph)
    values = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    model.fit_predict(values[:2], [0.0, 1.0], values[2:], [2.0], values[2:])

    assert fits[0].columns.tolist() == ["x0", "fe_income_per_age"]


def test_target_transform_trains_in_transformed_scale_and_inverts_predictions(
    monkeypatch,
):
    payload = json.loads(SEED.read_text())
    payload["target"] = {
        "code": "return np.log1p(y)",
        "inverse_code": "return np.expm1(predictions)",
    }
    graph = validator.FeatureGraph.model_validate(payload)
    fits = []

    class FakeRegressor:
        def __init__(self, **kwargs):
            pass

        def fit(self, X, y, **kwargs):
            fits.append(np.asarray(y).copy())
            return self

        def get_best_iteration(self):
            return 1

        def predict(self, X):
            return np.full(len(X), np.log1p(4.0))

    monkeypatch.setattr(validator, "CatBoostRegressor", FakeRegressor)
    monkeypatch.setattr(validator.tabular_data, "load_dataset", lambda name: _Dataset())
    model = validator.FeatureGraphModel(graph)
    values = np.arange(32, dtype=float).reshape(4, 8)
    predictions = model.fit_predict(
        values[:2], [0.0, 3.0], values[2:3], [8.0], values[3:]
    )

    np.testing.assert_allclose(fits[0], np.log1p([0.0, 3.0]))
    np.testing.assert_allclose(fits[1], np.log1p([0.0, 3.0, 8.0]))
    np.testing.assert_allclose(predictions, [4.0])


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


def test_fixed_estimator_uses_evolved_features_and_is_deterministic(monkeypatch):
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(160, 8))
    y_train = X_train[:, 0] * 2.0 + rng.normal(scale=0.1, size=160)
    X_val = rng.normal(size=(40, 8))
    y_val = X_val[:, 0] * 2.0 + rng.normal(scale=0.1, size=40)
    X_query = rng.normal(size=(30, 8))

    monkeypatch.setattr(validator.tabular_data, "load_dataset", lambda name: _Dataset())
    graph = validator.FeatureGraph.model_validate(json.loads(SEED.read_text()))
    ours = validator.FeatureGraphModel(graph).fit_predict(
        X_train, y_train, X_val, y_val, X_query
    )
    repeated = validator.FeatureGraphModel(graph).fit_predict(
        X_train, y_train, X_val, y_val, X_query
    )

    assert ours.shape == (len(X_query),)
    np.testing.assert_array_equal(ours, repeated)


def test_validate_emits_every_behavior_axis_on_the_success_path(monkeypatch):
    monkeypatch.setattr(validator.tabular_data, "load_dataset", lambda name: _Dataset())
    monkeypatch.setattr(validator, "build", lambda name: _Problem())

    metrics, _ = validator.validate(json.loads(SEED.read_text()))

    assert set(validator._INVALID) <= set(metrics)
