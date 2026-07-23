"""GigaEvo validator for FeatureGraph JSON genomes."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
import re
import sys
from typing import cast

from catboost import CatBoostClassifier, CatBoostRegressor, Pool
import numpy as np
import pandas as pd

_SOURCE_PATH = globals().get("__file__")
_PROBLEM_DIR = (
    Path(_SOURCE_PATH).resolve().parent if _SOURCE_PATH else Path(sys.path[0]).resolve()
)
_TABULAR_COMMON = _PROBLEM_DIR.parent / "tabular" / "_common"
if str(_TABULAR_COMMON) not in sys.path:
    sys.path.insert(0, str(_TABULAR_COMMON))

import tabular_data  # noqa: E402
from tabular_problem import build  # noqa: E402

from problems.dag_tab.execution import (  # noqa: E402
    GraphTriplet,
    assert_split_invariant,
    assert_target_round_trip,
    execute_graph_triplet,
    inverse_target,
    transform_target,
)
from problems.dag_tab.graph import FeatureGraph, FeatureValueKind  # noqa: E402

_INVALID = {
    "fitness": -1.0,
    "is_valid": 0.0,
    "cv_score_std": 2.0,
    "graph_node_count": 0.0,
    "graph_max_depth": 0.0,
    "generated_feature_count": 0.0,
    "local_lipschitz_p95": 4.0,
    "ood_delta_slope": 2.0,
}


class ValidationFailureReason(StrEnum):
    SCHEMA = "schema"
    EXECUTION = "execution"
    NON_FINITE = "non_finite"
    BATCH_PURITY = "batch_purity"
    DETERMINISM = "determinism"
    OWN_TARGET_INVARIANCE = "own_target_invariance"
    TARGET_ROUND_TRIP = "target_round_trip"
    MODEL_FIT = "model_fit"
    SAMPLE_WEIGHT = "sample_weight"
    DATASET_CONTRACT = "dataset_contract"
    UNKNOWN = "unknown"


def _validation_failure_reason(exc: Exception, stage: str) -> ValidationFailureReason:
    message = str(exc).lower()
    if stage == "schema":
        return ValidationFailureReason.SCHEMA
    if stage == "dataset_contract":
        return ValidationFailureReason.DATASET_CONTRACT
    if stage == "model_fit":
        return (
            ValidationFailureReason.SAMPLE_WEIGHT
            if "sample_weight" in message
            else ValidationFailureReason.MODEL_FIT
        )
    if "sample_weight" in message:
        return ValidationFailureReason.SAMPLE_WEIGHT
    if "own-target leakage" in message:
        return ValidationFailureReason.OWN_TARGET_INVARIANCE
    if "non-deterministic" in message:
        return ValidationFailureReason.DETERMINISM
    if "split-dependent" in message:
        return ValidationFailureReason.BATCH_PURITY
    if "target" in message and (
        "round-trip" in message or "transform" in message or "inverse" in message
    ):
        return ValidationFailureReason.TARGET_ROUND_TRIP
    if "finite" in message or "contains inf" in message:
        return ValidationFailureReason.NON_FINITE
    if stage == "behavioral_probes":
        return ValidationFailureReason.EXECUTION
    return ValidationFailureReason.UNKNOWN


def _failure_artifact(exc: Exception, stage: str) -> dict[str, object]:
    message = str(exc)
    node_match = re.search(r"\bnode ([A-Za-z][A-Za-z0-9_]*)", message)
    artifact: dict[str, object] = {
        "error": f"{type(exc).__name__}: {message}",
        "validation_failure_reason": _validation_failure_reason(exc, stage).value,
        "validation_failure_stage": stage,
    }
    if node_match is not None:
        artifact["validation_failure_node"] = node_match.group(1)
    return artifact


def _frame(values: np.ndarray, columns: list[str]) -> pd.DataFrame:
    array = np.asarray(values)
    if array.ndim != 2 or array.shape[1] != len(columns):
        raise ValueError(
            f"dataset matrix shape {array.shape} does not match {len(columns)} raw columns"
        )
    return pd.DataFrame(array, columns=columns)


class FeatureGraphModel:
    """Fixed estimator whose evolvable component is the feature graph."""

    def __init__(self, graph: FeatureGraph):
        self.graph = graph
        dataset = tabular_data.load_dataset(graph.dataset)
        self.task_type = dataset.task_type
        self.n_classes = dataset.n_classes
        self.raw_column_types: dict[str, FeatureValueKind] = {
            f"x{spec.index}": (
                "numerical"
                if spec.kind == "numeric"
                else cast(FeatureValueKind, spec.kind)
            )
            for spec in getattr(dataset, "columns", ())
        }

    def _feature_kind(self, column: str) -> FeatureValueKind:
        if column in self.raw_column_types:
            return self.raw_column_types[column]
        if column in self.graph.raw_columns:
            return "numerical"
        for node in self.graph.nodes:
            if column in node.output_cols:
                return node.output_type(column)
        raise ValueError(f"unknown feature column {column!r}")

    def _cat_features(self, frame: pd.DataFrame) -> list[int]:
        return [
            index
            for index, column in enumerate(frame.columns)
            if self._feature_kind(column) in {"categorical", "binary"}
        ]

    def _extract_sample_weights(
        self, triplet: GraphTriplet
    ) -> tuple[GraphTriplet, np.ndarray | None, np.ndarray | None]:
        frames = [triplet.fit.copy(), triplet.validation.copy(), triplet.query.copy()]
        if "sample_weight" not in triplet.fit.columns:
            return GraphTriplet(*frames), None, None

        weights: list[np.ndarray] = []
        for label, frame in zip(("fit", "validation", "query"), frames):
            values = pd.to_numeric(frame.pop("sample_weight"), errors="raise").to_numpy(
                dtype=float
            )
            if not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError(
                    f"{label} sample_weight must contain finite non-negative values"
                )
            weights.append(values)
        if len(weights[0]) and not np.any(weights[0] > 0):
            raise ValueError(
                "fit sample_weight must contain at least one positive value"
            )
        return GraphTriplet(*frames), weights[0], weights[1]

    def _prepare_catboost_triplet(self, triplet: GraphTriplet) -> GraphTriplet:
        frames = [triplet.fit.copy(), triplet.validation.copy(), triplet.query.copy()]
        for column in triplet.fit.columns:
            kind = self._feature_kind(column)
            if kind in {"categorical", "binary"}:
                fit_values = {
                    str(value)
                    for value in triplet.fit[column].tolist()
                    if not pd.isna(value)
                }

                def encode(value):
                    if pd.isna(value):
                        return "__MISSING__"
                    text = str(value)
                    return text if text in fit_values else "__UNKNOWN__"

                for frame in frames:
                    frame[column] = frame[column].map(encode)
            else:
                for frame in frames:
                    frame[column] = pd.to_numeric(frame[column], errors="raise")
        return GraphTriplet(*frames)

    def _transform(self, X_train, y_train, X_val, X_query) -> GraphTriplet:
        fit = _frame(X_train, self.graph.raw_columns)
        target = np.asarray(y_train)
        combined = execute_graph_triplet(
            self.graph,
            fit,
            _frame(X_val, self.graph.raw_columns),
            _frame(X_query, self.graph.raw_columns),
            y_fit=target,
        )
        # The rows the model trains on come from an execution with nothing appended,
        # which is the shape the own-target probe verifies. No finite set of probed
        # appended batches can certify every shape a fold produces, so rather than
        # sample harder, the training rows are computed where there is no appended
        # batch to condition on at all. An honest node returns the same values here.
        empty = fit.iloc[:0].copy()
        fit_only = execute_graph_triplet(
            self.graph, fit, empty, empty, y_fit=target
        ).fit
        return GraphTriplet(fit_only, combined.validation, combined.query)

    @staticmethod
    def _eval_pool(
        features: pd.DataFrame,
        target: np.ndarray,
        cat_features: list[int],
        weights: np.ndarray | None,
    ):
        if weights is None:
            return (features, target)
        return Pool(features, target, cat_features=cat_features, weight=weights)

    @staticmethod
    def _fit(model, features, target, *, weights=None, eval_set=None):
        kwargs = {}
        if weights is not None:
            kwargs["sample_weight"] = weights
        if eval_set is not None:
            kwargs["eval_set"] = eval_set
        return model.fit(features, target, **kwargs)

    def _search_fit_predict(self, X_train, y_train, X_val, y_val, X_query):
        """Fit one early-stopped fold model and predict without train+validation refit."""

        train_y = np.asarray(y_train)
        val_y = np.asarray(y_val)
        if self.graph.target is not None and self.task_type != tabular_data.REGRESSION:
            raise ValueError("target transforms are supported for regression only")
        if self.graph.target is not None:
            assert_target_round_trip(self.graph.target, train_y)
        search_triplet, train_weight, val_weight = self._extract_sample_weights(
            self._transform(X_train, train_y, X_val, X_query)
        )
        search_features = self._prepare_catboost_triplet(search_triplet)
        train_x = search_features.fit
        val_x = search_features.validation
        query_x = search_features.query
        cat_features = self._cat_features(train_x)
        params = {
            "learning_rate": 0.05,
            "depth": 6,
            "random_seed": 0,
            "thread_count": 4,
            "logging_level": "Silent",
            "allow_writing_files": False,
            "cat_features": cat_features,
        }

        if self.task_type == tabular_data.REGRESSION:
            search_train_y = transform_target(self.graph.target, train_y, train_y)
            search_val_y = transform_target(self.graph.target, train_y, val_y)
            model = CatBoostRegressor(
                iterations=2000,
                early_stopping_rounds=50,
                **params,
            )
            self._fit(
                model,
                train_x,
                search_train_y,
                weights=train_weight,
                eval_set=self._eval_pool(val_x, search_val_y, cat_features, val_weight),
            )
            predictions = inverse_target(
                self.graph.target, train_y, model.predict(query_x)
            )
            return predictions, max(1, model.get_best_iteration() + 1), params

        train_y = train_y.astype(int)
        val_y = val_y.astype(int)
        if self.n_classes is None or self.n_classes < 2:
            raise ValueError("classification dataset must declare n_classes >= 2")
        n_classes = int(self.n_classes)
        observed_classes = np.unique(np.concatenate([train_y, val_y]))
        if np.any(observed_classes < 0) or np.any(observed_classes >= n_classes):
            raise ValueError(
                f"classification labels {observed_classes.tolist()} fall outside "
                f"declared class universe [0, {n_classes})"
            )
        classification_params = dict(params)
        if n_classes > 2:
            classification_params.update(
                {"loss_function": "MultiClass", "classes_count": n_classes}
            )
        model = CatBoostClassifier(
            iterations=2000,
            early_stopping_rounds=50,
            **classification_params,
        )
        self._fit(
            model,
            train_x,
            train_y,
            weights=train_weight,
            eval_set=self._eval_pool(val_x, val_y, cat_features, val_weight),
        )
        probabilities = np.zeros((query_x.shape[0], n_classes))
        probabilities[:, model.classes_.astype(int)] = model.predict_proba(query_x)
        return (
            probabilities,
            max(1, model.get_best_iteration() + 1),
            classification_params,
        )

    def fit_predict(self, X_train, y_train, X_val, y_val, X_query):
        _, best_iterations, params = self._search_fit_predict(
            X_train, y_train, X_val, y_val, X_query
        )
        train_y = np.asarray(y_train)
        val_y = np.asarray(y_val)
        if self.task_type != tabular_data.REGRESSION:
            train_y = train_y.astype(int)
            val_y = val_y.astype(int)
        fit_X = np.concatenate([np.asarray(X_train), np.asarray(X_val)])
        fit_y = np.concatenate([train_y, val_y])
        empty = np.asarray(X_val)[:0]
        final_triplet, fit_weight, _ = self._extract_sample_weights(
            self._transform(fit_X, fit_y, empty, X_query)
        )
        final_features = self._prepare_catboost_triplet(final_triplet)
        fit_x = final_features.fit
        query_x = final_features.query

        if self.task_type == tabular_data.REGRESSION:
            final_y = transform_target(self.graph.target, fit_y, fit_y)
            model = CatBoostRegressor(iterations=best_iterations, **params)
            self._fit(model, fit_x, final_y, weights=fit_weight)
            predictions = model.predict(query_x)
            return inverse_target(self.graph.target, fit_y, predictions)

        model = CatBoostClassifier(iterations=best_iterations, **params)
        self._fit(model, fit_x, fit_y, weights=fit_weight)
        probabilities = np.zeros((query_x.shape[0], int(self.n_classes)))
        probabilities[:, model.classes_.astype(int)] = model.predict_proba(query_x)
        return probabilities


def _factory(graph: FeatureGraph):
    return lambda: FeatureGraphModel(graph)


def validate(payload):
    """Validate and score a decoded FeatureGraph JSON document."""
    stage = "schema"
    try:
        graph = FeatureGraph.model_validate(payload)
        stage = "dataset_contract"
        dataset = tabular_data.load_dataset(graph.dataset)
        expected = [f"x{i}" for i in range(dataset.X_train.shape[1])]
        if graph.raw_columns != expected:
            raise ValueError(
                f"raw_columns must exactly match dataset columns {expected}; "
                f"got {graph.raw_columns}"
            )
        stage = "behavioral_probes"
        sample_size = min(1024, len(dataset.X_train))
        assert_split_invariant(
            graph,
            _frame(dataset.X_train[:sample_size], graph.raw_columns),
            np.asarray(dataset.y_train[:sample_size]),
        )
        if graph.target is not None:
            if dataset.task_type != tabular_data.REGRESSION:
                raise ValueError("target transforms are supported for regression only")
            assert_target_round_trip(
                graph.target, np.asarray(dataset.y_train[:sample_size])
            )
        stage = "model_fit"
        metrics, evaluation_artifact = build(graph.dataset).validate(_factory(graph))
        metrics.update(
            {
                "graph_node_count": float(len(graph.nodes)),
                "graph_max_depth": float(graph.depth),
                "generated_feature_count": float(len(graph.feature_output_columns)),
            }
        )
        artifact = {
            "dataset": graph.dataset,
            "output_columns": graph.output_columns,
            "graph_node_count": len(graph.nodes),
        }
        if isinstance(evaluation_artifact, dict):
            artifact.update(evaluation_artifact)
        return metrics, artifact
    except Exception as exc:
        return dict(_INVALID), _failure_artifact(exc, stage)


def score_on_test(payload):
    """Score on the untouched test split under the exact `problems/tabular` protocol."""
    graph = FeatureGraph.model_validate(payload)
    return build(graph.dataset).score_on_test(_factory(graph))
