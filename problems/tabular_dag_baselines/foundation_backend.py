"""Shared fixed-context protocol for pretrained tabular foundation models."""

from __future__ import annotations

import numpy as np

from problems.dag_tab.execution import (
    assert_target_round_trip,
    inverse_target,
    transform_target,
)
from problems.tabular_dag_baselines.gpu_pool import random_gpu_lease, release_cuda
from problems.tabular_dag_baselines.model_base import BaselineFeatureGraphModel


class InContextFeatureGraphModel(BaselineFeatureGraphModel):
    """FeatureGraph adapter for a frozen pretrained in-context estimator."""

    gpu_model_name = "foundation"

    def __init__(self, graph, *, device: str | None = None, config=None):
        super().__init__(graph)
        self.device = device
        self.config = config
        self.last_fit_summary: dict[str, object] = {}

    def _make_estimator(self, categorical_indices: list[int]):
        raise NotImplementedError

    def _fit_predict_on_device(self, X_train, y_train, X_val, y_val, X_query):
        train_y = np.asarray(y_train)
        validation_y = np.asarray(y_val)
        if self.graph.target is not None and self.task_type != "regression":
            raise ValueError("target transforms are supported for regression only")
        if self.graph.target is not None:
            assert_target_round_trip(self.graph.target, train_y)
        if self.task_type != "regression":
            train_y = train_y.astype(int)
            validation_y = validation_y.astype(int)
            self._validate_classification_labels(train_y, validation_y)

        # Foundation models have no dataset-specific early-stopping phase.  The
        # canonical final context is therefore train+validation, exactly like
        # the refit stage of the trained estimators.
        fit_X = np.concatenate([np.asarray(X_train), np.asarray(X_val)])
        fit_y = np.concatenate([train_y, validation_y])
        empty = np.asarray(X_val)[:0]
        final_triplet, _, _ = self._strip_sample_weights(
            self._transform(fit_X, fit_y, empty, X_query)
        )
        features = self._prepare_model_triplet(final_triplet)
        categorical = self._categorical_columns(features.fit.columns)
        categorical_indices = [
            index
            for index, column in enumerate(features.fit.columns)
            if column in categorical
        ]
        estimator = self._make_estimator(categorical_indices)
        target = (
            transform_target(self.graph.target, fit_y, fit_y)
            if self.task_type == "regression"
            else fit_y
        )
        estimator.fit(features.fit, target)
        self.last_fit_summary = {
            "context_rows": int(len(features.fit)),
            "categorical_features": len(categorical_indices),
        }
        if self.task_type == "regression":
            return inverse_target(
                self.graph.target, fit_y, estimator.predict(features.query)
            )
        return self._full_probabilities(estimator, features.query)

    def fit_predict(self, X_train, y_train, X_val, y_val, X_query):
        if self.device is not None:
            return self._fit_predict_on_device(X_train, y_train, X_val, y_val, X_query)
        with random_gpu_lease(self.gpu_model_name) as lease:
            try:
                self.device = lease.device
                return self._fit_predict_on_device(
                    X_train, y_train, X_val, y_val, X_query
                )
            finally:
                release_cuda(lease)
