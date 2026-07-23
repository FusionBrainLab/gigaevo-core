"""RealMLP-TD adapter for the canonical FeatureGraph execution contract."""

from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np
import pandas as pd

from problems.dag_tab.execution import (
    GraphTriplet,
    assert_target_round_trip,
    inverse_target,
    transform_target,
)
from problems.tabular_dag_baselines.gpu_pool import random_gpu_lease, release_cuda
from problems.tabular_dag_baselines.model_base import BaselineFeatureGraphModel

_PREFIX = "GIGAEVO_REALMLP_"


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    value = int(os.environ.get(_PREFIX + name, str(default)))
    if value < minimum:
        raise ValueError(f"{_PREFIX + name} must be >= {minimum}; got {value}")
    return value


@dataclass(frozen=True)
class RealMLPConfig:
    max_epochs: int = 256
    batch_size: int = 256
    predict_batch_size: int = 1024
    n_threads: int = 4
    n_ens: int = 1
    seed: int = 0

    @classmethod
    def from_env(cls) -> RealMLPConfig:
        return cls(
            max_epochs=_env_int("MAX_EPOCHS", 256),
            batch_size=_env_int("BATCH_SIZE", 256),
            predict_batch_size=_env_int("PREDICT_BATCH_SIZE", 1024),
            n_threads=_env_int("N_THREADS", 4),
            n_ens=_env_int("N_ENS", 1),
            seed=_env_int("SEED", 0, minimum=0),
        )


class RealMLPFeatureGraphModel(BaselineFeatureGraphModel):
    """FeatureGraph model using PyTabKit's fixed RealMLP-TD defaults."""

    estimator_name = "RealMLP-TD"

    def __init__(self, graph, *, device: str | None = None, config=None):
        super().__init__(graph)
        self.device = device
        self.config = config or RealMLPConfig.from_env()
        self.last_fit_summary: dict[str, object] = {}

    def _model_class(self):
        from pytabkit import RealMLP_TD_Classifier, RealMLP_TD_Regressor

        return (
            RealMLP_TD_Regressor
            if self.task_type == "regression"
            else RealMLP_TD_Classifier
        )

    def _model_kwargs(self) -> dict[str, object]:
        return {
            "device": self.device,
            "random_state": self.config.seed,
            "n_cv": 1,
            "n_refit": 0,
            "n_repeats": 1,
            "n_epochs": self.config.max_epochs,
            "batch_size": self.config.batch_size,
            "predict_batch_size": self.config.predict_batch_size,
            "n_threads": self.config.n_threads,
            "n_ens": self.config.n_ens,
            "verbosity": 0,
        }

    def _stabilize_search_categories(self, features: GraphTriplet) -> GraphTriplet:
        """Prevent PyTabKit from learning category cardinality from validation.

        RealMLP's sklearn wrapper concatenates ``X`` and ``X_val`` before its
        internal ordinal encoder is fitted.  Keep validation useful for early
        stopping while mapping validation-only levels to a deterministic level
        observed in the fitting rows.  The final train+validation refit learns
        its vocabulary afresh and therefore keeps those levels normally.
        """

        fit = features.fit.copy()
        validation = features.validation.copy()
        query = features.query.copy()
        for column in self._categorical_columns(fit.columns):
            fit_values = fit[column].astype(object)
            counts = fit_values.value_counts(dropna=False)
            if counts.empty:
                raise ValueError(
                    f"RealMLP categorical feature {column!r} has no fitting rows"
                )
            fallback = counts.index[0]
            observed = set(fit_values.tolist())
            validation_values = validation[column].astype(object)
            unseen = ~validation_values.isin(observed)
            if unseen.any():
                validation_values.loc[unseen] = fallback
            validation[column] = pd.Categorical(
                validation_values,
                categories=fit[column].cat.categories,
            )
        return GraphTriplet(fit, validation, query)

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
            validation_only = np.setdiff1d(np.unique(validation_y), np.unique(train_y))
            if len(validation_only):
                raise ValueError(
                    "RealMLP validation labels contain classes absent from the "
                    f"training rows: {validation_only.tolist()}"
                )

        search_triplet, _, _ = self._strip_sample_weights(
            self._transform(X_train, train_y, X_val, X_query)
        )
        # RealMLP's own documentation requires manual imputation of missing
        # numerical values; medians and category vocabularies are fit-local.
        search_features = self._prepare_model_triplet(
            search_triplet, impute_numeric=True
        )
        search_features = self._stabilize_search_categories(search_features)
        categorical = self._categorical_columns(search_features.fit.columns)
        if self.task_type == "regression":
            search_train_y = transform_target(self.graph.target, train_y, train_y)
            search_validation_y = transform_target(
                self.graph.target, train_y, validation_y
            )
        else:
            search_train_y = train_y
            search_validation_y = validation_y

        model_class = self._model_class()
        search_model = model_class(**self._model_kwargs())
        search_model.fit(
            search_features.fit,
            search_train_y,
            X_val=search_features.validation,
            y_val=search_validation_y,
            cat_col_names=categorical,
        )
        try:
            stop_epoch = search_model.fit_params_["stop_epoch"]
        except (AttributeError, KeyError) as exc:
            raise RuntimeError(
                "RealMLP search did not report a selected epoch"
            ) from exc

        fit_X = np.concatenate([np.asarray(X_train), np.asarray(X_val)])
        fit_y = np.concatenate([train_y, validation_y])
        empty = np.asarray(X_val)[:0]
        final_triplet, _, _ = self._strip_sample_weights(
            self._transform(fit_X, fit_y, empty, X_query)
        )
        final_features = self._prepare_model_triplet(final_triplet, impute_numeric=True)
        categorical = self._categorical_columns(final_features.fit.columns)
        final_kwargs = {
            **self._model_kwargs(),
            "val_fraction": 0.0,
            "stop_epoch": stop_epoch,
        }
        final_model = model_class(**final_kwargs)
        final_y = (
            transform_target(self.graph.target, fit_y, fit_y)
            if self.task_type == "regression"
            else fit_y
        )
        final_model.fit(
            final_features.fit,
            final_y,
            val_idxs=np.empty(0, dtype=int),
            cat_col_names=categorical,
        )
        selected = (
            int(next(iter(stop_epoch.values())))
            if isinstance(stop_epoch, dict)
            else int(stop_epoch)
        )
        self.last_fit_summary = {"best_epochs": selected}
        if self.task_type == "regression":
            return inverse_target(
                self.graph.target,
                fit_y,
                final_model.predict(final_features.query),
            )
        return self._full_probabilities(final_model, final_features.query)

    def fit_predict(self, X_train, y_train, X_val, y_val, X_query):
        if self.device is not None:
            return self._fit_predict_on_device(X_train, y_train, X_val, y_val, X_query)
        with random_gpu_lease("realmlp") as lease:
            try:
                self.device = lease.device
                return self._fit_predict_on_device(
                    X_train, y_train, X_val, y_val, X_query
                )
            finally:
                release_cuda(lease)


FeatureGraphModel = RealMLPFeatureGraphModel
