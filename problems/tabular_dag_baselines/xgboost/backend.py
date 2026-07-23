"""XGBoost adapter for the canonical FeatureGraph execution contract."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os

import numpy as np

from problems.dag_tab.execution import (
    assert_target_round_trip,
    inverse_target,
    transform_target,
)
from problems.tabular_dag_baselines.model_base import BaselineFeatureGraphModel

_PREFIX = "GIGAEVO_XGBOOST_"


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    value = int(os.environ.get(_PREFIX + name, str(default)))
    if value < minimum:
        raise ValueError(f"{_PREFIX + name} must be >= {minimum}; got {value}")
    return value


def _env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    value = float(os.environ.get(_PREFIX + name, str(default)))
    if not math.isfinite(value) or value < minimum:
        raise ValueError(
            f"{_PREFIX + name} must be finite and >= {minimum}; got {value}"
        )
    return value


@dataclass(frozen=True)
class XGBoostConfig:
    learning_rate: float = 0.05
    max_depth: int = 6
    max_estimators: int = 2000
    early_stopping_rounds: int = 50
    n_jobs: int = 4
    seed: int = 0

    @classmethod
    def from_env(cls) -> XGBoostConfig:
        return cls(
            learning_rate=_env_float("LEARNING_RATE", 0.05),
            max_depth=_env_int("MAX_DEPTH", 6),
            max_estimators=_env_int("MAX_ESTIMATORS", 2000),
            early_stopping_rounds=_env_int("EARLY_STOPPING_ROUNDS", 50, minimum=0),
            n_jobs=_env_int("N_JOBS", 4),
            seed=_env_int("SEED", 0, minimum=0),
        )


class XGBoostFeatureGraphModel(BaselineFeatureGraphModel):
    """FeatureGraph model with fixed level-wise histogram boosting."""

    estimator_name = "XGBoost"
    supports_sample_weight = True

    def __init__(self, graph, *, config: XGBoostConfig | None = None):
        super().__init__(graph)
        self.config = config or XGBoostConfig.from_env()
        self.last_fit_summary: dict[str, object] = {}

    def _params(self) -> dict[str, object]:
        return {
            "learning_rate": self.config.learning_rate,
            "max_depth": self.config.max_depth,
            "tree_method": "hist",
            "random_state": self.config.seed,
            "n_jobs": self.config.n_jobs,
            "verbosity": 0,
            "enable_categorical": True,
        }

    def _fit_search(
        self,
        features,
        train_y,
        validation_y,
        fit_weight,
        validation_weight,
    ):
        from xgboost import XGBClassifier, XGBRegressor

        params = self._params()
        if self.task_type == "regression":
            model = XGBRegressor(
                n_estimators=self.config.max_estimators,
                early_stopping_rounds=self.config.early_stopping_rounds,
                **params,
            )
        else:
            if int(self.n_classes) > 2:
                params.update(objective="multi:softprob", num_class=int(self.n_classes))
            model = XGBClassifier(
                n_estimators=self.config.max_estimators,
                early_stopping_rounds=self.config.early_stopping_rounds,
                **params,
            )
        kwargs: dict[str, object] = {
            "eval_set": [(features.validation, validation_y)],
            "verbose": False,
        }
        if fit_weight is not None:
            kwargs["sample_weight"] = fit_weight
        if validation_weight is not None:
            kwargs["sample_weight_eval_set"] = [validation_weight]
        model.fit(features.fit, train_y, **kwargs)
        best = getattr(model, "best_iteration", None)
        best_iterations = self.config.max_estimators if best is None else int(best) + 1
        return best_iterations, params

    def fit_predict(self, X_train, y_train, X_val, y_val, X_query):
        from xgboost import XGBClassifier, XGBRegressor

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

        search_triplet, fit_weight, validation_weight = self._strip_sample_weights(
            self._transform(X_train, train_y, X_val, X_query)
        )
        search_features = self._prepare_model_triplet(search_triplet)
        if self.task_type == "regression":
            search_train_y = transform_target(self.graph.target, train_y, train_y)
            search_validation_y = transform_target(
                self.graph.target, train_y, validation_y
            )
        else:
            search_train_y = train_y
            search_validation_y = validation_y
        best_iterations, params = self._fit_search(
            search_features,
            search_train_y,
            search_validation_y,
            fit_weight,
            validation_weight,
        )

        fit_X = np.concatenate([np.asarray(X_train), np.asarray(X_val)])
        fit_y = np.concatenate([train_y, validation_y])
        empty = np.asarray(X_val)[:0]
        final_triplet, final_weight, _ = self._strip_sample_weights(
            self._transform(fit_X, fit_y, empty, X_query)
        )
        final_features = self._prepare_model_triplet(final_triplet)
        if self.task_type == "regression":
            model = XGBRegressor(n_estimators=best_iterations, **params)
            final_y = transform_target(self.graph.target, fit_y, fit_y)
        else:
            model = XGBClassifier(n_estimators=best_iterations, **params)
            final_y = fit_y
        kwargs: dict[str, object] = {}
        if final_weight is not None:
            kwargs["sample_weight"] = final_weight
        model.fit(final_features.fit, final_y, **kwargs)
        self.last_fit_summary = {"best_iterations": best_iterations}
        if self.task_type == "regression":
            return inverse_target(
                self.graph.target, fit_y, model.predict(final_features.query)
            )
        return self._full_probabilities(model, final_features.query)


FeatureGraphModel = XGBoostFeatureGraphModel
