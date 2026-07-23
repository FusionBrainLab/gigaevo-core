"""LightGBM adapter for the canonical FeatureGraph execution contract."""

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

_PREFIX = "GIGAEVO_LIGHTGBM_"


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
class LightGBMConfig:
    learning_rate: float = 0.05
    num_leaves: int = 63
    max_estimators: int = 2000
    early_stopping_rounds: int = 50
    n_jobs: int = 4
    seed: int = 0

    @classmethod
    def from_env(cls) -> LightGBMConfig:
        return cls(
            learning_rate=_env_float("LEARNING_RATE", 0.05),
            num_leaves=_env_int("NUM_LEAVES", 63, minimum=2),
            max_estimators=_env_int("MAX_ESTIMATORS", 2000),
            early_stopping_rounds=_env_int("EARLY_STOPPING_ROUNDS", 50, minimum=0),
            n_jobs=_env_int("N_JOBS", 4),
            seed=_env_int("SEED", 0, minimum=0),
        )


class LightGBMFeatureGraphModel(BaselineFeatureGraphModel):
    """FeatureGraph model with fixed leaf-wise histogram boosting."""

    estimator_name = "LightGBM"
    supports_sample_weight = True

    def __init__(self, graph, *, config: LightGBMConfig | None = None):
        super().__init__(graph)
        self.config = config or LightGBMConfig.from_env()
        self.last_fit_summary: dict[str, object] = {}

    def _params(self) -> dict[str, object]:
        return {
            "learning_rate": self.config.learning_rate,
            "num_leaves": self.config.num_leaves,
            "random_state": self.config.seed,
            "n_jobs": self.config.n_jobs,
            "verbose": -1,
            "deterministic": True,
            "force_col_wise": True,
        }

    def _fit_search(
        self,
        features,
        train_y,
        validation_y,
        fit_weight,
        validation_weight,
    ):
        import lightgbm as lgb

        params = self._params()
        categorical = self._categorical_columns(features.fit.columns)
        if self.task_type == "regression":
            model = lgb.LGBMRegressor(n_estimators=self.config.max_estimators, **params)
        else:
            if int(self.n_classes) > 2:
                params.update(objective="multiclass", num_class=int(self.n_classes))
            model = lgb.LGBMClassifier(
                n_estimators=self.config.max_estimators, **params
            )
        fit_kwargs: dict[str, object] = {
            "eval_set": [(features.validation, validation_y)],
            "categorical_feature": categorical,
        }
        if fit_weight is not None:
            fit_kwargs["sample_weight"] = fit_weight
        if validation_weight is not None:
            fit_kwargs["eval_sample_weight"] = [validation_weight]
        if self.config.early_stopping_rounds > 0:
            fit_kwargs["callbacks"] = [
                lgb.early_stopping(self.config.early_stopping_rounds, verbose=False)
            ]
        model.fit(features.fit, train_y, **fit_kwargs)
        return model, int(model.best_iteration_ or self.config.max_estimators), params

    def fit_predict(self, X_train, y_train, X_val, y_val, X_query):
        import lightgbm as lgb

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
        _, best_iterations, params = self._fit_search(
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
        categorical = self._categorical_columns(final_features.fit.columns)
        if self.task_type == "regression":
            model = lgb.LGBMRegressor(n_estimators=best_iterations, **params)
            final_y = transform_target(self.graph.target, fit_y, fit_y)
        else:
            model = lgb.LGBMClassifier(n_estimators=best_iterations, **params)
            final_y = fit_y
        kwargs: dict[str, object] = {"categorical_feature": categorical}
        if final_weight is not None:
            kwargs["sample_weight"] = final_weight
        model.fit(final_features.fit, final_y, **kwargs)
        self.last_fit_summary = {"best_iterations": best_iterations}
        if self.task_type == "regression":
            return inverse_target(
                self.graph.target, fit_y, model.predict(final_features.query)
            )
        return self._full_probabilities(model, final_features.query)


FeatureGraphModel = LightGBMFeatureGraphModel
