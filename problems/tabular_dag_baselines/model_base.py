"""Shared estimator-boundary helpers for FeatureGraph baselines."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from problems.dag_tab.execution import GraphTriplet
from problems.dag_tab.validate import FeatureGraphModel as _CatBoostFeatureGraphModel


def _category_key(value: object) -> tuple[str, str]:
    return type(value).__qualname__, str(value)


class BaselineFeatureGraphModel(_CatBoostFeatureGraphModel):
    """CatBoost-independent helpers built on the canonical graph executor."""

    estimator_name = "baseline"
    supports_sample_weight = False

    def _categorical_columns(self, columns: Sequence[str]) -> list[str]:
        return [
            column
            for column in columns
            if self._feature_kind(column) in {"categorical", "binary"}
        ]

    def _prepare_model_triplet(
        self,
        triplet: GraphTriplet,
        *,
        impute_numeric: bool = False,
    ) -> GraphTriplet:
        """Produce aligned frames with fit-local categories and optional medians."""

        frames = [triplet.fit.copy(), triplet.validation.copy(), triplet.query.copy()]
        expected = list(triplet.fit.columns)
        for label, frame in zip(("fit", "validation", "query"), frames):
            if list(frame.columns) != expected:
                raise ValueError(f"{label} feature columns differ from fit columns")

        categorical = set(self._categorical_columns(expected))
        for column in expected:
            if column in categorical:
                observed: list[str] = []
                seen: set[tuple[str, str]] = set()
                for value in triplet.fit[column].tolist():
                    if pd.isna(value):
                        continue
                    key = _category_key(value)
                    if key not in seen:
                        seen.add(key)
                        observed.append(f"{key[0]}:{key[1]}")
                categories = [*observed, "__MISSING__", "__UNKNOWN__"]
                fit_keys = set(observed)

                def encode(value: object) -> str:
                    if pd.isna(value):
                        return "__MISSING__"
                    key = _category_key(value)
                    token = f"{key[0]}:{key[1]}"
                    return token if token in fit_keys else "__UNKNOWN__"

                for frame in frames:
                    frame[column] = pd.Categorical(
                        frame[column].map(encode), categories=categories
                    )
                continue

            arrays: list[np.ndarray] = []
            for label, frame in zip(("fit", "validation", "query"), frames):
                values = pd.to_numeric(frame[column], errors="raise").to_numpy(
                    dtype=np.float64
                )
                if np.isinf(values).any():
                    raise ValueError(
                        f"{label} numerical feature {column!r} contains infinity"
                    )
                arrays.append(values)
            if impute_numeric:
                fit_values = arrays[0]
                median = (
                    float(np.nanmedian(fit_values))
                    if np.isfinite(fit_values).any()
                    else 0.0
                )
                arrays = [
                    np.where(np.isnan(values), median, values) for values in arrays
                ]
            for frame, values in zip(frames, arrays):
                frame[column] = values
        return GraphTriplet(*frames)

    def _strip_sample_weights(
        self, triplet: GraphTriplet
    ) -> tuple[GraphTriplet, np.ndarray | None, np.ndarray | None]:
        stripped, fit_weight, validation_weight = self._extract_sample_weights(triplet)
        if not self.supports_sample_weight and fit_weight is not None:
            raise ValueError(
                f"sample_weight is not supported by the fixed {self.estimator_name} baseline"
            )
        return stripped, fit_weight, validation_weight

    def _validate_classification_labels(self, *targets: np.ndarray) -> None:
        if self.n_classes is None or self.n_classes < 2:
            raise ValueError("classification dataset must declare n_classes >= 2")
        arrays = [
            np.asarray(target, dtype=int) for target in targets if target is not None
        ]
        if not arrays:
            raise ValueError("classification requires at least one target array")
        observed = np.unique(np.concatenate(arrays))
        if np.any(observed < 0) or np.any(observed >= int(self.n_classes)):
            raise ValueError(
                f"classification labels {observed.tolist()} fall outside declared "
                f"class universe [0, {int(self.n_classes)})"
            )

    def _full_probabilities(self, model, features) -> np.ndarray:
        probabilities = np.asarray(model.predict_proba(features), dtype=float)
        classes = np.asarray(model.classes_, dtype=int)
        if probabilities.ndim == 1:
            probabilities = np.column_stack([1.0 - probabilities, probabilities])
        full = np.zeros((len(features), int(self.n_classes)), dtype=float)
        full[:, classes] = probabilities
        return full
