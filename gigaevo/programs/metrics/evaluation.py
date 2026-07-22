"""Standard evaluator-reported uncertainty stored on programs.

Validators may place a ``_evaluation_measurements`` mapping in their artifact.
The artifact-aware validator stage validates and removes that reserved namespace,
then stores the normalized mapping on ``Program.metadata``.  Downstream outcome
code can therefore consume evaluator uncertainty without knowing whether it came
from cross-validation, repeated runs, a bootstrap, or an analytic calculation.
"""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

EVALUATION_MEASUREMENTS_ARTIFACT_KEY = "_evaluation_measurements"
EVALUATION_MEASUREMENTS_METADATA_KEY = "evaluation_measurements"


class EvaluationMeasurement(BaseModel):
    """One metric's point estimate and evaluator-reported uncertainty.

    Artifact input omits ``value`` because the authoritative point estimate is
    already present in the validator's metrics dictionary.  The routing stage
    binds that value before persisting the record on the program.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        str_strip_whitespace=True,
    )

    value: float | None = None
    se: float | None = Field(default=None, ge=0.0)
    sample_sd: float | None = Field(default=None, ge=0.0)
    n: int | None = Field(default=None, ge=1)
    method: str = Field(min_length=1)

    @model_validator(mode="after")
    def _uncertainty_contract(self) -> EvaluationMeasurement:
        if (self.se is None) == (self.sample_sd is None):
            raise ValueError("exactly one of se or sample_sd must be provided")
        if self.sample_sd is not None and (self.n is None or self.n < 2):
            raise ValueError("sample_sd requires n >= 2")
        return self

    @property
    def standard_error(self) -> float:
        """Return uncertainty normalized to a standard error."""

        if self.se is not None:
            return self.se
        if self.sample_sd is None or self.n is None:  # guarded by validation
            raise RuntimeError("invalid sample standard-deviation measurement")
        return self.sample_sd / math.sqrt(self.n)

    def bind(self, value: float) -> EvaluationMeasurement:
        """Attach the authoritative metric value before durable storage."""

        return self.model_copy(update={"value": float(value)})


def normalize_evaluation_measurements(
    metrics: dict[str, float], raw: Any
) -> dict[str, dict[str, Any]]:
    """Validate an artifact measurement mapping and bind metric values."""

    if not isinstance(raw, dict):
        raise ValueError(
            f"{EVALUATION_MEASUREMENTS_ARTIFACT_KEY} must be a metric mapping"
        )
    normalized: dict[str, dict[str, Any]] = {}
    for metric_key, payload in raw.items():
        if not isinstance(metric_key, str) or not metric_key:
            raise ValueError("evaluation measurement keys must be non-empty strings")
        if metric_key not in metrics:
            raise ValueError(
                f"evaluation measurement references missing metric {metric_key!r}"
            )
        try:
            metric_value = float(metrics[metric_key])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"evaluation measurement metric {metric_key!r} is not numeric"
            ) from exc
        if not math.isfinite(metric_value):
            raise ValueError(
                f"evaluation measurement metric {metric_key!r} is not finite"
            )
        measurement = EvaluationMeasurement.model_validate(payload).bind(metric_value)
        normalized[metric_key] = measurement.model_dump(mode="json", exclude_none=True)
    return normalized


def reported_standard_error(
    raw_measurements: Any,
    *,
    metric_key: str,
    expected_value: float,
) -> float | None:
    """Read one coherent stored measurement, or return unknown uncertainty."""

    if not isinstance(raw_measurements, dict):
        return None
    payload = raw_measurements.get(metric_key)
    if not isinstance(payload, dict):
        return None
    try:
        measurement = EvaluationMeasurement.model_validate(payload)
    except Exception:
        return None
    if measurement.value is None or not math.isclose(
        measurement.value,
        expected_value,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        return None
    standard_error = measurement.standard_error
    return standard_error if math.isfinite(standard_error) else None
