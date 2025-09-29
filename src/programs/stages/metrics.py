"""Metrics-related stages for MetaEvolve.

This module contains small, focused stages for handling metrics:
- EnsureMetricsStage: populate/validate metrics and enforce basic invariants
- NormalizeMetricsStage: compute normalized variants and a simple aggregate
"""

from datetime import datetime
import math
from typing import Any, Callable

from loguru import logger

from src.exceptions import StageError
from src.programs.program import Program, ProgramStageResult, StageState
from src.programs.stages.base import Stage
from src.programs.utils import build_stage_result
from src.programs.metrics.context import MetricsContext
from src.runner.stage_registry import StageRegistry


@StageRegistry.register(
    description="Populate and validate metrics for a program"
)
class EnsureMetricsStage(Stage):
    """Populate and validate metrics for a program.

    The stage will:
    - Read metrics from a previous stage's output (a dict)
    - Fallback to a factory when missing/incomplete
    - Coerce values to float, ensure finiteness, and clamp to bounds
    - Store results on the program
    """

    def __init__(
        self,
        metrics_factory: dict[str, Any] | Callable[[], dict[str, Any]],
        metrics_context: MetricsContext,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.metrics_factory = metrics_factory
        # Always derive required keys from metrics context
        self.required_keys = set(metrics_context.specs.keys())
        self.metrics_context = metrics_context

    @classmethod
    def mandatory_inputs(cls) -> list[str]:
        return []

    @classmethod
    def optional_inputs(cls) -> list[str]:
        return ["validation_result"]


    async def _execute_stage(
        self, program: Program, started_at: datetime
    ) -> ProgramStageResult:
        # Get validation result by semantic name, fallback to factory if not available
        validation_data = self.get_input_optional("validation_result")
        if validation_data is not None and isinstance(validation_data, dict):
            metrics_input, src = validation_data, "validation_stage"
        else:
            metrics_input, src = self._get_factory_metrics(), "factory"

        final_metrics = self._process_metrics(metrics_input)
        program.add_metrics(final_metrics)


        logger.debug(
            f"[{self.stage_name}] Program {program.id}: Updated {len(final_metrics)} metrics from {src}"
        )

        return build_stage_result(
            status=StageState.COMPLETED,
            started_at=started_at,
            output={
                "stored_metrics": True,
                "metrics_count": len(final_metrics),
                "metrics_keys": list(final_metrics.keys()),
                "metrics_source": src,
            },
        )

    def _has_required_keys(self, metrics: dict[str, Any]) -> bool:
        """Check that all required keys are present in the given mapping."""
        if not self.required_keys:
            return True
        return len(self.required_keys - set(metrics.keys())) == 0

    def _get_factory_metrics(self) -> dict[str, Any]:
        """Safely call/copy the metrics factory and return a dict."""
        metrics = (
            self.metrics_factory()
            if callable(self.metrics_factory)
            else dict(self.metrics_factory)
        )
        if not isinstance(metrics, dict):
            raise StageError(
                f"Factory must return a dictionary, got {type(metrics).__name__}",
                stage_name=self.stage_name,
                stage_type="metrics_factory",
            )
        return metrics

    def _coerce_and_clamp(self, key: str, raw_value: Any) -> float:
        """Convert to float, ensure finiteness, and clamp to bounds for a single metric."""
        try:
            value = float(raw_value)
        except Exception as exc:
            raise StageError(
                f"Metric '{key}' must be numeric, got {type(raw_value).__name__}",
                stage_name=self.stage_name,
                stage_type="metrics_type",
            ) from exc

        if not math.isfinite(value):
            raise StageError(
                f"Metric '{key}' must be finite, got {value}",
                stage_name=self.stage_name,
                stage_type="metrics_finiteness",
            )

        # Only apply bounds if the metric is defined in the context
        if (bounds := self.metrics_context.get_bounds(key)) is not None:
            lo, hi = bounds
            if lo is not None and value < lo:
                value = lo
            if hi is not None and value > hi:
                value = hi
        return value

    def _process_metrics(self, metrics: dict[str, Any]) -> dict[str, float]:
        """Select required keys and normalize each value via _coerce_and_clamp."""
        required = list(self.required_keys)
        available = list(metrics.keys())
        logger.debug(
            f"[{self.stage_name}] Required={required} | available={available}"
        )

        # Presence check first for clear error messages
        missing = [k for k in required if metrics.get(k) is None]
        if missing:
            raise StageError(
                f"Missing required metric keys: {missing}",
                stage_name=self.stage_name,
                stage_type="metrics_required",
            )

        # Coerce and clamp
        return {k: self._coerce_and_clamp(k, metrics.get(k)) for k in required}


@StageRegistry.register(
    description="Compute normalized metrics in [0,1] using bounds and orientation"
)
class NormalizeMetricsStage(Stage):
    """Compute normalized metrics in [0,1] using bounds and orientation.

    - For each metric with both bounds present, compute (value - lo) / (hi - lo)
      then flip if higher_is_better is False. Clamp to [0,1].
    - Optionally compute an aggregate score as the mean of normalized metrics.
    """

    def __init__(
        self,
        metrics_context: MetricsContext,
        aggregate_key: str = "normalized_score",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.metrics_context = metrics_context
        self.aggregate_key = aggregate_key

    async def _execute_stage(
        self, program: Program, started_at: datetime
    ) -> ProgramStageResult:
        normalized: dict[str, float] = {}
        for key, _spec in self.metrics_context.specs.items():
            if (bounds := self.metrics_context.get_bounds(key)) is None:
                continue
            lo, hi = bounds
            if lo is None or hi is None or hi <= lo:
                continue
            raw = program.metrics.get(key)
            try:
                v = float(raw)
            except Exception:
                continue
            ratio = (v - lo) / (hi - lo)
            ratio = max(0.0, min(1.0, ratio))
            if not self.metrics_context.is_higher_better(key):
                ratio = 1.0 - ratio
            normalized[f"{key}_norm"] = ratio

        if normalized:
            # Simple aggregate = mean of normalized values
            aggregate = sum(normalized.values()) / len(normalized)
            program.add_metrics({**normalized, self.aggregate_key: aggregate})
        else:
            program.add_metrics({})

        return build_stage_result(
            status=StageState.COMPLETED,
            started_at=started_at,
            output={
                "normalized_keys": list(normalized.keys()),
                "aggregate": program.metrics.get(self.aggregate_key),
            },
        )

    # NormalizeMetricsStage intentionally contains only normalization logic
