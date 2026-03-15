# gigaevo/programs/stages/metrics_stages.py
from __future__ import annotations

import math
from typing import Callable, Optional

from loguru import logger

from gigaevo.programs.core_types import StageIO, VoidInput
from gigaevo.programs.metrics.context import MetricsContext
from gigaevo.programs.program import Program
from gigaevo.programs.stages.base import Stage
from gigaevo.programs.stages.common import FloatDictContainer
from gigaevo.programs.stages.stage_registry import StageRegistry


class EnsureMetricsInputs(StageIO):
    """
    Optional candidate metrics coming from a prior stage.
    If absent, a factory will be used.
    """

    candidate: Optional[FloatDictContainer]


@StageRegistry.register(
    description="Populate & validate metrics; coerce to float and clamp to bounds"
)
class EnsureMetricsStage(Stage):
    """
    - Reads candidate metrics from DAG input (optional).
    - Falls back to metrics_factory when absent.
    - Coerces to float, ensures finiteness, clamps using MetricsContext bounds,
      while respecting sentinel values (not clamped).
    - Stores validated metrics on Program and returns them.
    """

    InputsModel = EnsureMetricsInputs
    OutputModel = FloatDictContainer

    def __init__(
        self,
        *,
        metrics_factory: dict[str, float] | Callable[[], dict[str, float]],
        metrics_context: MetricsContext,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.metrics_factory = metrics_factory
        self.ctx = metrics_context
        self.required_keys = set(self.ctx.specs.keys())

    async def compute(self, program: Program) -> StageIO:
        # Write sentinel fallbacks immediately as a safety net.  If candidate
        # processing raises below, the sentinel values are already on the program
        # and will be persisted by the DAG's error-handling path.  This ensures
        # the program always has all required metric keys (including the primary
        # fitness key) even when upstream stages fail.
        sentinel_metrics = self._get_factory_metrics()
        program.add_metrics(sentinel_metrics)

        metrics_input = (
            self.params.candidate.data
            if self.params.candidate is not None
            else sentinel_metrics
        )

        final_metrics = self._process_metrics(metrics_input)
        program.add_metrics(final_metrics)

        logger.debug(
            "[{}] Stored {} validated metrics on program {}",
            type(self).__name__,
            len(final_metrics),
            program.id[:8],
        )
        return FloatDictContainer(data=final_metrics)

    def _get_factory_metrics(self) -> dict[str, float]:
        metrics = (
            self.metrics_factory()
            if callable(self.metrics_factory)
            else dict(self.metrics_factory)
        )
        return metrics

    def _process_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
        missing = [k for k in self.required_keys if metrics.get(k) is None]
        if missing:
            raise ValueError(f"Missing required metric keys: {missing}")

        out: dict[str, float] = {}
        for k in self.required_keys:
            out[k] = self._coerce_and_clamp(k, metrics.get(k))
        return out

    def _coerce_and_clamp(self, key: str, value: float) -> float:
        # Attempt numeric coercion before the isfinite check so that string
        # values (e.g. "0.85") produce a clear ValueError pointing back to the
        # metric key, not a cryptic TypeError from math.isfinite().
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Metric '{key}' must be numeric, "
                f"got {type(value).__name__!r}: {value!r}"
            ) from exc
        if not math.isfinite(value):
            raise ValueError(f"Metric '{key}' must be finite, got {value}")

        spec = self.ctx.specs.get(key)
        if spec is None:
            # No spec in context → keep value as-is
            return value

        # Sentinel values are preserved (no clamping)
        if spec.is_sentinel(value):
            return value

        bounds = self.ctx.get_bounds(key)
        if bounds is None:
            return value

        lo, hi = bounds
        if lo is not None and value < lo:
            value = lo
        if hi is not None and value > hi:
            value = hi
        return value


@StageRegistry.register(
    description="Normalize metrics to [0,1] using bounds & orientation; optional aggregate"
)
class NormalizeMetricsStage(Stage):
    """
    For each metric that has finite (lo, hi) in MetricsContext:
      norm = clamp01((value - lo) / (hi - lo)), flipped if higher_is_better=False.
    Optionally emits an aggregate mean under `aggregate_key` if at least one normalized metric exists.
    """

    InputsModel = VoidInput
    OutputModel = FloatDictContainer

    def __init__(
        self,
        *,
        metrics_context: MetricsContext,
        aggregate_key: str = "normalized_score",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ctx = metrics_context
        self.aggregate_key = aggregate_key

    async def compute(self, program: Program) -> StageIO:
        normalized: dict[str, float] = {}

        for key, _spec in self.ctx.specs.items():
            bounds = self.ctx.get_bounds(key)
            if bounds is None:
                continue
            lo, hi = bounds
            if lo is None or hi is None or hi <= lo:
                continue

            v = program.metrics.get(key)
            if v is None:
                continue

            # Sentinel values carry the "failed run" signal. Normalising them would
            # map -1e5 → 0.0, making a failed run indistinguishable from a zero-score
            # run. Skip normalization so the sentinel is preserved downstream.
            if _spec.is_sentinel(v):
                continue

            ratio = (v - lo) / (hi - lo)
            # clamp
            ratio = 0.0 if ratio < 0.0 else 1.0 if ratio > 1.0 else ratio
            if not self.ctx.is_higher_better(key):
                ratio = 1.0 - ratio

            normalized[f"{key}_norm"] = ratio

        # Optional aggregate
        result = dict(normalized)
        if normalized:
            result[self.aggregate_key] = sum(normalized.values()) / len(normalized)

        program.add_metrics(result)
        return FloatDictContainer(data=result)
