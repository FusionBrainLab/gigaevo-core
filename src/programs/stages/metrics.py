"""Metrics-related stages for MetaEvolve."""

from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Callable, Union, Tuple
import math

from loguru import logger

from src.exceptions import (
    MetaEvolveError,
    StageError,
    ValidationError,
)
from src.programs.program import Program, ProgramStageResult, StageState, MIN_METRIC_VALUE, MAX_METRIC_VALUE
from src.programs.utils import build_stage_result
from src.programs.stages.base import Stage


class FactoryMetricsStage(Stage):
    """Metrics stage that updates program metrics from validation output with factory fallback."""

    def __init__(
        self,
        stage_to_extract_metrics: str,
        metrics_factory: Union[Dict[str, Any], Callable[[], Dict[str, Any]]],
        required_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.stage_to_extract_metrics = stage_to_extract_metrics
        self.metrics_factory = metrics_factory
        self.required_keys = set(required_keys) if required_keys else set()
        self._requires_code = False

    async def _execute_stage(
        self, program: Program, started_at: datetime
    ) -> ProgramStageResult:
        logger.debug(
            f"[{self.stage_name}] Program {program.id}: Updating metrics from {self.stage_to_extract_metrics}"
        )

        try:
            metrics_dict, source = self._get_metrics(program)
            final_metrics = self._process_metrics(program, metrics_dict)
            program.add_metrics(final_metrics)

            logger.debug(
                f"[{self.stage_name}] Program {program.id}: Updated {len(final_metrics)} metrics from {source}"
            )

            return build_stage_result(
                status=StageState.COMPLETED,
                started_at=started_at,
                output={
                    "stored_metrics": True,
                    "metrics_count": len(final_metrics),
                    "metrics_keys": list(final_metrics.keys()),
                    "metrics_source": source,
                },
            )

        except MetaEvolveError:
            raise
        except Exception as e:
            raise StageError(
                f"Metrics update failed: {e}",
                stage_name=self.stage_name,
                stage_type="metrics_factory",
                cause=e,
            )

    def _get_metrics(self, program: Program) -> Tuple[Dict[str, Any], str]:
        """Get metrics from previous stage or factory. Returns (metrics_dict, source)."""
        prev_result = program.stage_results.get(self.stage_to_extract_metrics)
        
        if prev_result and prev_result.is_completed() and isinstance(prev_result.output, dict):
            metrics = prev_result.output
            if self._has_required_keys(metrics):
                return metrics, "previous_stage"
            else:
                logger.debug(f"[{self.stage_name}] Previous stage missing required keys, using factory")
        
        factory_metrics = self._get_factory_metrics()
        return factory_metrics, "factory"

    def _has_required_keys(self, metrics: Dict[str, Any]) -> bool:
        """Check if metrics has all required keys."""
        if not self.required_keys:
            return True
        missing = self.required_keys - set(metrics.keys())
        return len(missing) == 0

    def _get_factory_metrics(self) -> Dict[str, Any]:
        """Get metrics from factory function or dict."""
        try:
            if callable(self.metrics_factory):
                metrics = self.metrics_factory()
            else:
                metrics = self.metrics_factory.copy()
            
            if not isinstance(metrics, dict):
                raise ValidationError(
                    f"Factory must return a dictionary, got {type(metrics).__name__}",
                    field="metrics_factory",
                    value=metrics,
                )
            return metrics
            
        except Exception as e:
            raise StageError(
                f"Factory function failed: {e}",
                stage_name=self.stage_name,
                stage_type="metrics_factory",
                cause=e,
            )

    def _process_metrics(self, program: Program, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and filter metrics to only include required keys."""
        # Clean metrics (handle unsupported types and non-finite values)
        metrics, dropped = clean_metrics(
            program=program,
            raw_metrics=metrics,
            stage_name=self.stage_name,
        )
        if dropped:
            logger.debug(f"[{self.stage_name}] Dropped {len(dropped)} metrics due to type validation")

        # Filter to only required keys
        return filter_required_keys(
            metrics=metrics,
            required=self.required_keys,
            stage_name=self.stage_name,
        )


def clean_metrics(
    *,
    program: Program,
    raw_metrics: Dict[str, Any],
    stage_name: str,
    stringify_unsupported: bool = True,
    drop_non_finite: bool = True,
    metadata_key: str = "dropped_metrics",
) -> Tuple[Dict[str, Any], List[str]]:
    """Sanitize metrics dictionary by handling unsupported types and non-finite values."""

    cleaned = dict(raw_metrics)
    dropped = []

    for key in list(cleaned.keys()):
        val = cleaned[key]

        if not isinstance(val, (int, float, str, bool, type(None))):
            if stringify_unsupported:
                logger.warning(
                    f"[{stage_name}] Metric '{key}' has unsupported type {type(val).__name__}; converting to str"
                )
                cleaned[key] = str(val)
            else:
                logger.warning(f"[{stage_name}] Dropping unsupported metric '{key}'")
                cleaned.pop(key)
                dropped.append(key)
            continue

        if isinstance(val, float) and not math.isfinite(val):
            if drop_non_finite:
                logger.warning(f"[{stage_name}] Dropping non-finite metric '{key}': {val}")
                cleaned.pop(key)
                dropped.append(key)
            else:
                clamped = MAX_METRIC_VALUE if val > 0 else MIN_METRIC_VALUE
                cleaned[key] = clamped
                logger.warning(f"[{stage_name}] Clamped non-finite metric '{key}' -> {clamped}")

    if dropped:
        try:
            existing = program.get_metadata(metadata_key) or []
            program.set_metadata(metadata_key, sorted(set(existing) | set(dropped)))
        except Exception as exc:
            logger.debug(f"[{stage_name}] Failed to persist {metadata_key}: {exc}")

    return cleaned, dropped


def filter_required_keys(
    *,
    metrics: Dict[str, Any],
    required: Set[str],
    stage_name: str,
) -> Dict[str, Any]:
    """Return subset of metrics restricted to required keys only."""

    if not required:
        return dict(metrics)

    filtered = {}
    for k, v in metrics.items():
        if k in required:
            filtered[k] = v
        else:
            logger.debug(f"[{stage_name}] Ignoring metric key not in required set: {k}")
    return filtered
