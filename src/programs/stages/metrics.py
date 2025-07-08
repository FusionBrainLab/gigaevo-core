"""Metrics-related stages for MetaEvolve."""

from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Callable, Union, Tuple
import math

from loguru import logger

from src.exceptions import (
    MetaEvolveError,
    StageError,
    ValidationError,
    ensure_not_none,
)
from src.programs.program import Program, ProgramStageResult, StageState, MIN_METRIC_VALUE, MAX_METRIC_VALUE
from src.programs.utils import build_stage_result
from src.programs.stages.base import Stage


class UpdateMetricsStage(Stage):
    """
    Bulletproof stage that updates program metrics from validation output.

    This stage takes the output of a previous validation stage and stores it
    to program.metrics with comprehensive validation and error handling.
    """

    def __init__(
        self,
        prev_stage_name: str,
        required_keys: Optional[List[str]] = None,
        optional_keys: Optional[List[str]] = None,
        validate_types: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.prev_stage_name = ensure_not_none(
            prev_stage_name, "prev_stage_name"
        )
        self.required_keys = set(required_keys) if required_keys else set()
        self.optional_keys = set(optional_keys) if optional_keys else set()
        self.validate_types = validate_types
        self._requires_code = False

    async def _execute_stage(
        self, program: Program, started_at: datetime
    ) -> ProgramStageResult:
        logger.debug(
            f"[{self.stage_name}] Program {program.id}: Updating metrics from {self.prev_stage_name}"
        )

        try:
            # Get previous stage result
            prev_result: Optional[ProgramStageResult] = (
                program.stage_results.get(self.prev_stage_name)
            )
            if not prev_result or not prev_result.is_completed():
                raise StageError(
                    f"Previous stage '{self.prev_stage_name}' did not complete successfully",
                    stage_name=self.stage_name,
                    stage_type="metrics_update",
                )

            # Validate output is a dictionary
            if not isinstance(prev_result.output, dict):
                raise ValidationError(
                    f"Previous stage output must be a dictionary, got {type(prev_result.output).__name__}",
                    field="prev_stage_output",
                    value=prev_result.output,
                )

            metrics_dict = prev_result.output

            # Sanitize / validate
            if self.validate_types:
                metrics_dict, dropped_metrics = clean_metrics(
                    program=program,
                    raw_metrics=metrics_dict,
                    stage_name=self.stage_name,
                )
                if dropped_metrics:
                    logger.debug(
                        f"[{self.stage_name}] Dropped {len(dropped_metrics)} metrics due to type validation"
                    )

            # Validate required keys
            if self.required_keys:
                missing_keys = self.required_keys - set(metrics_dict.keys())
                if missing_keys:
                    raise ValidationError(
                        f"Validator output missing required metrics: {', '.join(sorted(missing_keys))}",
                        field="required_metrics",
                        value=missing_keys,
                    )

            # Restrict to expected keys
            filtered_metrics = filter_allowed_keys(
                metrics=metrics_dict,
                allowed=self.required_keys | self.optional_keys,
                stage_name=self.stage_name,
            )

            # Update program metrics
            program.add_metrics(filtered_metrics)

            logger.debug(
                f"[{self.stage_name}] Program {program.id}: Updated {len(filtered_metrics)} metrics"
            )

            return build_stage_result(
                status=StageState.COMPLETED,
                started_at=started_at,
                output={
                    "stored_metrics": True,
                    "metrics_count": len(filtered_metrics),
                    "metrics_keys": list(filtered_metrics.keys()),
                },
            )

        except MetaEvolveError:
            raise
        except Exception as e:
            raise StageError(
                f"Metrics update failed: {e}",
                stage_name=self.stage_name,
                stage_type="metrics_update",
                cause=e,
            )


class FactoryMetricsStage(Stage):
    """
    Metrics stage similar to UpdateMetricsStage but uses a factory function 
    to provide default metrics when the previous stage fails or is missing.
    
    This stage always runs (using execution_order_deps) and always populates
    program.metrics with either real metrics from validation or factory defaults.
    """

    def __init__(
        self,
        prev_stage_name: str,
        metrics_factory: Union[Dict[str, Any], Callable[[], Dict[str, Any]]],
        required_keys: Optional[List[str]] = None,
        optional_keys: Optional[List[str]] = None,
        validate_types: bool = True,
        prefer_factory_on_partial: bool = False,
        prefer_factory_on_invalid_type: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.prev_stage_name = ensure_not_none(
            prev_stage_name, "prev_stage_name"
        )
        self.metrics_factory = metrics_factory
        self.required_keys = set(required_keys) if required_keys else set()
        self.optional_keys = set(optional_keys) if optional_keys else set()
        self.validate_types = validate_types
        self.prefer_factory_on_partial = prefer_factory_on_partial
        self.prefer_factory_on_invalid_type = prefer_factory_on_invalid_type
        self._requires_code = False

    async def _execute_stage(
        self, program: Program, started_at: datetime
    ) -> ProgramStageResult:
        logger.debug(
            f"[{self.stage_name}] Program {program.id}: Updating metrics from {self.prev_stage_name} with factory fallback"
        )

        try:
            metrics_dict = None
            metrics_source = "factory"  # Default assumption
            
            # Try to get metrics from previous stage
            prev_result: Optional[ProgramStageResult] = (
                program.stage_results.get(self.prev_stage_name)
            )
            
            if prev_result and prev_result.is_completed() and isinstance(prev_result.output, dict):
                # Previous stage succeeded and has dictionary output
                candidate_metrics = prev_result.output
                # Sanitize candidate metrics
                if self.validate_types and candidate_metrics:
                    candidate_metrics, dropped_metrics = clean_metrics(
                        program=program,
                        raw_metrics=candidate_metrics,
                        stage_name=self.stage_name,
                    )
                    if dropped_metrics:
                        logger.debug(
                            f"[{self.stage_name}] Dropped {len(dropped_metrics)} metrics due to type validation"
                        )
                        if self.prefer_factory_on_invalid_type:
                            metrics_dict = None
                            metrics_source = "factory_invalid"
                
                # Check if we have required keys (if specified)
                if self.required_keys:
                    missing_keys = self.required_keys - set(candidate_metrics.keys())
                    if missing_keys:
                        logger.debug(
                            f"[{self.stage_name}] Previous stage missing required keys {missing_keys}, using factory"
                        )
                        if self.prefer_factory_on_partial:
                            metrics_dict = None  # Will use factory
                        else:
                            # Use partial metrics and let factory fill gaps later
                            metrics_dict = candidate_metrics
                            metrics_source = "partial"
                    else:
                        # All required keys present
                        metrics_dict = candidate_metrics
                        metrics_source = "previous_stage"
                else:
                    # No required keys specified, use whatever we got
                    metrics_dict = candidate_metrics
                    metrics_source = "previous_stage"
            else:
                logger.debug(
                    f"[{self.stage_name}] Previous stage '{self.prev_stage_name}' failed or missing, using factory"
                )

            # Get factory metrics if needed
            factory_metrics = None
            if metrics_dict is None or metrics_source == "partial":
                try:
                    if callable(self.metrics_factory):
                        factory_metrics = self.metrics_factory()
                    else:
                        factory_metrics = self.metrics_factory.copy()
                    
                    if not isinstance(factory_metrics, dict):
                        raise ValidationError(
                            f"Factory must return a dictionary, got {type(factory_metrics).__name__}",
                            field="metrics_factory",
                            value=factory_metrics,
                        )
                        
                except Exception as e:
                    raise StageError(
                        f"Factory function failed: {e}",
                        stage_name=self.stage_name,
                        stage_type="metrics_factory",
                        cause=e,
                    )

            # Combine metrics based on source
            if metrics_dict is None:
                # Use factory entirely
                final_metrics = factory_metrics
                metrics_source = "factory"
            elif metrics_source == "partial":
                # Merge factory defaults with partial metrics
                final_metrics = factory_metrics.copy()
                final_metrics.update(metrics_dict)
                metrics_source = "merged"
            else:
                # Use previous stage metrics
                final_metrics = metrics_dict

            # Sanitize final metrics
            if self.validate_types:
                final_metrics, dropped_metrics = clean_metrics(
                    program=program,
                    raw_metrics=final_metrics,
                    stage_name=self.stage_name,
                )
                if dropped_metrics:
                    logger.debug(
                        f"[{self.stage_name}] Dropped {len(dropped_metrics)} metrics due to type validation"
                    )

            # Restrict to expected keys
            filtered_metrics = filter_allowed_keys(
                metrics=final_metrics,
                allowed=self.required_keys | self.optional_keys,
                stage_name=self.stage_name,
            )

            # Update program metrics
            program.add_metrics(filtered_metrics)

            logger.debug(
                f"[{self.stage_name}] Program {program.id}: Updated {len(filtered_metrics)} metrics from {metrics_source}"
            )

            return build_stage_result(
                status=StageState.COMPLETED,
                started_at=started_at,
                output={
                    "stored_metrics": True,
                    "metrics_count": len(filtered_metrics),
                    "metrics_keys": list(filtered_metrics.keys()),
                    "metrics_source": metrics_source,
                },
            )

        except MetaEvolveError:
            raise
        except Exception as e:
            raise StageError(
                f"Factory metrics update failed: {e}",
                stage_name=self.stage_name,
                stage_type="metrics_factory",
                cause=e,
            )


# ---------------------------------------------------------------------------
# Helper utilities (shared by all metric stages)
# ---------------------------------------------------------------------------


def clean_metrics(
    *,
    program: Program,
    raw_metrics: Dict[str, Any],
    stage_name: str,
    stringify_unsupported: bool = True,
    drop_non_finite: bool = True,
    metadata_key: str = "dropped_metrics",
) -> Tuple[Dict[str, Any], List[str]]:
    """Sanitise a metrics dictionary.

    1. Converts unsupported value types (non-primitive) to string when
       *stringify_unsupported* is True.
    2. Drops NaN/±Inf when *drop_non_finite* is True so downstream code never
       sees non-finite numbers.
    3. Records the list of dropped keys in *program.metadata[metadata_key]* so
       debugging or filtering logic can inspect them later.

    Returns the cleaned metrics and list of keys that were removed.
    """

    cleaned: Dict[str, Any] = dict(raw_metrics)  # shallow copy
    dropped: List[str] = []

    for key in list(cleaned.keys()):
        val = cleaned[key]

        # 1. Unsupported types
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

        # 2. Non-finite floats
        if isinstance(val, float) and not math.isfinite(val):
            if drop_non_finite:
                logger.warning(
                    f"[{stage_name}] Dropping non-finite metric '{key}': {val}"
                )
                cleaned.pop(key)
                dropped.append(key)
            else:
                # Clamp to bounds if we keep them
                clamped = MAX_METRIC_VALUE if val > 0 else MIN_METRIC_VALUE
                cleaned[key] = clamped
                logger.warning(
                    f"[{stage_name}] Clamped non-finite metric '{key}' -> {clamped}"
                )

    # Record dropped keys for traceability
    if dropped:
        try:
            existing: List[str] = program.get_metadata(metadata_key) or []
            program.set_metadata(metadata_key, sorted(set(existing) | set(dropped)))
        except Exception as exc:
            logger.debug(f"[{stage_name}] Failed to persist {metadata_key}: {exc}")

    return cleaned, dropped


def filter_allowed_keys(
    *,
    metrics: Dict[str, Any],
    allowed: Set[str],
    stage_name: str,
) -> Dict[str, Any]:
    """Return subset of *metrics* restricted to *allowed*; if *allowed* empty
    the original dict is returned."""

    if not allowed:
        return dict(metrics)

    filtered: Dict[str, Any] = {}
    for k, v in metrics.items():
        if k in allowed:
            filtered[k] = v
        else:
            logger.debug(f"[{stage_name}] Ignoring unexpected metric key: {k}")
    return filtered
