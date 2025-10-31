from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator

__all__ = ["MetricSpec", "MetricsContext"]

# Constants
MAX_VALUE_DEFAULT: float = 1e5
MIN_VALUE_DEFAULT: float = -1e5
EPSILON: float = 1e-6
VALIDITY_KEY: str = "is_valid"
DEFAULT_DECIMALS: int = 5


class MetricSpec(BaseModel):
    description: str
    decimals: int = DEFAULT_DECIMALS
    is_primary: bool = False
    higher_is_better: bool
    unit: str | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None
    include_in_prompts: bool = True
    significant_change: float | None = None
    sentinel_value: float | None = None

    @model_validator(mode="after")
    def _set_default_sentinel_value(self) -> MetricSpec:
        """Set default sentinel value based on optimization direction."""
        if self.sentinel_value is None:
            self.sentinel_value = (
                MIN_VALUE_DEFAULT if self.higher_is_better else MAX_VALUE_DEFAULT
            )
        return self

    @model_validator(mode="after")
    def _validate_sentinel_bounds(self) -> MetricSpec:
        """Validate that sentinel value is outside the bounds interval (exclusive)."""
        if (
            self.lower_bound is not None
            and self.upper_bound is not None
            and self.sentinel_value is not None
        ):
            if self.lower_bound < self.sentinel_value < self.upper_bound:
                raise ValueError(
                    f"Sentinel value {self.sentinel_value} must be outside bounds "
                    f"[{self.lower_bound}, {self.upper_bound}]"
                )
        return self

    def is_sentinel(self, value: float) -> bool:
        """Check if a value is the sentinel value for this metric."""
        return (
            self.sentinel_value is not None
            and abs(value - self.sentinel_value) < EPSILON
        )


class MetricsContext(BaseModel):
    """Centralized definition of metrics and their properties.

    Holds primary optimization metric and any additional metrics that may be
    displayed in prompts. Provides consistent access to descriptions and
    formatting preferences.
    """

    # All metric specs keyed by metric name. Must include exactly one primary metric.
    specs: dict[str, MetricSpec] = Field(default_factory=dict)

    # Optional explicit display order. If empty, order is: primary first, then others sorted by key.
    display_order: list[str] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def _validate_primary_spec(self) -> MetricsContext:
        """Validate exactly one primary metric exists."""
        primary_specs = [s for s in self.specs.values() if s.is_primary]
        if len(primary_specs) != 1:
            raise ValueError(
                f"Exactly one MetricSpec must have is_primary=True, found {len(primary_specs)}"
            )
        return self

    def get_primary_spec(self) -> MetricSpec:
        """Get the MetricSpec for the primary metric.

        Returns:
            The MetricSpec marked as primary
        """
        for spec in self.specs.values():
            if spec.is_primary:
                return spec

    def get_primary_key(self) -> str:
        """Get the key of the primary metric.

        Returns:
            The metric key marked as primary
        """
        for key, spec in self.specs.items():
            if spec.is_primary:
                return key

    def get_description(self, key: str) -> str:
        """Get the description for a metric.

        Args:
            key: The metric key

        Returns:
            The metric description
        """
        return self.specs[key].description

    def get_decimals(self, key: str) -> int:
        """Get the decimal precision for a metric.

        Args:
            key: The metric key

        Returns:
            Number of decimal places to display

        """
        return self.specs[key].decimals

    def metrics_descriptions(self) -> dict[str, str]:
        """Return mapping of metric key -> description for all known metrics."""
        return {k: v.description for k, v in self.specs.items()}

    def prompt_keys(self) -> list[str]:
        """Return ordered metric keys intended for prompts.

        Order rules:
        - If display_order is set, use it (filtered to known specs)
        - Else: primary first, then others sorted
        - Always filter by include_in_prompts flag
        """
        if self.display_order:
            ordered = [k for k in self.display_order if k in self.specs]
        else:
            primary_key = self.get_primary_key()
            remaining = [k for k in sorted(self.specs.keys()) if k != primary_key]
            ordered = [primary_key] + remaining
        return [k for k in ordered if self.specs[k].include_in_prompts]

    def additional_metrics(self) -> dict[str, str]:
        """Return mapping of non-primary metrics that have descriptions."""
        primary_key = self.get_primary_key()
        return {
            k: spec.description for k, spec in self.specs.items() if k != primary_key
        }

    def get_sentinels(self) -> dict[str, float]:
        """Get worst-case sentinel values for all metrics.

        Returns:
            Dictionary mapping metric keys to their sentinel values
        """
        return {k: spec.sentinel_value for k, spec in self.specs.items()}

    def get_bounds(self, key: str) -> tuple[float, float] | None:
        """Get the bounds for a metric if defined.

        Args:
            key: The metric key

        Returns:
            Tuple of (lower_bound, upper_bound) or None if not fully defined

        Raises:
            KeyError: If metric key not found
        """
        spec = self.specs[key]
        if spec.lower_bound is None or spec.upper_bound is None:
            return None
        return (spec.lower_bound, spec.upper_bound)

    def is_higher_better(self, key: str) -> bool:
        """Check if higher values are better for a metric.

        Args:
            key: The metric key

        Returns:
            True if higher is better, False otherwise

        Raises:
            KeyError: If metric key not found
        """
        return self.specs[key].higher_is_better

    @classmethod
    def from_descriptions(
        cls,
        *,
        primary_key: str,
        primary_description: str,
        higher_is_better: bool = True,
        additional_metrics: dict[str, str] | None = None,
        additional_metrics_higher_is_better: dict[str, bool] | None = None,
        decimals: int = DEFAULT_DECIMALS,
        per_metric_decimals: dict[str, int] | None = None,
        display_order: list[str] | None = None,
    ) -> MetricsContext:
        """Convenience constructor from simple description mappings.

        By default, all additional metrics are higher_is_better=True.

        Args:
            primary_key: Key for the primary optimization metric
            primary_description: Description of the primary metric
            higher_is_better: Whether higher is better for primary metric
            additional_metrics: Optional mapping of additional metric keys to descriptions
            additional_metrics_higher_is_better: Optional per-metric optimization direction
            decimals: Default decimal precision for all metrics
            per_metric_decimals: Optional per-metric decimal precision overrides
            display_order: Optional explicit display order for metrics

        Returns:
            New MetricsContext instance
        """
        specs: dict[str, MetricSpec] = {}
        specs[primary_key] = MetricSpec(
            description=primary_description,
            decimals=(per_metric_decimals or {}).get(primary_key, decimals),
            is_primary=True,
            higher_is_better=higher_is_better,
        )
        for k, desc in (additional_metrics or {}).items():
            specs[k] = MetricSpec(
                description=desc,
                decimals=(per_metric_decimals or {}).get(k, decimals),
                higher_is_better=(additional_metrics_higher_is_better or {}).get(
                    k, True
                ),
            )
        return cls(
            specs=specs,
            display_order=display_order or [],
        )

    @classmethod
    def from_dict(
        cls,
        *,
        specs: dict[str, dict[str, Any]],
        display_order: list[str] | None = None,
    ) -> MetricsContext:
        """Create MetricsContext from a dictionary of metric key -> spec fields.

        Args:
            specs: Dictionary mapping metric keys to spec field dictionaries
            display_order: Optional explicit display order for metrics

        Returns:
            New MetricsContext instance
        """
        built: dict[str, MetricSpec] = {}
        for key, data in specs.items():
            data = dict(data)
            built[key] = MetricSpec(**data)
        return cls(specs=built, display_order=display_order or [])
