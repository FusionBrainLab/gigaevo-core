from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

MAX_VALUE_DEFAULT = 1e7
MIN_VALUE_DEFAULT = -1e7
VALIDITY_KEY = "is_valid"


class MetricSpec(BaseModel):
    description: str
    decimals: int = 5
    is_primary: bool = False
    higher_is_better: bool
    unit: str | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None
    include_in_prompts: bool = True
    significant_change: float | None = None


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
    def _validate_primary_spec(self) -> "MetricsContext":
        primary_specs = [s for s in self.specs.values() if s.is_primary]
        if len(primary_specs) != 1:
            raise ValueError(f"Exactly one MetricSpec must have is_primary=True, found {len(primary_specs)}")
        return self

    # Accessors for primary metric
    def get_primary_spec(self) -> MetricSpec:
        for spec in self.specs.values():
            if spec.is_primary:
                return spec
        # Should be unreachable due to validator
        raise ValueError("Primary MetricSpec not found")
    
    def get_primary_key(self) -> str:
        """Get the key of the primary metric."""
        for key, spec in self.specs.items():
            if spec.is_primary:
                return key
        # Should be unreachable due to validator
        raise ValueError("Primary MetricSpec not found")

    def get_description(self, key: str) -> str | None:
        spec = self.specs.get(key)
        return spec.description if spec else None

    def get_decimals(self, key: str) -> int:
        spec = self.specs.get(key)
        return spec.decimals if spec else 5

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
        return {k: spec.description for k, spec in self.specs.items() if k != primary_key}

    def get_worst_with_coalesce(self) -> dict[str, float]: 
        def coalesce_none(lower_bound: float | None, upper_bound: float | None, higher_is_better: bool) -> float:
            if higher_is_better:
                return MIN_VALUE_DEFAULT if lower_bound is None else lower_bound
            else:
                return MAX_VALUE_DEFAULT if upper_bound is None else upper_bound
        return {k: coalesce_none(spec.lower_bound, spec.upper_bound, spec.higher_is_better) for k, spec in self.specs.items()}

    def get_bounds(self, key: str) -> tuple[float, float] | None:
        spec = self.specs.get(key)
        if not spec or spec.lower_bound is None or spec.upper_bound is None:
            return None
        return (spec.lower_bound, spec.upper_bound)

    def is_higher_better(self, key: str) -> bool:
        spec = self.specs.get(key)
        if not spec:
            raise KeyError(f"Unknown metric key: {key}")
        return spec.higher_is_better

    @classmethod
    def from_descriptions(
        cls,
        *,
        primary_key: str,
        primary_description: str,
        higher_is_better: bool = True,
        additional_metrics: dict[str, str] | None = None,
        decimals: int = 5,
        per_metric_decimals: dict[str, int] | None = None,
        display_order: list[str] | None = None,
    ) -> "MetricsContext":
        """Convenience constructor from simple description mappings."""
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
                higher_is_better=True,
            )
        return cls(
            specs=specs,
            display_order=display_order or [],
        )

    @classmethod
    def from_dict(
        cls,
        *,
        specs: dict[str, dict[str, object]],
        display_order: list[str] | None = None,
    ) -> "MetricsContext":
        """Create MetricsContext from a dictionary of metric key -> spec fields."""
        built: dict[str, MetricSpec] = {}
        for key, data in specs.items():
            # Remove any key field from data since MetricSpec no longer has it
            data = dict(data)
            data.pop("key", None)  # Remove key if present
            built[key] = MetricSpec(**data)
        return cls(specs=built, display_order=display_order or [])


