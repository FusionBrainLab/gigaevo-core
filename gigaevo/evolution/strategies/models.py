from enum import Enum
import math
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field, computed_field, field_validator, model_validator


class SelectionMode(Enum):
    """Selection modes for elite choosing."""

    RANDOM = "random"
    FITNESS_PROPORTIONAL = "fitness_proportional"
    TOURNAMENT = "tournament"


class BinningType(Enum):
    """Different binning strategies for behavior space discretization."""

    LINEAR = "linear"  # Standard linear binning
    LOGARITHMIC = "logarithmic"  # Log-scaled bins for exponential distributions
    SQUARE_ROOT = "square_root"  # Square root scaling for moderate non-linearity
    QUANTILE = "quantile"  # Quantile-based binning (requires pre-computed data)


class BehaviorSpace(BaseModel):
    """Enhanced discretized behavior space for MAP-Elites with multiple binning strategies."""

    feature_bounds: dict[str, tuple[float, float]] = Field(
        description="Bounds for each behavior feature (min, max)"
    )
    resolution: dict[str, int] = Field(
        description="Discretization resolution for each behavior feature"
    )
    binning_types: dict[str, "BinningType"] = Field(
        default_factory=dict,
        description="Binning strategy for each behavior feature (defaults to LINEAR)",
    )

    @computed_field
    @property
    def total_cells(self) -> int:
        """Calculate total number of cells in the behavior space."""
        total = 1
        for key in self.behavior_keys:
            if key in self.resolution:
                total *= self.resolution[key]
        return total

    @field_validator("feature_bounds")
    @classmethod
    def validate_feature_bounds(cls, v):
        for key, (min_val, max_val) in v.items():
            if min_val > max_val:
                raise ValueError(
                    f"Invalid bounds for {key}: min ({min_val}) must be <= max ({max_val})"
                )
        return v

    @field_validator("resolution")
    @classmethod
    def validate_resolution(cls, v):
        for key, res in v.items():
            if res <= 0:
                raise ValueError(f"Resolution for {key} must be positive, got {res}")
        return v

    @field_validator("binning_types")
    @classmethod
    def validate_binning_types(cls, v):
        for key, binning_type in v.items():
            if not isinstance(binning_type, BinningType):
                raise ValueError(f"Invalid binning type for {key}: {binning_type}")
        return v

    @model_validator(mode="after")
    def validate_consistency(self):
        missing_bounds = set(self.behavior_keys) - set(self.feature_bounds.keys())
        if missing_bounds:
            raise ValueError(f"Missing feature bounds for: {missing_bounds}")

        missing_resolution = set(self.behavior_keys) - set(self.resolution.keys())
        if missing_resolution:
            raise ValueError(f"Missing resolution for: {missing_resolution}")

        # Validate log binning bounds
        for key in self.behavior_keys:
            if self.get_binning_type(key) == BinningType.LOGARITHMIC:
                min_val, max_val = self.feature_bounds[key]
                if min_val <= 0:
                    raise ValueError(
                        f"Logarithmic binning requires positive bounds for {key}, got min={min_val}"
                    )

        if self.total_cells > 1_000_000:
            logger.warning(
                f"Large behavior space with {self.total_cells:,} cells may impact performance"
            )

        return self

    @computed_field
    @property
    def behavior_keys(self) -> list[str]:
        return list(self.resolution.keys())

    def get_binning_type(self, behavior_key: str) -> "BinningType":
        """Get the binning type for a behavior key, defaulting to LINEAR."""
        return self.binning_types.get(behavior_key, BinningType.LINEAR)

    def get_cell(self, metrics: dict[str, float]) -> tuple[int, ...]:
        """Map program metrics to discrete cell coordinates using appropriate binning."""
        coordinates = []
        details = []

        for behavior_key in self.behavior_keys:
            if behavior_key not in metrics:
                raise KeyError(
                    f"Missing required behavior key '{behavior_key}' in program metrics"
                )
            value = metrics[behavior_key]
            coordinate = self._map_value_to_coordinate(behavior_key, value)
            coordinates.append(coordinate)

            # Track mapping details for debug logging
            min_val, max_val = self.feature_bounds[behavior_key]
            binning_type = self.get_binning_type(behavior_key)
            details.append(
                f"{behavior_key}={value:.3f}->{coordinate} (bounds=[{min_val:.1f},{max_val:.1f}], {binning_type.value})"
            )

        cell = tuple(coordinates)
        logger.debug(
            "BehaviorSpace: mapped to cell {} | {}",
            cell,
            " | ".join(details),
        )

        return cell

    def _map_value_to_coordinate(self, behavior_key: str, value: float) -> int:
        """Map a single value to its coordinate using the appropriate binning strategy."""
        min_val, max_val = self.feature_bounds[behavior_key]
        num_bins = self.resolution[behavior_key]
        binning_type = self.get_binning_type(behavior_key)

        # Clamp value to bounds
        value = max(min_val, min(value, max_val))

        # Apply appropriate binning strategy
        if binning_type == BinningType.LINEAR:
            coordinate = self._linear_binning(value, min_val, max_val, num_bins)
        elif binning_type == BinningType.LOGARITHMIC:
            coordinate = self._logarithmic_binning(value, min_val, max_val, num_bins)
        elif binning_type == BinningType.SQUARE_ROOT:
            coordinate = self._square_root_binning(value, min_val, max_val, num_bins)
        else:
            # Fallback to linear
            logger.warning(
                f"Unknown binning type {binning_type} for {behavior_key}, using linear"
            )
            coordinate = self._linear_binning(value, min_val, max_val, num_bins)

        return max(0, min(coordinate, num_bins - 1))

    def _linear_binning(
        self, value: float, min_val: float, max_val: float, num_bins: int
    ) -> int:
        """Standard linear binning."""
        if max_val == min_val:
            return 0
        normalized = (value - min_val) / (max_val - min_val)
        return int(normalized * num_bins)

    def _logarithmic_binning(
        self, value: float, min_val: float, max_val: float, num_bins: int
    ) -> int:
        """Logarithmic binning for exponential distributions (e.g., complexity, entropy)."""
        if max_val == min_val:
            return 0

        safe_value = max(value, min_val)

        log_min = math.log(min_val)
        log_max = math.log(max_val)
        log_value = math.log(safe_value)

        normalized = (log_value - log_min) / (log_max - log_min)
        return int(normalized * num_bins)

    def _square_root_binning(
        self, value: float, min_val: float, max_val: float, num_bins: int
    ) -> int:
        """Square root binning for moderate non-linear distributions."""
        if max_val == min_val:
            return 0

        sqrt_min = math.sqrt(max(min_val, 0))
        sqrt_max = math.sqrt(max_val)
        sqrt_value = math.sqrt(max(value, 0))

        normalized = (sqrt_value - sqrt_min) / (sqrt_max - sqrt_min)
        return int(normalized * num_bins)

    def get_bin_centers(self, behavior_key: str) -> list[float]:
        """Get the center values of each bin for a behavior key."""
        min_val, max_val = self.feature_bounds[behavior_key]
        num_bins = self.resolution[behavior_key]
        binning_type = self.get_binning_type(behavior_key)

        centers = []
        for i in range(num_bins):
            normalized_pos = (i + 0.5) / num_bins

            if binning_type == BinningType.LINEAR:
                center = min_val + normalized_pos * (max_val - min_val)
            elif binning_type == BinningType.LOGARITHMIC:
                log_min = math.log(min_val)
                log_max = math.log(max_val)
                log_center = log_min + normalized_pos * (log_max - log_min)
                center = math.exp(log_center)
            elif binning_type == BinningType.SQUARE_ROOT:
                sqrt_min = math.sqrt(min_val)
                sqrt_max = math.sqrt(max_val)
                sqrt_center = sqrt_min + normalized_pos * (sqrt_max - sqrt_min)
                center = sqrt_center**2
            else:
                center = min_val + normalized_pos * (max_val - min_val)

            centers.append(center)

        return centers

    def describe_binning(self) -> dict[str, dict[str, Any]]:
        """Get a description of the binning configuration for each behavior key."""
        description = {}
        for key in self.behavior_keys:
            min_val, max_val = self.feature_bounds[key]
            num_bins = self.resolution[key]
            binning_type = self.get_binning_type(key)

            description[key] = {
                "bounds": (min_val, max_val),
                "resolution": num_bins,
                "binning_type": binning_type.value,
                "bin_centers": self.get_bin_centers(key),
                "bin_widths": self._calculate_bin_widths(key),
            }
        return description

    def _calculate_bin_widths(self, behavior_key: str) -> list[float]:
        """Calculate the width of each bin for a behavior key."""
        min_val, max_val = self.feature_bounds[behavior_key]
        num_bins = self.resolution[behavior_key]
        binning_type = self.get_binning_type(behavior_key)

        widths = []
        for i in range(num_bins):
            lower_norm = i / num_bins
            upper_norm = (i + 1) / num_bins

            if binning_type == BinningType.LINEAR:
                lower = min_val + lower_norm * (max_val - min_val)
                upper = min_val + upper_norm * (max_val - min_val)
            elif binning_type == BinningType.LOGARITHMIC:
                log_min = math.log(min_val)
                log_max = math.log(max_val)
                log_lower = log_min + lower_norm * (log_max - log_min)
                log_upper = log_min + upper_norm * (log_max - log_min)
                lower = math.exp(log_lower)
                upper = math.exp(log_upper)
            elif binning_type == BinningType.SQUARE_ROOT:
                sqrt_min = math.sqrt(min_val)
                sqrt_max = math.sqrt(max_val)
                sqrt_lower = sqrt_min + lower_norm * (sqrt_max - sqrt_min)
                sqrt_upper = sqrt_min + upper_norm * (sqrt_max - sqrt_min)
                lower = sqrt_lower**2
                upper = sqrt_upper**2
            else:
                # Fallback to linear
                lower = min_val + lower_norm * (max_val - min_val)
                upper = min_val + upper_norm * (max_val - min_val)

            widths.append(upper - lower)

        return widths
