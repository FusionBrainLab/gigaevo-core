from __future__ import annotations

import abc
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, computed_field, model_validator


class BinningStrategy(BaseModel, abc.ABC):
    """Abstract base class for behavior space binning strategies."""

    min_val: float
    max_val: float
    num_bins: int = Field(gt=0)

    @model_validator(mode="after")
    def check_bounds(self) -> "BinningStrategy":
        if self.min_val > self.max_val:
            raise ValueError(
                f"Invalid bounds: min ({self.min_val}) > max ({self.max_val})"
            )
        return self

    def _clamp(self, value: float) -> float:
        return max(self.min_val, min(value, self.max_val))

    @abc.abstractmethod
    def get_index(self, value: float) -> int:
        """Map a value to a 0-based bin index."""

    @abc.abstractmethod
    def get_bin_center(self, index: int) -> float:
        """Get the center value of a bin."""

    @abc.abstractmethod
    def get_bin_width(self, index: int) -> float:
        """Get the width of a bin."""

    @abc.abstractmethod
    def get_bin_edges(self, index: int) -> tuple[float, float]:
        """Get the (lower, upper) edges of a bin."""

    def update_bounds(
        self, new_min: float | None = None, new_max: float | None = None
    ) -> bool:
        """Update bounds if new values are outside current range (expand) OR to tighten (shrink).

        Args:
            new_min: If provided, update min_val.
            new_max: If provided, update max_val.

        Returns:
            bool: True if bounds changed, False otherwise.
        """
        changed = False

        # Logic update: we now allow shrinking (setting min > old_min or max < old_max)
        # The caller is responsible for ensuring this doesn't exclude valid data.

        if new_min is not None and new_min != self.min_val:
            self.min_val = new_min
            changed = True

        if new_max is not None and new_max != self.max_val:
            self.max_val = new_max
            changed = True

        if changed:
            # Re-validate post-update state
            if self.min_val > self.max_val:
                raise ValueError(
                    f"Invalid bounds update: {self.min_val} > {self.max_val}"
                )

        return changed


class LinearBinning(BinningStrategy):
    """Standard linear binning."""

    type: Literal["linear"] = "linear"

    def get_index(self, value: float) -> int:
        if self.max_val == self.min_val:
            return 0
        value = self._clamp(value)
        normalized = (value - self.min_val) / (self.max_val - self.min_val)
        # Handle edge case where value == max_val, should be in last bin
        idx = int(normalized * self.num_bins)
        return min(idx, self.num_bins - 1)

    def get_bin_edges(self, index: int) -> tuple[float, float]:
        step = (self.max_val - self.min_val) / self.num_bins
        lower = self.min_val + index * step
        upper = self.min_val + (index + 1) * step
        return lower, upper

    def get_bin_center(self, index: int) -> float:
        lower, upper = self.get_bin_edges(index)
        return (lower + upper) / 2

    def get_bin_width(self, index: int) -> float:
        return (self.max_val - self.min_val) / self.num_bins


# With only one strategy, we technically don't need a Union/Annotated,
# but keeping the structure allows for future strategies.
BinningStrategyType = Annotated[
    LinearBinning,
    Field(discriminator="type"),
]


class BehaviorSpace(BaseModel):
    """Static behavior space (fixed bounds)."""

    bins: dict[str, BinningStrategyType] = Field(
        description="Map of behavior name to its binning strategy"
    )

    @computed_field
    @property
    def behavior_keys(self) -> list[str]:
        return list(self.bins.keys())

    @computed_field
    @property
    def total_cells(self) -> int:
        total = 1
        for b in self.bins.values():
            total *= b.num_bins
        return total

    @model_validator(mode="after")
    def check_dimensionality(self) -> "BehaviorSpace":
        if not self.bins:
            raise ValueError("Behavior space must have at least one dimension")
        return self

    def get_cell(self, metrics: dict[str, float]) -> tuple[int, ...]:
        """Map program metrics to a cell coordinate."""
        coordinates = []

        for key, strategy in self.bins.items():
            if key not in metrics:
                raise KeyError(f"Missing required behavior key '{key}'")

            val = metrics[key]
            idx = strategy.get_index(val)
            coordinates.append(idx)

        return tuple(coordinates)

    def describe(self) -> dict[str, Any]:
        """Return human-readable description of the space."""
        return {
            key: {
                "type": strategy.type,
                "min": strategy.min_val,
                "max": strategy.max_val,
                "bins": strategy.num_bins,
                "centers": [
                    strategy.get_bin_center(i) for i in range(strategy.num_bins)
                ],
            }
            for key, strategy in self.bins.items()
        }


class DynamicBehaviorSpace(BehaviorSpace):
    """Dynamic behavior space that can expand/shrink based on observed metrics.

    The initial min/max values act as hard limits (used for clamping in BinningStrategy).
    Dynamic adjustments stay within these initial bounds.

    When a value exceeds current bounds, only the violated bound is adjusted (with margin).
    For example, if a value is above max, only max is pushed up; min stays unchanged.
    """

    expansion_buffer_ratio: float = Field(
        default=0.1,
        ge=0.0,
        description="Ratio of range to add as buffer when expanding/shrinking",
    )

    # Store initial bounds as hard limits for clamping
    _initial_bounds: dict[str, tuple[float, float]] = {}

    def model_post_init(self, __context) -> None:
        """Store initial bounds after model initialization."""
        super().model_post_init(__context)
        # Capture initial bounds for each dimension
        self._initial_bounds = {
            key: (strategy.min_val, strategy.max_val)
            for key, strategy in self.bins.items()
        }

    def _calculate_margin(self, current_range: float) -> float:
        """Calculate margin based on current range."""
        return (
            current_range * self.expansion_buffer_ratio if current_range > 0 else 1e-5
        )

    def _clamp_to_initial_bounds(self, key: str, value: float) -> float:
        """Clamp a value to the initial (hard) bounds for this dimension."""
        if key not in self._initial_bounds:
            return value
        initial_min, initial_max = self._initial_bounds[key]
        return max(initial_min, min(value, initial_max))

    def _calculate_bounds_for_value(
        self, key: str, value: float, strategy: BinningStrategy
    ) -> tuple[float | None, float | None]:
        """Calculate new bounds for a single value.

        Only adjusts the bound that is violated (if any).
        New bounds are clamped to initial bounds.

        Returns:
            (new_min, new_max) where None means "don't change"
        """
        current_range = strategy.max_val - strategy.min_val
        margin = self._calculate_margin(current_range)

        new_min, new_max = None, None

        # Only adjust min if value is BELOW current min
        if value < strategy.min_val:
            new_min = self._clamp_to_initial_bounds(key, value - margin)

        # Only adjust max if value is ABOVE current max
        if value > strategy.max_val:
            new_max = self._clamp_to_initial_bounds(key, value + margin)

        return new_min, new_max

    def _calculate_bounds_for_batch(
        self, key: str, values: list[float], strategy: BinningStrategy
    ) -> tuple[float, float]:
        """Calculate optimized bounds for a batch of values.

        Tightens bounds to observed range with margin on both sides.
        New bounds are clamped to initial bounds.

        Returns:
            (new_min, new_max) - always returns both values
        """
        obs_min, obs_max = min(values), max(values)

        # Calculate margin based on observed range
        value_range = obs_max - obs_min
        margin = self._calculate_margin(value_range)

        # Calculate new bounds with margin, clamped to initial bounds
        new_min = self._clamp_to_initial_bounds(key, obs_min - margin)
        new_max = self._clamp_to_initial_bounds(key, obs_max + margin)

        return new_min, new_max

    def update_bounds(
        self, new_bounds: dict[str, tuple[float | None, float | None]]
    ) -> bool:
        """Update bounds for multiple dimensions at once.

        Args:
            new_bounds: Dictionary mapping key -> (new_min, new_max)

        Returns:
            bool: True if any dimension changed
        """
        changed = False
        for key, (new_min, new_max) in new_bounds.items():
            if key in self.bins:
                if self.bins[key].update_bounds(new_min, new_max):
                    changed = True
        return changed

    def check_and_expand(self, metrics: dict[str, float]) -> bool:
        """Check if metrics are out of bounds and expand if necessary.

        Only expands bounds that are not marked as fixed.

        Returns:
            bool: True if any dimension was expanded.
        """
        expanded = False
        for key, strategy in self.bins.items():
            if key not in metrics:
                continue  # get_cell will raise error later

            val = metrics[key]
            new_min, new_max = self._calculate_bounds_for_value(key, val, strategy)

            if new_min is not None or new_max is not None:
                if strategy.update_bounds(new_min, new_max):
                    expanded = True

        return expanded

    def calculate_optimized_bounds(
        self, metrics_batch: list[dict[str, float]]
    ) -> dict[str, tuple[float, float]]:
        """Calculate new optimized bounds based on a batch of metrics.

        Respects fixed bounds - only optimizes bounds that are marked as dynamic.

        Returns:
            Dictionary mapping key -> (new_min, new_max)
        """
        if not metrics_batch:
            return {}

        new_bounds = {}
        for key, strategy in self.bins.items():
            values = [m[key] for m in metrics_batch if key in m]
            if not values:
                continue

            new_min, new_max = self._calculate_bounds_for_batch(key, values, strategy)
            new_bounds[key] = (new_min, new_max)

        return new_bounds
