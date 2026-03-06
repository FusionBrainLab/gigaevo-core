"""Tests for gigaevo/evolution/strategies/models.py — BehaviorSpace and binning."""

import pytest

from gigaevo.evolution.strategies.models import (
    BehaviorSpace,
    DynamicBehaviorSpace,
    LinearBinning,
)


class TestLinearBinning:
    def test_basic_binning(self):
        b = LinearBinning(min_val=0.0, max_val=10.0, num_bins=10)
        assert b.get_index(0.0) == 0
        assert b.get_index(5.0) == 5
        assert b.get_index(9.9) == 9

    def test_edge_max(self):
        b = LinearBinning(min_val=0.0, max_val=10.0, num_bins=10)
        assert b.get_index(10.0) == 9  # clamped to last bin

    def test_edge_min(self):
        b = LinearBinning(min_val=0.0, max_val=10.0, num_bins=10)
        assert b.get_index(0.0) == 0

    def test_clamp_below(self):
        b = LinearBinning(min_val=0.0, max_val=10.0, num_bins=10)
        assert b.get_index(-5.0) == 0

    def test_clamp_above(self):
        b = LinearBinning(min_val=0.0, max_val=10.0, num_bins=10)
        assert b.get_index(100.0) == 9

    def test_equal_bounds(self):
        b = LinearBinning(min_val=5.0, max_val=5.0, num_bins=3)
        assert b.get_index(5.0) == 0
        assert b.get_index(100.0) == 0

    def test_bin_edges(self):
        b = LinearBinning(min_val=0.0, max_val=10.0, num_bins=5)
        lower, upper = b.get_bin_edges(0)
        assert lower == pytest.approx(0.0)
        assert upper == pytest.approx(2.0)
        lower, upper = b.get_bin_edges(4)
        assert lower == pytest.approx(8.0)
        assert upper == pytest.approx(10.0)

    def test_bin_center(self):
        b = LinearBinning(min_val=0.0, max_val=10.0, num_bins=5)
        assert b.get_bin_center(0) == pytest.approx(1.0)
        assert b.get_bin_center(2) == pytest.approx(5.0)

    def test_bin_width(self):
        b = LinearBinning(min_val=0.0, max_val=10.0, num_bins=5)
        assert b.get_bin_width(0) == pytest.approx(2.0)

    def test_update_bounds(self):
        b = LinearBinning(min_val=0.0, max_val=10.0, num_bins=5)
        changed = b.update_bounds(new_min=-5.0)
        assert changed is True
        assert b.min_val == -5.0

    def test_update_bounds_no_change(self):
        b = LinearBinning(min_val=0.0, max_val=10.0, num_bins=5)
        changed = b.update_bounds(new_min=0.0)
        assert changed is False

    def test_update_bounds_invalid_raises(self):
        b = LinearBinning(min_val=0.0, max_val=10.0, num_bins=5)
        with pytest.raises(ValueError, match="Invalid bounds"):
            b.update_bounds(new_min=20.0)

    def test_invalid_min_gt_max_raises(self):
        with pytest.raises(ValueError, match="Invalid bounds"):
            LinearBinning(min_val=10.0, max_val=0.0, num_bins=5)

    def test_get_index_bin_center_roundtrip(self):
        """For each bin index, get_bin_center returns a value that maps back to the same bin."""
        b = LinearBinning(min_val=0.0, max_val=10.0, num_bins=10)
        for idx in range(b.num_bins):
            center = b.get_bin_center(idx)
            assert b.get_index(center) == idx


class TestBehaviorSpace:
    def test_1d_get_cell(self):
        space = BehaviorSpace(
            bins={"x": LinearBinning(min_val=0.0, max_val=10.0, num_bins=10)}
        )
        assert space.get_cell({"x": 5.0}) == (5,)

    def test_2d_get_cell(self):
        space = BehaviorSpace(
            bins={
                "x": LinearBinning(min_val=0.0, max_val=10.0, num_bins=10),
                "y": LinearBinning(min_val=0.0, max_val=1.0, num_bins=5),
            }
        )
        cell = space.get_cell({"x": 5.0, "y": 0.5})
        assert len(cell) == 2
        assert cell[0] == 5
        assert cell[1] == 2  # 0.5 in [0,1] with 5 bins -> bin 2

    def test_missing_key_raises(self):
        space = BehaviorSpace(
            bins={"x": LinearBinning(min_val=0.0, max_val=10.0, num_bins=10)}
        )
        with pytest.raises(KeyError, match="Missing required behavior key"):
            space.get_cell({"y": 5.0})

    def test_behavior_keys(self):
        space = BehaviorSpace(
            bins={
                "a": LinearBinning(min_val=0.0, max_val=1.0, num_bins=2),
                "b": LinearBinning(min_val=0.0, max_val=1.0, num_bins=3),
            }
        )
        assert set(space.behavior_keys) == {"a", "b"}

    def test_total_cells(self):
        space = BehaviorSpace(
            bins={
                "a": LinearBinning(min_val=0.0, max_val=1.0, num_bins=2),
                "b": LinearBinning(min_val=0.0, max_val=1.0, num_bins=3),
            }
        )
        assert space.total_cells == 6

    def test_empty_bins_raises(self):
        with pytest.raises(ValueError, match="at least one dimension"):
            BehaviorSpace(bins={})


class TestDynamicBehaviorSpace:
    def test_expand_above(self):
        # Use tight inner bounds with wide initial bounds to allow expansion
        space = DynamicBehaviorSpace(
            bins={"x": LinearBinning(min_val=0.0, max_val=100.0, num_bins=10)},
            expansion_buffer_ratio=0.1,
        )
        # Shrink bounds first so there's room to expand
        space.bins["x"].update_bounds(new_max=10.0)
        expanded = space.check_and_expand({"x": 15.0})
        assert expanded is True
        assert space.bins["x"].max_val > 10.0

    def test_expand_below(self):
        space = DynamicBehaviorSpace(
            bins={"x": LinearBinning(min_val=-100.0, max_val=10.0, num_bins=10)},
            expansion_buffer_ratio=0.1,
        )
        # Shrink min up first
        space.bins["x"].update_bounds(new_min=0.0)
        expanded = space.check_and_expand({"x": -5.0})
        assert expanded is True
        assert space.bins["x"].min_val < 0.0

    def test_within_bounds_no_expand(self):
        space = DynamicBehaviorSpace(
            bins={"x": LinearBinning(min_val=0.0, max_val=10.0, num_bins=10)},
        )
        expanded = space.check_and_expand({"x": 5.0})
        assert expanded is False

    def test_clamp_to_initial_bounds(self):
        space = DynamicBehaviorSpace(
            bins={"x": LinearBinning(min_val=0.0, max_val=100.0, num_bins=10)},
            expansion_buffer_ratio=0.1,
        )
        # Value far above initial max -> clamped to initial max
        space.check_and_expand({"x": 200.0})
        assert space.bins["x"].max_val == pytest.approx(100.0)

    def test_batch_optimize(self):
        space = DynamicBehaviorSpace(
            bins={"x": LinearBinning(min_val=0.0, max_val=100.0, num_bins=10)},
            expansion_buffer_ratio=0.1,
        )
        metrics_batch = [{"x": 20.0}, {"x": 30.0}, {"x": 40.0}]
        bounds = space.calculate_optimized_bounds(metrics_batch)
        assert "x" in bounds
        new_min, new_max = bounds["x"]
        # obs_min=20, obs_max=40, range=20, margin=2
        # new_min = max(0, 20-2) = 18; new_max = min(100, 40+2) = 42
        assert new_min == pytest.approx(18.0)
        assert new_max == pytest.approx(42.0)

    def test_batch_optimize_empty(self):
        space = DynamicBehaviorSpace(
            bins={"x": LinearBinning(min_val=0.0, max_val=10.0, num_bins=10)},
        )
        assert space.calculate_optimized_bounds([]) == {}

    def test_multi_dim_update(self):
        space = DynamicBehaviorSpace(
            bins={
                "x": LinearBinning(min_val=0.0, max_val=10.0, num_bins=5),
                "y": LinearBinning(min_val=0.0, max_val=10.0, num_bins=5),
            },
        )
        changed = space.update_bounds({"x": (None, 8.0), "y": (2.0, None)})
        assert changed is True
        assert space.bins["x"].max_val == 8.0
        assert space.bins["y"].min_val == 2.0

    def test_expand_idempotent(self):
        """Expanding twice with the same out-of-bounds value doesn't change bounds the second time."""
        space = DynamicBehaviorSpace(
            bins={"x": LinearBinning(min_val=-100.0, max_val=100.0, num_bins=10)},
            expansion_buffer_ratio=0.1,
        )
        space.bins["x"].update_bounds(new_min=0.0, new_max=10.0)

        # First expansion
        expanded1 = space.check_and_expand({"x": 15.0})
        assert expanded1 is True
        max_after_first = space.bins["x"].max_val

        # Second expansion with same value
        expanded2 = space.check_and_expand({"x": 15.0})
        assert expanded2 is False
        assert space.bins["x"].max_val == max_after_first

    def test_batch_optimize_single_point(self):
        """Single data point: bounds are tight around it (with margin)."""
        space = DynamicBehaviorSpace(
            bins={"x": LinearBinning(min_val=0.0, max_val=100.0, num_bins=10)},
            expansion_buffer_ratio=0.1,
        )
        bounds = space.calculate_optimized_bounds([{"x": 50.0}])
        assert "x" in bounds
        new_min, new_max = bounds["x"]
        # obs_min == obs_max == 50, range=0, margin=1e-5
        # new_min = max(0, 50-1e-5) ≈ 50; new_max = min(100, 50+1e-5) ≈ 50
        assert new_min == pytest.approx(50.0, abs=1e-3)
        assert new_max == pytest.approx(50.0, abs=1e-3)
