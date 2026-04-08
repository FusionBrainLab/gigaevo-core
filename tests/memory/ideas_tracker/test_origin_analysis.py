"""Tests for the origin_analysis subpackage."""
from __future__ import annotations

import math

import pytest

from gigaevo.memory.ideas_tracker.utils.origin_analysis.statistics import (
    elite_threshold_by_top_k,
    mad,
    nancount,
    nanmedian,
    nanquantile,
    nanrate_bool,
    percentile_rank,
    robust_median,
    robust_quantile,
)


class TestRobustMedian:
    def test_odd_list(self):
        assert robust_median([1.0, 3.0, 5.0]) == 3.0

    def test_even_list(self):
        assert robust_median([1.0, 2.0, 3.0, 4.0]) == 2.5

    def test_empty_returns_nan(self):
        assert math.isnan(robust_median([]))

    def test_single_element(self):
        assert robust_median([7.0]) == 7.0


class TestRobustQuantile:
    def test_q0_returns_min(self):
        assert robust_quantile([1.0, 2.0, 3.0], 0.0) == 1.0

    def test_q1_returns_max(self):
        assert robust_quantile([1.0, 2.0, 3.0], 1.0) == 3.0

    def test_q0_5_returns_median(self):
        assert robust_quantile([1.0, 2.0, 3.0], 0.5) == 2.0

    def test_empty_returns_nan(self):
        assert math.isnan(robust_quantile([], 0.5))


class TestMad:
    def test_known_values(self):
        # median=3, deviations=[2,1,0,1,2], mad=1
        result = mad([1.0, 2.0, 3.0, 4.0, 5.0])
        assert result == 1.0

    def test_empty_returns_nan(self):
        assert math.isnan(mad([]))


class TestPercentileRank:
    def test_value_at_max(self):
        assert percentile_rank([1.0, 2.0, 3.0], 3.0) == 1.0

    def test_value_at_min(self):
        assert percentile_rank([1.0, 2.0, 3.0], 0.5) == 0.0

    def test_empty_returns_nan(self):
        assert math.isnan(percentile_rank([], 1.0))

    def test_middle_value(self):
        assert percentile_rank([1.0, 2.0, 3.0], 2.0) == pytest.approx(2 / 3)


class TestEliteThreshold:
    def test_top_50_pct(self):
        threshold, count = elite_threshold_by_top_k([1.0, 2.0, 3.0, 4.0], 0.5)
        assert count == 2
        assert threshold == 3.0

    def test_empty_returns_nan(self):
        threshold, count = elite_threshold_by_top_k([], 0.1)
        assert math.isnan(threshold)
        assert count == 0


class TestNanHelpers:
    def test_nanmedian_skips_nan(self):
        assert nanmedian([1.0, float("nan"), 3.0]) == 2.0

    def test_nanmedian_all_nan(self):
        assert math.isnan(nanmedian([float("nan"), float("nan")]))

    def test_nanquantile_skips_nan(self):
        assert nanquantile([1.0, float("nan"), 3.0], 0.0) == 1.0

    def test_nanrate_bool_counts_gt_half(self):
        assert nanrate_bool([0.0, 1.0, 1.0]) == pytest.approx(2 / 3)

    def test_nanrate_bool_all_nan(self):
        assert math.isnan(nanrate_bool([float("nan")]))

    def test_nancount(self):
        assert nancount([1.0, float("nan"), 3.0]) == 2
