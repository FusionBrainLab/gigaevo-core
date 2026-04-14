"""Tests for gigaevo.utils.dataframes — outlier detection and DataFrame preparation."""

from __future__ import annotations

import pandas as pd

from gigaevo.utils.dataframes import (
    OutlierMethod,
    detect_outliers,
    prepare_iteration_dataframe,
)


class TestOutlierMethod:
    def test_enum_values(self):
        assert OutlierMethod.IQR == "iqr"
        assert OutlierMethod.MAD == "mad"
        assert OutlierMethod.ZSCORE == "zscore"
        assert OutlierMethod.PERCENTILE == "percentile"

    def test_from_string(self):
        assert OutlierMethod("iqr") == OutlierMethod.IQR


class TestDetectOutliers:
    def test_returns_mask_and_bounds(self):
        values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])
        mask, lower, upper, n = detect_outliers(values, method=OutlierMethod.IQR)
        assert isinstance(mask, pd.Series)
        assert n >= 1
        assert mask.iloc[-1]  # 100.0 should be an outlier

    def test_empty_series_returns_zero_outliers(self):
        values = pd.Series([], dtype=float)
        mask, lower, upper, n = detect_outliers(values)
        assert n == 0
        assert len(mask) == 0

    def test_nan_values_not_flagged(self):
        values = pd.Series([1.0, 2.0, float("nan"), 3.0, 4.0])
        mask, _, _, _ = detect_outliers(values, method=OutlierMethod.MAD)
        assert not mask.iloc[2]

    def test_side_low_only_flags_low(self):
        values = pd.Series([0.001, 1.0, 2.0, 3.0, 4.0, 5.0])
        mask, _, _, _ = detect_outliers(values, method=OutlierMethod.IQR, side="low")
        # Only low values should be flagged
        if mask.any():
            flagged_values = values[mask]
            assert all(v < values.median() for v in flagged_values)


class TestPrepareIterationDataframe:
    def test_basic_output_columns(self):
        df = pd.DataFrame(
            {
                "iteration": [0, 0, 1, 1, 2, 2],
                "metric_fitness": [0.5, 0.6, 0.7, 0.65, 0.8, 0.75],
            }
        )
        result = prepare_iteration_dataframe(df, remove_outliers=False)
        assert "iteration" in result.columns
        assert "metric_fitness" in result.columns
        assert "running_mean_fitness" in result.columns
        assert "frontier_fitness" in result.columns

    def test_empty_dataframe_returns_empty(self):
        df = pd.DataFrame({"iteration": [], "metric_fitness": []})
        result = prepare_iteration_dataframe(df)
        assert result.empty

    def test_sentinel_removal(self):
        df = pd.DataFrame(
            {
                "iteration": [0, 1, 2, 3, 4],
                "metric_fitness": [-1.0, 0.5, -1.0, 0.7, 0.8],
            }
        )
        result = prepare_iteration_dataframe(
            df, sentinel_value=-1.0, remove_outliers=False
        )
        assert -1.0 not in result["metric_fitness"].values

    def test_missing_fitness_col_returns_empty(self):
        df = pd.DataFrame({"iteration": [0, 1], "other_col": [0.5, 0.6]})
        result = prepare_iteration_dataframe(df)
        assert result.empty
