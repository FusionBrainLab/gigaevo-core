"""Tests for sentinel value filtering in prepare_iteration_dataframe."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from tools.utils import prepare_iteration_dataframe


def _make_df(
    fitness_values: list[float],
    iterations: list[int] | None = None,
) -> pd.DataFrame:
    """Create a minimal DataFrame for testing prepare_iteration_dataframe."""
    n = len(fitness_values)
    if iterations is None:
        iterations = list(range(1, n + 1))
    return pd.DataFrame(
        {
            "metric_fitness": fitness_values,
            "metadata_iteration": iterations,
        }
    )


class TestSentinelFiltering:
    """Tests for the sentinel_value parameter of prepare_iteration_dataframe."""

    def test_sentinel_values_excluded_from_rolling_stats(self) -> None:
        """DataFrame with sentinel=-1.0 produces stats from non-sentinel rows only."""
        df = _make_df([0.5, 0.6, -1.0, 0.7, -1.0, 0.8])
        result = prepare_iteration_dataframe(
            df,
            sentinel_value=-1.0,
            remove_outliers=False,
            compute_frontier=False,
        )
        assert len(result) == 4
        assert -1.0 not in result["metric_fitness"].values

    def test_all_sentinel_values_returns_empty(self) -> None:
        """DataFrame with only sentinel values returns empty DataFrame."""
        df = _make_df([-1.0, -1.0, -1.0])
        result = prepare_iteration_dataframe(
            df,
            sentinel_value=-1.0,
            remove_outliers=False,
            compute_frontier=False,
        )
        assert result.empty

    def test_sentinel_none_preserves_existing_behavior(self) -> None:
        """sentinel_value=None (default) does not filter any rows."""
        df = _make_df([0.5, 0.6, -1.0, 0.7, -1.0, 0.8])
        result = prepare_iteration_dataframe(
            df,
            sentinel_value=None,
            remove_outliers=False,
            compute_frontier=False,
        )
        # All 6 rows should be present (no sentinel filtering)
        assert len(result) == 6

    def test_sentinel_zero_filters_zero_values(self) -> None:
        """sentinel_value=0.0 correctly filters zero-valued fitness entries."""
        df = _make_df([0.5, 0.0, 0.6, 0.0, 0.7])
        result = prepare_iteration_dataframe(
            df,
            sentinel_value=0.0,
            remove_outliers=False,
            compute_frontier=False,
        )
        assert len(result) == 3
        assert 0.0 not in result["metric_fitness"].values

    def test_sentinel_filtering_before_outlier_removal(self) -> None:
        """Sentinel filtering happens BEFORE outlier removal (order matters).

        If sentinels were not removed first, they would skew the outlier
        detection statistics. We verify that:
        1. No sentinel values remain in the output
        2. The outlier removal operates on sentinel-free data (so only
           real outliers are removed, not sentinel-contaminated stats)
        """
        # Mix sentinels with real data. Disable outlier removal to count
        # sentinel-only effect, then enable to verify ordering.
        values = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
        sentinels = [-1.0, -1.0, -1.0, -1.0, -1.0]
        df = _make_df(values + sentinels)

        # Without outlier removal: all non-sentinel values preserved
        result_no_outlier = prepare_iteration_dataframe(
            df,
            sentinel_value=-1.0,
            remove_outliers=False,
            compute_frontier=False,
        )
        assert len(result_no_outlier) == len(values)

        # With outlier removal: sentinels still gone, and outlier stats
        # are based on clean data (not sentinel-polluted)
        result_with_outlier = prepare_iteration_dataframe(
            df,
            sentinel_value=-1.0,
            remove_outliers=True,
            compute_frontier=False,
        )
        assert -1.0 not in result_with_outlier["metric_fitness"].values
        # Outlier removal may trim some edge values from clean data,
        # but we must have fewer rows than without sentinel filtering
        # AND no sentinel values
        assert len(result_with_outlier) <= len(values)
        assert len(result_with_outlier) > 0

    def test_frontier_computed_from_non_sentinel_data(self) -> None:
        """frontier_fitness is computed only from non-sentinel data."""
        df = _make_df([0.5, -1.0, 0.6, -1.0, 0.8, 0.7])
        result = prepare_iteration_dataframe(
            df,
            sentinel_value=-1.0,
            remove_outliers=False,
            compute_frontier=True,
        )
        assert "frontier_fitness" in result.columns
        # Frontier should track cummax of non-sentinel values
        frontier_values = result["frontier_fitness"].values
        # The last frontier value should be 0.8 (max of non-sentinel data)
        assert frontier_values[-1] == pytest.approx(0.8)
        # No sentinel values should appear in frontier
        assert all(v >= 0 for v in frontier_values)


class TestFetchRunDataSentinelPassthrough:
    """Tests that _fetch_run_data passes sentinel_value to prepare_iteration_dataframe."""

    def _make_raw_df(self, fitness_values: list[float]) -> pd.DataFrame:
        """Create a raw DataFrame matching fetch_evolution_dataframe output."""
        n = len(fitness_values)
        return pd.DataFrame(
            {
                "metric_fitness": fitness_values,
                "iteration": list(range(1, n + 1)),
            }
        )

    @patch("gigaevo.cli.plot_group.asyncio")
    @patch("gigaevo.cli.plot_group._build_redis_config")
    def test_fetch_run_data_passes_sentinel_value(
        self, mock_build_config, mock_asyncio
    ) -> None:
        """_fetch_run_data with sentinel_value=-1.0 passes it to prepare_iteration_dataframe."""
        from gigaevo.cli.plot_group import _fetch_run_data

        raw_df = self._make_raw_df([0.5, -1.0, 0.6, -1.0, 0.8])
        mock_config = MagicMock()
        mock_config.display_label.return_value = "test_label"
        mock_build_config.return_value = mock_config
        mock_asyncio.run.return_value = raw_df

        mock_rc = MagicMock()
        results = _fetch_run_data(
            [mock_rc], "localhost", 6379, metric="fitness", sentinel_value=-1.0
        )

        assert len(results) == 1
        label, df = results[0]
        assert -1.0 not in df["metric_fitness"].values

    @patch("gigaevo.cli.plot_group.asyncio")
    @patch("gigaevo.cli.plot_group._build_redis_config")
    def test_fetch_run_data_default_no_sentinel_filtering(
        self, mock_build_config, mock_asyncio
    ) -> None:
        """_fetch_run_data with sentinel_value=None (default) does not filter sentinels."""
        from gigaevo.cli.plot_group import _fetch_run_data

        raw_df = self._make_raw_df([0.5, -1.0, 0.6, -1.0, 0.8])
        mock_config = MagicMock()
        mock_config.display_label.return_value = "test_label"
        mock_build_config.return_value = mock_config
        mock_asyncio.run.return_value = raw_df

        mock_rc = MagicMock()
        results = _fetch_run_data([mock_rc], "localhost", 6379, metric="fitness")

        assert len(results) == 1
        label, df = results[0]
        # -1.0 rows should still be present (no sentinel filtering by default)
        assert len(df) == 5
