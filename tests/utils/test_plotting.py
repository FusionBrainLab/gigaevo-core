"""Tests for gigaevo.utils.plotting — frontier annotation."""

from __future__ import annotations

from unittest.mock import MagicMock

from gigaevo.utils.plotting import annotate_frontier_points


class TestAnnotateFrontierPoints:
    def test_empty_arrays_no_error(self):
        ax = MagicMock()
        annotate_frontier_points(
            ax, [], [], minimize=False, max_annotations=3, color="blue"
        )
        ax.annotate.assert_not_called()

    def test_single_point_no_annotation(self):
        ax = MagicMock()
        annotate_frontier_points(
            ax, [0], [0.5], minimize=False, max_annotations=3, color="blue"
        )
        ax.annotate.assert_not_called()

    def test_flat_frontier_no_annotation(self):
        ax = MagicMock()
        annotate_frontier_points(
            ax,
            [0, 1, 2, 3],
            [0.5, 0.5, 0.5, 0.5],
            minimize=False,
            max_annotations=3,
            color="blue",
        )
        ax.annotate.assert_not_called()

    def test_significant_jumps_annotated(self):
        ax = MagicMock()
        x_vals = list(range(10))
        frontier_vals = [0.1, 0.1, 0.3, 0.3, 0.5, 0.5, 0.8, 0.8, 0.9, 0.9]
        annotate_frontier_points(
            ax,
            x_vals,
            frontier_vals,
            minimize=False,
            max_annotations=5,
            color="red",
        )
        assert ax.annotate.call_count > 0

    def test_max_annotations_limits_count(self):
        ax = MagicMock()
        x_vals = list(range(20))
        frontier_vals = [i * 0.05 for i in range(20)]
        annotate_frontier_points(
            ax,
            x_vals,
            frontier_vals,
            minimize=False,
            max_annotations=2,
            color="green",
        )
        assert ax.annotate.call_count <= 2
