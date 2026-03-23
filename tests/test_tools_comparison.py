from pathlib import Path

import pandas as pd

from tools.comparison import _select_frontier_improvements, plot_comparison


def test_select_frontier_improvements_maximize_skips_initial_point():
    df = pd.DataFrame(
        {
            "metadata_iteration": [1, 2, 3, 4, 5],
            "frontier_fitness": [0.1, 0.1, 0.2, 0.2, 0.35],
        }
    )

    improvements = _select_frontier_improvements(
        df, iteration_col="metadata_iteration", minimize=False
    )

    assert improvements["metadata_iteration"].tolist() == [3, 5]
    assert improvements["frontier_fitness"].tolist() == [0.2, 0.35]


def test_select_frontier_improvements_minimize_keeps_only_decreases():
    df = pd.DataFrame(
        {
            "metadata_iteration": [1, 2, 3, 4, 5],
            "frontier_fitness": [10.0, 10.0, 8.5, 8.5, 7.25],
        }
    )

    improvements = _select_frontier_improvements(
        df, iteration_col="metadata_iteration", minimize=True
    )

    assert improvements["metadata_iteration"].tolist() == [3, 5]
    assert improvements["frontier_fitness"].tolist() == [8.5, 7.25]


def test_plot_comparison_uses_custom_output_stem(tmp_path: Path):
    prepared_df = pd.DataFrame(
        {
            "metadata_iteration": [1, 2, 3],
            "running_mean_fitness": [0.1, 0.2, 0.3],
            "running_std_fitness": [0.01, 0.02, 0.03],
            "frontier_fitness": [0.1, 0.2, 0.3],
        }
    )

    saved_paths = plot_comparison(
        [("demo-run", prepared_df)],
        output_folder=tmp_path,
        output_stem="tracked_run",
        save_plots=True,
        show_plot=False,
    )

    assert saved_paths == (
        tmp_path / "tracked_run.png",
        tmp_path / "tracked_run.pdf",
    )
    assert (tmp_path / "tracked_run.png").exists()
    assert (tmp_path / "tracked_run.pdf").exists()
