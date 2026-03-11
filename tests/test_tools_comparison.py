import pandas as pd

from tools.comparison import _select_frontier_improvements


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
