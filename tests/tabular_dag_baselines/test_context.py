from pathlib import Path

from problems.dag_tab.problem_context import DagTabProblemContext

ROOT = Path(__file__).parents[2]


def test_nested_problem_context_includes_local_abi_and_canonical_dataset_text():
    context = DagTabProblemContext(
        ROOT / "problems" / "tabular_dag_baselines" / "lightgbm",
        dataset="california",
    )

    description = context.task_description

    assert "LightGBM" in description
    assert "SELECTED DATASET CONTEXT" in description
    assert "dataset id: california" in description
    assert "x0" in description
