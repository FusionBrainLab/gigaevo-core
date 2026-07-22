from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from problems.dag_tab.graph import FeatureGraph
from problems.dag_tab.problem_context import DagTabProblemContext
from problems.dag_tab.seed_loader import DagTabSeedLoader
from problems.tabular._common.tabular_data import ColumnSpec, Dataset


class _Storage:
    def __init__(self):
        self.added = []

    async def add(self, program):
        self.added.append(program)


def _dataset(name: str, task_type: str, n_classes: int | None, width: int) -> Dataset:
    rows = 6
    X = np.arange(rows * width, dtype=float).reshape(rows, width)
    y = np.arange(rows) % (n_classes or rows)
    return Dataset(
        name=name,
        task_type=task_type,
        n_classes=n_classes,
        X_train=X,
        y_train=y,
        X_val=X[:2],
        y_val=y[:2],
        X_test=X[:2],
        y_test=y[:2],
        columns=tuple(ColumnSpec(i, "numerical", None, None) for i in range(width)),
        train_size=rows,
    )


@pytest.mark.parametrize(
    ("name", "task_type", "n_classes", "width"),
    [
        ("regression", "regression", None, 8),
        ("binary", "binclass", 2, 14),
        ("multiclass", "multiclass", 7, 27),
    ],
)
async def test_seed_loader_builds_neutral_graph_for_any_task_type(
    monkeypatch, name, task_type, n_classes, width
):
    monkeypatch.setattr(
        "problems.dag_tab.seed_loader.load_dataset",
        lambda selected: _dataset(selected, task_type, n_classes, width),
    )
    storage = _Storage()

    programs = await DagTabSeedLoader(dataset=name).load(storage)

    graph = FeatureGraph.from_json(programs[0].code)
    assert graph.dataset == name
    assert graph.raw_columns == [f"x{i}" for i in range(width)]
    assert graph.nodes == []
    assert graph.estimator_columns == graph.raw_columns
    assert storage.added == programs


def test_context_combines_universal_abi_with_grouped_schema_only_dataset(
    tmp_path: Path,
):
    dag_tab = tmp_path / "dag_tab"
    tabular = tmp_path / "tabular" / "tabarena" / "adult"
    dag_tab.mkdir()
    tabular.mkdir(parents=True)
    (dag_tab / "task_description.txt").write_text("TASK\nUniversal FeatureGraph ABI")
    (dag_tab / "metrics.yaml").write_text("specs: {}")
    (tabular / "dataset_id.txt").write_text("tabarena-adult\n")
    (tabular / "task_description.txt").write_text(
        "TASK — TABULAR BINARY CLASSIFICATION (adult)\n\n"
        "DATASET — income prediction\n\n"
        "COLUMNS (assembled X[:, j])\n- [0] age\n- [1] hours\n"
    )

    description = DagTabProblemContext(
        dag_tab, dataset="tabarena-adult"
    ).task_description

    assert "Universal FeatureGraph ABI" in description
    assert "dataset id: tabarena-adult" in description
    assert "TABULAR BINARY CLASSIFICATION" in description
    assert "income prediction" in description
    assert "[0] age" in description
    assert "class Model:" not in description
