"""Import the official TabArena-v0.1 OpenML suite into the TabM layout.

Run with OpenML available, for example::

    uvx --from openml python import_tabarena.py DATA_ROOT \
        --openml-cache /some/dir/tabarena-openml
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any

import numpy as np
import openml  # type: ignore[import-not-found]
import pandas as pd

_SUITE = "tabarena-v0.1"
_PART_FOLDS = {"train": 2, "val": 1, "test": 0}


def _slug(name: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    if not value:
        raise ValueError(f"dataset name {name!r} has no filesystem-safe characters")
    return value


def _task_type(task: Any) -> tuple[str, tuple[str, ...] | None]:
    task_kind = task.task_type_id.name
    if task_kind == "SUPERVISED_REGRESSION":
        return "regression", None
    if task_kind != "SUPERVISED_CLASSIFICATION":
        raise ValueError(f"unsupported OpenML task type: {task_kind}")
    classes = tuple(str(label) for label in task.class_labels)
    if len(classes) < 2:
        raise ValueError(f"task {task.id} has fewer than two classes")
    return ("binclass" if len(classes) == 2 else "multiclass"), classes


def _encode_target(
    target: pd.Series, task_type: str, classes: tuple[str, ...] | None
) -> np.ndarray:
    if task_type == "regression":
        result = pd.to_numeric(target, errors="raise").to_numpy(dtype=np.float32)
        if np.isnan(result).any():
            raise ValueError("regression target contains missing values")
        return result

    assert classes is not None
    mapping = {label: index for index, label in enumerate(classes)}
    values = target.astype("string")
    if values.isna().any():
        raise ValueError("classification target contains missing values")
    unknown = sorted(set(values.astype(str)) - set(mapping))
    if unknown:
        raise ValueError(f"target contains labels absent from the task: {unknown}")
    return values.astype(str).map(mapping).to_numpy(dtype=np.int64)


def _official_fold_parts(task: Any, n_rows: int) -> dict[str, np.ndarray]:
    _repeats, folds, samples = task.get_split_dimensions()
    if folds != 3 or samples != 1:
        raise ValueError(
            f"task {task.id} has {folds} folds and {samples} samples; expected 3 and 1"
        )

    outer_train, official_test = task.get_train_test_split_indices(
        repeat=0, fold=0, sample=0
    )
    parts: dict[str, np.ndarray] = {}
    seen: set[int] = set()
    for part, fold in _PART_FOLDS.items():
        _outer_train, fold_test = task.get_train_test_split_indices(
            repeat=0, fold=fold, sample=0
        )
        index = np.asarray(fold_test, dtype=np.int64)
        if index.ndim != 1 or len(np.unique(index)) != len(index):
            raise ValueError(f"task {task.id} has invalid repeat-0 fold {fold}")
        overlap = seen.intersection(index.tolist())
        if overlap:
            raise ValueError(f"task {task.id} repeat-0 test folds overlap")
        seen.update(index.tolist())
        parts[part] = index

    if seen != set(range(n_rows)):
        raise ValueError(f"task {task.id} repeat-0 folds do not partition the dataset")
    if set(parts["test"].tolist()) != set(official_test):
        raise ValueError(f"task {task.id} test split differs from official r0f0")
    if set(parts["train"].tolist()) | set(parts["val"].tolist()) != set(outer_train):
        raise ValueError(f"task {task.id} train+val differs from official r0f0 train")
    return parts


def _categorical_array(frame: pd.DataFrame) -> np.ndarray:
    result = frame.astype(object)
    for column in result:
        result[column] = result[column].map(
            lambda value: None if pd.isna(value) else str(value)
        )
    return result.to_numpy(dtype=object)


def _is_binary(column: pd.Series) -> bool:
    values = column.dropna().unique()
    if len(values) == 0:
        return False
    try:
        return set(np.asarray(values, dtype=float).tolist()) <= {0.0, 1.0}
    except (TypeError, ValueError):
        return False


def convert_task(
    task: Any,
    dst_root: Path,
    *,
    prefix: str = "tabarena-",
    force: bool = False,
) -> Path:
    dataset = task.get_dataset()
    name = f"{prefix}{_slug(dataset.name)}"
    dst = dst_root / name
    if dst.exists() and not force:
        print(f"skip {dst} (already exists)")
        return dst

    X, target, categorical, _attribute_names = dataset.get_data(
        target=task.target_name,
        dataset_format="dataframe",
    )
    if not isinstance(X, pd.DataFrame) or not isinstance(target, pd.Series):
        raise TypeError(f"task {task.id} did not produce pandas data")
    if len(categorical) != X.shape[1]:
        raise ValueError(f"task {task.id} categorical metadata has the wrong width")

    task_type, classes = _task_type(task)
    y = _encode_target(target, task_type, classes)
    parts = _official_fold_parts(task, len(X))
    cat_flags = [
        bool(flag) or not pd.api.types.is_numeric_dtype(X[column].dtype)
        for column, flag in zip(X.columns, categorical)
    ]
    cat_columns = [column for column, flag in zip(X.columns, cat_flags) if flag]
    bin_columns = [
        column
        for column, flag in zip(X.columns, cat_flags)
        if not flag and _is_binary(X[column])
    ]
    num_columns = [
        column
        for column, flag in zip(X.columns, cat_flags)
        if not flag and column not in bin_columns
    ]

    tmp = dst_root / f".{name}.tabarena-import"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        for part, index in parts.items():
            split = X.iloc[index]
            numeric = (
                split[num_columns].to_numpy(dtype=np.float32)
                if num_columns
                else np.empty((len(split), 0), dtype=np.float32)
            )
            np.save(tmp / f"X_num_{part}.npy", numeric)
            if bin_columns:
                binary = split[bin_columns].to_numpy(dtype=np.float32)
                np.save(tmp / f"X_bin_{part}.npy", binary)
            if cat_columns:
                np.save(
                    tmp / f"X_cat_{part}.npy",
                    _categorical_array(split[cat_columns]),
                )
            np.save(tmp / f"Y_{part}.npy", y[index])

        info: dict[str, Any] = {
            "name": dataset.name,
            "id": f"tabarena-v0.1--openml-task-{task.id}--repeat-0",
            "task_type": task_type,
            "eval_metric": {
                "regression": "rmse",
                "binclass": "roc_auc",
                "multiclass": "log_loss",
            }[task_type],
            "n_num_features": len(num_columns),
            "n_bin_features": len(bin_columns),
            "n_cat_features": len(cat_columns),
            "train_size": len(parts["train"]),
            "val_size": len(parts["val"]),
            "test_size": len(parts["test"]),
            "assembled_column_names": [
                *(str(column) for column in num_columns),
                *(str(column) for column in bin_columns),
                *(str(column) for column in cat_columns),
            ],
            "source": {
                "benchmark": "TabArena-v0.1",
                "protocol": "TabArena-Lite-r0f0",
                "openml_suite_id": 457,
                "openml_task_id": task.id,
                "openml_dataset_id": dataset.id,
                "repeat": 0,
                "fold_assignment": _PART_FOLDS,
            },
        }
        if classes is not None:
            info["n_classes"] = len(classes)
            info["class_labels"] = list(classes)
        (tmp / "info.json").write_text(json.dumps(info, indent=4) + "\n")
        (tmp / "READY").touch()

        if dst.exists():
            shutil.rmtree(dst)
        tmp.rename(dst)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    print(dst)
    return dst


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dst", type=Path, help="TabM/GigaEvo data root")
    parser.add_argument("--openml-cache", type=Path)
    parser.add_argument("--prefix", default="tabarena-")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("OPENML_SKIP_PARQUET", "true")
    if args.openml_cache is not None:
        args.openml_cache.mkdir(parents=True, exist_ok=True)
        openml.config.set_root_cache_directory(str(args.openml_cache))
    args.dst.mkdir(parents=True, exist_ok=True)

    suite = openml.study.get_suite(_SUITE)
    print(f"Importing {len(suite.tasks)} tasks from {suite.name} (suite {suite.id})")
    for task_id in suite.tasks:
        task = openml.tasks.get_task(
            task_id,
            download_data=True,
            download_qualities=True,
            download_splits=True,
        )
        convert_task(task, args.dst, prefix=args.prefix, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
