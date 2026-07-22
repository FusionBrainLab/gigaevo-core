"""Materialize tabular problem directories from datasets on disk.

Run once (with GIGAEVO_TABULAR_DATA set) after the dataset dirs exist:
    python problems/tabular/_common/gen_task_descriptions.py
    python problems/tabular/_common/gen_task_descriptions.py --collection tabm
    python problems/tabular/_common/gen_task_descriptions.py --collection tabarena
    python problems/tabular/_common/gen_task_descriptions.py --collection tabred
The generated column block describes feature types and categorical code
vocabularies while compacting long runs of numbered feature names.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import yaml

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import tabular_data  # noqa: E402

_TABULAR_ROOT = _HERE.parent
_SEMANTICS_PATH = _HERE / "column_semantics.yaml"
_DATA_ENV = "GIGAEVO_TABULAR_DATA"
_COLLECTIONS = ("tabm", "tabred", "tabarena")

_DATASETS = {
    "regression": ["california", "house", "diamond", "black-friday", "microsoft"],
    "binclass": ["adult", "churn", "higgs-small"],
    "multiclass": ["otto", "covtype2"],
}

_HEADERS = {
    "regression": "TASK — TABULAR REGRESSION ({name})",
    "binclass": "TASK — TABULAR BINARY CLASSIFICATION ({name})",
    "multiclass": "TASK — TABULAR MULTICLASS CLASSIFICATION ({name}, {k} classes)",
}

_REG_SKELETON = """```python
import numpy as np

class Model:
    def fit_predict(self, X_train, y_train, X_val, y_val, X_query) -> np.ndarray:
        ...
        return preds  # 1D float array, shape (len(X_query),)

def entrypoint() -> type:
    return Model
```"""

_CLF_SKELETON = """```python
import numpy as np

class Model:
    def fit_predict(self, X_train, y_train, X_val, y_val, X_query) -> np.ndarray:
        ...
        return proba  # 2D float array, shape (len(X_query), n_classes); column j = P(class j)

def entrypoint() -> type:
    return Model
```"""

_SKELETON = {
    "regression": _REG_SKELETON,
    "binclass": _CLF_SKELETON,
    "multiclass": _CLF_SKELETON,
}


def _semantics() -> dict:
    if not _SEMANTICS_PATH.is_file():
        return {}
    return yaml.safe_load(_SEMANTICS_PATH.read_text()) or {}


def _data_root() -> Path:
    value = os.environ.get(_DATA_ENV)
    if not value:
        raise ValueError(f"{_DATA_ENV} must point at the tabular data root")
    root = Path(value)
    if not root.is_dir():
        raise ValueError(f"{_DATA_ENV}={value!r} is not a directory")
    return root


def _dataset_info(name: str) -> dict:
    return json.loads((_data_root() / name / "info.json").read_text())


def _collection_for(name: str, info: dict) -> str | None:
    identifier = str(info.get("id", ""))
    source = str((info.get("source") or {}).get("benchmark", "")).lower()
    if name.startswith("tabarena-") or source.startswith("tabarena"):
        return "tabarena"
    if identifier.endswith("--tabred-default") or source.startswith("tabred"):
        return "tabred"
    return "tabm"


def _discover(collection: str) -> dict[str, list[str]]:
    datasets: dict[str, list[str]] = {task_type: [] for task_type in _SKELETON}
    for folder in sorted(_data_root().iterdir()):
        if not folder.is_dir():
            continue
        info_path = folder / "info.json"
        if not info_path.is_file():
            continue
        info = json.loads(info_path.read_text())
        if _collection_for(folder.name, info) != collection:
            continue
        task_type = info.get("task_type")
        if task_type not in datasets:
            raise ValueError(f"{folder.name} has unsupported task_type={task_type!r}")
        datasets[task_type].append(folder.name)
    if not any(datasets.values()):
        raise ValueError(f"no {collection!r} datasets found in {_data_root()}")
    return datasets


def _metadata_columns(name: str) -> dict[int, dict[str, str]]:
    names = _dataset_info(name).get("assembled_column_names") or []
    return {index: {"name": str(value)} for index, value in enumerate(names)}


def render(name: str, task_type: str) -> str:
    ds = tabular_data.load_dataset(name)
    header = _HEADERS[task_type].format(name=name, k=ds.n_classes)
    sem = _semantics().get(name) or {}
    column_metadata = sem.get("columns") or _metadata_columns(name)
    cols = tabular_data.describe_columns(name, names=column_metadata)
    parts = [header, ""]
    if sem.get("source"):
        parts += [f"DATASET — {sem['source']}", ""]
    parts += [
        "CONTRACT",
        _SKELETON[task_type],
        "- Model() takes no constructor arguments",
        "- all predictions finite (no NaN, no inf)",
        "",
        cols,
        "",
        "PROTOCOL",
        "- (X_train, y_train) and (X_val, y_val) are both labeled and may both be used to fit; X_query is the unlabeled scoring slice — prediction quality on it is the only objective",
        "- categorical columns are integer codes; train-derived code vocabularies are listed above when available, and unseen values use -1",
        "",
        "CONSTRAINTS",
        "- fix all random seeds (numpy + library random_state/random_seed/seed)",
        "- no dataset peek (no fetch_*, openml, pickled caches)",
        "- no print()/stdout in fit_predict; silence library logs",
    ]
    return "\n".join(parts) + "\n"


def _ensure_link(path: Path, target: str, *, directory: bool = False) -> None:
    if path.is_symlink():
        if os.readlink(path) == target:
            return
        path.unlink()
    if path.exists():
        raise ValueError(f"refusing to replace existing {path}")
    path.symlink_to(target, target_is_directory=directory)


def materialize(name: str, task_type: str, *, collection: str | None = None) -> Path:
    visible_name = name.removeprefix("tabarena-") if collection == "tabarena" else name
    out = (
        _TABULAR_ROOT / collection / visible_name
        if collection
        else _TABULAR_ROOT / name
    )
    out.mkdir(parents=True, exist_ok=True)
    relative_common = os.path.relpath(_HERE, out)
    _ensure_link(out / "validate.py", f"{relative_common}/validate.py")
    _ensure_link(out / "score_test.py", f"{relative_common}/score_test.py")
    _ensure_link(out / "metrics.yaml", f"{relative_common}/metrics_{task_type}.yaml")
    _ensure_link(
        out / "initial_programs",
        f"{relative_common}/seeds/{task_type}",
        directory=True,
    )
    if collection:
        (out / "dataset_id.txt").write_text(f"{name}\n")
    (out / "task_description.txt").write_text(render(name, task_type))
    tabular_data._CACHE.pop(name, None)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", choices=_COLLECTIONS)
    args = parser.parse_args()

    datasets = _discover(args.collection) if args.collection else _DATASETS
    for task_type, names in datasets.items():
        for name in names:
            print(f"wrote {materialize(name, task_type, collection=args.collection)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
