"""Regenerate per-dataset task_description.txt from the dataset on disk.

Run once (with GIGAEVO_TABULAR_DATA set) after the dataset dirs exist:
    python problems/tabular/_common/gen_task_descriptions.py
The generated column block enumerates categorical value vocabularies so a
program can decode/one-hot the integer-coded categorical columns.
"""

from __future__ import annotations

from pathlib import Path
import sys

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import tabular_data  # noqa: E402

_TABULAR_ROOT = _HERE.parent

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

_RETURN = {
    "regression": "- return a 1D float array of length len(X_query)",
    "binclass": "- return a 2D array (len(X_query), n_classes) of class probabilities (column j = P(class j)); 1D int labels also accepted",
    "multiclass": "- return a 2D array (len(X_query), n_classes) of class probabilities (column j = P(class j)); 1D int labels also accepted",
}


def render(name: str, task_type: str) -> str:
    ds = tabular_data.load_dataset(name)
    header = _HEADERS[task_type].format(name=name, k=ds.n_classes)
    cols = tabular_data.describe_columns(name)
    parts = [
        header,
        "",
        "CONTRACT",
        "- entrypoint() -> Model class; Model() takes no arguments",
        "- Model().fit_predict(X_train, y_train, X_val, y_val, X_query) -> np.ndarray",
        _RETURN[task_type],
        "- all predictions finite (no NaN, no inf)",
        "",
        cols,
        "",
        "PROTOCOL",
        "- (X_train, y_train, X_val, y_val) are training data; X_query is the scoring slice",
        "- categorical columns are integer codes; one-hot or target-encode them if useful, or pass through to tree models",
        "",
        "CONSTRAINTS",
        "- fix all random seeds (numpy + library random_state/random_seed/seed)",
        "- no dataset peek (no fetch_*, openml, pickled caches)",
        "- no print()/stdout in fit_predict; silence library logs",
    ]
    return "\n".join(parts) + "\n"


def main() -> int:
    for task_type, names in _DATASETS.items():
        for name in names:
            out = _TABULAR_ROOT / name / "task_description.txt"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(render(name, task_type))
            print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
