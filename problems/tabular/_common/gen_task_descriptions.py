"""Regenerate per-dataset task_description.txt from the dataset on disk.

Run once (with GIGAEVO_TABULAR_DATA set) after the dataset dirs exist:
    python problems/tabular/_common/gen_task_descriptions.py
The generated column block enumerates categorical value vocabularies so a
program can decode/one-hot the integer-coded categorical columns.
"""

from __future__ import annotations

from pathlib import Path
import sys

import yaml

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import tabular_data  # noqa: E402

_TABULAR_ROOT = _HERE.parent
_SEMANTICS_PATH = _HERE / "column_semantics.yaml"

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

# General, model-agnostic guidance to steer the mutation LLM. Kept deliberately
# broad so it informs without prescribing one architecture (which would collapse
# population diversity).
_STRATEGY_COMMON = [
    "- feature engineering is the primary lever — go well beyond the raw columns: build ratios, interactions, per-unit normalizations, aggregates, binning, and target/count encodings",
    "- when the COLUMNS section gives semantic meaning, use it to design domain-specific features (combine related quantities, encode the structure the domain implies); creative, dataset-tailored transforms are encouraged",
    "- gradient-boosted trees (LightGBM/XGBoost/CatBoost) are a strong tabular baseline; other families are worth trying too — regularized linear models with rich feature engineering, kNN, and neural nets/TabM",
    "- blend diverse models (different families or seeds) — averaging or stacking usually beats any single learner",
    "- hold the validation split out to choose hyperparameters and early-stopping rounds; once those are fixed, optionally refit the final model on train+val combined before scoring X_query",
]

_STRATEGY_OUTPUT = {
    "regression": "- for a skewed target, consider a log/power transform of y and clip predictions back to the observed target range",
    "binclass": "- return well-calibrated probabilities and handle class imbalance if present",
    "multiclass": "- return well-calibrated per-class probabilities and handle class imbalance if present",
}


def _semantics() -> dict:
    if not _SEMANTICS_PATH.is_file():
        return {}
    return yaml.safe_load(_SEMANTICS_PATH.read_text()) or {}


def render(name: str, task_type: str) -> str:
    ds = tabular_data.load_dataset(name)
    header = _HEADERS[task_type].format(name=name, k=ds.n_classes)
    sem = _semantics().get(name) or {}
    cols = tabular_data.describe_columns(name, names=sem.get("columns"))
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
        "- categorical columns are integer codes; one-hot or target-encode them if useful, or pass through to tree models",
        "",
        "STRATEGY (general heuristics; adapt to the data)",
        *_STRATEGY_COMMON,
        _STRATEGY_OUTPUT[task_type],
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
