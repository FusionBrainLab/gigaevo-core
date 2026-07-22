from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path

import numpy as np

_DATA_ENV = "GIGAEVO_TABULAR_DATA"
REGRESSION = "regression"
BINCLASS = "binclass"
MULTICLASS = "multiclass"
_MAX_VOCAB_LISTED = 40


class TabularDataError(RuntimeError):
    pass


@dataclass(frozen=True)
class ColumnSpec:
    index: int
    kind: str  # "numeric" | "binary" | "categorical"
    cardinality: int | None
    vocabulary: tuple[str, ...] | None  # vocabulary[code] == original value


@dataclass(frozen=True)
class Dataset:
    name: str
    task_type: str
    n_classes: int | None
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    columns: tuple[ColumnSpec, ...]
    train_size: int


_CACHE: dict[str, Dataset] = {}


def _data_root() -> Path:
    root = os.environ.get(_DATA_ENV)
    if not root:
        raise TabularDataError(
            f"{_DATA_ENV} is not set; point it at the directory holding the tabm "
            f"dataset folders (e.g. ~/tabm-data/data)."
        )
    p = Path(root)
    if not p.is_dir():
        raise TabularDataError(f"{_DATA_ENV}={root!r} is not a directory.")
    return p


def _load_npy(folder: Path, stem: str) -> np.ndarray | None:
    path = folder / f"{stem}.npy"
    if not path.is_file():
        return None
    return np.load(path, allow_pickle=True)


def _category_key(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "__MISSING__"
    return str(value)


def _levels(column: np.ndarray) -> tuple[str, ...]:
    """Train-only vocabulary, ordered by the column's own dtype.

    Sorting the string form would renumber integer categories ("10" before
    "2"), silently changing every published code for datasets that store
    categories as integers. Missing values have no natural position and
    sort last.
    """

    present = [v for v in column if _category_key(v) != "__MISSING__"]
    levels = [_category_key(v) for v in np.unique(present)] if present else []
    if len(present) < column.shape[0]:
        levels.append("__MISSING__")
    return tuple(dict.fromkeys(levels))


def _encode_categoricals(
    cat_tr: np.ndarray, cat_va: np.ndarray, cat_te: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[str, ...]]]:
    """Fit data-inferred vocabularies on train and map unseen values to -1."""

    n_cols = cat_tr.shape[1]
    codes_tr = np.empty(cat_tr.shape, dtype=np.float64)
    codes_va = np.empty(cat_va.shape, dtype=np.float64)
    codes_te = np.empty(cat_te.shape, dtype=np.float64)
    vocab: list[tuple[str, ...]] = []
    for j in range(n_cols):
        levels = _levels(cat_tr[:, j])
        mapping = {value: code for code, value in enumerate(levels)}

        def encode(values: np.ndarray) -> np.ndarray:
            return np.asarray(
                [mapping.get(_category_key(value), -1) for value in values],
                dtype=np.float64,
            )

        codes_tr[:, j] = encode(cat_tr[:, j])
        codes_va[:, j] = encode(cat_va[:, j])
        codes_te[:, j] = encode(cat_te[:, j])
        vocab.append(levels)
    return codes_tr, codes_va, codes_te, vocab


def _assemble(num, bin_, cat) -> np.ndarray:
    blocks = [b.astype(np.float64) for b in (num, bin_, cat) if b is not None]
    if not blocks:
        raise TabularDataError("dataset has no feature columns")
    return np.hstack(blocks)


def _build_columns(
    n_num: int, n_bin: int, vocab: list[tuple[str, ...]]
) -> tuple[ColumnSpec, ...]:
    cols: list[ColumnSpec] = []
    idx = 0
    for _ in range(n_num):
        cols.append(ColumnSpec(idx, "numeric", None, None))
        idx += 1
    for _ in range(n_bin):
        cols.append(ColumnSpec(idx, "binary", None, None))
        idx += 1
    for levels in vocab:
        cols.append(ColumnSpec(idx, "categorical", len(levels), levels))
        idx += 1
    return tuple(cols)


def load_dataset(name: str) -> Dataset:
    if name in _CACHE:
        return _CACHE[name]
    folder = _data_root() / name
    if not folder.is_dir():
        raise TabularDataError(f"Dataset folder not found: {folder}")
    info = json.loads((folder / "info.json").read_text())
    task_type = info["task_type"]
    n_num = int(info.get("n_num_features", 0))
    n_bin = int(info.get("n_bin_features", 0))
    n_cat = int(info.get("n_cat_features", 0))
    if "n_classes" in info:
        n_classes: int | None = int(info["n_classes"])
    elif task_type == BINCLASS:
        n_classes = 2
    else:
        n_classes = None

    def split(s: str):
        return (
            _load_npy(folder, f"X_num_{s}"),
            _load_npy(folder, f"X_bin_{s}"),
            _load_npy(folder, f"X_cat_{s}"),
        )

    num_tr, bin_tr, cat_tr = split("train")
    num_va, bin_va, cat_va = split("val")
    num_te, bin_te, cat_te = split("test")

    vocab: list[tuple[str, ...]] = []
    if n_cat > 0 and cat_tr is not None:
        cat_tr, cat_va, cat_te, vocab = _encode_categoricals(cat_tr, cat_va, cat_te)
    else:
        cat_tr = cat_va = cat_te = None

    X_train = _assemble(num_tr, bin_tr, cat_tr)
    X_val = _assemble(num_va, bin_va, cat_va)
    X_test = _assemble(num_te, bin_te, cat_te)

    ycast = np.float64 if task_type == REGRESSION else np.int64
    y_train = np.asarray(_load_npy(folder, "Y_train"), dtype=ycast)
    y_val = np.asarray(_load_npy(folder, "Y_val"), dtype=ycast)
    y_test = np.asarray(_load_npy(folder, "Y_test"), dtype=ycast)

    for arr in (X_train, X_val, X_test, y_train, y_val, y_test):
        arr.setflags(write=False)
    ds = Dataset(
        name=name,
        task_type=task_type,
        n_classes=n_classes,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        columns=_build_columns(n_num, n_bin, vocab),
        train_size=X_train.shape[0],
    )
    _CACHE[name] = ds
    return ds


def column_spec(name: str) -> tuple[ColumnSpec, ...]:
    return load_dataset(name).columns


def _fmt_vocab(levels: tuple[str, ...]) -> str:
    if len(levels) <= _MAX_VOCAB_LISTED:
        body = " ".join(f"{code}={val}" for code, val in enumerate(levels))
        return f"{len(levels)} levels: {body}"
    head = " ".join(f"{code}={levels[code]}" for code in range(6))
    return f"{len(levels)} levels: {head} ... {len(levels) - 1}={levels[-1]}"


def _named_columns(cols: tuple[ColumnSpec, ...], names: dict) -> str:
    lines: list[str] = ["COLUMNS (assembled X[:, j], left to right)"]
    for c in cols:
        meta = names.get(c.index, names.get(str(c.index), {})) or {}
        label = str(meta.get("name", "")).strip()
        desc = str(meta.get("desc", "")).strip()
        if c.kind == "categorical":
            assert c.vocabulary is not None
            vocab_note = f"categorical (integer code), {_fmt_vocab(c.vocabulary)}"
            note = f"{desc} — {vocab_note}" if desc else vocab_note
        elif c.kind == "binary":
            note = desc or "binary (0/1)"
        else:
            note = desc or "numeric"
        if label:
            lines.append(f"- [{c.index}] {label:<11} {note}".rstrip())
        else:
            lines.append(f"- [{c.index}]{'':<6} {note}")
    return "\n".join(lines)


def describe_columns(name: str, names: dict | None = None) -> str:
    ds = load_dataset(name)
    if names:
        return _named_columns(ds.columns, names)
    lines: list[str] = ["COLUMNS (assembled X[:, j], left to right)"]
    i = 0
    cols = ds.columns
    while i < len(cols):
        c = cols[i]
        if c.kind in ("numeric", "binary"):
            j = i
            while j + 1 < len(cols) and cols[j + 1].kind == c.kind:
                j += 1
            span = f"[{c.index}]" if j == i else f"[{c.index}-{cols[j].index}]"
            label = "numeric" if c.kind == "numeric" else "binary (0/1)"
            lines.append(f"- {span:<10} {label}")
            i = j + 1
        else:
            assert c.vocabulary is not None
            lines.append(
                f"- [{c.index}]{'':<6} categorical (integer code), {_fmt_vocab(c.vocabulary)}"
            )
            i += 1
    return "\n".join(lines)
