"""Convert ``uvx tabred download`` output to the TabM dataset layout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

_PARTS = ("train", "val", "test")
_FEATURES = ("num", "bin", "cat")


def _task_type(info: dict, y: np.ndarray) -> tuple[str, int | None]:
    raw = str(info.get("task_type", info.get("task", {}).get("type", ""))).lower()
    if raw == "regression":
        return "regression", None

    classes = np.unique(y)
    if len(classes) < 2:
        raise ValueError("classification target must contain at least two classes")
    return ("binclass" if len(classes) == 2 else "multiclass"), len(classes)


def _load_indices(src: Path, n_rows: int) -> dict[str, np.ndarray]:
    split_dir = src / "splits" / "default"
    result: dict[str, np.ndarray] = {}
    seen: set[int] = set()
    for part in _PARTS:
        index = np.asarray(np.load(split_dir / f"{part}.npy"), dtype=np.int64)
        if index.ndim != 1 or len(np.unique(index)) != len(index):
            raise ValueError(f"invalid {part} indices in {split_dir}")
        if len(index) and (index.min() < 0 or index.max() >= n_rows):
            raise ValueError(f"out-of-range {part} indices in {split_dir}")
        overlap = seen.intersection(index.tolist())
        if overlap:
            raise ValueError(f"overlapping default splits in {split_dir}")
        seen.update(index.tolist())
        result[part] = index
    return result


def convert_dataset(src: Path, dst_root: Path, *, force: bool = False) -> Path:
    name = src.name
    dst = dst_root / name
    if dst.exists() and not force:
        raise FileExistsError(f"{dst} exists; pass --force to replace it")

    arrays: dict[str, np.ndarray] = {}
    for kind in _FEATURES:
        path = src / f"x_{kind}.npy"
        if path.is_file():
            array = np.load(path, mmap_mode="r")
            if array.ndim != 2:
                raise ValueError(f"{path} must be a 2-D array")
            arrays[kind] = array

    if not arrays:
        raise ValueError(f"{src} has no feature arrays")
    row_counts = {len(array) for array in arrays.values()}

    y = np.load(src / "y.npy", mmap_mode="r")
    if y.ndim != 1:
        raise ValueError(f"{src / 'y.npy'} must be a 1-D array")
    row_counts.add(len(y))
    if len(row_counts) != 1:
        raise ValueError(f"feature and target row counts differ in {src}")

    indices = _load_indices(src, len(y))
    native_info = json.loads((src / "info.json").read_text())
    task_type, n_classes = _task_type(native_info, y)
    if n_classes is not None:
        classes = np.unique(y)
        if not np.array_equal(classes, np.arange(n_classes)):
            raise ValueError(f"classification labels in {src} are not 0..K-1")

    tmp = dst_root / f".{name}.tabm-import"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        for part, index in indices.items():
            for kind, array in arrays.items():
                np.save(tmp / f"X_{kind}_{part}.npy", array[index])
            target = np.asarray(y[index])
            target = target.astype(
                np.float32 if task_type == "regression" else np.int64,
                copy=False,
            )
            np.save(tmp / f"Y_{part}.npy", target)

        info = {
            "name": name,
            "id": f"{name}--tabred-default",
            "task_type": task_type,
            "n_num_features": arrays.get("num", np.empty((0, 0))).shape[1],
            "n_bin_features": arrays.get("bin", np.empty((0, 0))).shape[1],
            "n_cat_features": arrays.get("cat", np.empty((0, 0))).shape[1],
            "train_size": len(indices["train"]),
            "val_size": len(indices["val"]),
            "test_size": len(indices["test"]),
        }
        if n_classes is not None:
            info["n_classes"] = n_classes
        (tmp / "info.json").write_text(json.dumps(info, indent=4) + "\n")
        (tmp / "READY").touch()

        if dst.exists():
            shutil.rmtree(dst)
        tmp.rename(dst)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return dst


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("src", type=Path, help="Directory produced by uvx tabred")
    parser.add_argument("dst", type=Path, help="TabM/GigaEvo data root")
    parser.add_argument("names", nargs="*", help="Dataset names (default: all present)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    names = args.names or sorted(
        path.name for path in args.src.iterdir() if (path / "info.json").is_file()
    )
    args.dst.mkdir(parents=True, exist_ok=True)
    for name in names:
        print(convert_dataset(args.src / name, args.dst, force=args.force))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
