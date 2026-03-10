"""Build per-task BM25 indices from MuSiQue dataset passages.

Despite the legacy filename, this script does not download Wikipedia.
It builds one local BM25 index per task/sample from dataset-provided passages.

Usage:
    python -m problems.chains.musique_retrieval.dataset.download_corpus
"""

import os
import sys
from pathlib import Path


def _ensure_repo_root_on_path() -> None:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "problems").is_dir():
            parent_str = str(parent)
            if parent_str not in sys.path:
                sys.path.insert(0, parent_str)
            return


def _collect_passages_by_task(raw_samples: list[dict]) -> dict[str, list[str]]:
    from problems.chains.musique_retrieval.shared_config import preprocess_sample

    passages_by_task: dict[str, list[str]] = {}
    for i, sample in enumerate(raw_samples):
        processed = preprocess_sample(sample, sample_idx=i)
        task_id = processed["task_id"]
        passages = processed["passages"]
        if task_id not in passages_by_task:
            passages_by_task[task_id] = passages
    return passages_by_task


def main() -> None:
    _ensure_repo_root_on_path()

    from problems.chains.musique_retrieval.shared_config import (
        DATASET_CONFIG,
        TASK_BM25S_INDEX_DIR,
        load_jsonl,
    )
    from problems.chains.musique_retrieval.utils.retrieval import build_task_indices

    train_path = Path(DATASET_CONFIG["train_path"])
    test_path = Path(DATASET_CONFIG["test_path"])
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "MuSiQue train/test files are missing. Run:\n"
            "  python -m problems.chains.musique_retrieval.dataset.load_dataset"
        )

    train_raw = load_jsonl(str(train_path))
    test_raw = load_jsonl(str(test_path))
    passages_by_task = _collect_passages_by_task(train_raw + test_raw)

    force = os.environ.get("MUSIQUE_TASK_INDEX_FORCE_REBUILD", "").strip() == "1"
    build_task_indices(
        passages_by_task,
        TASK_BM25S_INDEX_DIR,
        force_rebuild=force,
    )

    print(
        f"Per-task BM25 index is ready: {TASK_BM25S_INDEX_DIR} "
        f"({len(passages_by_task):,} tasks)"
    )


if __name__ == "__main__":
    main()
