"""Load MuSiQue dataset and save train/test JSONL files for evolution."""

import os
from pathlib import Path

from datasets import load_dataset

SEED = 42
TRAIN_SAMPLES = 1000
TEST_SAMPLES = 300

_DIR = Path(__file__).parent

HF_CANDIDATES: list[tuple[str, str | None]] = [
    ("dgslibisey/MuSiQue", None),
    ("KUNLP/MuSiQue", None),
    ("musique", "answerable"),
    ("musique", "all"),
]


def _load_source_dataset():
    """Load MuSiQue from local JSON/JSONL path or HuggingFace Hub."""
    local_path = os.environ.get("MUSIQUE_DATA_PATH")
    if local_path:
        ds = load_dataset("json", data_files=local_path, split="train")
        return ds, f"local:{local_path}"

    errors = []
    for dataset_name, dataset_config in HF_CANDIDATES:
        try:
            if dataset_config is None:
                ds = load_dataset(dataset_name, split="train")
                return ds, f"{dataset_name} (default)"
            ds = load_dataset(dataset_name, dataset_config, split="train")
            return ds, f"{dataset_name}/{dataset_config}"
        except Exception as exc:  # pragma: no cover
            errors.append(f"{dataset_name}/{dataset_config}: {exc}")

    msg = "\n".join(errors)
    raise RuntimeError(
        "Failed to load MuSiQue dataset from all known sources.\n"
        f"Tried:\n{msg}\n\n"
        "Set MUSIQUE_DATA_PATH=/path/to/musique.jsonl to use local data."
    )


def main():
    dataset, source = _load_source_dataset()
    dataset = dataset.shuffle(seed=SEED)

    required_samples = TRAIN_SAMPLES + TEST_SAMPLES
    if len(dataset) < required_samples:
        raise ValueError(
            f"Dataset has only {len(dataset)} samples; need at least {required_samples}"
        )

    train = dataset.select(range(TRAIN_SAMPLES))
    test = dataset.select(range(TRAIN_SAMPLES, TRAIN_SAMPLES + TEST_SAMPLES))

    train_path = str(_DIR / "MuSiQue_train.jsonl")
    test_path = str(_DIR / "MuSiQue_test.jsonl")
    train.to_json(train_path)
    test.to_json(test_path)

    print(
        f"Dataset ready from {source}: "
        f"{train_path} ({len(train):,} samples), "
        f"{test_path} ({len(test):,} samples)"
    )


if __name__ == "__main__":
    main()
