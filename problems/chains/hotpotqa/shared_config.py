"""Shared configuration for HotpotQA chain evolution experiments."""

import json
import random
from pathlib import Path


# --- LLM Configuration ---

LLM_CONFIG = {
    "model": "Qwen/Qwen3-8B",
    "max_cost": 10.0,
    "model_pricing": {
        "prompt": 0.05,
        "completion": 0.25,
    },
    "generation_kwargs": {
        "temperature": 0.6,
        "top_p": 0.95,
        "extra_body": {
            "top_k": 20,
        },
    },
    "client_kwargs": {
        "api_key": "None",
        "base_url": "http://10.225.185.92:8000/v1",
    },
}

# --- Dataset Configuration ---

_BASE_DIR = Path(__file__).parent

DATASET_CONFIG = {
    "train_path": str(_BASE_DIR / "dataset" / "HotpotQA_train.jsonl"),
    "test_path": str(_BASE_DIR / "dataset" / "HotpotQA_test.jsonl"),
    "target_field": "answer",
}

# --- Corpus Configuration ---

CORPUS_PATH = str(_BASE_DIR / "dataset" / "wiki17_abstracts.jsonl.gz")
BM25S_INDEX_DIR = str(_BASE_DIR / "dataset" / "bm25s_index")


# --- Data Loading Utilities ---


def load_jsonl(path: str) -> list[dict]:
    """Load JSONL file as list of dicts."""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def preprocess_sample(sample: dict) -> dict:
    """Preprocess a single sample for chain execution."""
    return {
        "question": sample["question"],
        "answer": sample["answer"],
    }


def outer_context_builder(sample: dict) -> str:
    """Build the data context string from a preprocessed sample.

    Returns a plain string (the question) used as outer_context in the chain.
    """
    return sample["question"]


def load_context(
    n_samples: int | None = None,
    n_train: int | None = None,
    n_val: int | None = None,
    n_pool: int | None = None,
    n_held_out_val: int | None = None,
    seed: int = 42,
) -> dict:
    """Load dataset for validation.

    Three modes:
    - Legacy (n_samples or default): returns single train_dataset.
    - Split (n_train + n_val): returns fixed train_dataset + val_dataset.
    - Pool (n_pool + n_held_out_val): returns train_pool + val_dataset for
      rotating random subset evaluation.

    BM25 index is not loaded here — the retrieval module lazy-loads it
    from disk inside the subprocess. Only paths are returned.
    """
    raw_samples = load_jsonl(DATASET_CONFIG["train_path"])

    if n_pool is not None and n_held_out_val is not None:
        total = raw_samples[: n_pool + n_held_out_val]
        processed = [preprocess_sample(s) for s in total]
        rng = random.Random(seed)
        indices = list(range(len(processed)))
        rng.shuffle(indices)
        return {
            "train_pool": [processed[i] for i in indices[:n_pool]],
            "val_dataset": [processed[i] for i in indices[n_pool:]],
            "target_field": DATASET_CONFIG["target_field"],
            "bm25s_index_dir": BM25S_INDEX_DIR,
            "corpus_path": CORPUS_PATH,
        }

    if n_train is not None and n_val is not None:
        total = raw_samples[: n_train + n_val]
        processed = [preprocess_sample(s) for s in total]
        rng = random.Random(seed)
        indices = list(range(len(processed)))
        rng.shuffle(indices)
        return {
            "train_dataset": [processed[i] for i in indices[:n_train]],
            "val_dataset": [processed[i] for i in indices[n_train:]],
            "target_field": DATASET_CONFIG["target_field"],
            "bm25s_index_dir": BM25S_INDEX_DIR,
            "corpus_path": CORPUS_PATH,
        }

    # Legacy mode
    n = n_samples if n_samples is not None else 300
    total = raw_samples[:n]
    processed = [preprocess_sample(s) for s in total]
    return {
        "train_dataset": processed,
        "target_field": DATASET_CONFIG["target_field"],
        "bm25s_index_dir": BM25S_INDEX_DIR,
        "corpus_path": CORPUS_PATH,
    }
