"""Shared configuration for HotpotQA chain evolution experiments."""

import json
import os
from pathlib import Path
import random
from typing import Literal

from tools.no_proxy import ensure_no_proxy

ensure_no_proxy()

# --- LLM Configuration ---
# All chain requests go through the LiteLLM proxy (INTERNAL_IP:4000).
# Start the proxy with: bash tools/litellm.sh --background
_CHAIN_URL = os.environ.get("HOTPOTQA_CHAIN_URL", "http://localhost:8000/v1")

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
        "api_key": "sk-gigaevo",
        "base_url": _CHAIN_URL,
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

_CORPUS_PKL = _BASE_DIR / "dataset" / "wiki17_abstracts.jsonl.passages.pkl"
_CORPUS_GZ = _BASE_DIR / "dataset" / "wiki17_abstracts.jsonl.gz"
CORPUS_PATH = str(_CORPUS_PKL if _CORPUS_PKL.exists() else _CORPUS_GZ)
BM25S_INDEX_DIR = str(_BASE_DIR / "dataset" / "bm25s_index")

# --- Retriever Configuration ---
# Switch retriever here. "bm25" uses the pre-built BM25s index (default).
# "colbert" uses ColBERTv2 loaded in-process via colbert-ai.
# Build the ColBERT index first: python dataset/build_colbert_index.py
RETRIEVER: Literal["bm25", "colbert"] = "bm25"
# ColBERT saves index to {root}/{experiment}/indexes/{name} where experiment="hotpotqa".
# ColBERTRetriever passes index_dir.parent as root and index_dir.name as index name.
# So index_dir.parent must be the repo-level experiments/ dir.
_REPO_ROOT = _BASE_DIR.parent.parent.parent  # .../problems/chains/hotpotqa -> repo root
COLBERT_INDEX_DIR = str(_REPO_ROOT / "experiments" / "colbert_index")
COLBERT_CHECKPOINT = "colbert-ir/colbertv2.0"


def build_retriever(k: int = 7):
    """Instantiate the retriever selected by the RETRIEVER constant.

    When RETRIEVER == "colbert", checks HOTPOTQA_COLBERT_SERVER_URL first.
    If set, returns a ColBERTServerRetriever that proxies to the running
    colbert_server.py process (recommended for exec_runner workers — avoids
    loading the 15-20 GB index in every subprocess).  Falls back to the
    in-process ColBERTRetriever if the env var is absent.
    """
    from problems.chains.hotpotqa.utils.retrieval import (
        BM25Retriever,
        ColBERTRetriever,
        ColBERTServerRetriever,
    )

    if RETRIEVER == "colbert":
        server_url = os.environ.get("HOTPOTQA_COLBERT_SERVER_URL", "")
        if server_url:
            return ColBERTServerRetriever(server_url, k=k)
        return ColBERTRetriever(COLBERT_INDEX_DIR, checkpoint=COLBERT_CHECKPOINT, k=k)
    return BM25Retriever(BM25S_INDEX_DIR, CORPUS_PATH, k=k)


# --- Data Loading Utilities ---


def load_jsonl(path: str) -> list[dict]:
    """Load JSONL file as list of dicts."""
    samples = []
    with open(path, encoding="utf-8") as f:
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
