"""Shared configuration for HoVer chain evolution experiments."""

import json
import os
from pathlib import Path

# --- Squid Proxy Fix (CRITICAL) ---
# The system Squid proxy intercepts Python HTTP to internal IPs.
# Must bypass before any HTTP imports.
_no_proxy = os.environ.get("NO_PROXY", "")
for _ip in [
    "INTERNAL_IP",
    "10.225.185.235",
    "INTERNAL_IP",
    "INTERNAL_IP",
    "INTERNAL_IP",
    "10.225.51.251",
]:
    if _ip not in _no_proxy:
        _no_proxy = ",".join(filter(None, [_no_proxy, _ip]))
os.environ["NO_PROXY"] = _no_proxy
os.environ["no_proxy"] = _no_proxy

# --- LLM Configuration ---

# HOVER_CHAIN_URL overrides the chain-execution endpoint at runtime.
# Each run should get its own endpoint to avoid contention:
#   export HOVER_CHAIN_URL="http://localhost:8000/v1"
_CHAIN_URL = os.environ.get("HOVER_CHAIN_URL", "http://localhost:8000/v1")

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
        "base_url": _CHAIN_URL,
    },
}

# --- Dataset Configuration ---

_BASE_DIR = Path(__file__).parent

DATASET_CONFIG = {
    "train_path": str(_BASE_DIR / "dataset" / "HoVer_train.jsonl"),
    "test_path": str(_BASE_DIR / "dataset" / "HoVer_test.jsonl"),
}

# --- Corpus Configuration ---

CORPUS_PATH = str(_BASE_DIR / "dataset" / "wiki17_abstracts.jsonl.passages.pkl")
BM25S_INDEX_DIR = str(_BASE_DIR / "dataset" / "bm25s_index")


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
        "claim": sample["claim"],
        "label": sample["label"],
        "supporting_facts": sample["supporting_facts"],
    }


def outer_context_builder(sample: dict) -> str:
    """Build the data context string from a preprocessed sample.

    Returns the claim text as outer_context for the chain.
    """
    return sample["claim"]


def load_context(n_samples: int = 300) -> dict:
    """Load dataset for validation.

    BM25 index is not loaded here — the retrieval module lazy-loads it
    from disk inside the subprocess. Only paths are returned.
    """
    raw_samples = load_jsonl(DATASET_CONFIG["train_path"])

    if n_samples is not None and n_samples < len(raw_samples):
        raw_samples = raw_samples[:n_samples]

    processed = [preprocess_sample(s) for s in raw_samples]

    return {
        "train_dataset": processed,
        "bm25s_index_dir": BM25S_INDEX_DIR,
        "corpus_path": CORPUS_PATH,
    }
