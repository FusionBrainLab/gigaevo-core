"""Shared configuration for HoVer chain evolution experiments."""

import json
import os
from pathlib import Path

from tools.no_proxy import ensure_no_proxy

ensure_no_proxy()

# --- LLM Configuration ---
# All chain requests go through the LiteLLM proxy (INTERNAL_IP:4000),
# which load-balances across backend servers defined in infrastructure.yaml.
# Start the proxy with: bash tools/litellm.sh --background

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
        "api_key": "sk-gigaevo",
        "base_url": _CHAIN_URL,
    },
}


def get_llm_config() -> dict:
    """Return LLM config. Load balancing is handled by litellm proxy."""
    return dict(LLM_CONFIG)


def release_chain_endpoint(url: str, *, success: bool = True) -> None:
    """No-op — load balancing is handled by litellm proxy."""
    pass

# --- Dataset Configuration ---

_BASE_DIR = Path(__file__).parent

DATASET_CONFIG = {
    "train_path": str(_BASE_DIR / "dataset" / "HoVer_train.jsonl"),
    "test_path": str(_BASE_DIR / "dataset" / "HoVer_test.jsonl"),
}

# --- Corpus Configuration ---

_CORPUS_PKL = _BASE_DIR / "dataset" / "wiki17_abstracts.jsonl.passages.pkl"
_CORPUS_GZ = _BASE_DIR / "dataset" / "wiki17_abstracts.jsonl.gz"
CORPUS_PATH = str(_CORPUS_PKL if _CORPUS_PKL.exists() else _CORPUS_GZ)
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
