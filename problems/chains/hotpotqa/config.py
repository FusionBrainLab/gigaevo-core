"""Configuration for HotpotQA CARL-aligned chain evolution."""

import json
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
        "max_tokens": 8192,
    },
    "client_kwargs": {
        "api_key": "None",
        "base_url": "http://localhost:8000/v1",
    },
}

# --- Dataset Configuration ---

_BASE_DIR = Path(__file__).parent

DATASET_CONFIG = {
    "train_path": str(_BASE_DIR / "dataset" / "HotpotQA_train.jsonl"),
    "test_path": str(_BASE_DIR / "dataset" / "HotpotQA_test.jsonl"),
    "target_field": "answer",
}

# --- CARL Chain Topology (static mode) ---

STATIC_CHAIN_TOPOLOGY = {
    "num_steps": 6,
    "steps": [
        {"number": 1, "step_type": "tool", "dependencies": [], "frozen": True},
        {"number": 2, "step_type": "llm", "dependencies": [1], "frozen": False},
        {"number": 3, "step_type": "llm", "dependencies": [2], "frozen": False},
        {"number": 4, "step_type": "tool", "dependencies": [3], "frozen": True},
        {"number": 5, "step_type": "llm", "dependencies": [2, 4], "frozen": False},
        {"number": 6, "step_type": "llm", "dependencies": [2, 5], "frozen": False},
    ],
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
        "target_field": DATASET_CONFIG["target_field"],
        "bm25s_index_dir": BM25S_INDEX_DIR,
        "corpus_path": CORPUS_PATH,
    }


def load_baseline() -> dict:
    """Load baseline chain specification from initial_programs/baseline.py.

    Returns:
        Dict with "system_prompt" and "steps" keys.
    """
    baseline_path = _BASE_DIR / "initial_programs" / "baseline.py"
    baseline_globals = {}
    exec(baseline_path.read_text(), baseline_globals)
    return baseline_globals["entrypoint"]()
