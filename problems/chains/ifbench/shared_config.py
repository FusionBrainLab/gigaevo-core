"""Shared configuration for IFBench chain evolution experiments."""

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
    },
    "client_kwargs": {
        "api_key": "None",
        "base_url": "http://10.225.185.92:8000/v1",
    },
}

# --- Dataset Configuration ---

_BASE_DIR = Path(__file__).parent

DATASET_CONFIG = {
    "train_path": str(_BASE_DIR / "dataset" / "IFBench_train.jsonl"),
    "test_path": str(_BASE_DIR / "dataset" / "IFBench_test.jsonl"),
}


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
        "prompt": sample["prompt"],
        "instruction_id_list": sample["instruction_id_list"],
        "kwargs": sample["kwargs"],
    }


def outer_context_builder(sample: dict) -> str:
    """Build the data context string from a preprocessed sample.

    Returns the user prompt (with embedded constraints) as outer_context.
    """
    return sample["prompt"]


def load_context(n_samples: int = 300) -> dict:
    """Load dataset for validation."""
    raw_samples = load_jsonl(DATASET_CONFIG["train_path"])

    if n_samples is not None and n_samples < len(raw_samples):
        raw_samples = raw_samples[:n_samples]

    processed = [preprocess_sample(s) for s in raw_samples]

    return {
        "train_dataset": processed,
    }
