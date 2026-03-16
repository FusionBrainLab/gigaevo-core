"""Shared configuration for HotpotQA-QA chain evolution.

Unlike chains/hotpotqa which uses BM25 retrieval, this problem provides
golden passages (plus distractors) in outer_context. No tools needed.
"""

import json
from pathlib import Path
import random

from problems.chains.hotpotqa_qa.utils.passages import select_passages

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
    "k_passages": 7,
}


# --- Data Loading Utilities ---


def load_jsonl(path: str) -> list[dict]:
    """Load JSONL file as list of dicts."""
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def preprocess_sample(
    sample: dict, k: int = 7, rng: random.Random | None = None
) -> dict:
    """Preprocess a single sample: select passages and format them.

    Returns dict with question, passages (formatted string), and answer.
    """
    passages = select_passages(sample, k=k, rng=rng)
    formatted_passages = "\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))

    return {
        "question": sample["question"],
        "passages": formatted_passages,
        "answer": sample["answer"],
    }


def outer_context_builder(sample: dict) -> str:
    """Build outer_context string from a preprocessed sample.

    Includes both question and passages so every LLM step sees them.
    """
    return f"Question: {sample['question']}\n\nPassages:\n{sample['passages']}"


def load_context(n_samples: int = 300, seed: int = 42) -> dict:
    """Load dataset for validation.

    Uses fixed seed for reproducible passage selection across runs.
    """
    raw_samples = load_jsonl(DATASET_CONFIG["train_path"])

    if n_samples is not None and n_samples < len(raw_samples):
        raw_samples = raw_samples[:n_samples]

    rng = random.Random(seed)

    processed = [
        preprocess_sample(s, k=DATASET_CONFIG["k_passages"], rng=rng)
        for s in raw_samples
    ]

    return {
        "train_dataset": processed,
        "target_field": DATASET_CONFIG["target_field"],
    }
