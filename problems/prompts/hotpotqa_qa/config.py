"""Configuration for HotpotQA prompt evolution."""

import json
from pathlib import Path
import random

import pandas as pd

_BASE_DIR = Path(__file__).parent

LLM_CONFIG = {
    "model": "Qwen/Qwen3-8B",
    "max_cost": 10.0,  # we don't constrain the cost here
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
        "max_tokens": 16384,
    },
    "client_kwargs": {
        "api_key": "sk-gigaevo",
        "base_url": "http://localhost:8000/v1",
    },
}

DATASET_CONFIG = {
    "train_path": str(_BASE_DIR / "dataset" / "HotpotQA_train.jsonl"),
    "test_path": str(_BASE_DIR / "dataset" / "HotpotQA_test.jsonl"),
    "required_placeholders": ["question", "passages"],
    "target_field": "answer",
    "k_passages": 7,
}


def load_jsonl(path: str) -> list[dict]:
    """Load JSONL file as list of dicts."""
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def format_passage(title: str, sentences: list[str]) -> str:
    """Format a single passage as 'Title | sentence1 sentence2 ...'"""
    return f"{title} | {' '.join(sentences)}"


def select_passages(
    sample: dict, k: int = 7, rng: random.Random | None = None
) -> list[str]:
    if rng is None:
        rng = random.Random()

    golden_titles = set(sample["supporting_facts"]["title"])

    context_titles = sample["context"]["title"]
    context_sentences = sample["context"]["sentences"]
    title_to_sentences = {
        title: sentences for title, sentences in zip(context_titles, context_sentences)
    }

    golden_passages = []
    distractor_passages = []

    for title in context_titles:
        formatted = format_passage(title, title_to_sentences[title])
        if title in golden_titles:
            golden_passages.append(formatted)
        else:
            distractor_passages.append(formatted)

    selected = golden_passages.copy()

    remaining_slots = k - len(selected)
    if remaining_slots > 0:
        rng.shuffle(distractor_passages)
        selected.extend(distractor_passages[:remaining_slots])

    rng.shuffle(selected)

    return selected


def preprocess_sample(
    sample: dict, k: int = 7, rng: random.Random | None = None
) -> dict:
    """Preprocess a single sample for prompt formatting."""
    passages = select_passages(sample, k=k, rng=rng)
    formatted_passages = "\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))

    return {
        "question": sample["question"],
        "passages": formatted_passages,
        "answer": sample["answer"],
    }


def load_context(n_samples: int = 300, seed: int = 42) -> dict:
    """Load dataset and return context for validation.

    Uses fixed seed to ensure reproducible dataset across validate() calls.
    """
    raw_samples = load_jsonl(DATASET_CONFIG["train_path"])

    if n_samples is not None and n_samples < len(raw_samples):
        raw_samples = raw_samples[:n_samples]

    # Single RNG with fixed seed for reproducibility
    rng = random.Random(seed)

    processed = [
        preprocess_sample(s, k=DATASET_CONFIG["k_passages"], rng=rng)
        for s in raw_samples
    ]

    train_dataset = pd.DataFrame(processed)

    return {
        "train_dataset": train_dataset,
        "available_placeholders": DATASET_CONFIG["required_placeholders"],
        "required_placeholders": DATASET_CONFIG["required_placeholders"],
        "target_field": DATASET_CONFIG["target_field"],
    }


def load_baseline() -> str:
    """Load baseline prompt template from initial_programs/baseline.py.

    Returns:
        Prompt template string.
    """
    baseline_path = _BASE_DIR / "initial_programs" / "baseline.py"
    baseline_globals = {}
    exec(baseline_path.read_text(), baseline_globals)
    return baseline_globals["entrypoint"]()
