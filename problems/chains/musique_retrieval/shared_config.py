"""Shared configuration for MuSiQue retrieval-based chain evolution."""

import hashlib
import json
import random
from pathlib import Path
from typing import Any


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
        "api_key": "sk-or-v1-REDACTED",
        "base_url": "https://openrouter.ai/api/v1",
    },
}

# --- Dataset Configuration ---

_BASE_DIR = Path(__file__).parent

DATASET_CONFIG = {
    "train_path": str(_BASE_DIR / "dataset" / "MuSiQue_train.jsonl"),
    "test_path": str(_BASE_DIR / "dataset" / "MuSiQue_test.jsonl"),
    "target_field": "answer",
    "aliases_field": "answer_aliases",
}

# --- Per-task Index Configuration ---

TASK_BM25S_INDEX_DIR = str(_BASE_DIR / "dataset" / "task_bm25s_index")


def load_jsonl(path: str) -> list[dict]:
    """Load JSONL file as list of dicts."""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def _clean_text(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(str(x).strip() for x in value if str(x).strip()).strip()
    if isinstance(value, str):
        return value.strip()
    return ""


def extract_passages(sample: dict) -> list[str]:
    """Extract all passages for a sample."""
    paragraphs = sample.get("paragraphs")
    passages: list[str] = []

    if isinstance(paragraphs, list):
        for i, paragraph in enumerate(paragraphs):
            if not isinstance(paragraph, dict):
                continue
            title = str(paragraph.get("title") or f"Paragraph {i + 1}")
            text = _clean_text(
                paragraph.get("paragraph_text")
                or paragraph.get("paragraph")
                or paragraph.get("text")
            )
            if text:
                passages.append(f"{title} | {text}")
        if passages:
            return passages

    context = sample.get("context")
    if isinstance(context, dict):
        titles = context.get("title")
        sentences = context.get("sentences")
        if isinstance(titles, list) and isinstance(sentences, list):
            for title, sents in zip(titles, sentences):
                title_str = str(title)
                text = _clean_text(sents)
                if text:
                    passages.append(f"{title_str} | {text}")

    return passages


def get_gold_answers(sample: dict) -> list[str]:
    """Return de-duplicated gold answers (primary answer + aliases)."""
    candidates: list[str] = []

    answer = sample.get("answer")
    if isinstance(answer, str) and answer.strip():
        candidates.append(answer.strip())

    aliases = sample.get("answer_aliases", [])
    if isinstance(aliases, str) and aliases.strip():
        candidates.append(aliases.strip())
    elif isinstance(aliases, list):
        for alias in aliases:
            if isinstance(alias, str) and alias.strip():
                candidates.append(alias.strip())

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = candidate.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def _infer_task_id(sample: dict, fallback_source: str) -> str:
    for key in ("id", "question_id", "uid", "_id"):
        value = sample.get(key)
        if value is not None and str(value).strip():
            return str(value)
    digest = hashlib.sha1(fallback_source.encode("utf-8")).hexdigest()[:16]
    return f"sample_{digest}"


def preprocess_sample(sample: dict, sample_idx: int = 0) -> dict:
    """Preprocess a single sample for retrieval-based chain execution."""
    question = str(sample.get("question", "")).strip()
    task_id = _infer_task_id(sample, fallback_source=f"{sample_idx}:{question}")
    answers = get_gold_answers(sample)
    passages = extract_passages(sample)

    return {
        "task_id": task_id,
        "question": question,
        "passages": passages,
        "answer": answers[0] if answers else "",
        "answer_aliases": answers,
    }


def outer_context_builder(sample: dict) -> str:
    """Build outer context string from a preprocessed sample."""
    return sample["question"]


def load_context(
    n_samples: int | None = None,
    n_train: int | None = None,
    n_val: int | None = None,
    n_pool: int | None = None,
    n_held_out_val: int | None = None,
    seed: int = 42,
) -> dict:
    """Load MuSiQue context for validation/evolution."""
    raw_samples = load_jsonl(DATASET_CONFIG["train_path"])

    if n_pool is not None and n_held_out_val is not None:
        total = raw_samples[: n_pool + n_held_out_val]
        processed = [preprocess_sample(s, i) for i, s in enumerate(total)]
        rng = random.Random(seed)
        indices = list(range(len(processed)))
        rng.shuffle(indices)
        return {
            "train_pool": [processed[i] for i in indices[:n_pool]],
            "val_dataset": [processed[i] for i in indices[n_pool:]],
            "target_field": DATASET_CONFIG["target_field"],
            "aliases_field": DATASET_CONFIG["aliases_field"],
            "task_index_dir": TASK_BM25S_INDEX_DIR,
        }

    if n_train is not None and n_val is not None:
        total = raw_samples[: n_train + n_val]
        processed = [preprocess_sample(s, i) for i, s in enumerate(total)]
        rng = random.Random(seed)
        indices = list(range(len(processed)))
        rng.shuffle(indices)
        return {
            "train_dataset": [processed[i] for i in indices[:n_train]],
            "val_dataset": [processed[i] for i in indices[n_train:]],
            "target_field": DATASET_CONFIG["target_field"],
            "aliases_field": DATASET_CONFIG["aliases_field"],
            "task_index_dir": TASK_BM25S_INDEX_DIR,
        }

    n = n_samples if n_samples is not None else 300
    processed = [preprocess_sample(s, i) for i, s in enumerate(raw_samples[:n])]
    return {
        "train_dataset": processed,
        "target_field": DATASET_CONFIG["target_field"],
        "aliases_field": DATASET_CONFIG["aliases_field"],
        "task_index_dir": TASK_BM25S_INDEX_DIR,
    }
