"""Shared config for the CARL summarizer chain problem."""

from __future__ import annotations

import json
import os
from pathlib import Path

PROBLEM_DIR = Path(__file__).resolve().parent

_CHAIN_URL = os.environ.get("SUMMARIZER_CHAIN_URL", "http://localhost:8000/v1")

LLM_CONFIG = {
    "model": "Qwen/Qwen3-235B-A22B-Instruct-2507",
    "max_cost": 10.0,
    "model_pricing": {"prompt": 0.05, "completion": 0.25},
    "generation_kwargs": {
        "temperature": 0.6,
        "top_p": 0.95,
        "max_tokens": 8192,
        "extra_body": {"top_k": 20},
    },
    "client_kwargs": {
        "api_key": os.environ.get("OPENAI_API_KEY", "sk-gigaevo"),
        "base_url": _CHAIN_URL,
    },
}


def get_llm_config() -> dict:
    return dict(LLM_CONFIG)


def load_dataset() -> list[dict]:
    rows = []
    with (PROBLEM_DIR / "data" / "eval.jsonl").open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def outer_context_builder(sample: dict) -> str:
    return f"Задание: {sample['task']}\n\nТекст:\n{sample['input']}"
