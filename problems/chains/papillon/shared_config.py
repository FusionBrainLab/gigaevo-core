"""Shared configuration for PAPILLON chain evolution experiments.

PAPILLON = Privacy-preserving query pipeline. Evolves a 3-step chain:
redact PII → external LLM → aggregate response.

Uses the PUPA dataset (Columbia-NLP/PUPA) with quality + leakage evaluation.
"""

import os
from pathlib import Path

import pandas as pd

# --- LLM Configuration (Qwen/Qwen3-8B, same as all chain problems) ---

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
        "base_url": "http://localhost:8000/v1",
    },
}

# --- Judge Configuration (GPT-4o-mini via OpenRouter, same as prompts/pupa) ---

JUDGE_CONFIG = {
    "model": "openai/gpt-4o-mini",
    "max_cost": 10.0,
    "model_pricing": {
        "prompt": 0.15,
        "completion": 0.60,
    },
    "generation_kwargs": {
        "temperature": 0.0,
    },
    "client_kwargs": {
        "api_key": os.environ.get("PROMPT_API_KEY"),
        "base_url": "https://openrouter.ai/api/v1",
        "proxy": f"socks5://{os.environ['PROXY_USER']}:"
        f"{os.environ['PROXY_PASS']}@"
        f"{os.environ['PROXY_HOST']}",
    },
}

# --- Dataset Configuration ---

_BASE_DIR = Path(__file__).parent

DATASET_CONFIG = {
    "train_path": str(_BASE_DIR / "dataset" / "PUPA_train.csv"),
    "test_path": str(_BASE_DIR / "dataset" / "PUPA_test.csv"),
    "target_field": "target_response",
    "pii_field": "pii_units",
}


# --- Data Loading Utilities ---


def outer_context_builder(sample: dict) -> str:
    """Build the data context string from a sample.

    Returns the original user query (which contains PII).
    The chain's first step should rewrite this to remove PII.
    """
    return sample["user_query"]


def load_context(n_samples: int = 150) -> dict:
    """Load dataset for validation.

    Args:
        n_samples: Number of training samples (default 150, lower than
            other problems because dual-LLM + judge pipeline is more expensive).

    Returns:
        Dict with train_dataset (list[dict]), target_field, pii_field.
    """
    train_df = pd.read_csv(DATASET_CONFIG["train_path"])

    if n_samples is not None and n_samples < len(train_df):
        train_df = train_df.head(n_samples).reset_index(drop=True)

    train_dataset = train_df.to_dict("records")

    return {
        "train_dataset": train_dataset,
        "target_field": DATASET_CONFIG["target_field"],
        "pii_field": DATASET_CONFIG["pii_field"],
    }
