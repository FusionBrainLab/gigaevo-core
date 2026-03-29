import os
from pathlib import Path

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

JUDGE_CONFIG = {
    "model": "openai/gpt-4o-mini",
    "max_cost": 10.0,  # we don't constrain the cost here
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
        "proxy": f"socks5://{os.environ.get('PROXY_USER')}:"
        f"{os.environ.get('PROXY_PASS')}@"
        f"{os.environ.get('PROXY_HOST')}",
    },
}

DATASET_CONFIG = {
    "train_path": str(_BASE_DIR / "dataset" / "PUPA_train.csv"),
    "test_path": str(_BASE_DIR / "dataset" / "PUPA_test.csv"),
    "required_placeholders": ["user_query"],
    "target_field": "target_response",
    "pii_field": "pii_units",
}


def load_context(n_samples: int = 300) -> dict:
    """Load dataset and return context for validation."""
    train_dataset = pd.read_csv(DATASET_CONFIG["train_path"])

    # Take first n_samples (already shuffled during loading)
    if n_samples is not None and n_samples < len(train_dataset):
        train_dataset = train_dataset.head(n_samples).reset_index(drop=True)

    return {
        "train_dataset": train_dataset,
        "available_placeholders": DATASET_CONFIG["required_placeholders"],
        "required_placeholders": DATASET_CONFIG["required_placeholders"],
        "target_field": DATASET_CONFIG["target_field"],
        "pii_field": DATASET_CONFIG["pii_field"],
    }


def load_baseline() -> str:
    """Load baseline prompt template from initial_programs/baseline.py."""
    baseline_path = _BASE_DIR / "initial_programs" / "baseline.py"
    baseline_globals = {}
    exec(baseline_path.read_text(), baseline_globals)
    return baseline_globals["entrypoint"]()
