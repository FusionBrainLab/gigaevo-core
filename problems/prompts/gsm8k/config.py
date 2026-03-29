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

DATASET_CONFIG = {
    "train_path": str(_BASE_DIR / "dataset" / "GSM8K_train.csv"),
    "test_path": str(_BASE_DIR / "dataset" / "GSM8K_test.csv"),
    "required_placeholders": ["question"],
    "target_field": "answer",
}


def load_context(n_samples: int = 300) -> dict:
    """Load dataset and return context for validation."""
    train_dataset = pd.read_csv(DATASET_CONFIG["train_path"])

    train_dataset = train_dataset[:n_samples].reset_index(drop=True)

    return {
        "train_dataset": train_dataset,
        "available_placeholders": DATASET_CONFIG["required_placeholders"],
        "required_placeholders": DATASET_CONFIG["required_placeholders"],
        "target_field": DATASET_CONFIG["target_field"],
    }


def load_baseline() -> str:
    """Load baseline prompt template from initial_programs/baseline.py."""
    baseline_path = _BASE_DIR / "initial_programs" / "baseline.py"
    baseline_globals = {}
    exec(baseline_path.read_text(), baseline_globals)
    return baseline_globals["entrypoint"]()
