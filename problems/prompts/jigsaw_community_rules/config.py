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

# Dataset configuration
DATASET_CONFIG = {
    "path": str(_BASE_DIR / "dataset" / "data.csv"),
    "required_placeholders": ["body", "rule"],
    "target_field": "rule_violation",
}


def load_context() -> dict:
    """Load dataset and return context for validation.

    Returns:
        dict: Context data containing:
            - train_dataset (pd.DataFrame): Dataset for evaluation
            - available_placeholders (list[str]): Column names usable in templates
            - required_placeholders (list[str]): Fields that MUST be in template
            - target_field (str): Target/label column name
    """
    train_dataset = pd.read_csv(DATASET_CONFIG["path"]).reset_index(drop=True)

    return {
        "train_dataset": train_dataset,
        "available_placeholders": list(train_dataset.columns),
        "required_placeholders": DATASET_CONFIG["required_placeholders"],
        "target_field": DATASET_CONFIG["target_field"],
    }
