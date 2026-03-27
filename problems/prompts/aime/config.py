from pathlib import Path

import pandas as pd

_BASE_DIR = Path(__file__).parent

LLM_CONFIG = {
    "model": "Qwen/Qwen3-8B",
    "max_cost": 10.0,  # we don't constrain the cost here
    "model_pricing": {
        "prompt": 0.05,  # Price per 1M prompt tokens
        "completion": 0.25,  # Price per 1M completion tokens
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
    "path": str(_BASE_DIR / "dataset" / "AIME_Dataset_2023_2025.csv"),
    "required_placeholders": ["problem"],
    "target_field": "answer",
}


def load_context(years=(2023, 2024), n_trials: int = 2) -> dict:
    """Load dataset and return context for validation.

    Returns:
        dict: Context data containing:
            - train_dataset (pd.DataFrame): Dataset for evaluation
            - available_placeholders (list[str]): Column names usable in templates
            - required_placeholders (list[str]): Fields that MUST be in template
            - target_field (str): Target/label column name
    """
    train_dataset = pd.read_csv(DATASET_CONFIG["path"])

    # Evaluate on a specific year
    train_dataset = train_dataset[train_dataset["Year"].isin(years)].reset_index(
        drop=True
    )

    # Evaluate multiple independent times on each problem to reduce variance
    train_dataset = pd.concat([train_dataset] * n_trials, ignore_index=True)

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
