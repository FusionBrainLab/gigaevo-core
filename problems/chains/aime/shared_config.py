"""Shared configuration for AIME chain evolution.

Math olympiad problem solving with up to 3 LLM reasoning steps.
No tools — pure LLM chain.
"""

from pathlib import Path

import pandas as pd

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
    "path": str(_BASE_DIR / "dataset" / "AIME_Dataset_2023_2025.csv"),
    "target_field": "answer",
}


# --- Data Loading Utilities ---


def outer_context_builder(sample: dict) -> str:
    """Build outer_context string from a preprocessed sample.

    Returns the problem text as the context for all chain steps.
    """
    return sample["problem"]


def load_context(years: tuple = (2023, 2024), n_trials: int = 2) -> dict:
    """Load dataset for validation.

    Args:
        years: Tuple of years to filter the dataset by.
        n_trials: Number of times to repeat each problem (reduces variance).

    Returns:
        Dict with train_dataset (list[dict]) and target_field.
    """
    df = pd.read_csv(DATASET_CONFIG["path"])

    df = df[df["Year"].isin(years)].reset_index(drop=True)

    # Repeat n_trials times to reduce variance
    df = pd.concat([df] * n_trials, ignore_index=True)

    # Convert to list[dict] for chain runner
    train_dataset = df.to_dict("records")

    return {
        "train_dataset": train_dataset,
        "target_field": DATASET_CONFIG["target_field"],
    }
