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
    "train_path": str(_BASE_DIR / "dataset" / "IFBench_train.jsonl"),
    "test_path": str(_BASE_DIR / "dataset" / "IFBench_test.jsonl"),
    "required_placeholders": ["prompt"],
}


def load_context(n_samples: int = 300) -> dict:
    """Load dataset and return context for validation.

    Returns:
        dict: Context data containing:
            - train_dataset (pd.DataFrame): Dataset for evaluation
            - available_placeholders (list[str]): Column names usable in templates
            - required_placeholders (list[str]): Fields that MUST be in template
    """
    train_dataset = pd.read_json(DATASET_CONFIG["train_path"], lines=True)[:n_samples]

    return {
        "train_dataset": train_dataset,
        "available_placeholders": DATASET_CONFIG["required_placeholders"],
        "required_placeholders": DATASET_CONFIG["required_placeholders"],
    }


def load_baseline() -> str:
    """Load baseline prompt template from initial_programs/baseline.py."""
    baseline_path = _BASE_DIR / "initial_programs" / "baseline.py"
    baseline_globals = {}
    exec(baseline_path.read_text(), baseline_globals)
    return baseline_globals["entrypoint"]()
