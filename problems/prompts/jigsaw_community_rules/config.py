import pandas as pd

# LLM client configuration
LLM_CONFIG = {
    "model": "Qwen/Qwen2.5-14B-Instruct",
    "max_cost": 1.0,  # Maximum cost budget per sample in dollars
    "model_pricing": {
        "prompt": 0.06,  # Price per 1M prompt tokens
        "completion": 0.24,  # Price per 1M completion tokens
    },
    "generation_kwargs": {
        "temperature": 0.7,
        "top_p": 0.8,
        "extra_body": {
            "top_k": 20,
        },
    },
    "client_kwargs": {
        "api_key": "token-classifier",
        "base_url": "http://localhost:7732/v1",
    },
}

# Dataset configuration
DATASET_CONFIG = {
    "path": "problems/prompts/jigsaw_community_rules/dataset/data.csv",
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
