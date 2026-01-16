import os
import pandas as pd

# LLM client configuration
# LLM_CONFIG = {
#     "model": "openai/gpt-4.1-nano",
#     "max_cost": 1.0,  # Maximum cost budget per sample in dollars
#     "model_pricing": {
#         "prompt": 0.10,  # Price per 1M prompt tokens
#         "completion": 0.40,  # Price per 1M completion tokens
#     },
#     "generation_kwargs": {
#         "temperature": 0.7,
#         "max_tokens": 16384,
#     },
#     "client_kwargs": {
#         "api_key": os.environ["OPENAI_API_KEY"],
#         "base_url": "https://openrouter.ai/api/v1",
#         "proxy": f"socks5://{os.environ['PROXY_USER']}:{os.environ['PROXY_PASS']}@{os.environ['PROXY_HOST']}"
#     },
# }

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
        "max_tokens": 8192
    },
    "client_kwargs": {
        "api_key": "token-classifier",
        "base_url": "http://localhost:7732/v1",
    },
}

# Dataset configuration
DATASET_CONFIG = {
    "path": "problems/prompts/aime/dataset/AIME_Dataset_2023_2025.csv",
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
    train_dataset = train_dataset[train_dataset["Year"].isin(years)].reset_index(drop=True)

    # Evaluate multiple independent times on each problem to reduce variance
    train_dataset = pd.concat([train_dataset] * n_trials, ignore_index=True)

    return {
        "train_dataset": train_dataset,
        "available_placeholders": list(train_dataset.columns),
        "required_placeholders": DATASET_CONFIG["required_placeholders"],
        "target_field": DATASET_CONFIG["target_field"],
    }
