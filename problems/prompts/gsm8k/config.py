import pandas as pd


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
        "max_tokens": 4096,
    },
    "client_kwargs": {
        "api_key": "None",
        "base_url": "http://localhost:8000/v1",
    },
}

DATASET_CONFIG = {
    "train_path": "problems/prompts/gsm8k/dataset/GSM8K_train.csv",
    "test_path": "problems/prompts/gsm8k/dataset/GSM8K_test.csv",
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
