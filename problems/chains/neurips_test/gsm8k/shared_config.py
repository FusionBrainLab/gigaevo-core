"""Shared configuration for neurips_test/gsm8k chain evolution.

LLM parameters are read from environment variables so the same codebase can be
used with any model/server without editing files:

    export OPENAI_API_KEY=sk-...
    export LLM_BASE_URL=http://host:port/v1
    export LLM_MODEL_NAME=Qwen/Qwen3-235B-A22B

Dataset is loaded from local JSONL files or downloaded from HuggingFace
(``openai/gsm8k``).  See dataset/load_dataset.py for details.
"""

import os

from problems.chains.neurips_test.gsm8k.dataset.load_dataset import load_gsm8k

# ---------------------------------------------------------------------------
# LLM configuration (env-driven)
# ---------------------------------------------------------------------------

LLM_CONFIG = {
    "model": os.environ.get(
        "CHAIN_LLM_MODEL_NAME", os.environ.get("LLM_MODEL_NAME", "Qwen/Qwen3-235B-A22B")
    ),
    "max_cost": 100.0,
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
        "api_key": os.environ.get("OPENAI_API_KEY", "sk-gigaevo"),
        "base_url": os.environ.get(
            "CHAIN_LLM_BASE_URL",
            os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1"),
        ),
    },
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def outer_context_builder(sample: dict) -> str:
    """Return the math question as the outer context."""
    return sample["question"]


def load_context(n_samples: int = 200) -> dict:
    """Load GSM8K training samples for validation.

    Args:
        n_samples: Number of training examples to use (default 200).

    Returns:
        Dict with ``"train_dataset"`` and ``"target_field"`` keys.
    """
    samples = load_gsm8k(split="train", n_samples=n_samples)
    return {
        "train_dataset": samples,
        "target_field": "answer",
    }
