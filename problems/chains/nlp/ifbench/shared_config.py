"""Shared configuration for nlp/ifbench chain evolution.

LLM parameters are read from environment variables so the same codebase can be
used with any model/server without editing files:

    export OPENAI_API_KEY=sk-...
    export LLM_BASE_URL=http://host:port/v1
    export LLM_MODEL_NAME=Qwen/Qwen3-235B-A22B

Dataset and evaluation utilities are reused from problems/chains/ifbench/.
"""

import os

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
# Dataset / evaluation — reuse from chains/ifbench
# ---------------------------------------------------------------------------

from problems.chains.ifbench.shared_config import (  # noqa: E402
    load_context,
    outer_context_builder,
)

__all__ = ["LLM_CONFIG", "load_context", "outer_context_builder"]
