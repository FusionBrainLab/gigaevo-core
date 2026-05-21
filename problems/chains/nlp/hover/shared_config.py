"""Shared configuration for nlp/hover chain evolution.

Chain execution (dataset evaluation) uses Qwen/Qwen3-8B via the LiteLLM proxy
(same as chains/hover experiments). Mutation/evolution uses Qwen3-235B configured
via the GigaEvo engine — this file only controls the chain runner.

Dataset and retrieval index are reused from problems/chains/hover/.
"""

import os

# ---------------------------------------------------------------------------
# LLM configuration — chain runner uses Qwen3-8B (fast, 8-node cluster)
# ---------------------------------------------------------------------------

_CHAIN_URL = os.environ.get(
    "HOVER_CHAIN_URL",
    os.environ.get("CHAIN_LLM_BASE_URL", "http://localhost:8000/v1"),
)

LLM_CONFIG = {
    "model": os.environ.get("CHAIN_LLM_MODEL_NAME", "Qwen/Qwen3-8B"),
    "max_cost": 100.0,
    "model_pricing": {
        "prompt": 0.05,
        "completion": 0.25,
    },
    "generation_kwargs": {
        "temperature": 0.6,
        "top_p": 0.95,
        "max_tokens": 8192,
        "extra_body": {
            "top_k": 20,
        },
    },
    "client_kwargs": {
        "api_key": "sk-gigaevo",
        "base_url": _CHAIN_URL,
    },
}

# ---------------------------------------------------------------------------
# Dataset / retrieval — reuse from chains/hover
# ---------------------------------------------------------------------------

from problems.chains.hover.shared_config import (
    load_context,
    outer_context_builder,
)

__all__ = ["LLM_CONFIG", "load_context", "outer_context_builder"]
