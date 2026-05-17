"""One-liner factory functions matching the shipped config/llm/*.yaml.

Phase 1 (:mod:`gigaevo.config.schemas.llm`) covered the core typed
shape; this module exports the deployment-specific endpoint presets
the YAMLs hardcode. Each builder returns a fully-validated
:class:`BanditRouterConfig` or :class:`EnsembleRouterConfig` that
experiment files compose with::

    from gigaevo.config.llm_presets import build_openrouter_bandit

    def build() -> ExperimentConfig:
        return ExperimentConfig(..., llm=build_openrouter_bandit())

Endpoints, model names, and timeouts default to the YAMLs' values;
overrides flow through keyword arguments so a sweep can pin a
specific model with ``build_openrouter_bandit(temperature=0.3)``.
"""

from __future__ import annotations

import os

from gigaevo.config.defaults import (
    DEFAULT_LLM_REQUEST_TIMEOUT_S,
    DEFAULT_LLM_TEMPERATURE,
)
from gigaevo.config.schemas import (
    BanditRouterConfig,
    ChatOpenAIConfig,
    EnsembleRouterConfig,
)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

OPENROUTER_FOUR_MODELS = (
    "google/gemini-2.5-flash",
    "google/gemini-3-flash-preview",
    "deepseek/deepseek-v3.2",
    "openai/gpt-4.1-mini",
)


def _chat_openai(
    model: str,
    *,
    base_url: str | None = OPENROUTER_BASE_URL,
    temperature: float = DEFAULT_LLM_TEMPERATURE,
    max_tokens: int = 16_384,
    request_timeout: float = DEFAULT_LLM_REQUEST_TIMEOUT_S,
) -> ChatOpenAIConfig:
    """Single-endpoint helper; the presets below compose it."""
    return ChatOpenAIConfig(
        model=model,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        request_timeout=request_timeout,
    )


def build_openrouter_ensemble(
    *,
    temperature: float = DEFAULT_LLM_TEMPERATURE,
    max_tokens: int = 16_384,
    request_timeout: float = DEFAULT_LLM_REQUEST_TIMEOUT_S,
    probabilities: list[float] | None = None,
) -> EnsembleRouterConfig:
    """Static 4-way OpenRouter ensemble matching
    ``config/llm/openrouter_ensemble.yaml``. Equal probabilities by
    default; override to bias one provider."""
    endpoints = [
        _chat_openai(
            m,
            temperature=temperature,
            max_tokens=max_tokens,
            request_timeout=request_timeout,
        )
        for m in OPENROUTER_FOUR_MODELS
    ]
    return EnsembleRouterConfig(
        models=endpoints,
        probabilities=probabilities or [0.25, 0.25, 0.25, 0.25],
    )


def build_openrouter_bandit(
    *,
    temperature: float = DEFAULT_LLM_TEMPERATURE,
    max_tokens: int = 16_384,
    request_timeout: float = DEFAULT_LLM_REQUEST_TIMEOUT_S,
    exploration_constant: float = 1.41,
    window_size: int = 100,
) -> BanditRouterConfig:
    """UCB1 bandit over the same 4 OpenRouter models matching
    ``config/llm/openrouter_bandit.yaml``. The bandit learns which
    model produces the best fitness improvements."""
    endpoints = [
        _chat_openai(
            m,
            temperature=temperature,
            max_tokens=max_tokens,
            request_timeout=request_timeout,
        )
        for m in OPENROUTER_FOUR_MODELS
    ]
    return BanditRouterConfig(
        models=endpoints,
        exploration_constant=exploration_constant,
        window_size=window_size,
    )


def build_single(
    model: str,
    *,
    base_url: str | None = None,
    temperature: float = DEFAULT_LLM_TEMPERATURE,
    max_tokens: int = 16_384,
    request_timeout: float = DEFAULT_LLM_REQUEST_TIMEOUT_S,
) -> EnsembleRouterConfig:
    """Single-endpoint preset matching ``config/llm/single.yaml``.

    Returned as an EnsembleRouterConfig with one model rather than a
    bandit; the runtime MultiModelRouter handles single-arm pools
    correctly without the bookkeeping overhead a bandit imposes."""
    return EnsembleRouterConfig(
        models=[
            _chat_openai(
                model,
                base_url=base_url,
                temperature=temperature,
                max_tokens=max_tokens,
                request_timeout=request_timeout,
            )
        ],
        probabilities=[1.0],
    )


def build_heterogeneous_bandit(
    model_1: str | None = None,
    model_2: str | None = None,
    *,
    base_url: str | None = None,
    temperature: float = 0.8,
    max_tokens: int = 16_384,
    exploration_constant: float = 1.41,
    window_size: int = 100,
) -> BanditRouterConfig:
    """Two-model bandit matching ``config/llm/heterogeneous_bandit.yaml``.
    Defaults read ``LLM_MODEL_1`` and ``LLM_MODEL_2`` from the
    environment, falling back to the YAML's hardcoded Llama / Qwen
    pair. ``base_url`` is required because heterogeneous endpoints
    typically point at self-hosted inference servers; deferred to the
    caller rather than defaulted."""
    if base_url is None:
        raise ValueError(
            "build_heterogeneous_bandit requires base_url; the "
            "heterogeneous preset is intended for self-hosted endpoints"
        )
    m1 = model_1 or os.environ.get(
        "LLM_MODEL_1", "meta-llama/Llama-3.3-70B-Instruct"
    )
    m2 = model_2 or os.environ.get("LLM_MODEL_2", "Qwen/Qwen2.5-72B-Instruct")
    return BanditRouterConfig(
        models=[
            _chat_openai(
                m1,
                base_url=base_url,
                temperature=temperature,
                max_tokens=max_tokens,
            ),
            _chat_openai(
                m2,
                base_url=base_url,
                temperature=temperature,
                max_tokens=max_tokens,
            ),
        ],
        exploration_constant=exploration_constant,
        window_size=window_size,
    )


def build_gemini_3_flash(
    *,
    temperature: float = DEFAULT_LLM_TEMPERATURE,
    max_tokens: int = 16_384,
) -> EnsembleRouterConfig:
    """Single-model preset matching ``config/llm/gemini3_flash.yaml``."""
    return build_single(
        "google/gemini-3-flash-preview",
        base_url=OPENROUTER_BASE_URL,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def build_gemini_31_pro(
    *,
    temperature: float = DEFAULT_LLM_TEMPERATURE,
    max_tokens: int = 16_384,
) -> EnsembleRouterConfig:
    """Single-model preset matching ``config/llm/gemini31_pro.yaml``."""
    return build_single(
        "google/gemini-3.1-pro",
        base_url=OPENROUTER_BASE_URL,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def build_gemini_25_pro(
    *,
    temperature: float = DEFAULT_LLM_TEMPERATURE,
    max_tokens: int = 16_384,
) -> EnsembleRouterConfig:
    """Single-model preset matching ``config/llm/gemini25_pro.yaml``."""
    return build_single(
        "google/gemini-2.5-pro",
        base_url=OPENROUTER_BASE_URL,
        temperature=temperature,
        max_tokens=max_tokens,
    )
