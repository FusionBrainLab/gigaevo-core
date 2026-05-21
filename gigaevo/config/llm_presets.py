"""One-liner factory functions for the shipped LLM endpoint presets.

Each builder returns a fully-validated :class:`BanditRouterConfig` or
:class:`EnsembleRouterConfig`. Experiment files compose them::

    from gigaevo.config.llm_presets import build_openrouter_bandit

    def build() -> ExperimentConfig:
        return ExperimentConfig(..., llm=build_openrouter_bandit())

Endpoints, model names, and timeouts have sensible defaults; overrides
flow through keyword arguments, so a sweep can pin a specific
parameter with ``build_openrouter_bandit(temperature=0.3)``.
"""

from __future__ import annotations

import os
from typing import Final

from gigaevo.config.defaults import (
    DEFAULT_LLM_MAX_TOKENS,
    DEFAULT_LLM_REQUEST_TIMEOUT_S,
    DEFAULT_LLM_TEMPERATURE,
)
from gigaevo.config.schemas import (
    BanditRouterConfig,
    ChatOpenAIConfig,
    EnsembleRouterConfig,
)

OPENROUTER_BASE_URL: Final[str] = "https://openrouter.ai/api/v1"

OPENROUTER_FOUR_MODELS: Final[tuple[str, ...]] = (
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
    max_tokens: int = DEFAULT_LLM_MAX_TOKENS,
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
    max_tokens: int = DEFAULT_LLM_MAX_TOKENS,
    request_timeout: float = DEFAULT_LLM_REQUEST_TIMEOUT_S,
    probabilities: list[float] | None = None,
) -> EnsembleRouterConfig:
    """Static 4-way OpenRouter ensemble. Equal probabilities by
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
    max_tokens: int = DEFAULT_LLM_MAX_TOKENS,
    request_timeout: float = DEFAULT_LLM_REQUEST_TIMEOUT_S,
    exploration_constant: float = 1.41,
    window_size: int = 100,
) -> BanditRouterConfig:
    """UCB1 bandit over the same 4 OpenRouter models. The bandit
    learns which model produces the best fitness improvements."""
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
    max_tokens: int = DEFAULT_LLM_MAX_TOKENS,
    request_timeout: float = DEFAULT_LLM_REQUEST_TIMEOUT_S,
) -> EnsembleRouterConfig:
    """Single-endpoint preset.

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
    """Two-model bandit. Defaults read ``LLM_MODEL_1`` and
    ``LLM_MODEL_2`` from the environment, falling back to a hardcoded
    Llama / Qwen pair. ``base_url`` is required because heterogeneous
    endpoints typically point at self-hosted inference servers, so it
    is deferred to the caller rather than defaulted."""
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
    max_tokens: int = DEFAULT_LLM_MAX_TOKENS,
) -> EnsembleRouterConfig:
    """Single-model preset: ``google/gemini-3-flash-preview`` on
    OpenRouter."""
    return build_single(
        "google/gemini-3-flash-preview",
        base_url=OPENROUTER_BASE_URL,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def build_gemini_31_pro(
    *,
    temperature: float = DEFAULT_LLM_TEMPERATURE,
    max_tokens: int = DEFAULT_LLM_MAX_TOKENS,
) -> EnsembleRouterConfig:
    """Single-model preset: ``google/gemini-3.1-pro-preview`` on
    OpenRouter. The ``-preview`` suffix is required; without it the
    slug resolves to a non-existent route and fails at the OpenRouter
    HTTP boundary."""
    return build_single(
        "google/gemini-3.1-pro-preview",
        base_url=OPENROUTER_BASE_URL,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def build_gemini_25_pro(
    *,
    temperature: float = DEFAULT_LLM_TEMPERATURE,
    max_tokens: int = DEFAULT_LLM_MAX_TOKENS,
) -> EnsembleRouterConfig:
    """Single-model preset: ``google/gemini-2.5-pro`` on OpenRouter."""
    return build_single(
        "google/gemini-2.5-pro",
        base_url=OPENROUTER_BASE_URL,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def build_openai(
    *,
    temperature: float = DEFAULT_LLM_TEMPERATURE,
    max_tokens: int = DEFAULT_LLM_MAX_TOKENS,
    probabilities: list[float] | None = None,
) -> EnsembleRouterConfig:
    """Two-model OpenAI heterogeneous ensemble: ``openai/gpt-5`` plus
    ``openai/gpt-5-mini`` at a 10/90 default split. Override
    ``probabilities`` to bias toward the larger model."""
    return EnsembleRouterConfig(
        models=[
            _chat_openai(
                "openai/gpt-5",
                temperature=temperature,
                max_tokens=max_tokens,
            ),
            _chat_openai(
                "openai/gpt-5-mini",
                temperature=temperature,
                max_tokens=max_tokens,
            ),
        ],
        probabilities=probabilities or [0.1, 0.9],
    )


def build_google(
    *,
    temperature: float = DEFAULT_LLM_TEMPERATURE,
    max_tokens: int = DEFAULT_LLM_MAX_TOKENS,
    probabilities: list[float] | None = None,
) -> EnsembleRouterConfig:
    """Two-model Google heterogeneous ensemble: ``gemini-2.5-flash``
    plus ``gemini-2.5-pro`` at a 10/90 default split."""
    return EnsembleRouterConfig(
        models=[
            _chat_openai(
                "google/gemini-2.5-flash",
                temperature=temperature,
                max_tokens=max_tokens,
            ),
            _chat_openai(
                "google/gemini-2.5-pro",
                temperature=temperature,
                max_tokens=max_tokens,
            ),
        ],
        probabilities=probabilities or [0.1, 0.9],
    )
