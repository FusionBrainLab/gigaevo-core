from __future__ import annotations

import os
from typing import Annotated, Literal

from langchain_openai import ChatOpenAI
from pydantic import Field, model_validator

from gigaevo.config.schemas._base import FrozenStrictModel
from gigaevo.llm.strict_chat_openai import strict_chat_openai


class ChatOpenAIConfig(FrozenStrictModel):
    """Single LLM endpoint configured for OpenAI-compatible servers.

    Field names mirror the ``ChatOpenAI`` constructor surface validated
    by :func:`strict_chat_openai`. ``api_key`` defaults to the runtime
    ``OPENAI_API_KEY`` environment variable; the after-validator refuses
    a ``None`` resolution with a typed error rather than letting the
    request reach the OpenAI HTTP boundary.
    """

    kind: Literal["chat_openai"] = "chat_openai"
    model: str
    api_key: str | None = Field(
        default_factory=lambda: os.environ.get("OPENAI_API_KEY"),
        repr=False,
    )
    base_url: str | None = None
    temperature: float = Field(default=0.5, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=1)
    request_timeout: float = Field(default=60.0, gt=0.0)

    @model_validator(mode="after")
    def require_api_key(self) -> "ChatOpenAIConfig":
        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable must be set, or "
                "api_key passed explicitly on ChatOpenAIConfig"
            )
        return self

    def build(self) -> ChatOpenAI:
        return strict_chat_openai(
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            request_timeout=self.request_timeout,
        )


class BanditRouterConfig(FrozenStrictModel):
    """UCB1 bandit-driven router over a static model pool."""

    kind: Literal["bandit"] = "bandit"
    models: list[ChatOpenAIConfig] = Field(min_length=1)
    skip_reward_on_acceptor_reject: bool = False
    exploration_constant: float = Field(default=1.4, gt=0.0)
    name: str = "default"


class EnsembleRouterConfig(FrozenStrictModel):
    """Probability-weighted router. ``probabilities`` length must match ``models``."""

    kind: Literal["ensemble"] = "ensemble"
    models: list[ChatOpenAIConfig] = Field(min_length=1)
    probabilities: list[float] | None = None
    name: str = "default"

    @model_validator(mode="after")
    def probabilities_aligned(self) -> "EnsembleRouterConfig":
        if self.probabilities is None:
            return self
        if len(self.probabilities) != len(self.models):
            raise ValueError(
                f"probabilities length ({len(self.probabilities)}) "
                f"must equal models length ({len(self.models)})"
            )
        if any(p <= 0 for p in self.probabilities):
            raise ValueError("all probabilities must be positive")
        return self


LLMConfig = Annotated[
    BanditRouterConfig | EnsembleRouterConfig,
    Field(discriminator="kind"),
]
