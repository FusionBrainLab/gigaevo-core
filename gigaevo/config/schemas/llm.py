from __future__ import annotations

import os
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import Field, model_validator

from gigaevo.config.schemas._base import FrozenStrictModel

if TYPE_CHECKING:
    from langchain_core.runnables import Runnable
    from langchain_openai import ChatOpenAI

    from gigaevo.utils.trackers.base import LogWriter


class ChatOpenAIConfig(FrozenStrictModel):
    """Single LLM endpoint configured for OpenAI-compatible servers.

    Field names mirror the ``ChatOpenAI`` constructor surface validated
    by :func:`strict_chat_openai`. ``api_key`` defaults to the runtime
    ``OPENAI_API_KEY`` environment variable; the after-validator refuses
    a ``None`` resolution with a typed error rather than letting the
    request reach the OpenAI HTTP boundary.

    ``api_key`` is excluded from ``__repr__`` and from ``model_dump``
    so the secret never lands in the dumped ``config.json``, in
    experiment-id hashes, or in log lines. The default factory re-reads
    ``OPENAI_API_KEY`` on every load, so a round trip through JSON
    pulls the current ambient key rather than the one captured at the
    time of the original construction.
    """

    kind: Literal["chat_openai"] = "chat_openai"
    model: str = Field(min_length=1)
    api_key: str | None = Field(
        default_factory=lambda: os.environ.get("OPENAI_API_KEY"),
        repr=False,
        exclude=True,
    )
    base_url: str | None = None
    temperature: float = Field(default=0.5, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=1)
    request_timeout: float = Field(default=60.0, gt=0.0)

    @model_validator(mode="after")
    def require_api_key(self) -> ChatOpenAIConfig:
        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable must be set, or "
                "api_key passed explicitly on ChatOpenAIConfig"
            )
        return self

    def build(self) -> ChatOpenAI:
        from gigaevo.llm.strict_chat_openai import strict_chat_openai

        return strict_chat_openai(
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            request_timeout=self.request_timeout,
        )


class BanditRouterConfig(FrozenStrictModel):
    """UCB1 bandit-driven router over a static model pool.

    ``fitness_key`` and ``higher_is_better`` are typically supplied at
    construction by the experiment's ``ProblemContext`` via the
    cross-field validator on the root config; declared here as Optional
    so the schema can validate in isolation but ``build()`` requires
    them to be resolved to non-None values.
    """

    kind: Literal["bandit"] = "bandit"
    # ``default_factory=list`` lets tyro render help text for the
    # inactive ``LLMConfig`` union branch. ``validate_default=True``
    # promotes ``min_length=1`` to fire when a caller constructs
    # ``BanditRouterConfig()`` directly without a ``models`` argument;
    # the constructor exits with a typed error instead of silently
    # accepting a zero-arm bandit.
    models: list[ChatOpenAIConfig] = Field(
        default_factory=list, min_length=1, validate_default=True
    )
    skip_reward_on_acceptor_reject: bool = False
    # ``1.41`` ~= sqrt(2), the canonical UCB1 exploration constant
    # balancing exploration vs exploitation for bounded reward signals.
    exploration_constant: float = Field(default=1.41, gt=0.0)
    window_size: int = Field(default=100, ge=1)
    name: str = Field(default="default", min_length=1)

    def build(
        self,
        *,
        fitness_key: str,
        higher_is_better: bool = True,
        writer: LogWriter | None = None,
    ) -> Runnable:
        from gigaevo.llm.bandit import BanditModelRouter

        endpoints = [m.build() for m in self.models]
        uniform = [1.0 / len(endpoints)] * len(endpoints)
        return BanditModelRouter(
            endpoints,
            uniform,
            writer=writer,
            name=self.name,
            exploration_constant=self.exploration_constant,
            window_size=self.window_size,
            fitness_key=fitness_key,
            higher_is_better=higher_is_better,
        )


class EnsembleRouterConfig(FrozenStrictModel):
    """Probability-weighted router (the runtime ``MultiModelRouter``).

    When ``probabilities`` is ``None`` the runtime applies a uniform
    distribution over ``models``. When provided, the after-validator
    enforces length parity and positivity; the runtime normalises the
    weights to a probability distribution.
    """

    kind: Literal["ensemble"] = "ensemble"
    # ``default_factory=list`` lets tyro render help text for the
    # inactive ``LLMConfig`` union branch. ``validate_default=True``
    # promotes ``min_length=1`` to fire when a caller constructs
    # ``EnsembleRouterConfig()`` directly without a ``models``
    # argument; the constructor exits with a typed error instead of
    # silently accepting a zero-model ensemble.
    models: list[ChatOpenAIConfig] = Field(
        default_factory=list, min_length=1, validate_default=True
    )
    probabilities: list[float] | None = None
    name: str = Field(default="default", min_length=1)

    @model_validator(mode="after")
    def probabilities_aligned(self) -> EnsembleRouterConfig:
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

    def build(self, *, writer: LogWriter | None = None) -> Runnable:
        from gigaevo.llm.models import MultiModelRouter

        endpoints = [m.build() for m in self.models]
        weights = self.probabilities or [1.0 / len(endpoints)] * len(endpoints)
        return MultiModelRouter(endpoints, weights, writer=writer, name=self.name)


LLMConfig = Annotated[
    BanditRouterConfig | EnsembleRouterConfig,
    Field(discriminator="kind"),
]
