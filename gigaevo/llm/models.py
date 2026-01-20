from collections.abc import AsyncIterator, Iterator
import os
import random
from typing import Any

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_openai import ChatOpenAI
from langfuse.langchain import CallbackHandler
from loguru import logger

from gigaevo.llm.token_tracking import TokenTracker
from gigaevo.utils.trackers.base import LogWriter


def _create_langfuse_handler() -> CallbackHandler | None:
    """Create Langfuse handler if credentials are configured."""
    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        return None

    handler = CallbackHandler()
    handler.client.flush_at = 1
    handler.client.flush_interval = 1
    logger.info("[MultiModelRouter] Langfuse tracing enabled")
    return handler


def _with_langfuse(
    config: RunnableConfig | None,
    handler: CallbackHandler | None,
    model_name: str | None = None,
) -> RunnableConfig:
    """Add Langfuse handler and metadata to config."""
    if handler is None:
        return config

    cfg = dict(config or {})
    callbacks = cfg.setdefault("callbacks", [])
    if handler not in callbacks:
        callbacks.append(handler)

    if model_name:
        cfg.setdefault("metadata", {})["selected_model"] = model_name

    return cfg


class MultiModelRouter(Runnable):
    """Probabilistic model router with token tracking and Langfuse tracing.

    Example:
        >>> router = MultiModelRouter(
        ...     [ChatOpenAI(model="gpt-4"), ChatOpenAI(model="gpt-3.5-turbo")],
        ...     [0.8, 0.2],
        ...     writer=metrics_writer,
        ...     name="mutation",  # metrics go to llm/tokens/mutation/...
        ... )
        >>> response = await router.ainvoke("Hello!")
        >>> structured = router.with_structured_output(MySchema)
    """

    def __init__(
        self,
        models: list[ChatOpenAI],
        probabilities: list[float],
        writer: LogWriter | None = None,
        name: str = "default",
    ):
        if len(models) != len(probabilities):
            raise ValueError(
                f"Length mismatch: {len(models)} models, {len(probabilities)} probabilities"
            )
        if any(p <= 0 for p in probabilities):
            raise ValueError("All probabilities must be positive")

        self.models = models
        self.model_names = [m.model_name for m in models]
        self.probabilities = [p / sum(probabilities) for p in probabilities]

        self._tracker = TokenTracker(
            name=name,
            writer=writer.bind(path=["llm", "tokens"]) if writer else None,
        )
        self._langfuse = _create_langfuse_handler()

        logger.info(
            "[MultiModelRouter:{}] Initialized with {} models", name, len(models)
        )

    def _select(self) -> tuple[ChatOpenAI, str]:
        """Select a model based on probabilities."""
        idx = random.choices(range(len(self.models)), weights=self.probabilities)[0]
        return self.models[idx], self.model_names[idx]

    def _config(self, config: RunnableConfig | None, model_name: str) -> RunnableConfig:
        return _with_langfuse(config, self._langfuse, model_name)

    def invoke(
        self, input: LanguageModelInput, config: RunnableConfig | None = None, **kwargs
    ) -> BaseMessage:
        model, name = self._select()
        response = model.invoke(input, self._config(config, name), **kwargs)
        self._tracker.track(response, name)
        return response

    async def ainvoke(
        self, input: LanguageModelInput, config: RunnableConfig | None = None, **kwargs
    ) -> BaseMessage:
        model, name = self._select()
        response = await model.ainvoke(input, self._config(config, name), **kwargs)
        self._tracker.track(response, name)
        return response

    def stream(
        self, input: LanguageModelInput, config: RunnableConfig | None = None, **kwargs
    ) -> Iterator[BaseMessage]:
        model, name = self._select()
        last = None
        for chunk in model.stream(input, self._config(config, name), **kwargs):
            last = chunk
            yield chunk
        if last:
            self._tracker.track(last, name)

    async def astream(
        self, input: LanguageModelInput, config: RunnableConfig | None = None, **kwargs
    ) -> AsyncIterator[BaseMessage]:
        model, name = self._select()
        last = None
        async for chunk in model.astream(input, self._config(config, name), **kwargs):
            last = chunk
            yield chunk
        if last:
            self._tracker.track(last, name)

    def with_structured_output(
        self, schema: Any, **kwargs
    ) -> "_StructuredOutputRouter":
        """Create a router that returns parsed Pydantic models with token tracking."""
        wrapped = [
            m.with_structured_output(schema, include_raw=True, **kwargs)
            for m in self.models
        ]
        return _StructuredOutputRouter(
            wrapped, self.model_names, self.probabilities, self._langfuse, self._tracker
        )


class _StructuredOutputRouter(Runnable):
    """Router for structured output with token tracking from raw responses."""

    def __init__(
        self,
        models: list,
        model_names: list[str],
        probabilities: list[float],
        langfuse: CallbackHandler | None,
        tracker: TokenTracker,
    ):
        self._models = models
        self._names = model_names
        self._probs = probabilities
        self._langfuse = langfuse
        self._tracker = tracker

    def _select(self) -> tuple[Any, str]:
        idx = random.choices(range(len(self._models)), weights=self._probs)[0]
        return self._models[idx], self._names[idx]

    def _config(self, config: RunnableConfig | None, model_name: str) -> RunnableConfig:
        return _with_langfuse(config, self._langfuse, model_name)

    def _process(self, response: dict, name: str) -> Any:
        if raw := response.get("raw"):
            self._tracker.track(raw, name)
        return response.get("parsed")

    def invoke(
        self, input: LanguageModelInput, config: RunnableConfig | None = None, **kwargs
    ) -> Any:
        model, name = self._select()
        return self._process(
            model.invoke(input, self._config(config, name), **kwargs), name
        )

    async def ainvoke(
        self, input: LanguageModelInput, config: RunnableConfig | None = None, **kwargs
    ) -> Any:
        model, name = self._select()
        return self._process(
            await model.ainvoke(input, self._config(config, name), **kwargs), name
        )
