from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Iterator
from contextvars import ContextVar
import os
import random
from typing import TYPE_CHECKING, Any

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_openai import ChatOpenAI
from langfuse.langchain import CallbackHandler
from loguru import logger

from gigaevo.llm.token_tracking import TokenTracker
from gigaevo.utils.trackers.base import LogWriter

if TYPE_CHECKING:
    from gigaevo.programs.program import Program


_selected_model_var: ContextVar[str | None] = ContextVar("selected_model", default=None)


def get_selected_model() -> str | None:
    """Return the last selected model name for the current async context."""
    return _selected_model_var.get()


def _remember_selected_model(model_name: str) -> None:
    _selected_model_var.set(model_name)


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
        self._task_model_map: dict[int, str] = {}
        self._name = name

        self._tracker = TokenTracker(
            name=name,
            writer=writer.bind(path=["llm", "tokens"]) if writer else None,
        )
        self._langfuse = _create_langfuse_handler()

        model_desc = ", ".join(
            f"{n} ({p:.0%})" for n, p in zip(self.model_names, self.probabilities)
        )
        logger.info(
            "[MultiModelRouter:{}] Initialized with {} models: {}",
            name,
            len(models),
            model_desc,
        )
        # Log base URLs for debugging server connectivity
        for m in models:
            # ChatOpenAI exposes base_url as a property (langchain 0.1+)
            base_url = getattr(m, "base_url", None)
            if base_url:
                logger.info(
                    "[MultiModelRouter:{}] Model {} at {}", name, m.model_name, base_url
                )

        self._verify_models()

    def _verify_models(self) -> None:
        """Best-effort startup probe — verify configured models exist on servers."""
        import json as _json
        import urllib.request

        checked: set[str] = set()
        for model in self.models:
            base_url = getattr(model, "base_url", None) or getattr(
                model, "openai_api_base", None
            )
            if not base_url or base_url in checked:
                continue
            checked.add(base_url)
            try:
                url = f"{base_url}/models"
                req = urllib.request.Request(url, method="GET")
                with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
                    data = _json.loads(resp.read())
                available = [d["id"] for d in data.get("data", [])]
                for m in self.models:
                    m_url = getattr(m, "base_url", None) or getattr(
                        m, "openai_api_base", None
                    )
                    if m_url == base_url:
                        if m.model_name in available:
                            logger.info(
                                "[MultiModelRouter:{}] Model {} verified on {}",
                                self._name,
                                m.model_name,
                                base_url,
                            )
                        else:
                            logger.warning(
                                "[MultiModelRouter:{}] Model {} NOT FOUND on {}. Available: {}",
                                self._name,
                                m.model_name,
                                base_url,
                                available,
                            )
            except Exception as exc:
                logger.warning(
                    "[MultiModelRouter:{}] Cannot verify models at {}: {}",
                    self._name,
                    base_url,
                    exc,
                )

    @staticmethod
    def _current_task_id() -> int | None:
        """Return ``id(asyncio.current_task())`` or *None* outside an event loop."""
        try:
            task = asyncio.current_task()
        except RuntimeError:
            return None
        return id(task) if task is not None else None

    def _select(self) -> tuple[ChatOpenAI, str]:
        """Select a model based on probabilities."""
        idx = random.choices(range(len(self.models)), weights=self.probabilities)[0]
        model, name = self.models[idx], self.model_names[idx]
        _remember_selected_model(name)
        tid = self._current_task_id()
        if tid is not None:
            self._task_model_map[tid] = name
        return model, name

    def get_last_model(self) -> str | None:
        """Return the model name selected in the most recent ``_select()`` call for the current async task."""
        tid = self._current_task_id()
        if tid is not None:
            return self._task_model_map.pop(tid, None)
        return None

    def on_mutation_outcome(
        self,
        program: Program,
        parents: list[Program],
        outcome: Any = None,
    ) -> None:
        """Callback when a mutated program completes evaluation. Override for feedback."""

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

    def with_structured_output(self, schema: Any, **kwargs) -> _StructuredOutputRouter:
        """Create a router that returns parsed Pydantic models with token tracking."""
        wrapped = [
            m.with_structured_output(schema, include_raw=True, **kwargs)
            for m in self.models
        ]
        return _StructuredOutputRouter(
            wrapped,
            self.model_names,
            self.probabilities,
            self._langfuse,
            self._tracker,
            task_model_map=self._task_model_map,
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
        task_model_map: dict[int, str] | None = None,
        select_override: Callable[[], tuple[Any, str]] | None = None,
    ):
        self._models = models
        self._names = model_names
        self._probs = probabilities
        self._langfuse = langfuse
        self._tracker = tracker
        self._task_model_map = task_model_map
        self._select_override = select_override

    def _select(self) -> tuple[Any, str]:
        if self._select_override is not None:
            return self._select_override()
        idx = random.choices(range(len(self._models)), weights=self._probs)[0]
        model, name = self._models[idx], self._names[idx]
        _remember_selected_model(name)
        if self._task_model_map is not None:
            tid = MultiModelRouter._current_task_id()
            if tid is not None:
                self._task_model_map[tid] = name
        return model, name

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
