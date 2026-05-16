from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Iterator
from contextvars import ContextVar
import os
import random
import threading
import time
from typing import TYPE_CHECKING, Any, cast

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


# ---------------------------------------------------------------------------
# Startup model-verification helpers
# ---------------------------------------------------------------------------
#
# Multiple ``MultiModelRouter`` instances and multiple driver processes
# all call ``_verify_models`` from ``__init__``.  Without coordination
# that synchronous burst against ``GET {base_url}/models`` is a textbook
# thundering herd at boot.  Within a single process the cache below
# collapses repeat probes; across processes the jitter spreads them.

_VERIFY_CACHE_TTL_SUCCESS_S = 300.0
"""Cache successful probes for 5 minutes.  Re-probing more often than
this adds no value — the upstream's model list changes on the timescale
of deployments."""

_VERIFY_CACHE_TTL_FAILURE_S = 30.0
"""Cache failures briefly so retries are still possible without
re-burning the jitter sleep on every router instantiation."""

_VERIFY_JITTER_MAX_S = 2.0
"""Each probe sleeps a uniform [0, max] seconds before issuing the GET so
concurrent router constructions land staggered, not synchronously."""

# Maps ``base_url`` -> (cached_at_monotonic, available_models_or_None).
_verify_cache: dict[str, tuple[float, frozenset[str] | None]] = {}
_verify_cache_lock = threading.Lock()


def _model_base_url(model: ChatOpenAI) -> str | None:
    """Pick the OpenAI-compatible base URL off a langchain model object.

    Different langchain-openai versions surface this as ``base_url`` or
    ``openai_api_base``; check both."""
    return getattr(model, "base_url", None) or getattr(model, "openai_api_base", None)


def _model_api_key(model: ChatOpenAI) -> str | None:
    """Return the API key as plaintext for the ``Authorization`` header.

    ``ChatOpenAI.openai_api_key`` is a ``pydantic.SecretStr``;
    ``.get_secret_value()`` unwraps it.  ``None`` if the model has no
    key configured (in-cluster vLLM / SGLang typically need none)."""
    secret = getattr(model, "openai_api_key", None)
    if secret is None:
        return None
    try:
        return secret.get_secret_value()
    except AttributeError:
        return str(secret) or None


def _fetch_available_models_at(
    base_url: str, api_key: str | None
) -> frozenset[str] | None:
    """Return the set of model ids advertised by an OpenAI-compatible server.

    Process-wide TTL-cached + jittered.  Sends ``Authorization: Bearer``
    when an API key is configured so authenticated providers
    (openai.com, OpenRouter) return real results instead of 401.

    Returns ``None`` on probe failure; the caller is expected to skip
    verification logging for that base URL rather than treat ``None``
    as "no models available".
    """
    now = time.monotonic()
    with _verify_cache_lock:
        cached = _verify_cache.get(base_url)
        if cached is not None:
            cached_at, cached_value = cached
            ttl = (
                _VERIFY_CACHE_TTL_SUCCESS_S
                if cached_value is not None
                else _VERIFY_CACHE_TTL_FAILURE_S
            )
            if (now - cached_at) < ttl:
                return cached_value

    # Stagger probes across coexisting routers / drivers.
    time.sleep(random.uniform(0, _VERIFY_JITTER_MAX_S))

    from gigaevo.infra.requests_factory import make_requests_session

    headers: dict[str, str] | None = None
    if api_key:
        headers = {"Authorization": f"Bearer {api_key}"}
    session = make_requests_session("model_verify", timeout=(5.0, 10.0))
    available: frozenset[str] | None
    try:
        response = session.get(f"{base_url}/models", headers=headers)
        response.raise_for_status()
        data = response.json()
        available = frozenset(
            d["id"] for d in data.get("data", []) if isinstance(d, dict) and "id" in d
        )
    except Exception as exc:
        logger.warning(
            "[MultiModelRouter] Cannot verify models at {}: {}", base_url, exc
        )
        available = None
    finally:
        session.close()

    with _verify_cache_lock:
        _verify_cache[base_url] = (time.monotonic(), available)
    return available


def _create_langfuse_handler() -> CallbackHandler | None:
    """Create Langfuse handler if credentials are configured."""
    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        return None

    handler = CallbackHandler()
    handler.client.flush_at = 1  # type: ignore[attr-defined]
    handler.client.flush_interval = 1  # type: ignore[attr-defined]
    logger.info("[MultiModelRouter] Langfuse tracing enabled")
    return handler


def _with_langfuse(
    config: RunnableConfig | None,
    handler: CallbackHandler | None,
    model_name: str | None = None,
) -> RunnableConfig | None:
    """Add Langfuse handler and metadata to config."""
    if handler is None:
        return config

    cfg: dict[str, Any] = dict(config or {})
    callbacks: list[Any] = cfg.setdefault("callbacks", [])
    if handler not in callbacks:
        callbacks.append(handler)

    if model_name:
        metadata: dict[str, Any] = cfg.setdefault("metadata", {})
        metadata["selected_model"] = model_name

    return cast(RunnableConfig, cfg)


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
        """Best-effort startup probe — verify configured models exist on servers.

        Routed through :func:`_fetch_available_models_at`, which adds a
        process-wide TTL cache + random jitter before each probe so
        multiple routers and multiple drivers don't synchronously
        hammer the inference server's ``/models`` endpoint at boot.
        Sends the model's API key as ``Authorization: Bearer …`` so the
        probe works against authenticated providers (OpenAI, OpenRouter)
        instead of silently logging 401s and continuing.
        """
        checked: set[str] = set()
        for model in self.models:
            base_url = _model_base_url(model)
            # ``isinstance(str)`` short-circuits mock objects that return
            # truthy ``MagicMock`` attributes — those models never have a
            # real URL to probe, and exercising the network path with a
            # mock-stringified URL just produces noise in test logs.
            if not isinstance(base_url, str) or base_url in checked:
                continue
            checked.add(base_url)
            api_key = _model_api_key(model)
            available = _fetch_available_models_at(base_url, api_key)
            if available is None:
                continue  # probe failure already logged at the helper layer
            for m in self.models:
                if _model_base_url(m) != base_url:
                    continue
                if m.model_name in available:
                    logger.info(
                        "[MultiModelRouter:{}] Model {} verified on {}",
                        self._name,
                        m.model_name,
                        base_url,
                    )
                else:
                    logger.warning(
                        "[MultiModelRouter:{}] Model {} NOT FOUND on {}. "
                        "Available: {}",
                        self._name,
                        m.model_name,
                        base_url,
                        sorted(available),
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

    def _config(
        self, config: RunnableConfig | None, model_name: str
    ) -> RunnableConfig | None:
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

    def _config(
        self, config: RunnableConfig | None, model_name: str
    ) -> RunnableConfig | None:
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
