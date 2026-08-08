from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Iterator
from contextvars import ContextVar
import json
import os
import random
from typing import TYPE_CHECKING, Any, Literal, cast
import urllib.error
import urllib.request

from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_openai import ChatOpenAI
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from loguru import logger
from pydantic import BaseModel

from gigaevo.llm.io_dump import PromptIODumpHandler, create_io_dump_handler
from gigaevo.llm.token_tracking import TokenTracker, TokenUsage
from gigaevo.utils.trackers.base import LogWriter

if TYPE_CHECKING:
    from gigaevo.programs.program import Program


_selected_model_var: ContextVar[str | None] = ContextVar("selected_model", default=None)
_last_token_usage_var: ContextVar[TokenUsage | None] = ContextVar(
    "last_token_usage", default=None
)


def get_selected_model() -> str | None:
    """Return the last selected model name for the current async context."""
    return _selected_model_var.get()


def get_last_token_usage() -> TokenUsage | None:
    """Return token usage from the most recent LLM call in the current async context.

    Populated by ``MultiModelRouter`` and ``_StructuredOutputRouter`` after every
    invocation that yields a response with usage metadata. ``None`` if the last
    response had no usage info (e.g. stream chunk without metadata).
    """
    return _last_token_usage_var.get()


def _remember_selected_model(model_name: str) -> None:
    _selected_model_var.set(model_name)


def _remember_token_usage(response: Any) -> None:
    usage = TokenUsage.from_response(response)
    if usage is not None:
        _last_token_usage_var.set(usage)


def _extract_content_text(content: str | list[str | dict]) -> str:
    if isinstance(content, str):
        return content
    return "".join(
        part if isinstance(part, str) else part.get("text") or "" for part in content
    )


def _parsed_to_text(parsed: Any) -> str:
    if isinstance(parsed, BaseModel):
        return parsed.model_dump_json()
    if isinstance(parsed, (dict, list)):
        return json.dumps(parsed)
    if parsed is None:
        return ""
    return str(parsed)


def _create_langfuse_handler() -> CallbackHandler | None:
    """Create Langfuse handler if credentials are configured.

    In langfuse v4 the handler is a thin wrapper over the global ``Langfuse``
    client returned by ``get_client()``; flush tuning must be set on the
    client at construction time. Constructing the ``Langfuse`` client here
    installs it as the singleton the handler picks up.
    """
    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        return None

    Langfuse(flush_at=1, flush_interval=1)
    handler = CallbackHandler()
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


def _with_io_dump(
    config: RunnableConfig | None, handler: PromptIODumpHandler | None
) -> RunnableConfig | None:
    """Attach the LLM I/O dump handler to the callback list."""
    if handler is None:
        return config

    cfg: dict[str, Any] = dict(config or {})
    callbacks: list[Any] = cfg.setdefault("callbacks", [])
    if handler not in callbacks:
        callbacks.append(handler)

    return cast(RunnableConfig, cfg)


_StructuredMethod = Literal["function_calling", "json_mode", "json_schema"]
_AUTO_STRUCTURED_METHOD = "auto"
_STRUCTURED_METHOD_CHAIN: tuple[_StructuredMethod, ...] = (
    "json_schema",
    "function_calling",
    "json_mode",
)


def _schema_for_method(schema: Any, method: str | None) -> Any:
    """Translate the named JSON-schema wrapper into the selected wire format."""
    if (
        method == "function_calling"
        and isinstance(schema, dict)
        and isinstance(schema.get("schema"), dict)
    ):
        function = {
            "name": str(schema.get("name") or schema["schema"].get("title", "output")),
            "parameters": schema["schema"],
        }
        if schema.get("description"):
            function["description"] = schema["description"]
        return function
    return schema


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
        models: list[BaseChatModel],
        probabilities: list[float],
        writer: LogWriter | None = None,
        name: str = "default",
        structured_output_method: str | None = None,
        max_concurrent: int | None = None,
    ):
        if len(models) != len(probabilities):
            raise ValueError(
                f"Length mismatch: {len(models)} models, {len(probabilities)} probabilities"
            )
        if any(p <= 0 for p in probabilities):
            raise ValueError("All probabilities must be positive")
        if max_concurrent is not None and max_concurrent < 1:
            raise ValueError(
                f"max_concurrent must be >= 1 or None, got {max_concurrent}"
            )

        self.models = models
        self.model_names = [m.model_name for m in models]
        self.probabilities = [p / sum(probabilities) for p in probabilities]
        self._task_model_map: dict[int, str] = {}
        self._name = name
        self._structured_output_method = structured_output_method
        self._negotiated_method: dict[int, _StructuredMethod] = {}
        self._max_concurrent = max_concurrent
        self._semaphore: asyncio.Semaphore | None = None

        self._tracker = TokenTracker(
            name=name,
            writer=writer.bind(path=["llm", "tokens"]) if writer else None,
        )
        self._langfuse = _create_langfuse_handler()
        self._io_dump = create_io_dump_handler(name)

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
        """Startup probe — verify configured models exist on their servers.

        Raises ``RuntimeError`` if any base URL is unreachable. Missing model
        names on a reachable server are logged as warnings.
        """
        by_base_url: dict[str, list[ChatOpenAI]] = {}
        for model in self.models:
            base_url = getattr(model, "base_url", None) or getattr(
                model, "openai_api_base", None
            )
            if not isinstance(base_url, str):
                continue
            by_base_url.setdefault(base_url, []).append(model)

        for base_url, models in by_base_url.items():
            # Use the key actually configured on the model so the probe
            # auths the same way as real requests. Fall back to env vars.
            secret: Any = getattr(models[0], "openai_api_key", None)
            if hasattr(secret, "get_secret_value"):
                model_token = secret.get_secret_value()
            elif isinstance(secret, str):
                model_token = secret
            else:
                model_token = ""
            token = (
                model_token
                or os.getenv("OPENAI_API_KEY", "")
                or os.getenv("OPENAI_API_TOKEN", "")
            )
            url = f"{base_url.rstrip('/')}/models"
            req = urllib.request.Request(
                url, method="GET", headers={"Authorization": f"Bearer {token}"}
            )
            try:
                with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
                    payload = json.loads(resp.read())
            except (urllib.error.URLError, OSError) as exc:
                raise RuntimeError(
                    f"[MultiModelRouter:{self._name}] Cannot reach LLM endpoint "
                    f"at {base_url}: {exc}"
                ) from exc

            available = {entry["id"] for entry in payload.get("data", [])}
            for model in models:
                if model.model_name in available:
                    logger.info(
                        "[MultiModelRouter:{}] Model {} verified on {}",
                        self._name,
                        model.model_name,
                        base_url,
                    )
                else:
                    logger.warning(
                        "[MultiModelRouter:{}] Model {} NOT FOUND on {}. Available: {}",
                        self._name,
                        model.model_name,
                        base_url,
                        sorted(available),
                    )

    def get_semaphore(self) -> asyncio.Semaphore | None:
        """Lazy-init shared semaphore that caps concurrent async LLM calls.

        Returns ``None`` when no cap was configured. Otherwise the same
        :class:`asyncio.Semaphore` instance is returned on every call —
        both this router's ``ainvoke``/``astream`` paths and the
        structured-output router obtained via
        :meth:`with_structured_output` share it, so the cap is global
        across all async LLM calls routed through this instance.

        Must be called from inside a running event loop (the standard
        ``asyncio.Semaphore`` constructor binds to the current loop).
        """
        if self._max_concurrent is None:
            return None
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self._max_concurrent)
            logger.info(
                "[MultiModelRouter:{}] LLM concurrency capped at {}",
                self._name,
                self._max_concurrent,
            )
        return self._semaphore

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
        config = _with_langfuse(config, self._langfuse, model_name)
        return _with_io_dump(config, self._io_dump)

    def invoke(
        self, input: LanguageModelInput, config: RunnableConfig | None = None, **kwargs
    ) -> BaseMessage:
        model, name = self._select()
        response = model.invoke(input, self._config(config, name), **kwargs)
        self._tracker.track(response, name)
        _remember_token_usage(response)
        return response

    async def ainvoke(
        self, input: LanguageModelInput, config: RunnableConfig | None = None, **kwargs
    ) -> BaseMessage:
        sema = self.get_semaphore()
        if sema is None:
            return await self._ainvoke_inner(input, config, **kwargs)
        async with sema:
            return await self._ainvoke_inner(input, config, **kwargs)

    async def _ainvoke_inner(
        self, input: LanguageModelInput, config: RunnableConfig | None, **kwargs
    ) -> BaseMessage:
        model, name = self._select()
        response = await model.ainvoke(input, self._config(config, name), **kwargs)
        self._tracker.track(response, name)
        _remember_token_usage(response)
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
            _remember_token_usage(last)

    async def astream(
        self, input: LanguageModelInput, config: RunnableConfig | None = None, **kwargs
    ) -> AsyncIterator[BaseMessage]:
        sema = self.get_semaphore()
        if sema is None:
            async for chunk in self._astream_inner(input, config, **kwargs):
                yield chunk
            return
        async with sema:
            async for chunk in self._astream_inner(input, config, **kwargs):
                yield chunk

    async def _astream_inner(
        self, input: LanguageModelInput, config: RunnableConfig | None, **kwargs
    ) -> AsyncIterator[BaseMessage]:
        model, name = self._select()
        last = None
        async for chunk in model.astream(input, self._config(config, name), **kwargs):
            last = chunk
            yield chunk
        if last:
            self._tracker.track(last, name)
            _remember_token_usage(last)

    def generate(
        self,
        data: str,
        *,
        schema: dict[str, Any] | None = None,
    ) -> tuple[str, Any, int | None, float | None]:
        """LLMServiceProtocol entrypoint for the memory subsystem (A-MEM/GAM).

        Returns ``(content, raw_response, total_tokens, cost)``. Token usage
        is booked by the router's :class:`TokenTracker` on both paths, so the
        trailing tuple slots are always ``None`` — they exist only for the
        vendor protocol shape, and no consumer reads them. With a ``schema``
        the call delegates to :meth:`with_structured_output`, so a parse
        failure raises ``ValueError`` like the rest of the stack — memory
        callers retry (dedup) or absorb via the read-path guard.
        """
        if schema is None:
            raw = self.invoke(data)
            return _extract_content_text(raw.content), raw, None, None

        parsed = self.with_structured_output(schema).invoke(data)
        return _parsed_to_text(parsed), parsed, None, None

    def with_structured_output(self, schema: Any, **kwargs) -> Runnable:
        """Create a router that returns parsed Pydantic models with token tracking.

        If the router was constructed with ``structured_output_method`` (e.g.
        ``"json_schema"`` or ``"function_calling"``), that method is forwarded
        to each underlying model. The ``"auto"`` sentinel instead negotiates a
        working method per model at call time (see
        :class:`_AutoStructuredOutputRouter`). Explicit ``method`` in
        ``**kwargs`` wins and bypasses negotiation.
        """
        if (
            self._structured_output_method == _AUTO_STRUCTURED_METHOD
            and "method" not in kwargs
        ):
            return _AutoStructuredOutputRouter(self, schema)
        if self._structured_output_method is not None:
            kwargs.setdefault("method", self._structured_output_method)
        method = kwargs.get("method")
        wire_schema = _schema_for_method(schema, method)
        wrapped = [
            m.with_structured_output(wire_schema, include_raw=True, **kwargs)
            for m in self.models
        ]
        return _StructuredOutputRouter(
            wrapped,
            self.model_names,
            self.probabilities,
            self._langfuse,
            self._tracker,
            task_model_map=self._task_model_map,
            semaphore_factory=self.get_semaphore,
            io_dump=self._io_dump,
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
        semaphore_factory: Callable[[], asyncio.Semaphore | None] | None = None,
        io_dump: PromptIODumpHandler | None = None,
    ):
        self._models = models
        self._names = model_names
        self._probs = probabilities
        self._langfuse = langfuse
        self._tracker = tracker
        self._task_model_map = task_model_map
        self._select_override = select_override
        self._semaphore_factory = semaphore_factory
        self._io_dump = io_dump

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
        config = _with_langfuse(config, self._langfuse, model_name)
        return _with_io_dump(config, self._io_dump)

    def _process(self, response: dict, name: str) -> Any:
        raw = response.get("raw")
        if raw:
            self._tracker.track(raw, name)
            _remember_token_usage(raw)
        parsed = response.get("parsed")
        if parsed is None:
            content = _extract_content_text(getattr(raw, "content", "") or "")
            excerpt = (content[:500] + "…") if len(content) > 500 else content
            raise ValueError(
                f"[{name}] Structured output parse failed: raw response had no parsable schema. "
                f"content_excerpt={excerpt!r}"
            )
        return parsed

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
        sema = self._semaphore_factory() if self._semaphore_factory else None
        if sema is None:
            model, name = self._select()
            return self._process(
                await model.ainvoke(input, self._config(config, name), **kwargs), name
            )
        async with sema:
            model, name = self._select()
            return self._process(
                await model.ainvoke(input, self._config(config, name), **kwargs), name
            )


class _AutoStructuredOutputRouter(Runnable):
    """Negotiates a structured-output method per model, caching the winner.

    For each selected model, tries the methods in ``_STRUCTURED_METHOD_CHAIN``
    until one both transports and parses, then caches it on the base router's
    ``_negotiated_method`` so later calls skip the dead methods. A model that
    only serves ``function_calling`` self-heals; one that rejects forced
    ``tool_choice`` stays on ``json_schema``.
    """

    def __init__(self, router: MultiModelRouter, schema: Any) -> None:
        self._router = router
        self._schema = schema

    def _select(self) -> tuple[ChatOpenAI, str]:
        idx = random.choices(
            range(len(self._router.models)), weights=self._router.probabilities
        )[0]
        return self._router.models[idx], self._router.model_names[idx]

    def _methods_for(self, model_id: int) -> tuple[_StructuredMethod, ...]:
        cached = self._router._negotiated_method.get(model_id)
        return (cached,) if cached is not None else _STRUCTURED_METHOD_CHAIN

    def _remember(self, model_id: int, name: str, method: _StructuredMethod) -> None:
        if self._router._negotiated_method.get(model_id) == method:
            return
        self._router._negotiated_method[model_id] = method
        logger.info(
            "[MultiModelRouter:{}] structured-output method for {} auto-resolved to {!r}",
            self._router._name,
            name,
            method,
        )

    def _bound_router(
        self, model: ChatOpenAI, name: str, method: _StructuredMethod
    ) -> _StructuredOutputRouter:
        wrapped = model.with_structured_output(
            _schema_for_method(self._schema, method),
            include_raw=True,
            method=method,
        )
        return _StructuredOutputRouter(
            [wrapped],
            [name],
            [1.0],
            self._router._langfuse,
            self._router._tracker,
            task_model_map=self._router._task_model_map,
            semaphore_factory=self._router.get_semaphore,
            io_dump=self._router._io_dump,
        )

    def _log_failure(
        self, method: _StructuredMethod, name: str, exc: Exception
    ) -> None:
        logger.debug(
            "[MultiModelRouter:{}] structured method {!r} failed for {}: {}",
            self._router._name,
            method,
            name,
            exc,
        )

    def invoke(
        self, input: LanguageModelInput, config: RunnableConfig | None = None, **kwargs
    ) -> Any:
        model, name = self._select()
        model_id = id(model)
        last_exc: Exception | None = None
        for method in self._methods_for(model_id):
            try:
                result = self._bound_router(model, name, method).invoke(
                    input, config, **kwargs
                )
                self._remember(model_id, name, method)
                return result
            except Exception as exc:
                last_exc = exc
                self._log_failure(method, name, exc)
        raise cast(Exception, last_exc)

    async def ainvoke(
        self, input: LanguageModelInput, config: RunnableConfig | None = None, **kwargs
    ) -> Any:
        model, name = self._select()
        model_id = id(model)
        last_exc: Exception | None = None
        for method in self._methods_for(model_id):
            try:
                result = await self._bound_router(model, name, method).ainvoke(
                    input, config, **kwargs
                )
                self._remember(model_id, name, method)
                return result
            except Exception as exc:
                last_exc = exc
                self._log_failure(method, name, exc)
        raise cast(Exception, last_exc)
