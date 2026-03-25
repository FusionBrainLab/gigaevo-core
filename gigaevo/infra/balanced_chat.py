"""Drop-in ``ChatOpenAI`` replacement with Redis-coordinated load balancing.

``BalancedChatOpenAI`` holds N ``ChatOpenAI`` instances (one per endpoint) and
routes each request to the least-loaded healthy endpoint via ``EndpointPool``.
It is a ``ChatOpenAI`` subclass so ``MultiModelRouter`` can use it directly —
just replace ``ChatOpenAI(base_url=X)`` with
``BalancedChatOpenAI(endpoints=[X,Y,Z])``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
import time
from typing import Any

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_openai import ChatOpenAI
from loguru import logger

from gigaevo.infra.endpoint_pool import EndpointPool
from gigaevo.infra.pool_metrics import PoolMetricsTracker
from gigaevo.utils.trackers.base import LogWriter

_DEFAULT_REDIS_URL = "redis://localhost:6379/15"


class BalancedChatOpenAI(ChatOpenAI):
    """ChatOpenAI that load-balances across multiple endpoints via Redis.

    Holds one ``ChatOpenAI`` instance per endpoint.  On each call, acquires
    the least-loaded endpoint from the shared ``EndpointPool``, delegates to
    the corresponding client, and releases on completion.

    Usage in Hydra config::

        _target_: gigaevo.infra.balanced_chat.BalancedChatOpenAI
        model: ${model_name}
        endpoints:
          - "http://server-a:8777/v1"
          - "http://server-b:8777/v1"
        pool_name: "mutation"
    """

    def __init__(
        self,
        *,
        endpoints: list[str],
        pool_name: str = "mutation",
        redis_url: str = _DEFAULT_REDIS_URL,
        cooldown_secs: int = 60,
        writer: LogWriter | None = None,
        **kwargs: Any,
    ) -> None:
        # Initialize base ChatOpenAI with first endpoint (for model_name etc.)
        super().__init__(base_url=endpoints[0], **kwargs)

        self._endpoints = list(endpoints)
        self._pool = EndpointPool(
            pool_name=pool_name,
            endpoints=endpoints,
            redis_url=redis_url,
            cooldown_secs=cooldown_secs,
        )
        self._metrics = PoolMetricsTracker(pool_name=pool_name, writer=writer)

        # One ChatOpenAI client per endpoint
        self._clients: dict[str, ChatOpenAI] = {}
        for ep in endpoints:
            self._clients[ep] = ChatOpenAI(base_url=ep, **kwargs)

        logger.info(
            "[BalancedChatOpenAI:{}] {} endpoints: {}",
            pool_name,
            len(endpoints),
            ", ".join(endpoints),
        )

    # ------------------------------------------------------------------
    # Core invoke/ainvoke — delegate to selected endpoint
    # ------------------------------------------------------------------

    def invoke(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> BaseMessage:
        endpoint = self._pool.acquire_sync()
        t0 = time.perf_counter()
        try:
            result = self._clients[endpoint].invoke(input, config, **kwargs)
            latency = (time.perf_counter() - t0) * 1000
            self._pool.release_sync(endpoint, latency)
            self._metrics.record(endpoint, latency, success=True)
            return result
        except Exception:
            latency = (time.perf_counter() - t0) * 1000
            self._pool.mark_unhealthy_sync(endpoint)
            self._metrics.record(endpoint, latency, success=False)
            raise

    async def ainvoke(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> BaseMessage:
        endpoint = await self._pool.acquire()
        t0 = time.perf_counter()
        try:
            result = await self._clients[endpoint].ainvoke(input, config, **kwargs)
            latency = (time.perf_counter() - t0) * 1000
            await self._pool.release(endpoint, latency)
            self._metrics.record(endpoint, latency, success=True)
            return result
        except Exception:
            latency = (time.perf_counter() - t0) * 1000
            await self._pool.mark_unhealthy(endpoint)
            self._metrics.record(endpoint, latency, success=False)
            raise

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def stream(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Iterator[BaseMessage]:
        endpoint = self._pool.acquire_sync()
        t0 = time.perf_counter()
        try:
            yield from self._clients[endpoint].stream(input, config, **kwargs)
            latency = (time.perf_counter() - t0) * 1000
            self._pool.release_sync(endpoint, latency)
            self._metrics.record(endpoint, latency, success=True)
        except Exception:
            latency = (time.perf_counter() - t0) * 1000
            self._pool.mark_unhealthy_sync(endpoint)
            self._metrics.record(endpoint, latency, success=False)
            raise

    async def astream(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[BaseMessage]:
        endpoint = await self._pool.acquire()
        t0 = time.perf_counter()
        try:
            async for chunk in self._clients[endpoint].astream(input, config, **kwargs):
                yield chunk
            latency = (time.perf_counter() - t0) * 1000
            await self._pool.release(endpoint, latency)
            self._metrics.record(endpoint, latency, success=True)
        except Exception:
            latency = (time.perf_counter() - t0) * 1000
            await self._pool.mark_unhealthy(endpoint)
            self._metrics.record(endpoint, latency, success=False)
            raise

    # ------------------------------------------------------------------
    # Structured output
    # ------------------------------------------------------------------

    def with_structured_output(
        self, schema: Any, **kwargs: Any
    ) -> _BalancedStructuredOutput:
        """Return a balanced wrapper around per-endpoint structured output chains."""
        chains = {
            ep: client.with_structured_output(schema, include_raw=True, **kwargs)
            for ep, client in self._clients.items()
        }
        return _BalancedStructuredOutput(
            chains=chains,
            pool=self._pool,
            metrics=self._metrics,
        )


class _BalancedStructuredOutput(Runnable):
    """Structured output wrapper that routes via EndpointPool."""

    def __init__(
        self,
        chains: dict[str, Any],
        pool: EndpointPool,
        metrics: PoolMetricsTracker,
    ) -> None:
        self._chains = chains
        self._pool = pool
        self._metrics = metrics

    def invoke(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        endpoint = self._pool.acquire_sync()
        t0 = time.perf_counter()
        try:
            result = self._chains[endpoint].invoke(input, config, **kwargs)
            latency = (time.perf_counter() - t0) * 1000
            self._pool.release_sync(endpoint, latency)
            self._metrics.record(endpoint, latency, success=True)
            return result
        except Exception:
            latency = (time.perf_counter() - t0) * 1000
            self._pool.mark_unhealthy_sync(endpoint)
            self._metrics.record(endpoint, latency, success=False)
            raise

    async def ainvoke(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        endpoint = await self._pool.acquire()
        t0 = time.perf_counter()
        try:
            result = await self._chains[endpoint].ainvoke(input, config, **kwargs)
            latency = (time.perf_counter() - t0) * 1000
            await self._pool.release(endpoint, latency)
            self._metrics.record(endpoint, latency, success=True)
            return result
        except Exception:
            latency = (time.perf_counter() - t0) * 1000
            await self._pool.mark_unhealthy(endpoint)
            self._metrics.record(endpoint, latency, success=False)
            raise
