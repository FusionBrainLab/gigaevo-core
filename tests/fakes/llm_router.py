"""Test double for the MultiModelRouter surface used by the memory subsystem."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from langchain_core.messages import AIMessage


class FakeStructuredRouter:
    def __init__(self, router: FakeMemoryRouter, schema: type, kwargs: dict) -> None:
        self.router = router
        self.schema = schema
        self.kwargs = kwargs

    def invoke(self, messages: Any) -> Any:
        if not self.router.allow_sync:
            raise AssertionError("sync structured invoke in async test path")
        self.router.record("structured", self.schema, messages)
        return self.router.respond(self.schema, messages)

    async def ainvoke(self, messages: Any) -> Any:
        self.router.in_flight += 1
        self.router.max_in_flight = max(
            self.router.max_in_flight, self.router.in_flight
        )
        try:
            await asyncio.sleep(self.router.delay_s)
            self.router.record("structured_async", self.schema, messages)
            return self.router.respond(self.schema, messages)
        finally:
            self.router.in_flight -= 1


class FakeMemoryRouter:
    """
    Mimics the MultiModelRouter methods the memory writer call path uses:
    invoke/ainvoke returning an AIMessage, and with_structured_output
    returning a runnable whose (a)invoke yields ``respond(schema, messages)``.

    Args:
        respond: Callable (schema, messages) -> parsed instance for structured
            calls. Defaults to constructing ``schema()`` with no arguments.
        text: Plain-text response (str, or callable(messages) -> str).
        delay_s: Async sleep per structured call, for concurrency assertions.
        allow_sync: When False, sync structured calls raise AssertionError.
    """

    def __init__(
        self,
        respond: Callable[[type, Any], Any] | None = None,
        text: str | Callable[[Any], str] = "",
        delay_s: float = 0.0,
        allow_sync: bool = True,
    ) -> None:
        self.respond = respond or (lambda schema, messages: schema())
        self.text = text
        self.delay_s = delay_s
        self.allow_sync = allow_sync
        self.calls: list[tuple[str, type | None, Any]] = []
        self.in_flight = 0
        self.max_in_flight = 0

    def record(self, kind: str, schema: type | None, messages: Any) -> None:
        self.calls.append((kind, schema, messages))

    def message_text(self, messages: Any) -> str:
        return self.text(messages) if callable(self.text) else self.text

    def invoke(self, messages: Any) -> AIMessage:
        self.record("invoke", None, messages)
        return AIMessage(content=self.message_text(messages))

    async def ainvoke(self, messages: Any) -> AIMessage:
        self.record("ainvoke", None, messages)
        return AIMessage(content=self.message_text(messages))

    def with_structured_output(self, schema: type, **kwargs) -> FakeStructuredRouter:
        return FakeStructuredRouter(self, schema, kwargs)

    def user_texts(self) -> list[str]:
        return [
            dict(messages).get("user", "")
            for _, _, messages in self.calls
            if isinstance(messages, list)
        ]
