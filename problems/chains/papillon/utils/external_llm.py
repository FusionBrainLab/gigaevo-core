"""Batched external LLM tool for PAPILLON chain execution.

The chain runner calls tool functions via asyncio.to_thread(), so tools
run in a thread pool where asyncio.run() works cleanly. This module
creates a batched tool function that wraps async LLM calls.
"""

import asyncio
from collections.abc import Callable

from problems.chains.client import LLMClient


def make_external_llm_fn(llm_config: dict) -> Callable[[list[dict]], list[str]]:
    """Create a batched tool function that wraps async LLM calls.

    The returned function accepts a list of resolved kwargs dicts (each
    with a "query" key) and returns a list of LLM response strings.

    Args:
        llm_config: LLMClient configuration dict.

    Returns:
        Function with signature (items: list[dict]) -> list[str].
    """
    client = LLMClient(**llm_config)

    def external_llm(items: list[dict]) -> list[str]:
        async def _run():
            return list(
                await asyncio.gather(
                    *(client.copy()(item["query"]) for item in items)
                )
            )

        return asyncio.run(_run())

    return external_llm
