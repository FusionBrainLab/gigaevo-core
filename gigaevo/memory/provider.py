"""Memory provider abstraction for the DAG memory-selection stage.

The provider is a strategy object consumed by ``MemoryContextStage``.

- ``NullMemoryProvider`` — no-op, returns an empty selection.
- ``SelectorMemoryProvider`` — delegates to ``MemorySelectorAgent``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from loguru import logger

from gigaevo.llm.agents.memory_selector import MemorySelection
from gigaevo.programs.program import Program

if TYPE_CHECKING:
    from gigaevo.llm.agents.memory_selector import MemorySelectorAgent


class MemoryProvider(ABC):
    """Strategy interface for selecting memory cards for a program."""

    @abstractmethod
    async def select_cards(
        self,
        program: Program,
        *,
        task_description: str,
        metrics_description: str,
    ) -> MemorySelection:
        """Select memory cards relevant to this program."""


class NullMemoryProvider(MemoryProvider):
    """No-op provider. Returns an empty selection on every call."""

    async def select_cards(
        self,
        program: Program,
        *,
        task_description: str,
        metrics_description: str,
    ) -> MemorySelection:
        return MemorySelection(cards=[], card_ids=[])


class SelectorMemoryProvider(MemoryProvider):
    """Delegates card selection to ``MemorySelectorAgent``.

    Supports all backends the selector itself supports (API, local, GAM).
    The selector agent is created lazily on first use to keep construction
    cheap. ``checkpoint_dir`` and ``namespace`` are forwarded directly to
    ``MemorySelectorAgent`` (no environment-variable indirection).
    """

    def __init__(
        self,
        *,
        max_cards: int = 3,
        checkpoint_dir: str | None = None,
        namespace: str | None = None,
    ) -> None:
        self._max_cards = max_cards
        self._checkpoint_dir = checkpoint_dir
        self._namespace = namespace
        self._selector: MemorySelectorAgent | None = None

    def _get_selector(self) -> MemorySelectorAgent:
        if self._selector is None:
            from gigaevo.llm.agents.memory_selector import MemorySelectorAgent

            logger.info(
                "[SelectorMemoryProvider] Creating MemorySelectorAgent "
                "(checkpoint_dir={}, namespace={}, use_api=False)",
                self._checkpoint_dir,
                self._namespace,
            )
            self._selector = MemorySelectorAgent(
                checkpoint_dir=self._checkpoint_dir,
                namespace=self._namespace,
                use_api=False,
            )
        return self._selector

    async def select_cards(
        self,
        program: Program,
        *,
        task_description: str,
        metrics_description: str,
    ) -> MemorySelection:
        selector = self._get_selector()
        return await selector.select(
            input=[program],
            mutation_mode="rewrite",
            task_description=task_description,
            metrics_description=metrics_description,
            memory_text="",
            max_cards=self._max_cards,
        )
