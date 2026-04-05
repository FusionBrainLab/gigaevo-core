"""Memory provider abstraction for Hydra-injected memory selection.

The provider is a strategy object injected into the DAG pipeline via Hydra.
- ``NullMemoryProvider`` — no-op, returns empty selection (default: ``memory=none``)
- ``SelectorMemoryProvider`` — delegates to ``MemorySelectorAgent`` (``memory=local`` or ``memory=api``)
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
    """Abstract memory provider injected via Hydra."""

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
    """No-op provider. Returns empty selection. Default when ``memory=none``."""

    async def select_cards(
        self,
        program: Program,
        *,
        task_description: str,
        metrics_description: str,
    ) -> MemorySelection:
        return MemorySelection(cards=[], card_ids=[])


class SelectorMemoryProvider(MemoryProvider):
    """Delegates to ``MemorySelectorAgent``. Supports all backends (API, local, GAM).

    The selector agent is created lazily on first use to avoid heavy initialization
    at Hydra config resolution time.

    Optional ``checkpoint_dir`` and ``namespace`` override the corresponding
    values in ``config/memory_backend.yaml`` at runtime, passed directly
    to ``MemorySelectorAgent`` — no environment variable hacks needed.
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
