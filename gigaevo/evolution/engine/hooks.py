"""Post-run hooks for EvolutionEngine.

``PostRunHook`` is the ABC; concrete implementations are injected via Hydra.

- ``NullPostRunHook`` — no-op (default: ``ideas_tracker=none``)
- ``IdeaTracker`` — analyses programs and classifies ideas (``ideas_tracker=default``)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gigaevo.database.program_storage import ProgramStorage


class PostRunHook(ABC):
    """Hook called by EvolutionEngine after the evolution loop completes."""

    @abstractmethod
    async def on_run_complete(self, storage: ProgramStorage) -> None:
        """Called once after evolution finishes, before storage is closed."""


class NullPostRunHook(PostRunHook):
    """No-op hook. Default when ``ideas_tracker=none``."""

    async def on_run_complete(self, storage: ProgramStorage) -> None:
        pass
