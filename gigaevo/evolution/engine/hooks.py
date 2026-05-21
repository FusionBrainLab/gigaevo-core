"""Post-run hooks for EvolutionEngine.

``PostRunHook`` is the ABC consumed by ``EvolutionEngine`` after the
generation loop finishes.

- ``NullPostRunHook`` — no-op.
- ``IdeaTracker`` (in ``gigaevo.memory.ideas_tracker``) — analyses
  programs and classifies improvement ideas.
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
    """No-op hook. Returns immediately without touching storage."""

    async def on_run_complete(self, storage: ProgramStorage) -> None:
        pass
