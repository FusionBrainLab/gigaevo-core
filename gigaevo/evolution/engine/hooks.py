"""Post-run hooks for EvolutionEngine.

``PostRunHook`` is the ABC; concrete implementations are injected via Hydra.

- ``NullPostRunHook`` — no-op (writer off: ``memory=none``)
- ``IncrementalPostRunHook`` — abstract extension that also supports mid-run
  refreshes via ``run_increment`` (required by ``LiveMemoryRefreshHook``)
- ``MemoryWriter`` — authors memory cards from run results (writer on:
  ``memory=v2``)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gigaevo.database.program_storage import ProgramStorage
    from gigaevo.programs.program import Program


class PostRunHook(ABC):
    """Hook called by EvolutionEngine after the evolution loop completes."""

    @abstractmethod
    async def on_run_complete(self, storage: ProgramStorage) -> None:
        """Called once after evolution finishes, before storage is closed."""


class IncrementalPostRunHook(PostRunHook):
    """PostRunHook that can additionally refresh its state mid-run.

    ``LiveMemoryRefreshHook`` requires this interface — wiring a plain
    ``PostRunHook`` (e.g. ``NullPostRunHook``) as a live tracker is a
    composition error and is rejected at construction time.
    """

    @abstractmethod
    async def run_increment(
        self,
        programs: list[Program],
        *,
        posterior_programs: list[Program] | None = None,
    ) -> None:
        """Incremental refresh over the current program pool mid-run."""


class NullPostRunHook(PostRunHook):
    """No-op hook. Default when the writer is off (``memory=none``)."""

    async def on_run_complete(self, storage: ProgramStorage) -> None:
        pass
