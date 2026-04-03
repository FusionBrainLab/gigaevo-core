"""DAG stage that selects memory cards via the injected MemoryProvider."""

from __future__ import annotations

from typing import Any

from gigaevo.evolution.mutation.constants import (
    MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY,
)
from gigaevo.memory.provider import MemoryProvider
from gigaevo.programs.program import Program
from gigaevo.programs.stages.base import Stage
from gigaevo.programs.stages.cache_handler import NO_CACHE
from gigaevo.programs.stages.common import StageIO, StringContainer
from gigaevo.programs.stages.stage_registry import StageRegistry


class MemoryContextInputs(StageIO):
    """Inputs for MemoryContextStage (currently none required)."""

    pass


@StageRegistry.register(description="Select memory cards for mutation context")
class MemoryContextStage(Stage):
    """Select memory cards via the injected MemoryProvider.

    Always present in the DAG. When the provider is NullMemoryProvider,
    this stage returns an empty string instantly (no-op).

    Writes selected card IDs into program metadata for tracking.
    """

    InputsModel = MemoryContextInputs
    OutputModel = StringContainer
    cache_handler = NO_CACHE

    def __init__(
        self,
        *,
        memory_provider: MemoryProvider,
        task_description: str,
        metrics_description: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._provider = memory_provider
        self._task_description = task_description
        self._metrics_description = metrics_description

    async def compute(self, program: Program) -> StageIO:
        selection = await self._provider.select_cards(
            program,
            task_description=self._task_description,
            metrics_description=self._metrics_description,
        )

        if selection.cards:
            program.set_metadata(
                MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY, selection.card_ids
            )
            return StringContainer(data="\n\n".join(selection.cards))

        return StringContainer(data="")
