"""DAG stage that selects memory cards via the injected MemoryProvider."""

from __future__ import annotations

from typing import Any

from loguru import logger

from gigaevo.evolution.mutation.constants import (
    MUTATION_MEMORY_CANDIDATE_SLATE_METADATA_KEY,
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


class MemoryExposureCounter:
    """Run-lifetime exposure telemetry for card injection.

    The DAG builds a fresh ``MemoryContextStage`` per program, so the counter
    must be a single object shared through the pipeline-builder closure.
    """

    summary_every: int = 50

    def __init__(self) -> None:
        self.attempts = 0
        self.non_empty = 0

    def record(self, *, program_id: str, card_ids: list[str]) -> None:
        self.attempts += 1
        if card_ids:
            self.non_empty += 1
            if self.non_empty == 1:
                logger.info(
                    "[Memory][Exposure] FIRST_INJECTION program={} ids={}",
                    program_id[:8],
                    card_ids,
                )
        if self.attempts % self.summary_every == 0:
            logger.info(
                "[Memory][Exposure] attempts={} non_empty={} ({:.0f}%)",
                self.attempts,
                self.non_empty,
                100.0 * self.non_empty / self.attempts,
            )


@StageRegistry.register(description="Select memory cards for mutation context")
class MemoryContextStage(Stage):
    """Select memory cards via the injected MemoryProvider.

    Always present in the DAG. When the provider is NullMemoryProvider,
    this stage returns an empty string instantly (no-op).

    Always writes selected ids and slate metadata, empty lists included —
    this NO_CACHE stage re-runs on every parent requeue, and a stale slate
    left behind by a previous run would hand phantom credit to children.
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
        exposure: MemoryExposureCounter | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._provider = memory_provider
        self._task_description = task_description
        self._metrics_description = metrics_description
        self._exposure = exposure if exposure is not None else MemoryExposureCounter()

    async def compute(self, program: Program) -> StageIO:
        # Erase any slate left by a prior run of this NO_CACHE stage BEFORE the
        # provider call: a select_cards failure (e.g. stage timeout) must not
        # leave a requeued parent carrying a stale slate that would hand phantom
        # credit to its children. Overwritten with the real selection on success.
        program.set_metadata(MUTATION_MEMORY_CANDIDATE_SLATE_METADATA_KEY, [])
        program.set_metadata(MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY, [])

        selection = await self._provider.select_cards(
            program,
            task_description=self._task_description,
            metrics_description=self._metrics_description,
        )

        program.set_metadata(
            MUTATION_MEMORY_CANDIDATE_SLATE_METADATA_KEY,
            [bid.model_dump() for bid in selection.slate],
        )
        program.set_metadata(
            MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY, list(selection.card_ids)
        )
        self._exposure.record(program_id=program.id, card_ids=selection.card_ids)

        if selection.cards:
            logger.info(
                "[Memory][ContextStage] Selected {} card(s) for {} (ids={})",
                len(selection.cards),
                program.id[:8],
                selection.card_ids,
            )
            numbered = [
                f"[card {i}] id={cid}\n{text}"
                for i, (text, cid) in enumerate(
                    zip(selection.cards, selection.card_ids), start=1
                )
            ]
            return StringContainer(data="\n\n".join(numbered))

        logger.info(
            "[Memory][ContextStage] Empty selection for {} (candidates={})",
            program.id[:8],
            len(selection.slate),
        )
        return StringContainer(data="")
