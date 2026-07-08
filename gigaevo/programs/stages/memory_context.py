"""DAG stage that selects memory cards via the injected MemoryProvider."""

from __future__ import annotations

from collections.abc import Sequence
import random
from typing import Any, cast

from loguru import logger

from gigaevo.evolution.mutation.constants import (
    MUTATION_MEMORY_CANDIDATE_SLATE_METADATA_KEY,
    MUTATION_MEMORY_NO_CARD_CONTROL_METADATA_KEY,
    MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY,
)
from gigaevo.evolution.mutation.context import (
    CompositeMutationContext,
    EvolutionaryStatisticsMutationContext,
    MemoryMutationContext,
)
from gigaevo.memory.provider import MemoryProvider
from gigaevo.programs.metrics.context import MetricsContext
from gigaevo.programs.program import Program
from gigaevo.programs.stages.base import Stage
from gigaevo.programs.stages.cache_handler import NO_CACHE
from gigaevo.programs.stages.collector import EvolutionaryStatistics
from gigaevo.programs.stages.common import StageIO, StringContainer
from gigaevo.programs.stages.stage_registry import StageRegistry


class MemoryContextInputs(StageIO):
    """Inputs for MemoryContextStage.

    Both optional: the stage conditions extra-memory card selection on the
    fresh this-pass lineage card (``intra_card``) and the live evolutionary
    snapshot (``evolutionary_statistics``). Absent on a cold seed's first pass.
    """

    intra_card: StringContainer | None = None
    evolutionary_statistics: EvolutionaryStatistics | None = None


class MemoryExposureCounter:
    """Run-lifetime exposure telemetry for card injection.

    The DAG builds a fresh ``MemoryContextStage`` per program, so the counter
    must be a single object shared through the pipeline-builder closure.
    """

    summary_every: int = 50

    def __init__(self) -> None:
        self.attempts = 0
        self.non_empty = 0

    def record(self, *, program_id: str, card_ids: Sequence[str]) -> None:
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
        metrics_context: MetricsContext | None = None,
        exposure: MemoryExposureCounter | None = None,
        fresh_context_reorder: bool = True,
        reverse_repack: bool = False,
        no_card_control_probability: float = 0.0,
        no_card_control_rng: Any | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if not 0.0 <= no_card_control_probability <= 1.0:
            raise ValueError(
                "no_card_control_probability must be in [0, 1], "
                f"got {no_card_control_probability}"
            )
        self._provider = memory_provider
        self._task_description = task_description
        self._metrics_description = metrics_description
        self._metrics_context = metrics_context
        self._exposure = exposure if exposure is not None else MemoryExposureCounter()
        self._fresh_context_reorder = fresh_context_reorder
        self._reverse_repack = reverse_repack
        self._no_card_control_probability = no_card_control_probability
        self._no_card_control_rng = (
            no_card_control_rng if no_card_control_rng is not None else random.Random()
        )

    def _build_parent_context(self) -> str:
        # Mirror the fresh this-pass mutation context the mutation agent will
        # see, so the GAM selects cards conditioned on the lineage card + live
        # evolutionary snapshot instead of a one-pass-stale metadata block.
        contexts: list[Any] = []
        params = cast(MemoryContextInputs, self.params)
        intra_card = params.intra_card
        if intra_card is not None and intra_card.data:
            contexts.append(MemoryMutationContext(memory_block=intra_card.data))
        evo = params.evolutionary_statistics
        if evo is not None and self._metrics_context is not None:
            contexts.append(
                EvolutionaryStatisticsMutationContext(
                    evolutionary_statistics=evo,
                    metrics_context=self._metrics_context,
                )
            )
        if not contexts:
            return ""
        return CompositeMutationContext(contexts=contexts).format()

    async def compute(self, program: Program) -> StageIO:
        # Erase any slate left by a prior run of this NO_CACHE stage BEFORE the
        # provider call: a select_cards failure (e.g. stage timeout) must not
        # leave a requeued parent carrying a stale slate that would hand phantom
        # credit to its children. Overwritten with the real selection on success.
        program.set_metadata(MUTATION_MEMORY_CANDIDATE_SLATE_METADATA_KEY, [])
        program.set_metadata(MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY, [])
        program.set_metadata(MUTATION_MEMORY_NO_CARD_CONTROL_METADATA_KEY, False)

        # Arm B (reorder off): pass None so the selector falls back to the stale
        # parent.metadata[MUTATION_CONTEXT] block — the pre-reorder behaviour.
        parent_context = (
            self._build_parent_context() if self._fresh_context_reorder else None
        )
        selection = await self._provider.select_cards(
            program,
            task_description=self._task_description,
            metrics_description=self._metrics_description,
            parent_context=parent_context,
        )

        program.set_metadata(
            MUTATION_MEMORY_CANDIDATE_SLATE_METADATA_KEY,
            [bid.model_dump() for bid in selection.slate],
        )

        withheld_for_control = bool(selection.card_ids) and (
            self._no_card_control_rng.random() < self._no_card_control_probability
        )
        if withheld_for_control:
            program.set_metadata(MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY, [])
            program.set_metadata(MUTATION_MEMORY_NO_CARD_CONTROL_METADATA_KEY, True)
            self._exposure.record(program_id=program.id, card_ids=())
            logger.info(
                "[Memory][ContextStage] Withheld {} selected card(s) for no-card "
                "control on {} (ids={})",
                len(selection.card_ids),
                program.id[:8],
                selection.card_ids,
            )
            return StringContainer(data="")

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
            pairs = list(zip(selection.cards, selection.card_ids))
            # Reverse repacking (lost-in-the-middle mitigation): present cards
            # worst-first so the strongest candidate sits last, nearest the
            # downstream instruction. Selection-order metadata is untouched.
            if self._reverse_repack:
                pairs.reverse()
            numbered = [
                f"[card {i}] id={cid}\n{text}"
                for i, (text, cid) in enumerate(pairs, start=1)
            ]
            return StringContainer(data="\n\n".join(numbered))

        logger.info(
            "[Memory][ContextStage] Empty selection for {} (candidates={})",
            program.id[:8],
            len(selection.slate),
        )
        return StringContainer(data="")
