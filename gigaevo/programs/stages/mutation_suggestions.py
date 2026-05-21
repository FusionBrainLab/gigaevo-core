"""MutationSuggestionStage — descriptive memory cards + ancestral trail → actionable suggestions.

This stage is the bridge between purely-descriptive memory (intra lineage
card + cross-population memory cards) and the mutator. The cards say "what
was tried and how it fared"; the ancestral trail says "what gains the
lineage has made over recent generations"; this stage says "given that,
here is what the next mutation should change, citing literal anchors and
concrete substitutes". The mutator then consumes its structured output via
the PROGRAM INSIGHTS section of the mutation prompt.

**Why split from IntraMemoryStage.** The intra stage is descriptive only —
it summarises history but doesn't prescribe. Keeping prescription in a
separate stage lets us:

* Cache the (expensive) descriptive card aggressively on children-id
  changes, and re-prescribe (cheap deltas in input hash) whenever the
  parent's neighbourhood evolves — without re-doing the lineage clustering.
* Pipe in additional signals (ancestral momentum trail) that only matter
  for prescription, not description.
* Plug in different prescribers (e.g. an exploit-mode vs. explore-mode
  variant) without touching the history-summary stage.

**Caching.** ``intra_card`` and ``memory_cards`` are first-class
``InputsModel`` fields, so the framework's ``InputHashCache`` folds them
into the cache key. Re-running on the same parent yields a cache HIT
whenever both cards are unchanged. The ancestral trail is computed
internally from storage; trail changes ride on the children-id change that
already invalidates the intra card upstream, so the chain stays consistent.

**Soft-fail.** When the LLM call raises, the stage returns an empty
``ProgramInsights(insights=[])`` so the downstream mutator still receives
a valid (but empty) structured input — the mutation prompt's PROGRAM
INSIGHTS section just collapses to "(none)" without breaking the run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from langchain_openai import ChatOpenAI
from loguru import logger

from gigaevo.database.program_storage import ProgramStorage
from gigaevo.llm.agents.insights import ProgramInsights
from gigaevo.llm.models import MultiModelRouter
from gigaevo.programs.core_types import ProgramStageResult, StageIO
from gigaevo.programs.metrics.context import MetricsContext
from gigaevo.programs.program import Program
from gigaevo.programs.stages.collector import EvolutionaryStatistics
from gigaevo.programs.stages.common import StringContainer
from gigaevo.programs.stages.insights import InsightsOutput
from gigaevo.programs.stages.langgraph_stage import LangGraphStage
from gigaevo.programs.stages.lineage_memory import (
    DEFAULT_TRAIL_MAX_ANCESTORS,
    DEFAULT_TRAIL_MAX_DEPTH,
    collect_ancestral_trail,
)
from gigaevo.programs.stages.stage_registry import StageRegistry


class MutationSuggestionInputs(StageIO):
    """Inputs to ``MutationSuggestionStage``.

    ``intra_card`` is the per-parent lineage card emitted by
    ``IntraMemoryStage``; ``memory_cards`` is the rendered cross-population
    block emitted by ``MemoryContextStage``. Either may be absent (e.g. seed
    programs with no children yet, runs without an extra-memory hook). When
    absent, the corresponding section of the user prompt collapses cleanly
    to empty.

    ``evolutionary_statistics`` is the run-level window snapshot emitted by
    ``EvolutionaryStatisticsCollector``. It tells the suggester where the
    parent sits in the whole run (plateau / streak / invalid streak) so the
    suggester can reshape PRIORITY across its grounded suggestions (cards,
    code, trail). Absent on the very first iteration of a run.

    The ancestral trail is NOT in this model — it is computed inside the
    stage from storage on each invocation so we don't have to materialise
    a long trail object into the cache key. The intra card's id-list-driven
    invalidation already covers "the lineage has evolved" cases.
    """

    intra_card: StringContainer | None = None
    memory_cards: StringContainer | None = None
    evolutionary_statistics: EvolutionaryStatistics | None = None


# Output is ``InsightsOutput`` (not a separate class) so the same
# ``MutationContextInputs.insights`` slot accepts suggestions interchangeably
# with the legacy ``InsightsStage`` output — no changes needed downstream.
MutationSuggestionOutput = InsightsOutput


@StageRegistry.register(
    description=(
        "LLM-driven actionable mutation suggestions from memory cards + ancestral trail"
    )
)
class MutationSuggestionStage(LangGraphStage):
    """LangGraph wrapper around ``MutationSuggestionAgent``.

    Inputs (validated into ``MutationSuggestionInputs``): optional intra
    lineage card + optional memory cards block. The ancestral trail is
    collected from storage inside ``compute``. Output: ``InsightsOutput``
    — same shape as ``InsightsStage`` so ``MutationContextStage.insights``
    accepts either source interchangeably.
    """

    InputsModel: type[StageIO] = MutationSuggestionInputs
    OutputModel: type[StageIO] = InsightsOutput

    def __init__(
        self,
        *,
        llm: ChatOpenAI | MultiModelRouter,
        storage: ProgramStorage,
        metrics_context: MetricsContext,
        task_description: str,
        max_insights: int = 7,
        trail_max_depth: int = DEFAULT_TRAIL_MAX_DEPTH,
        trail_max_ancestors: int = DEFAULT_TRAIL_MAX_ANCESTORS,
        prompts_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        self._max_insights = max_insights
        self._storage = storage
        self._metrics_context = metrics_context
        self._trail_max_depth = trail_max_depth
        self._trail_max_ancestors = trail_max_ancestors
        from gigaevo.llm.agents.factories import create_mutation_suggestion_agent

        super().__init__(
            agent=create_mutation_suggestion_agent(
                llm,
                task_description,
                metrics_context,
                max_insights,
                prompts_dir=prompts_dir,
            ),
            program_kwarg="program",
            **kwargs,
        )

    async def preprocess(
        self, program: Program, params: StageIO
    ) -> dict[str, Any] | ProgramStageResult:
        """Unwrap optional StringContainer inputs into bare strings.

        Empty / whitespace-only cards collapse to ``None`` so the agent
        omits the section entirely instead of rendering an empty header.
        """
        p = cast(MutationSuggestionInputs, params)
        intra = (
            p.intra_card.data.strip()
            if p.intra_card is not None and p.intra_card.data
            else ""
        ) or None
        cards = (
            p.memory_cards.data.strip()
            if p.memory_cards is not None and p.memory_cards.data
            else ""
        ) or None
        return {
            "intra_card": intra,
            "memory_cards": cards,
            "evolutionary_statistics": p.evolutionary_statistics,
        }

    async def compute(self, program: Program) -> InsightsOutput:
        prep = await self.preprocess(program, self.params)
        if isinstance(prep, ProgramStageResult):
            return prep  # type: ignore[return-value]
        kwargs = dict(prep)

        # Trail is computed from storage on every invocation. It rides on the
        # upstream intra-card cache-invalidation for "new children evaluated"
        # (which updates this parent's ancestry's children-id sets too).
        try:
            trail = await collect_ancestral_trail(
                program,
                self._storage,
                self._metrics_context,
                max_depth=self._trail_max_depth,
                max_total_ancestors=self._trail_max_ancestors,
            )
        except Exception:
            logger.opt(exception=True).warning(
                "[MutationSuggestionStage] ancestral trail walk failed for {}; "
                "proceeding with empty trail",
                program.id[:8],
            )
            trail = []
        kwargs["ancestral_trail"] = trail

        kwargs[self.program_kwarg] = program  # type: ignore[index]
        try:
            insights = await self.agent.arun(**kwargs)
        except Exception:
            logger.opt(exception=True).warning(
                "[MutationSuggestionStage] LLM call failed for {}; returning "
                "empty suggestions",
                program.id[:8],
            )
            insights = ProgramInsights(insights=[])
        return InsightsOutput(insights=insights)
