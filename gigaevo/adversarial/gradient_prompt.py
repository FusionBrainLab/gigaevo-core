"""GradientInPromptStage: inject D's best code into G's mutation prompt.

Reads D's best improvement from the opponent archive and formats it as
"Improvement Strategy from Opponent" in G's mutation prompt. Textual
gradient mechanism for Arm C — G's LLM sees D's strategy as a hint.

Unlike CompositionInjection (Arm A), no direct code injection into G's
population. G's LLM decides how to incorporate D's strategy.

When a DGImprovementTracker is provided, selects the D that most improved
the specific G program being mutated (per-program selection). Falls back
to global best D when no per-program data exists.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

from gigaevo.adversarial.opponent_provider import (
    OpponentArchiveProvider,
    OpponentProgram,
)
from gigaevo.programs.core_types import VoidInput
from gigaevo.programs.program import Program
from gigaevo.programs.stages.base import Stage
from gigaevo.programs.stages.cache_handler import NO_CACHE
from gigaevo.programs.stages.common import StringContainer

if TYPE_CHECKING:
    from gigaevo.adversarial.dg_tracker import DGImprovementTracker

_GRADIENT_HEADER = """\
## Improvement Strategy from Opponent

The following Improver program has been the most effective at finding
improvements to Constructor configurations like yours. Study its approach
to understand what weaknesses it exploits, then evolve your point placement
strategy to be resistant to these attacks.

**Improver fitness**: {fitness:.5f}\
"""

_GRADIENT_BLOCK = """\
```python
{code}
```

Use this information to guide your mutation — but generate your own original
point placement code. Do not copy the Improver's code directly.\
"""


class GradientInPromptStage(Stage):
    """Inject D's best improvement code into G's mutation prompt.

    When dg_tracker is provided, selects the D that most improved the
    specific G being mutated. Falls back to global best D otherwise.
    """

    InputsModel = VoidInput
    OutputModel = StringContainer
    cache_handler = NO_CACHE

    def __init__(
        self,
        *,
        opponent_provider: OpponentArchiveProvider,
        dg_tracker: DGImprovementTracker | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._provider = opponent_provider
        self._dg_tracker = dg_tracker

    async def compute(self, program: Program) -> StringContainer:
        d_best = await self._select_best_d(program)
        if d_best is None:
            logger.info("[GradientInPrompt] no D programs available -- skipping")
            return StringContainer(data="")

        logger.info(
            "[GradientInPrompt] injecting D into G prompt "
            "(d_id={} fitness={:.5f} source={})",
            d_best.program_id,
            d_best.fitness,
            "per-program" if self._dg_tracker else "global",
        )
        header = _GRADIENT_HEADER.format(fitness=d_best.fitness)
        block = _GRADIENT_BLOCK.format(code=d_best.code.strip())
        return StringContainer(data=f"{header}\n\n{block}")

    async def _select_best_d(self, program: Program) -> OpponentProgram | None:
        """Select the best D for this specific G program.

        Priority:
        1. Per-program best D from DGImprovementTracker (if tracker configured and has data)
        2. Global best D from opponent_provider.get_top_k(1) (fallback)
        """
        if self._dg_tracker is not None:
            best = await self._dg_tracker.get_best_d_for_g(program.id)
            if best is not None:
                d_id, delta = best
                programs = await self._provider.get_programs_by_ids([d_id])
                if programs:
                    logger.debug(
                        "[GradientInPrompt] per-program D: d={} delta={:.6f} for g={}",
                        d_id,
                        delta,
                        program.id,
                    )
                    return programs[0]
                logger.debug(
                    "[GradientInPrompt] per-program D {} not in archive, falling back to global",
                    d_id,
                )

        top = await self._provider.get_top_k(1, higher_is_better=True)
        return top[0] if top else None
