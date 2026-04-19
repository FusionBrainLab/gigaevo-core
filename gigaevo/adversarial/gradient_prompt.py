"""GradientInPromptStage: inject D's best code into G's mutation prompt.

Reads D's best improvement from the opponent archive and formats it as
"Improvement Strategy from Opponent" in G's mutation prompt. Textual
gradient mechanism for Arm C — G's LLM sees D's strategy as a hint.

Unlike CompositionInjection (Arm A), no direct code injection into G's
population. G's LLM decides how to incorporate D's strategy.

Selection semantics
-------------------
- **Tracker configured (v3 default):** selects the D that most improved the
  specific G program being mutated via ``DGImprovementTracker.get_best_d_for_g``.
  When the tracker has no per-G entry for this G, **no D is injected** — the
  stage returns an empty string (no global-top-1 fallback). Rationale: a D
  that is generically strong but has never specifically improved *this* G
  provides a misleading signal; a cleaner prompt beats a dishonest one.
  The per-G delta is threaded into the prompt header alongside D's fitness.
- **Tracker absent (legacy / non-adversarial experiments):** falls back to
  ``opponent_provider.get_top_k(1)`` — preserves the pre-v3 behavior.
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

_GRADIENT_HEADER_PER_G = """\
## Improvement Strategy from Opponent

The following Improver program has been the most effective at finding
improvements to Constructor configurations like yours. Study its approach
to understand what weaknesses it exploits, then evolve your point placement
strategy to be resistant to these attacks.

**Improver fitness**: {fitness:.5f}
**Per-program improvement on *this* Constructor**: {delta:+.5f}\
"""

_GRADIENT_HEADER_GLOBAL = """\
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
        selection = await self._select_best_d(program)
        if selection is None:
            logger.info(
                "[GradientInPrompt] no D to inject (tracker={}) -- skipping",
                "per-G" if self._dg_tracker is not None else "global",
            )
            return StringContainer(data="")

        d_best, delta = selection
        if delta is not None:
            header = _GRADIENT_HEADER_PER_G.format(fitness=d_best.fitness, delta=delta)
            source = "per-program"
        else:
            header = _GRADIENT_HEADER_GLOBAL.format(fitness=d_best.fitness)
            source = "global"

        logger.info(
            "[GradientInPrompt] injecting D into G prompt "
            "(d_id={} fitness={:.5f} delta={} source={})",
            d_best.program_id,
            d_best.fitness,
            f"{delta:+.5f}" if delta is not None else "n/a",
            source,
        )
        block = _GRADIENT_BLOCK.format(code=d_best.code.strip())
        return StringContainer(data=f"{header}\n\n{block}")

    async def _select_best_d(
        self, program: Program
    ) -> tuple[OpponentProgram, float | None] | None:
        """Select the best D for this specific G program.

        Returns ``(OpponentProgram, delta)`` where ``delta`` is the per-G
        improvement (positive), or ``None`` if no D should be injected.

        Semantics:
        - Tracker configured: query ``get_best_d_for_g``. If it has a per-G
          entry and the D is still in the archive, return ``(D, delta)``.
          Otherwise return ``None`` — **no global fallback** (v3). This
          keeps the adversarial signal honest: we only inject a D that has
          actually improved *this* G.
        - Tracker absent: fall back to ``opponent_provider.get_top_k(1)``
          and return ``(D, None)`` — delta is unknown, header omits it.
          Preserves pre-v3 behavior for experiments that do not wire a
          tracker.
        """
        if self._dg_tracker is not None:
            best = await self._dg_tracker.get_best_d_for_g(program.id)
            if best is None:
                logger.debug(
                    "[GradientInPrompt] no per-G tracker entry for g={}",
                    program.id,
                )
                return None
            d_id, delta = best
            programs = await self._provider.get_programs_by_ids([d_id])
            if not programs:
                logger.debug(
                    "[GradientInPrompt] per-G D {} not in archive for g={}"
                    " -- skipping (no global fallback in v3)",
                    d_id,
                    program.id,
                )
                return None
            logger.debug(
                "[GradientInPrompt] per-program D: d={} delta={:.6f} for g={}",
                d_id,
                delta,
                program.id,
            )
            return (programs[0], delta)

        top = await self._provider.get_top_k(1, higher_is_better=True)
        if not top:
            return None
        return (top[0], None)
