"""GradientInPromptStage: inject D's best code into G's mutation prompt.

Reads D's best improvement from the opponent archive and formats it as
"Improvement Strategy from Opponent" in G's mutation prompt. Textual
gradient mechanism for Arm C — G's LLM sees D's strategy as a hint.

Unlike CompositionInjection (Arm A), no direct code injection into G's
population. G's LLM decides how to incorporate D's strategy.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from gigaevo.adversarial.opponent_provider import OpponentArchiveProvider
from gigaevo.programs.core_types import VoidInput
from gigaevo.programs.program import Program
from gigaevo.programs.stages.base import Stage
from gigaevo.programs.stages.cache_handler import NO_CACHE
from gigaevo.programs.stages.common import StringContainer

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
    """Inject D's best improvement code into G's mutation prompt."""

    InputsModel = VoidInput
    OutputModel = StringContainer
    cache_handler = NO_CACHE

    def __init__(self, *, opponent_provider: OpponentArchiveProvider, **kwargs: Any):
        super().__init__(**kwargs)
        self._provider = opponent_provider

    async def compute(self, program: Program) -> StringContainer:  # noqa: ARG002
        top = await self._provider.get_top_k(1, higher_is_better=True)
        if not top:
            logger.info("[GradientInPrompt] no D programs (cold start) — skipping")
            return StringContainer(data="")

        d_best = top[0]
        logger.info(
            "[GradientInPrompt] injecting D's best into G prompt (fitness={:.5f})",
            d_best.fitness,
        )
        header = _GRADIENT_HEADER.format(fitness=d_best.fitness)
        block = _GRADIENT_BLOCK.format(code=d_best.code.strip())
        return StringContainer(data=f"{header}\n\n{block}")
