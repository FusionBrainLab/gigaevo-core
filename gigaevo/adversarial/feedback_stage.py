"""OpponentFeedbackStage: inject opponent source codes into mutation context.

Reads top-K opponent programs from the adversarial archive and formats their
source code as a structured text block, which is passed to MutationContextStage
via the 'formatted' slot. This gives the mutation LLM access to opponent code
(analogous to GAN gradients flowing between Generator and Discriminator).

Two roles:
  - "constructor": sees Improver code → "OPPONENT ATTACK REPORT"
  - "improver": sees Constructor code → "TARGET ANALYSIS REPORT"

Opponents are ranked by fitness descending (top-K most successful opponents).
"""

from __future__ import annotations

from typing import Any, Literal

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

_CONSTRUCTOR_HEADER = """\
=== OPPONENT ATTACK REPORT (top-{k} Improvers by fitness) ===
The following Improver programs have been the most effective at finding
improvements to Constructor configurations. Study their attack strategies
to harden your point placement against these approaches.\
"""

_IMPROVER_HEADER = """\
=== TARGET ANALYSIS REPORT (top-{k} Constructors by fitness) ===
The following Constructor programs have resisted improvement attempts most
effectively. Study their point placement strategies to identify weaknesses
that your improvement algorithm can exploit.\
"""

_OPPONENT_BLOCK = """\
--- Opponent {i} (fitness={fitness:.5f}) ---
```python
{code}
```\
"""


class OpponentFeedbackStage(Stage):
    """Inject top-K opponent source codes into the mutation prompt.

    Fetches opponent programs from OpponentArchiveProvider, selects the top-K
    by fitness (descending), and formats them into a structured text block.
    Returns a StringContainer fed to MutationContextStage.formatted.

    On cold start (empty archive), returns an empty string so the mutation
    LLM sees no feedback block (graceful degradation).

    Args:
        opponent_provider: Archive provider to fetch opponent programs from.
        k: Number of top opponents to include (default: 3).
        role: "constructor" (sees Improver code) or "improver" (sees Constructor
            code). Controls the report header and framing.
    """

    InputsModel = VoidInput
    OutputModel = StringContainer
    cache_handler = NO_CACHE  # opponents evolve — never cache

    def __init__(
        self,
        *,
        opponent_provider: OpponentArchiveProvider,
        k: int = 3,
        role: Literal["constructor", "improver"] = "constructor",
        higher_is_better: bool = True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._provider = opponent_provider
        self._k = k
        self._role = role
        self._higher_is_better = higher_is_better

    async def compute(self, program: Program) -> StringContainer:  # noqa: ARG002
        top_k = await self._provider.get_top_k(
            self._k, higher_is_better=self._higher_is_better
        )

        if not top_k:
            logger.info(
                "[OpponentFeedback] no opponents in archive (cold start) — skipping feedback"
            )
            return StringContainer(data="")

        if len(top_k) < self._k:
            logger.warning(
                "[OpponentFeedback] requested k={} but only {} opponents available (sparse archive)",
                self._k,
                len(top_k),
            )
        logger.info(
            "[OpponentFeedback] role={} k={} opponents selected (top fitness={:.5f})",
            self._role,
            len(top_k),
            top_k[0].fitness,
        )

        return StringContainer(data=self._format_report(top_k))

    def _format_report(self, opponents: list[OpponentProgram]) -> str:
        k_actual = len(opponents)
        if self._role == "constructor":
            header = _CONSTRUCTOR_HEADER.format(k=k_actual)
        else:
            header = _IMPROVER_HEADER.format(k=k_actual)

        blocks = [
            _OPPONENT_BLOCK.format(i=i, fitness=opp.fitness, code=opp.code.strip())
            for i, opp in enumerate(opponents, start=1)
        ]

        return "\n\n".join([header] + blocks)
