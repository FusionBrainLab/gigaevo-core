"""SourceCodeInjectionStage: inject G's source code into D's mutation context.

Receives opponent IDs from FetchOpponentIdsStage (the SAME opponents D
evaluates against), fetches their source code via get_programs_by_ids,
ranks by fitness descending, and shows the top-l in D's mutation prompt.

Parameters:
    source_prompt_k (l): how many opponent source codes to show (l <= k).
    Default: 1.

This gives D white-box access to G's strategy. D can reason about HOW G
places points, not just WHERE. The source code shown is from the exact
opponents D evaluates against — no conceptual leakage.
"""

from __future__ import annotations

from typing import Any, cast

from loguru import logger

from gigaevo.adversarial.opponent_provider import OpponentArchiveProvider
from gigaevo.programs.core_types import StageIO
from gigaevo.programs.program import Program
from gigaevo.programs.stages.base import Stage
from gigaevo.programs.stages.cache_handler import NO_CACHE
from gigaevo.programs.stages.common import Box, StringContainer

_SOURCE_HEADER = """\
=== TARGET CONSTRUCTOR SOURCE CODE ({n} of {total} opponents, best by fitness) ===
The following is the source code of the Constructor program(s) you are
trying to improve. Study the point placement strategy to find weaknesses
that your improvement algorithm can exploit. You have white-box access
to the algorithm — use this to craft targeted improvements.\
"""

_SOURCE_BLOCK = """\
--- Constructor (fitness={fitness:.5f}) ---
```python
{code}
```\
"""


class SourceCodeInjectionInput(StageIO):
    """Input: opponent IDs from FetchOpponentIdsStage."""

    opponent_ids: Box[Any]


class SourceCodeInjectionStage(Stage):
    """Inject source code of D's evaluation opponents into D's mutation prompt.

    Receives opponent_ids from FetchOpponentIdsStage (shared with
    FetchOpponentResultsStage). Fetches full OpponentProgram objects,
    ranks by fitness descending, and formats top-l as a text block.

    On cold start (empty IDs or no programs found), returns empty string.
    """

    InputsModel = SourceCodeInjectionInput
    OutputModel = StringContainer
    cache_handler = NO_CACHE

    def __init__(
        self,
        *,
        opponent_provider: OpponentArchiveProvider,
        source_prompt_k: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._provider = opponent_provider
        self._l = source_prompt_k

    async def compute(self, program: Program) -> StringContainer:  # noqa: ARG002
        params = cast(SourceCodeInjectionInput, self.params)
        ids: list[str] = params.opponent_ids.data

        if not ids:
            logger.info("[SourceCodeInjection] no opponent IDs — skipping")
            return StringContainer(data="")

        programs = await self._provider.get_programs_by_ids(ids)
        if not programs:
            logger.info("[SourceCodeInjection] no programs found for IDs — skipping")
            return StringContainer(data="")

        ranked = sorted(programs, key=lambda p: p.fitness, reverse=True)
        top_l = ranked[: self._l]

        logger.info(
            "[SourceCodeInjection] showing {}/{} opponents (top fitness={:.5f})",
            len(top_l),
            len(programs),
            top_l[0].fitness,
        )

        header = _SOURCE_HEADER.format(n=len(top_l), total=len(programs))
        blocks = [
            _SOURCE_BLOCK.format(fitness=p.fitness, code=p.code.strip()) for p in top_l
        ]
        return StringContainer(data="\n\n".join([header] + blocks))
