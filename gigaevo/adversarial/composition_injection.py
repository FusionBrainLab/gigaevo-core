"""CompositionInjectionHook: inject D's best improvement into G's DAG flow.

After each generation, reads D's best improvement program and submits it
as a tagged mutation candidate into G's population. The injected program
goes through G's full DAG pipeline (validation, fitness eval).

Tagged with mutation_type="d_improvement" for lineage tracking.
This is the Lamarckian transfer mechanism for Arm A.
"""

from __future__ import annotations

from loguru import logger

from gigaevo.adversarial.opponent_provider import OpponentArchiveProvider
from gigaevo.database.program_storage import ProgramStorage
from gigaevo.programs.program import Program


class CompositionInjectionHook:
    """Post-sync hook: inject D's best improvement into G's population.

    Args:
        d_provider: Opponent archive provider pointing to D's archive.
        g_storage: ProgramStorage connected to G's Redis DB.
    """

    def __init__(
        self,
        d_provider: OpponentArchiveProvider,
        g_storage: ProgramStorage,
    ):
        self._d_provider = d_provider
        self._g_storage = g_storage

    async def inject(self) -> str | None:
        """Read D's best and submit to G's population. Returns program ID or None."""
        top = await self._d_provider.get_top_k(1, higher_is_better=True)
        if not top:
            logger.info("[CompositionInjection] D archive empty — no injection")
            return None

        d_best = top[0]
        program = Program(
            code=d_best.code,
            metadata={
                "mutation_type": "d_improvement",
                "d_source_id": d_best.program_id,
                "d_fitness": d_best.fitness,
            },
        )
        await self._g_storage.add(program)

        logger.info(
            "[CompositionInjection] mutation_type=d_improvement "
            "d_fitness={:.5f} code_len={} injected_id={}",
            d_best.fitness,
            len(d_best.code),
            program.id,
        )
        return program.id
