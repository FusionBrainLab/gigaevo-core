"""CompositionInjectionHook: compose D's improvement of G's output as a valid G program.

After each generation, reads D's best improvement program, executes it
against a randomly chosen G program's output, and (if the result differs)
injects the composed program into G's population.

The composed program is a standalone Python script that:
1. Embeds G's original point configuration as _G_POINTS
2. Contains D's code with entrypoint renamed to _d_entrypoint
3. Defines a new entrypoint() that applies D's improve function to G's points

Tagged with mutation_type="d_improvement" for lineage tracking.
This is the Lamarckian transfer mechanism for Arm A.
"""

from __future__ import annotations

import random
import re
from typing import Any

from loguru import logger
import numpy as np

from gigaevo.adversarial.opponent_provider import OpponentArchiveProvider
from gigaevo.database.program_storage import ProgramStorage
from gigaevo.programs.program import Program
from gigaevo.programs.stages.python_executors.wrapper import (
    ExecRunnerError,
    run_exec_runner,
)

_SUBPROCESS_TIMEOUT_S = 30

WRAPPER_TEMPLATE = """\
import numpy as np

# G's original point configuration
_G_POINTS = np.array({g_points_repr}, dtype=np.float64)

# --- D's code (entrypoint renamed to _d_entrypoint) ---
{d_code_with_renamed_entrypoint}

def entrypoint():
    \"\"\"D's improvement of G's points, as a valid G program.\"\"\"
    improve_fn = _d_entrypoint()
    improved = improve_fn(_G_POINTS.copy())
    return np.asarray(improved, dtype=np.float64)
"""


class CompositionInjectionHook:
    """Post-step hook: compose D's improvement of G's output into G's population.

    Args:
        d_provider: Opponent archive provider pointing to D's archive.
        g_storage: ProgramStorage connected to G's Redis DB (read and write).
        dg_tracker: Optional tracker for recording D-G improvement pairs.
    """

    def __init__(
        self,
        d_provider: OpponentArchiveProvider,
        g_storage: ProgramStorage,
        dg_tracker: Any | None = None,
    ):
        self._d_provider = d_provider
        self._g_storage = g_storage
        self._dg_tracker = dg_tracker

    async def __call__(self) -> None:
        """Hook-compatible interface: delegates to inject()."""
        await self.inject()

    @staticmethod
    def _compose_g_program(d_code: str, g_points_list: list[list[float]]) -> str:
        """Produce a standalone G program that applies D's improve to G's points.

        Args:
            d_code: D's full code containing an entrypoint() that returns
                a callable improve(pts) -> improved_pts.
            g_points_list: G's point configuration as a nested list.

        Returns:
            A standalone Python string defining entrypoint() -> (N,2) ndarray.
        """
        d_code_renamed = re.sub(
            r"^(def\s+)entrypoint(\s*\()",
            r"\1_d_entrypoint\2",
            d_code,
            count=1,
            flags=re.MULTILINE,
        )
        return WRAPPER_TEMPLATE.format(
            g_points_repr=repr(g_points_list),
            d_code_with_renamed_entrypoint=d_code_renamed,
        )

    async def inject(self) -> str | None:
        """Compose D's best improvement of a G program and inject if improved.

        Returns the injected program ID, or None if injection was skipped.
        """
        top_d = await self._d_provider.get_top_k(1, higher_is_better=True)
        if not top_d:
            logger.info("[CompositionInjection] D archive empty -- no injection")
            return None

        d_best = top_d[0]

        g_programs = await self._g_storage.get_all()
        if not g_programs:
            logger.info("[CompositionInjection] G archive empty -- no injection")
            return None

        g_prog = random.choice(g_programs)

        try:
            g_output, _, _ = await run_exec_runner(
                code=g_prog.code,
                function_name="entrypoint",
                timeout=_SUBPROCESS_TIMEOUT_S,
            )
        except (ExecRunnerError, TimeoutError, Exception) as e:
            logger.warning(
                "[CompositionInjection] G program {} exec failed: {}",
                g_prog.id[:8],
                e,
            )
            return None

        g_points = np.asarray(g_output, dtype=np.float64)
        g_points_list = g_points.tolist()

        composed_code = self._compose_g_program(d_best.code, g_points_list)

        try:
            composed_output, _, _ = await run_exec_runner(
                code=composed_code,
                function_name="entrypoint",
                timeout=_SUBPROCESS_TIMEOUT_S,
            )
        except (ExecRunnerError, TimeoutError, Exception) as e:
            logger.warning(
                "[CompositionInjection] Composed program exec failed: {}", e
            )
            return None

        composed_points = np.asarray(composed_output, dtype=np.float64)

        if np.array_equal(composed_points, g_points):
            logger.info(
                "[CompositionInjection] No improvement: composed output == G output"
            )
            return None

        program = Program(
            code=composed_code,
            metadata={
                "mutation_type": "d_improvement",
                "d_source_id": d_best.program_id,
                "g_source_id": g_prog.id,
                "d_fitness": d_best.fitness,
            },
        )
        await self._g_storage.add(program)

        logger.info(
            "[CompositionInjection] mutation_type=d_improvement "
            "d_id={} g_id={} d_fitness={:.5f} injected_id={}",
            d_best.program_id,
            g_prog.id[:8],
            d_best.fitness,
            program.id,
        )

        if self._dg_tracker is not None:
            try:
                await self._dg_tracker.record_improvement(
                    d_id=d_best.program_id,
                    g_id=g_prog.id,
                    injected_id=program.id,
                )
            except Exception as e:
                logger.warning(
                    "[CompositionInjection] dg_tracker.record_improvement failed: {}",
                    e,
                )

        return program.id
