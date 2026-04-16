"""CompositionInjectionHook: Lamarckian transfer via D ∘ G code composition.

After each generation, iterates ALL G programs in G's archive. For each G,
looks up the best D for that specific G (via DGImprovementTracker). If the
(D, G) pair has not been injected before, composes a new G program whose
entrypoint() returns D(G()) and adds it to G's storage. The pair is then
permanently marked as injected — never composed again.

Composition is CODE-level, not data-level: G's code is renamed to
_g_entrypoint, D's code is renamed to _d_entrypoint, and a new entrypoint()
chains them. MAP-Elites evaluates the result; no pre-check on fitness.

If no D-improvement is recorded for any G in the archive, no programs are
injected this iteration.

Tagged with mutation_type="d_improvement" for lineage tracking.
This is the Lamarckian transfer mechanism for Arm A.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from loguru import logger

from gigaevo.adversarial.opponent_provider import OpponentArchiveProvider
from gigaevo.database.program_storage import ProgramStorage
from gigaevo.programs.program import Program

if TYPE_CHECKING:
    from gigaevo.adversarial.dg_tracker import DGImprovementTracker


WRAPPER_TEMPLATE = """\
# --- G's code (entrypoint renamed to _g_entrypoint) ---
{g_code_renamed}

# --- D's code (entrypoint renamed to _d_entrypoint) ---
{d_code_renamed}

def entrypoint():
    \"\"\"Lamarckian composition: D applied to G's output.\"\"\"
    g_output = _g_entrypoint()
    d_callable = _d_entrypoint()
    return d_callable(g_output)
"""


def _rename_entrypoint(code: str, new_name: str) -> str:
    """Rename the first top-level `def entrypoint(...)` to `def {new_name}(...)`."""
    return re.sub(
        r"^(def\s+)entrypoint(\s*\()",
        rf"\1{new_name}\2",
        code,
        count=1,
        flags=re.MULTILINE,
    )


class CompositionInjectionHook:
    """Post-step hook: compose D ∘ G for every (D, G) pair the tracker has improved.

    On each invocation, walks all G programs in storage and for each one looks up
    the best D recorded by the tracker. If a D exists and the pair has not yet
    been injected, composes the chained-entrypoint program and inserts it into
    G's storage. Pair is then permanently marked — never repeated.

    Args:
        d_provider: Opponent archive provider pointing to D's archive.
        g_storage: ProgramStorage connected to G's Redis DB (read and write).
        dg_tracker: DGImprovementTracker for per-G best-D lookup and dedup.
    """

    def __init__(
        self,
        d_provider: OpponentArchiveProvider,
        g_storage: ProgramStorage,
        dg_tracker: DGImprovementTracker,
    ):
        self._d_provider = d_provider
        self._g_storage = g_storage
        self._dg_tracker = dg_tracker

    async def __call__(self) -> None:
        """Hook-compatible interface: delegates to inject_all()."""
        await self.inject_all()

    @staticmethod
    def _compose(g_code: str, d_code: str) -> str:
        """Build a standalone G program: entrypoint() returns D(G())."""
        return WRAPPER_TEMPLATE.format(
            g_code_renamed=_rename_entrypoint(g_code, "_g_entrypoint"),
            d_code_renamed=_rename_entrypoint(d_code, "_d_entrypoint"),
        )

    async def inject_all(self) -> list[str]:
        """Compose D ∘ G for every G in storage with a recorded best-D.

        Returns the list of injected program IDs (may be empty).
        """
        g_programs = await self._g_storage.get_all()
        if not g_programs:
            logger.info("[CompositionInjection] G archive empty -- no injections")
            return []

        logger.info(
            "[CompositionInjection] scanning {} G programs for D-improvements",
            len(g_programs),
        )

        injected_ids: list[str] = []
        n_no_d = 0
        n_dedup = 0
        n_d_missing = 0
        n_compose_error = 0

        for g_prog in g_programs:
            g_id = g_prog.id
            best = await self._dg_tracker.get_best_d_for_g(g_id)
            if best is None:
                n_no_d += 1
                continue
            d_id, delta = best

            if await self._dg_tracker.is_pair_injected(d_id, g_id):
                n_dedup += 1
                logger.debug(
                    "[CompositionInjection] skip dedup d={} g={} (already injected)",
                    d_id,
                    g_id[:8],
                )
                continue

            d_programs = await self._d_provider.get_programs_by_ids([d_id])
            if not d_programs:
                n_d_missing += 1
                logger.info(
                    "[CompositionInjection] skip d={} g={}: D no longer in archive",
                    d_id,
                    g_id[:8],
                )
                continue
            d_best = d_programs[0]

            try:
                composed_code = self._compose(g_prog.code, d_best.code)
            except Exception as e:
                n_compose_error += 1
                logger.warning(
                    "[CompositionInjection] compose failed d={} g={}: {}",
                    d_id,
                    g_id[:8],
                    e,
                )
                continue

            program = Program(
                code=composed_code,
                metadata={
                    "mutation_type": "d_improvement",
                    "d_source_id": d_best.program_id,
                    "g_source_id": g_id,
                    "d_fitness": d_best.fitness,
                    "tracked_delta": float(delta),
                },
            )
            await self._g_storage.add(program)
            await self._dg_tracker.mark_pair_injected(d_id, g_id)
            injected_ids.append(program.id)
            logger.info(
                "[CompositionInjection] mutation_type=d_improvement "
                "d_id={} g_id={} d_fitness={:.5f} tracked_delta={:.6f} injected_id={}",
                d_id,
                g_id[:8],
                d_best.fitness,
                float(delta),
                program.id,
            )

        logger.info(
            "[CompositionInjection] iteration summary: injected={} skip_no_d={} "
            "skip_dedup={} skip_d_missing={} compose_errors={} archive_size={}",
            len(injected_ids),
            n_no_d,
            n_dedup,
            n_d_missing,
            n_compose_error,
            len(g_programs),
        )
        return injected_ids
