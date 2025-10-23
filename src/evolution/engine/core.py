from __future__ import annotations

import asyncio
import gc
from datetime import datetime, timezone

from loguru import logger

from src.database.program_storage import ProgramStorage
from src.evolution.engine.config import EngineConfig
from src.evolution.engine.metrics import EngineMetrics
from src.evolution.engine.mutation import generate_mutations
from src.evolution.mutation.base import MutationOperator
from src.evolution.strategies.base import EvolutionStrategy
from src.exceptions import EvolutionError
from src.programs.program import Program
from src.programs.program_state import ProgramState
from src.programs.state_manager import ProgramStateManager

__all__ = ["EvolutionEngine"]


class EvolutionEngine:
    """
    Lean evolution loop:
    - All *program state updates* go through ProgramStateManager.
    - Storage is used for reads (and mutations persist), never for state writes.
    """

    def __init__(
        self,
        storage: ProgramStorage,
        strategy: EvolutionStrategy,
        mutation_operator: MutationOperator,
        config: EngineConfig,
    ):
        self.storage = storage
        self.strategy = strategy
        self.mutation_operator = mutation_operator
        self.config = config

        self._running = False
        self._paused = False
        self._consecutive_errors = 0
        self._last_pending_dags_state = None
        self._last_pending_dags_counts = None  # Track pending DAG counts to avoid spam logging
        self._last_logged_metrics_hash = None  # Track last logged metrics hash to avoid spam logging

        self.metrics = EngineMetrics()
        self.state = ProgramStateManager(self.storage)

        logger.info(
            "[EvolutionEngine] Init | strategy={}, acceptor={}",
            type(self.strategy).__name__,
            type(self.config.program_acceptor).__name__,
        )

    async def run(self) -> None:
        logger.info("[EvolutionEngine] Start")
        self._running, self._consecutive_errors = True, 0

        try:
            while self._running:
                if self._paused:
                    await asyncio.sleep(self.config.loop_interval)
                    continue

                if self._reached_generation_cap():
                    logger.info("[EvolutionEngine] Stop: max_generations={}", self.config.max_generations)
                    break

                try:
                    await asyncio.wait_for(self.evolve_step(), timeout=self.config.generation_timeout)
                    self._consecutive_errors = 0
                    self.metrics.last_generation_time = datetime.now(timezone.utc)

                    if self._every(self.metrics.total_generations, self.config.log_interval):
                        await self._log_metrics()
                    if self._every(self.metrics.total_generations, self.config.cleanup_interval):
                        gc.collect()

                except asyncio.TimeoutError:
                    self._on_error("Generation timeout")

                except Exception as exc:  # pylint: disable=broad-except
                    self._on_error(str(exc))
                    if self._consecutive_errors >= self.config.max_consecutive_errors:
                        logger.critical("[EvolutionEngine] Stop: {} consecutive errors", self._consecutive_errors)
                        break

                await asyncio.sleep(self.config.loop_interval)

        except KeyboardInterrupt:
            logger.info("[EvolutionEngine] Interrupted")
        finally:
            self._running = False
            logger.info("[EvolutionEngine] Stopped")

    async def evolve_step(self) -> None:
        try:
            await self._step()
        except Exception as exc:
            raise EvolutionError(f"Evolution step failed: {exc}") from exc

    async def _step(self) -> None:
        # Stage 1: Wait for all programs to be processed
        has_pending = await self._has_pending_dags()
        if has_pending:
            if not self._last_pending_dags_state:
                logger.debug("[EvolutionEngine] Skip: pending DAGs")
            self._last_pending_dags_state = True
            return
        else:
            self._last_pending_dags_state = False
        
        logger.debug("[EvolutionEngine] Stage 1: No pending DAGs")

        # Stage 2: Ingest completed programs
        completed = await self.storage.get_all_by_status(ProgramState.DAG_PROCESSING_COMPLETED.value)
        newly_ingested = False
        if completed:
            logger.debug(f"[EvolutionEngine] Stage 2: Ingesting {len(completed)} completed programs")
            newly_ingested = await self._ingest_completed(completed)
            # No DAG_PROCESSING_COMPLETED programs after this step
        else:
            logger.debug("[EvolutionEngine] Stage 2: No completed programs to ingest")

        # Stage 3: Refresh all evolving programs (if any new programs were ingested)
        if newly_ingested:
            logger.debug("[EvolutionEngine] Stage 3: Refreshing all evolving programs")
            await self._reprocess()
            # After refresh, return to wait for DAGs to complete
            return

        # Stage 4: Mutate (only when no pending DAGs and no refresh needed)
        logger.debug("[EvolutionEngine] Stage 4: Selecting elites and mutating")
        elites = await self._select_elites()
        if elites:
            await self._mutate(elites)

        self.metrics.total_generations += 1

    async def _ingest_completed(self, programs: list[Program]) -> bool:
        """Validate and hand over completed programs to the strategy (state via StateManager only).
        
        For programs that were refreshing (previously EVOLVING, check if they're in the strategy),
        restore them to EVOLVING. For new programs, try to add them to evolution.
        
        Returns:
            True if any programs were added to EVOLVING state, False otherwise.
        """
        logger.info("[EvolutionEngine] Ingest {} program(s)", len(programs))

        added = 0
        restored = 0
        rej_valid = 0
        rej_strategy = 0

        # We fan out state changes as tasks; the lock lives inside ProgramStateManager.
        state_tasks: list[asyncio.Task] = []

        # Get all program IDs currently in evolution to check for refreshing programs
        evolving_program_ids = {p.id for p in await self.strategy.get_program_ids()}
        
        for prog in programs:
            try:
                accepted = await asyncio.to_thread(self.config.program_acceptor.is_accepted, prog)
                if not accepted:
                    rej_valid += 1
                    state_tasks.append(asyncio.create_task(self._set_state(prog, ProgramState.DISCARDED)))
                    continue

                is_already_evolving = prog.id in evolving_program_ids
                
                if is_already_evolving:
                    restored += 1
                    state_tasks.append(asyncio.create_task(self._set_state(prog, ProgramState.EVOLVING)))
                elif await self.strategy.add(prog):
                    added += 1
                    state_tasks.append(asyncio.create_task(self._set_state(prog, ProgramState.EVOLVING)))
                else:
                    rej_strategy += 1
                    state_tasks.append(asyncio.create_task(self._set_state(prog, ProgramState.DISCARDED)))

            except Exception as exc:
                # Best-effort discard on per-item failure; still go through StateManager.
                logger.debug("[EvolutionEngine] Ingest fail for {}: {}", prog.id, exc)
                state_tasks.append(asyncio.create_task(self._set_state(prog, ProgramState.DISCARDED)))

        if state_tasks:
            await asyncio.gather(*state_tasks, return_exceptions=True)

        self.metrics.programs_processed += added
        self.metrics.novel_programs_per_generation.append(added)
        logger.info(
            "[EvolutionEngine] Ingest done | added={}, restored={}, rejected_validation={}, rejected_strategy={}",
            added, restored, rej_valid, rej_strategy
        )
        
        return added > 0

    async def _select_elites(self) -> list[Program]:
        try:
            elites = await self.strategy.select_elites(total=self.config.max_elites_per_generation)
            logger.debug("[EvolutionEngine] Elites selected: {}", len(elites))
            return elites
        except Exception as exc:
            logger.error("[EvolutionEngine] Elite selection error: {}", exc)
            return []

    async def _reprocess(self) -> None:
        """Reprocess ALL evolving programs to refresh non-cacheable stages.
        
        After new programs finish processing, get all programs in EVOLVING state
        and submit them for reprocessing to refresh dynamic context like lineage.
        Marks them as FRESH so they go through DAG again.
        """

        programs_to_refresh = await self.strategy.get_program_ids()
        
        logger.debug(f"[EvolutionEngine] Found {len(programs_to_refresh)} programs in EVOLVING state to refresh")
        
        reprocessed_count = 0
        state_tasks: list[asyncio.Task] = []
        
        for program in programs_to_refresh:
            state_tasks.append(asyncio.create_task(self._set_state(program, ProgramState.FRESH)))
            reprocessed_count += 1
        
        if state_tasks:
            await asyncio.gather(*state_tasks, return_exceptions=True)
            
        logger.info(
            "[EvolutionEngine] Submitted {} programs for refresh",
            reprocessed_count,
        )

    async def _mutate(self, elites: list[Program]) -> None:
        logger.debug("[EvolutionEngine] Mutate from {} elite(s)", len(elites))
        try:
            created = await generate_mutations(
                elites,
                mutator=self.mutation_operator,
                storage=self.storage,
                parent_selector=self.config.parent_selector,
                limit=self.config.max_mutations_per_generation,
                iteration=self.metrics.total_generations,
            )
            self.metrics.mutations_created += created
        except Exception as exc:
            logger.error("[EvolutionEngine] Mutation error: {}", exc)

    async def _has_pending_dags(self) -> bool:
        """Backpressure: defer if fresh/processing programs exist (read-only)."""
        fresh_programs = await self.storage.get_all_by_status(ProgramState.FRESH.value)
        proc_programs = await self.storage.get_all_by_status(ProgramState.DAG_PROCESSING_STARTED.value)
        fresh = len(fresh_programs)
        proc = len(proc_programs)
        
        if fresh or proc:
            current_counts = (fresh, proc)
            # Only log when the counts change to avoid spam
            if self._last_pending_dags_counts != current_counts:
                logger.debug("[EvolutionEngine] Pending DAGs: fresh={}, processing={}", fresh, proc)
                self._last_pending_dags_counts = current_counts
            return True
        else:
            # Reset counts tracking when no pending DAGs
            self._last_pending_dags_counts = None
            return False

    async def _set_state(self, program: Program, state: ProgramState) -> None:
        """Single write path for program state updates."""
        await self.state.set_program_state(program, state)

    def _reached_generation_cap(self) -> bool:
        cap = self.config.max_generations
        return cap is not None and self.metrics.total_generations >= cap

    @staticmethod
    def _every(i: int, n: int) -> bool:
        return n > 0 and i % n == 0

    def _on_error(self, msg: str) -> None:
        self._consecutive_errors += 1
        self.metrics.errors_encountered += 1
        logger.error("[EvolutionEngine] Error #{}: {}", self._consecutive_errors, msg)

    async def _log_metrics(self) -> None:
        current_metrics_hash = hash(self.metrics)
        
        # Only log when metrics actually change to avoid spam
        if self._last_logged_metrics_hash != current_metrics_hash:
            m = self.metrics.to_dict()
            metrics_str = " | ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in m.items())
            logger.info(f"[EvolutionEngine] | {metrics_str}")
            self._last_logged_metrics_hash = current_metrics_hash

    def stop(self) -> None:
        """Request the main loop to exit."""
        self._running = False

    def pause(self) -> None:
        """Pause new work; the run loop keeps idling."""
        self._paused = True

    def resume(self) -> None:
        """Resume from a paused state."""
        self._paused = False

    def is_running(self) -> bool:
        return self._running

    async def get_status(self) -> dict[str, object]:
        """Light, non-blocking status for UIs/health checks."""
        return {
            "running": self._running,
            "paused": self._paused,
            "consecutive_errors": self._consecutive_errors,
            **self.metrics.to_dict(),
        }


