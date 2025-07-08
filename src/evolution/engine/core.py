from __future__ import annotations

"""Core implementation of EvolutionEngine split out from the previous monolithic
`src/evolution/engine.py`.

The public API is re-exported by the package `src.evolution.engine`, so external
imports remain unchanged:

    from src.evolution.engine import EvolutionEngine, EngineConfig

Only internal layout changed; logic is identical (except for import paths).
"""

import asyncio
import gc
from typing import Any, Dict, List, Optional, Set

from loguru import logger
import random
from datetime import datetime, timezone

from src.database.program_storage import ProgramStorage
from src.evolution.mutation.base import MutationOperator
from src.evolution.strategies.base import EvolutionStrategy
from src.exceptions import EvolutionError, ensure_not_none
from src.programs.program import Program
from src.programs.state_manager import ProgramStateManager
from src.programs.program_state import ProgramState

# ---------------------------------------------------------------------------
# Local imports from the new modular sub-package
# ---------------------------------------------------------------------------
from .config import EngineConfig
from .metrics import EngineMetrics
from .validation import ProgramValidationResult, validate_program
from .mutation import generate_mutations

__all__ = [
    "EvolutionEngine",
]


class EvolutionEngine:
    """Focused evolution engine with critical reliability improvements."""

    def __init__(
        self,
        storage: ProgramStorage,
        strategy: EvolutionStrategy,
        mutation_operator: MutationOperator,
        config: Optional[EngineConfig] = None,
    ):
        # ------------------------------------------------------------------
        # Dependencies & basic state
        # ------------------------------------------------------------------
        self.storage: ProgramStorage = ensure_not_none(storage, "storage")
        self.strategy: EvolutionStrategy = ensure_not_none(strategy, "strategy")
        self.mutation_operator: MutationOperator = ensure_not_none(
            mutation_operator, "mutation_operator"
        )
        self.config: EngineConfig = config or EngineConfig()

        self._running = False
        self._paused = False
        self._consecutive_errors = 0
        self.metrics = EngineMetrics()
        self.program_state_manager = ProgramStateManager(storage)

        logger.info(
            f"[EvolutionEngine] Initialized with strategy: {type(strategy).__name__}"
        )

    # ------------------------------------------------------------------
    # Main run-loop helpers
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Main evolution loop with focused error handling."""
        logger.info("[EvolutionEngine] Starting evolution loop")
        self._running = True
        self._consecutive_errors = 0

        try:
            while self._running:
                if self._paused:
                    await asyncio.sleep(self.config.loop_interval)
                    continue

                try:
                    generation_start = datetime.now(timezone.utc)

                    # Run generation with timeout
                    await asyncio.wait_for(
                        self.evolve_step(),
                        timeout=self.config.generation_timeout,
                    )

                    # Update metrics & housekeeping
                    self.metrics.last_generation_time = datetime.now(
                        timezone.utc
                    )
                    self._consecutive_errors = 0

                    # Periodic logging and cleanup
                    if (
                        self.metrics.total_generations % self.config.log_interval
                        == 0
                    ):
                        await self._log_metrics()

                    if (
                        self.metrics.total_generations
                        % self.config.cleanup_interval
                        == 0
                    ):
                        gc.collect()

                    # Prometheus metrics
                    from .prometheus import EnginePrometheusExporter

                    EnginePrometheusExporter.inc_generation()

                except asyncio.TimeoutError:
                    self._handle_error("Generation timeout")

                except Exception as exc:  # pylint: disable=broad-except
                    self._handle_error(str(exc))

                    if (
                        self._consecutive_errors
                        >= self.config.max_consecutive_errors
                    ):
                        logger.critical(
                            f"[EvolutionEngine] Stopping due to {self._consecutive_errors} consecutive errors"
                        )
                        break

                await asyncio.sleep(self.config.loop_interval)

        except KeyboardInterrupt:
            logger.info("[EvolutionEngine] Received interrupt signal – stopping")
        finally:
            self._running = False
            logger.info("[EvolutionEngine] Evolution loop stopped")

    async def evolve_step(self) -> None:
        """Single evolution step with improved error handling."""
        logger.debug("[EvolutionEngine] Performing evolution step")

        try:
            # Check if we should skip entire evolution step due to incomplete DAG processing
            if await self._should_skip_evolution_due_to_incomplete_dags():
                logger.info("[EvolutionEngine] Skipping evolution step - waiting for DAG processing to complete")
                return

            # Load and validate programs
            novel_programs = await self.storage.get_all_by_status(ProgramState.DAG_PROCESSING_COMPLETED.value)

            # Process novel programs with validation
            if novel_programs:
                await self._process_novel_programs_safely(novel_programs)

            # Select elites and create mutations (only when we have complete fitness data)
            elites = await self._select_elites_safely()
            random.shuffle(elites)
            if elites:
                await self._generate_mutations_safely(elites)

            # Update generation counter
            self.metrics.total_generations += 1

            # Prometheus metrics
            from .prometheus import EnginePrometheusExporter

            EnginePrometheusExporter.inc_generation()

        except Exception as exc:  # pylint: disable=broad-except
            raise EvolutionError(f"Evolution step failed: {exc}") from exc



    async def _process_novel_programs_safely(
        self, novel_programs: List[Program]
    ) -> None:
        """Process novel programs with validation and error recovery."""
        logger.info(
            f"[EvolutionEngine] Processing {len(novel_programs)} novel programs"
        )

        # No additional hash-based duplicate filtering – process programs as-is.
        if not novel_programs:
            return

        added = 0
        for prog in novel_programs:
            try:
                # Validate program; discard on failure
                if not await self._validate_program_critical(prog):
                    await self.program_state_manager.set_program_state(prog, ProgramState.DISCARDED)
                    continue

                # Attempt to add to strategy; discard if strategy refuses
                success = await self.strategy.add(prog)
                if success:
                    added += 1
                    await self.program_state_manager.set_program_state(prog, ProgramState.EVOLVING)
                else:
                    await self.program_state_manager.set_program_state(prog, ProgramState.DISCARDED)

            except Exception as exc:
                logger.error(f"[EvolutionEngine] Failed to process program {prog.id}: {exc}")
                # Best effort: mark as discarded to avoid endless retries
                try:
                    await self.program_state_manager.set_program_state(prog, ProgramState.DISCARDED)
                except Exception:
                    pass

        self.metrics.programs_processed += added
        logger.info(f"[EvolutionEngine] Successfully added {added}/{len(novel_programs)} programs.")

    async def _validate_program_critical(self, program: Program) -> bool:
        """Critical program validation – only check essential requirements."""
        validation_result = await self._validate_program_detailed(program)

        if self.config.log_validation_failures and not validation_result.is_valid:
            logger.warning(f"[EvolutionEngine] {validation_result.contract_summary}")
            logger.debug(f"[EvolutionEngine] {validation_result.detailed_message}")
        elif validation_result.is_valid:
            logger.debug(f"[EvolutionEngine] {validation_result.contract_summary}")

        return validation_result.is_valid

    async def _validate_program_detailed(self, program: Program) -> ProgramValidationResult:
        """Detailed program validation with comprehensive feedback."""
        # Delegate to standalone helper; keep async for compatibility
        validation_result = await asyncio.to_thread(
            validate_program,
            program,
            required_behavior_keys=self.config.required_behavior_keys,
        )

        if self.config.log_validation_failures and not validation_result.is_valid:
            logger.warning(f"[EvolutionEngine] {validation_result.contract_summary}")
            logger.debug(f"[EvolutionEngine] {validation_result.detailed_message}")
        elif validation_result.is_valid:
            logger.debug(f"[EvolutionEngine] {validation_result.contract_summary}")

        return validation_result

    async def _select_elites_safely(self) -> List[Program]:
        """Select elite programs with error handling."""
        try:
            elites = await self.strategy.select_elites(
                total=self.config.max_elites_per_generation
            )
            logger.debug(
                f"[EvolutionEngine] Selected {len(elites)} elites"
            )
            return elites
        except Exception as exc:  # pylint: disable=broad-except
            logger.error(f"[EvolutionEngine] Elite selection failed: {exc}")
            return []

    async def _generate_mutations_safely(self, elites: List[Program]):
        """Generate mutations from elites with error recovery.

        This implementation mirrors the previous monolith and therefore:
        1. Off-loads heavy mutation computation to a background thread
           via ``asyncio.to_thread`` (unit-tests patch this helper).
        2. Persists each mutated program via ``self.storage.add`` so the
           broader system can pick them up.
        """
        try:
            logger.debug(
                f"[EvolutionEngine] Generating mutations from {len(elites)} elites"
            )

            mutation_count = await generate_mutations(
                elites,
                mutator=self.mutation_operator,
                storage=self.storage,
                limit=self.config.max_mutations_per_generation,
            )

            self.metrics.mutations_created += mutation_count

        except Exception as exc:  # pylint: disable=broad-except
            logger.error(f"[EvolutionEngine] Mutation generation failed: {exc}")

    async def _should_skip_evolution_due_to_incomplete_dags(self) -> bool:
        """Check if evolution step should be skipped due to incomplete DAG processing.
        
        This implements backpressure control to ensure evolution decisions are made
        only when complete fitness information is available.
        """
        try:
            # Efficiently check for incomplete DAGs by examining all program states
            # This avoids multiple Redis queries and provides better performance
            all_programs = await self.storage.get_all()
            
            # Count programs in incomplete states
            fresh_count = sum(1 for p in all_programs if p.state == ProgramState.FRESH)
            processing_count = sum(1 for p in all_programs if p.state == ProgramState.DAG_PROCESSING_STARTED)
            
            pending_count = fresh_count + processing_count
            
            # Skip evolution if there are ANY incomplete DAGs
            # This ensures we have complete fitness information before making evolution decisions
            if pending_count > 0:
                logger.debug(
                    f"[EvolutionEngine] Found {pending_count} incomplete DAGs "
                    f"({fresh_count} fresh, {processing_count} processing). "
                    f"Waiting for complete fitness data before evolution."
                )
                return True
            else:
                logger.debug("[EvolutionEngine] All DAGs completed. Proceeding with evolution step.")
                return False
                
        except Exception as exc:
            logger.error(f"[EvolutionEngine] Failed to check DAG completion status: {exc}")
            # On error, be conservative and skip evolution to avoid bad decisions
            return True

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------

    def _handle_error(self, error_msg: str):
        self._consecutive_errors += 1
        self.metrics.errors_encountered += 1

        # Prometheus metrics
        from .prometheus import EnginePrometheusExporter

        EnginePrometheusExporter.inc_error()

        logger.error(
            f"[EvolutionEngine] Error #{self._consecutive_errors}: {error_msg}"
        )

    async def _log_metrics(self):
        try:
            metrics = self.metrics.to_dict()
            logger.info(
                f"[EvolutionEngine] Iteration | {metrics['total_generations']}: "
                f"Processed {metrics['programs_processed']} programs, "
                f"Mutations {metrics['mutations_created']}, "
                f"Errors {metrics['errors_encountered']}"
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.error(f"[EvolutionEngine] Error logging metrics: {exc}")

    # ------------------------------------------------------------------
    # External control helpers
    # ------------------------------------------------------------------

    def pause(self) -> None:
        if not self._running:
            raise EvolutionError("Cannot pause when engine is not running")
        self._paused = True
        logger.info("[EvolutionEngine] Engine paused")

    def resume(self) -> None:
        if not self._running:
            raise EvolutionError("Cannot resume when engine is not running")
        self._paused = False
        logger.info("[EvolutionEngine] Engine resumed")

    def stop(self) -> None:
        self._running = False
        logger.info("[EvolutionEngine] Stop requested")

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    def is_running(self) -> bool:
        return self._running

    def is_paused(self) -> bool:
        return self._paused

    def get_metrics(self) -> Dict[str, Any]:
        return self.metrics.to_dict()

    async def get_status(self) -> Dict[str, Any]:
        return {
            "running": self._running,
            "paused": self._paused,
            "consecutive_errors": self._consecutive_errors,
            "metrics": self.get_metrics(),
            "config": self.config.model_dump(),
        } 