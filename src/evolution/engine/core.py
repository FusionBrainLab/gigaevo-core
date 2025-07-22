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
from typing import List, Optional

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
from src.evolution.engine.config import EngineConfig
from src.evolution.engine.metrics import EngineMetrics
from src.evolution.engine.validation import validate_program
from src.evolution.engine.mutation import generate_mutations
from src.evolution.engine.prometheus import EnginePrometheusExporter


__all__ = [
    "EvolutionEngine",
]


class EvolutionEngine:

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

                if self.config.max_generations is not None and self.metrics.total_generations >= self.config.max_generations:
                    logger.info(f"[EvolutionEngine] Reached max_generations limit ({self.config.max_generations}) - stopping evolution")
                    break

                try:

                    await asyncio.wait_for(
                        self.evolve_step(),
                        timeout=self.config.generation_timeout,
                    )

                    self.metrics.last_generation_time = datetime.now(
                        timezone.utc
                    )
                    self._consecutive_errors = 0

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
            if await self._should_skip_evolution_due_to_incomplete_dags():
                logger.info("[EvolutionEngine] Skipping evolution step - waiting for DAG processing to complete")
                return

            novel_programs = await self.storage.get_all_by_status(ProgramState.DAG_PROCESSING_COMPLETED.value)

            if novel_programs:
                await self._process_novel_programs(novel_programs)

            elites = await self._select_elites()
            random.shuffle(elites)
            if elites:
                await self._generate_mutations(elites)

            self.metrics.total_generations += 1

            EnginePrometheusExporter.inc_generation()

        except Exception as exc:  # pylint: disable=broad-except
            raise EvolutionError(f"Evolution step failed: {exc}") from exc



    async def _process_novel_programs(
        self, novel_programs: List[Program]
    ) -> None:
        """Process novel programs with validation and error recovery."""
        logger.info(
            f"[EvolutionEngine] Processing {len(novel_programs)} novel programs"
        )

        if not novel_programs:
            return

        added = 0
        for prog in novel_programs:
            try:
                if not await self._validate_program(prog):
                    await self.program_state_manager.set_program_state(prog, ProgramState.DISCARDED)
                    continue

                success = await self.strategy.add(prog)
                if success:
                    added += 1
                    await self.program_state_manager.set_program_state(prog, ProgramState.EVOLVING)
                else:
                    await self.program_state_manager.set_program_state(prog, ProgramState.DISCARDED)

            except Exception as exc:
                logger.error(f"[EvolutionEngine] Failed to process program {prog.id}: {exc}")
                try:
                    await self.program_state_manager.set_program_state(prog, ProgramState.DISCARDED)
                except Exception:
                    pass

        self.metrics.programs_processed += added
        logger.info(f"[EvolutionEngine] Successfully added {added}/{len(novel_programs)} programs.")


    async def _validate_program(self, program: Program) -> bool:
        validation_result = await asyncio.to_thread(
            validate_program,
            program,
            required_behavior_keys=self.config.required_behavior_keys,
        )

        if not validation_result.is_valid:
            logger.warning(f"[EvolutionEngine] {validation_result.contract_summary}")
            logger.debug(f"[EvolutionEngine] {validation_result.detailed_message}")
        elif validation_result.is_valid:
            logger.debug(f"[EvolutionEngine] {validation_result.contract_summary}")

        return validation_result

    async def _select_elites(self) -> List[Program]:
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

    async def _generate_mutations(self, elites: List[Program]):
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
                parent_selector=self.config.parent_selector,
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
            all_programs = await self.storage.get_all()
            fresh_count = sum(1 for p in all_programs if p.state == ProgramState.FRESH)
            processing_count = sum(1 for p in all_programs if p.state == ProgramState.DAG_PROCESSING_STARTED)
            
            pending_count = fresh_count + processing_count
            
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
            return True

    def _handle_error(self, error_msg: str):
        self._consecutive_errors += 1
        self.metrics.errors_encountered += 1
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
