from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING

from loguru import logger

from gigaevo.database.program_storage import ProgramStorage
from gigaevo.database.state_manager import ProgramStateManager
from gigaevo.evolution.engine.config import EngineConfig
from gigaevo.evolution.engine.metrics import EngineMetrics
from gigaevo.evolution.engine.mutation import generate_mutations
from gigaevo.evolution.mutation.base import MutationOperator
from gigaevo.evolution.strategies.base import EvolutionStrategy
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState
from gigaevo.utils.metrics_collector import start_metrics_collector
from gigaevo.utils.metrics_tracker import MetricsTracker
from gigaevo.utils.trackers.base import LogWriter

if TYPE_CHECKING:
    from typing import Any


class EvolutionEngine:
    """
      1) Wait until no DAGs are running (idle)
      2) Select elites & create mutants
      3) Wait for mutants' DAGs to finish (idle again)
      4) Ingest completed mutants
      5) Refresh all evolving programs (EVOLVING -> FRESH)
      6) Wait for refresh DAGs to finish (idle)
    All state writes go through ProgramStateManager; storage is read-oriented here.
    """

    def __init__(
        self,
        storage: ProgramStorage,
        strategy: EvolutionStrategy,
        mutation_operator: MutationOperator,
        config: EngineConfig,
        writer: LogWriter,
        metrics_tracker: MetricsTracker,
    ):
        self.storage = storage
        self.strategy = strategy
        self.mutation_operator = mutation_operator
        self.config = config
        self._writer = writer.bind(path=["evolution_engine"])

        self._running = False
        self._paused = False
        self._last_pending_dags_counts: tuple[int, int] | None = None

        self._task: asyncio.Task | None = None
        self._metrics_collector_task: asyncio.Task | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

        self.metrics = EngineMetrics()
        self.state = ProgramStateManager(self.storage)
        self._metrics_tracker = metrics_tracker

        logger.info(
            "[EvolutionEngine] Init | strategy={}, acceptor={}",
            type(self.strategy).__name__,
            type(self.config.program_acceptor).__name__,
        )

    def start(self) -> None:
        """Start the evolution engine in a background task."""
        if self._task and not self._task.done():
            return
        self._loop = asyncio.get_running_loop()
        self._running = True
        self._task = asyncio.create_task(self.run(), name="evolution-engine")
        self._metrics_tracker.start(self._loop)

        async def _collect_metrics() -> dict[str, Any]:
            out = self.metrics.model_dump(mode="json")
            strategy_metrics = await self.strategy.get_metrics()
            if strategy_metrics:
                out.update(strategy_metrics.to_dict())
            return out

        self._metrics_collector_task = start_metrics_collector(
            writer=self._writer,
            collect_fn=_collect_metrics,
            interval=self.config.metrics_collection_interval,
            stop_flag=lambda: not self._running,
            task_name="evolution-metrics-collector",
        )
        logger.info("[EvolutionEngine] Task started")

    async def stop(self) -> None:
        """Stop the evolution engine and await task completion."""
        self._running = False
        task = self._task
        self._task = None
        if task:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        if self._metrics_collector_task:
            self._metrics_collector_task.cancel()
            self._metrics_collector_task = None

        if self._metrics_tracker:
            await self._metrics_tracker.stop()

        await self.storage.close()

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def is_running(self) -> bool:
        return self._running

    @property
    def task(self) -> asyncio.Task | None:
        return self._task

    async def run(self) -> None:
        logger.info("[EvolutionEngine] Start")
        self._running = True
        try:
            while self._running:
                if self._paused:
                    await asyncio.sleep(self.config.loop_interval)
                    continue

                if self._reached_generation_cap():
                    logger.info(
                        "[EvolutionEngine] Stop: max_generations={}",
                        self.config.max_generations,
                    )
                    break

                try:
                    timeout = self.config.generation_timeout
                    if timeout:
                        await asyncio.wait_for(self.step(), timeout=timeout)
                    else:
                        await self.step()
                except asyncio.TimeoutError:
                    # One step took too long; log and continue the loop.
                    logger.warning(
                        "[EvolutionEngine] step() timed out after {}s", timeout
                    )
                except asyncio.CancelledError:
                    # Propagate so shutdown stays clean.
                    raise
                except Exception as e:
                    # Don’t crash the engine on a single bad step; just log and continue.
                    logger.exception("[EvolutionEngine] step() failed: {}", e)

                await asyncio.sleep(self.config.loop_interval)
        except asyncio.CancelledError:
            # Task is being cancelled during shutdown.
            logger.debug("[EvolutionEngine] run() cancelled")
            raise
        finally:
            self._running = False
            logger.info("[EvolutionEngine] Stopped")

    async def step(self) -> None:
        """One generation step (idle → mutate → idle → ingest → refresh → idle)."""
        # Phase 1: wait until engine is idle (no FRESH/PROCESSING programs)
        await self._await_idle()
        logger.debug("[EvolutionEngine] Phase 1: Idle confirmed")

        # Phase 2: select elites & create mutants
        elites = await self._select_elites_for_mutation()
        created = await self._create_mutants(elites) if elites else 0
        logger.debug("[EvolutionEngine] Phase 2: Created {} mutant(s)", created)

        # Phase 3: wait for the mutants' DAGs to finish
        await self._await_idle()
        logger.debug("[EvolutionEngine] Phase 3: Mutant DAGs finished (idle)")

        # Phase 4: ingest newly completed programs (typically the mutants)
        await self._ingest_completed_programs()
        logger.debug("[EvolutionEngine] Phase 4: Ingestion done")

        # Phase 5: refresh all evolving programs (to re-run lineage/descendant-aware stages)
        refreshed = await self._refresh_evolving_programs()
        logger.debug("[EvolutionEngine] Phase 5: Refreshed {} program(s)", refreshed)

        # Phase 6: wait for refresh DAGs to finish
        if refreshed:
            await self._await_idle()
            logger.debug("[EvolutionEngine] Phase 6: Refresh DAGs finished (idle)")

        self.metrics.total_generations += 1

    async def _await_idle(self) -> None:
        """Block until there are no programs in FRESH or DAG_PROCESSING_STARTED."""
        while await self._has_active_dags():
            await asyncio.sleep(self.config.loop_interval)

    async def _select_elites_for_mutation(self) -> list[Program]:
        elites = await self.strategy.select_elites(
            total=self.config.max_elites_per_generation
        )
        logger.debug("[EvolutionEngine] Elites selected: {}", len(elites))
        self.metrics.record_elite_selection_metrics(len(elites), 0)
        return elites

    async def _create_mutants(self, elites: list[Program]) -> int:
        logger.debug("[EvolutionEngine] Mutate from {} elite(s)", len(elites))
        created = await generate_mutations(
            elites,
            mutator=self.mutation_operator,
            storage=self.storage,
            state_manager=self.state,
            parent_selector=self.config.parent_selector,
            limit=self.config.max_mutations_per_generation,
            iteration=self.metrics.total_generations,
        )
        self.metrics.record_mutation_metrics(created, 0)
        return created

    async def _ingest_completed_programs(self) -> None:
        """
        Validate and hand over any DAG_PROCESSING_COMPLETED programs to the strategy.
        Restores already-known programs to EVOLVING; adds new ones if accepted; otherwise discards.
        """
        completed = await self.storage.get_all_by_status(
            ProgramState.DAG_PROCESSING_COMPLETED.value
        )
        if not completed:
            logger.debug("[EvolutionEngine] No completed programs to ingest")
            return

        logger.info("[EvolutionEngine] Ingest {} program(s)", len(completed))
        logger.debug(
            "[EvolutionEngine] Program IDs: {}",
            [p.id for p in completed[:8]] + (["..."] if len(completed) > 10 else []),
        )

        added = 0
        restored = 0
        rej_valid = 0
        rej_strategy = 0

        state_tasks: list[asyncio.Task] = []
        evolving_program_ids = set(await self.strategy.get_program_ids())

        for prog in completed:
            if prog.id in evolving_program_ids:
                # for evolving programs, we just restore them to the evolving state
                restored += 1
                state_tasks.append(
                    asyncio.create_task(self._set_state(prog, ProgramState.EVOLVING))
                )
            elif not self.config.program_acceptor.is_accepted(prog):
                # rejected by basic checks
                rej_valid += 1
                logger.debug(
                    "[EvolutionEngine] Program {} rejected by acceptor",
                    prog.id,
                )
                state_tasks.append(
                    asyncio.create_task(self._set_state(prog, ProgramState.DISCARDED))
                )
            elif await self.strategy.add(prog):
                # accepted by strategy (i.e, routed to an island)
                added += 1
                logger.debug(
                    "[EvolutionEngine] Program {} added to strategy",
                    prog.id,
                )
                state_tasks.append(
                    asyncio.create_task(self._set_state(prog, ProgramState.EVOLVING))
                )
            else:
                # rejected by strategy / validation
                rej_strategy += 1
                logger.debug(
                    "[EvolutionEngine] Program {} rejected by strategy",
                    prog.id,
                )
                state_tasks.append(
                    asyncio.create_task(self._set_state(prog, ProgramState.DISCARDED))
                )

        if state_tasks:
            await asyncio.gather(*state_tasks, return_exceptions=True)

        self.metrics.programs_processed += added
        self.metrics.record_ingestion_metrics(added, restored, rej_valid, rej_strategy)
        logger.info(
            "[EvolutionEngine] Ingest done | added={}, restored={}, rejected_validation={}, rejected_strategy={}",
            added,
            restored,
            rej_valid,
            rej_strategy,
        )

    async def _refresh_evolving_programs(self) -> int:
        """Flip all EVOLVING programs to FRESH so lineage/descendant-aware stages re-run."""
        program_ids_to_refresh = await self.strategy.get_program_ids()

        if not program_ids_to_refresh:
            return 0

        programs_to_refresh = await self.storage.mget(program_ids_to_refresh)

        tasks: list[asyncio.Task] = []
        for program in programs_to_refresh:
            tasks.append(
                asyncio.create_task(self._set_state(program, ProgramState.FRESH))
            )

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        count = len(tasks)
        if count:
            logger.info("[EvolutionEngine] Submitted {} program(s) for refresh", count)
            self.metrics.record_reprocess_metrics(count)
        return count

    async def _has_active_dags(self) -> bool:
        """True if any programs are FRESH or DAG_PROCESSING_STARTED (i.e., engine not idle)."""
        fresh = await self.storage.count_by_status(ProgramState.FRESH.value)
        proc = await self.storage.count_by_status(
            ProgramState.DAG_PROCESSING_STARTED.value
        )

        if fresh or proc:
            current_counts = (fresh, proc)
            if self._last_pending_dags_counts != current_counts:
                logger.debug(
                    "[EvolutionEngine] Pending DAGs: fresh={}, processing={}",
                    fresh,
                    proc,
                )
                self._last_pending_dags_counts = current_counts
            return True

        self._last_pending_dags_counts = None
        return False

    async def _set_state(self, program: Program, state: ProgramState) -> None:
        await self.state.set_program_state(program, state)

    def _reached_generation_cap(self) -> bool:
        cap = self.config.max_generations
        return cap is not None and self.metrics.total_generations >= cap
