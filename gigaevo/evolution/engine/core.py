from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
import contextlib
import time
import json
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from gigaevo.database.program_storage import ProgramStorage
from gigaevo.database.state_manager import ProgramStateManager
from gigaevo.evolution.engine.config import EngineConfig
from gigaevo.evolution.engine.metrics import EngineMetrics
from gigaevo.evolution.engine.mutation import generate_mutations
from gigaevo.evolution.mutation.base import MutationOperator
from gigaevo.evolution.mutation.mutation_operator import (
    LLMMutationOperator,
)
from gigaevo.evolution.strategies.base import EvolutionStrategy
from gigaevo.llm.bandit import BanditModelRouter, MutationOutcome
from gigaevo.programs.metrics.context import VALIDITY_KEY
from gigaevo.programs.program import EXCLUDE_STAGE_RESULTS, Program
from gigaevo.programs.program_state import ProgramState
from gigaevo.utils.metrics_collector import start_metrics_collector
from gigaevo.utils.metrics_tracker import MetricsTracker
from gigaevo.utils.trackers.base import LogWriter

if TYPE_CHECKING:
    from typing import Any

# Redis run-state field names (used for resume persistence)
_RUN_STATE_TOTAL_GENERATIONS = "engine:total_generations"


class EvolutionEngine:
    """
      1) Wait until no DAGs are running (idle)
      2) Select elites & create mutants
      3) Wait for mutants' DAGs to finish (idle again)
      4) Ingest completed mutants
      5) Refresh all archive programs (DONE -> QUEUED)
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
        pre_step_hook: Callable[[], Awaitable[None]] | None = None,
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

        # ETA tracking — set at the start of run()
        self._run_start_time: float | None = None
        self._run_start_gen: int = 0

        # Archive stagnation tracking
        self._prev_archive_size: int = 0
        self._stagnant_gens: int = 0

        self.metrics = EngineMetrics()
        self.state = ProgramStateManager(self.storage)
        self._metrics_tracker = metrics_tracker
        self._pre_step_hook = pre_step_hook

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
            if isinstance(self.mutation_operator, LLMMutationOperator) and isinstance(
                self.mutation_operator.llm_wrapper, BanditModelRouter
            ):
                out["bandit"] = self.mutation_operator.llm_wrapper.get_bandit_stats()
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
        logger.info(
            "[EvolutionEngine] Start | max_generations={} strategy={} acceptor={}"
            " | max_elites={} max_mutations={} loop_interval={}s",
            self.config.max_generations,
            type(self.strategy).__name__,
            type(self.config.program_acceptor).__name__,
            self.config.max_elites_per_generation,
            self.config.max_mutations_per_generation,
            self.config.loop_interval,
        )
        self._running = True
        self._run_start_time = time.monotonic()
        self._run_start_gen = self.metrics.total_generations
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
                    await self.step()
                except asyncio.CancelledError:
                    # Propagate so shutdown stays clean.
                    raise
                except Exception as e:
                    # Don’t crash the engine on a single bad step; just log and continue.
                    logger.exception("[EvolutionEngine] step() failed: {}", e)

                await asyncio.sleep(0)
        except asyncio.CancelledError:
            # Task is being cancelled during shutdown.
            logger.debug("[EvolutionEngine] run() cancelled")
            raise
        finally:
            self._running = False
            logger.info("[EvolutionEngine] Stopped")

    async def step(self) -> None:
        """One generation step (idle → mutate → idle → ingest → refresh → idle)."""
        gen = self.metrics.total_generations
        logger.info("[EvolutionEngine] ──────────── Generation {} ────────────", gen)
        step_t0 = time.monotonic()

        if self._pre_step_hook:
            await self._pre_step_hook()

        self.storage.snapshot.bump()

        # Phase 1: wait until engine is idle (no QUEUED/RUNNING programs)
        await self._await_idle()
        logger.debug("[EvolutionEngine] gen={} Phase 1: Idle confirmed", gen)

        # Phase 2: select elites & create mutants
        elites = await self._select_elites_for_mutation()
        mutation_ids = await self._create_mutants(elites) if elites else None
        logger.debug(
            "[EvolutionEngine] gen={} Phase 2: Created {} mutant(s)",
            gen,
            len(mutation_ids) if mutation_ids is not None else 0,
        )

        # Phase 3: wait for the mutants' DAGs to finish
        await self._await_idle()
        logger.debug(
            "[EvolutionEngine] gen={} Phase 3: Mutant DAGs finished (idle)", gen
        )

        # Phase 4: ingest newly completed programs (typically the mutants)
        await self._ingest_completed_programs(mutation_ids=mutation_ids)
        logger.debug("[EvolutionEngine] gen={} Phase 4: Ingestion done", gen)

        # Incremental bump: ingestion only changes program states (DONE→DISCARDED)
        # and set membership — not the data fields the collector reads (metrics,
        # lineage, generation).  Allows the snapshot to reuse cached Program
        # objects and only fetch newly added/removed IDs.
        self.storage.snapshot.bump(incremental=True)

        # Phase 5: refresh all archive programs (to re-run lineage/descendant-aware stages)
        refreshed = await self._refresh_archive_programs()
        logger.debug(
            "[EvolutionEngine] gen={} Phase 5: Refreshed {} program(s)", gen, refreshed
        )

        # Phase 6: wait for refresh DAGs to finish
        if refreshed:
            await self._await_idle()
            logger.debug(
                "[EvolutionEngine] gen={} Phase 6: Refresh DAGs finished (idle)", gen
            )

            # Phase 7: reindex archive with updated metrics (e.g., prompt fitness)
            await self.strategy.reindex_archive()
            logger.debug("[EvolutionEngine] gen={} Phase 7: Archive reindexed", gen)

        self.metrics.total_generations += 1
        await self.storage.save_run_state(
            _RUN_STATE_TOTAL_GENERATIONS, self.metrics.total_generations
        )

        # Log generation summary for easy diagnosis
        step_elapsed = time.monotonic() - step_t0
        archive_size = len(await self.strategy.get_program_ids())
        best_str = self._metrics_tracker.format_best_summary()
        eta_str = self._format_eta()

        archive_delta = archive_size - self._prev_archive_size
        self._prev_archive_size = archive_size
        if archive_delta == 0:
            self._stagnant_gens += 1
        else:
            self._stagnant_gens = 0

        logger.info(
            "[EvolutionEngine] gen={} done | elites={} mutants={} refreshed={}"
            " | archive={} ({:+d}){} ({:.1f}s){}",
            gen,
            len(elites),
            len(mutation_ids) if mutation_ids is not None else 0,
            refreshed,
            archive_size,
            archive_delta,
            best_str,
            step_elapsed,
            eta_str,
        )

        if self._stagnant_gens >= 5:
            logger.warning(
                "[EvolutionEngine] Archive stagnant for {} consecutive generations",
                self._stagnant_gens,
            )

    def _format_eta(self) -> str:
        """Return a compact ETA string based on elapsed time and generation progress."""
        current_gen = self.metrics.total_generations
        if (
            not self.config.max_generations
            or self._run_start_time is None
            or current_gen <= self._run_start_gen
        ):
            return ""
        elapsed_total = time.monotonic() - self._run_start_time
        gens_done = current_gen - self._run_start_gen
        avg_per_gen = elapsed_total / gens_done
        remaining = self.config.max_generations - current_gen
        eta_s = avg_per_gen * remaining
        progress_pct = current_gen / self.config.max_generations * 100
        return (
            f" | progress={progress_pct:.0f}%"
            f" ETA={eta_s / 60:.0f}min ({avg_per_gen:.1f}s/gen)"
        )

    async def _await_idle(self) -> None:
        """Block until there are no programs in QUEUED or RUNNING."""
        t0 = time.monotonic()
        ghost_checked = False
        while True:
            has_active = await self._has_active_dags()
            if not has_active:
                break

            elapsed = time.monotonic() - t0
            if elapsed > 30 and int(elapsed) % 60 < self.config.loop_interval:
                logger.info(
                    "[EvolutionEngine] gen={} Waiting for idle ({:.0f}s elapsed)",
                    self.metrics.total_generations,
                    elapsed,
                )
            # Ghost safety: after 30s, verify counts with full fetch (once)
            if elapsed > 30 and not ghost_checked:
                ghost_checked = True
                real_q = len(
                    await self.storage.get_all_by_status(
                        ProgramState.QUEUED.value,
                        exclude=EXCLUDE_STAGE_RESULTS,
                    )
                )
                real_r = len(
                    await self.storage.get_all_by_status(
                        ProgramState.RUNNING.value,
                        exclude=EXCLUDE_STAGE_RESULTS,
                    )
                )
                if real_q == 0 and real_r == 0:
                    # Clean up ghost IDs from status sets
                    queued_ids = await self.storage.get_ids_by_status(
                        ProgramState.QUEUED.value
                    )
                    running_ids = await self.storage.get_ids_by_status(
                        ProgramState.RUNNING.value
                    )
                    if queued_ids:
                        await self.storage.remove_ids_from_status_set(
                            ProgramState.QUEUED.value, queued_ids
                        )
                    if running_ids:
                        await self.storage.remove_ids_from_status_set(
                            ProgramState.RUNNING.value, running_ids
                        )
                    ghost_count = len(queued_ids) + len(running_ids)
                    logger.warning(
                        "[EvolutionEngine] Ghost IDs detected — SCARD says active "
                        "but no real programs found. Cleaned {} ghost ID(s) from "
                        "status sets. Breaking idle wait.",
                        ghost_count,
                    )
                    break
            await asyncio.sleep(self.config.loop_interval)

    async def _select_elites_for_mutation(self) -> list[Program]:
        elites = await self.strategy.select_elites(
            total=self.config.max_elites_per_generation
        )
        logger.debug(
            "[EvolutionEngine] gen={} Elites selected: {}",
            self.metrics.total_generations,
            len(elites),
        )
        self.metrics.record_elite_selection_metrics(len(elites), 0)
        return elites

    async def _create_mutants(self, elites: list[Program]) -> list[str]:
        """Create mutants and return their program IDs."""
        logger.debug(
            "[EvolutionEngine] gen={} Mutate from {} elite(s)",
            self.metrics.total_generations,
            len(elites),
        )
        mutation_ids = await generate_mutations(
            elites,
            mutator=self.mutation_operator,
            storage=self.storage,
            state_manager=self.state,
            parent_selector=self.config.parent_selector,
            limit=self.config.max_mutations_per_generation,
            iteration=self.metrics.total_generations,
            memory_used=False,
        )
        if self.config.memory_enabled:
            memory_instructions = self._read_memory_instructions()
            if memory_instructions:
                fitness_key = self.config.fitness_key
                if not fitness_key:
                    logger.warning(
                        "[EvolutionEngine] Memory enabled but fitness_key is not configured"
                    )
                    memory_parents = elites[: self.config.memory_top_n]
                else:
                    memory_parents = await self._select_top_programs_by_fitness(
                        self.config.memory_top_n
                    )
                if memory_parents:
                    memory_ids = await generate_mutations(
                        memory_parents,
                        mutator=self.mutation_operator,
                        storage=self.storage,
                        state_manager=self.state,
                        parent_selector=self.config.parent_selector,
                        limit=self.config.memory_top_n,
                        iteration=self.metrics.total_generations,
                        memory_instructions=memory_instructions,
                        memory_used=True,
                    )
                    mutation_ids.extend(memory_ids)
                else:
                    logger.debug(
                        "[EvolutionEngine] No eligible programs for memory mutations"
                    )
            else:
                logger.debug(
                    "[EvolutionEngine] Memory enabled but no instructions loaded"
                )

        self.metrics.record_mutation_metrics(len(mutation_ids), 0)
        return mutation_ids

    async def _ingest_completed_programs(
        self,
        *,
        mutation_ids: list[str] | None = None,
    ) -> None:
        """
        Validate and hand over any DONE programs to the strategy.
        Programs already in the archive stay DONE (they arrived from a refresh DAG).
        New programs are added if accepted, otherwise discarded.

        Args:
            mutation_ids: IDs of programs created during this generation's mutation
                phase.  When None (mutation was skipped), all non-archive DONE programs
                are deserialized and validated normally.  When a list (mutation ran),
                non-archive DONE programs that are NOT in this set are batch-discarded
                without deserialization — they are stale leftovers from previous
                generations or initial population.
        """
        # Fetch only IDs first (SMEMBERS — no deserialization), then filter
        # out archive programs before doing the expensive mget+deserialize.
        done_ids = await self.storage.get_ids_by_status(ProgramState.DONE.value)
        if not done_ids:
            logger.debug(
                "[EvolutionEngine] gen={} No completed programs to ingest",
                self.metrics.total_generations,
            )
            return

        archive_program_ids = set(await self.strategy.get_program_ids())
        non_archive_ids = [pid for pid in done_ids if pid not in archive_program_ids]

        if not non_archive_ids:
            logger.debug(
                "[EvolutionEngine] gen={} {} DONE programs all in archive, skipping",
                self.metrics.total_generations,
                len(done_ids),
            )
            return

        # Fast path: when mutation_ids are known, batch-discard stale DONE
        # programs (those not created this generation) without deserializing
        # them.  This avoids O(N) mget + from_dict on the initial population.
        if mutation_ids is not None:
            mutation_id_set = set(mutation_ids)
            stale_ids = [pid for pid in non_archive_ids if pid not in mutation_id_set]
            new_ids = [pid for pid in non_archive_ids if pid in mutation_id_set]
            if stale_ids:
                logger.info(
                    "[EvolutionEngine] gen={} Fast-discard {} stale DONE program(s)",
                    self.metrics.total_generations,
                    len(stale_ids),
                )
                try:
                    await self.storage.batch_move_status_sets(
                        stale_ids,
                        ProgramState.DONE.value,
                        ProgramState.DISCARDED.value,
                    )
                except Exception as e:
                    logger.error(
                        "[EvolutionEngine] gen={} stale batch discard failed: {}",
                        self.metrics.total_generations,
                        e,
                    )
        else:
            new_ids = non_archive_ids

        if not new_ids:
            return

        # Only deserialize the new (non-archive) programs.
        # Exclude stage_results (~10% of payload) — ingestion only needs
        # metrics, state, metadata, and lineage.  The merge strategy in
        # storage.update() preserves existing stage_results from Redis.
        completed = await self.storage.mget(new_ids, exclude=EXCLUDE_STAGE_RESULTS)
        # Filter to actual DONE state (mget may return stale status)
        completed = [p for p in completed if p.state == ProgramState.DONE]

        if not completed:
            return

        logger.info(
            "[EvolutionEngine] gen={} Ingest {} program(s) ({} in archive skipped)",
            self.metrics.total_generations,
            len(completed),
            len(done_ids) - len(new_ids),
        )
        logger.debug(
            "[EvolutionEngine] Program IDs: {}",
            [p.short_id for p in completed[:8]]
            + (["..."] if len(completed) > 8 else []),
        )

        added = 0
        rej_valid = 0
        rej_strategy = 0

        # Collect IDs of rejected programs for a single batch transition
        # at the end, instead of one Redis write per reject.
        reject_ids: list[str] = []

        for prog in completed:
            try:
                if not self.config.program_acceptor.is_accepted(prog):
                    # rejected by basic checks
                    rej_valid += 1
                    logger.info(
                        "[EvolutionEngine] Program {} REJECTED by acceptor (metrics={})",
                        prog.short_id,
                        prog.metrics,
                    )
                    await self._notify_hook(prog, MutationOutcome.REJECTED_ACCEPTOR)
                    reject_ids.append(prog.id)
                elif await self.strategy.add(prog):
                    # accepted by strategy — stays DONE until next refresh
                    added += 1
                    await self._notify_hook(prog, MutationOutcome.ACCEPTED)
                    logger.debug(
                        "[EvolutionEngine] Program {} added to strategy (metrics={})",
                        prog.short_id,
                        prog.metrics,
                    )
                else:
                    # rejected by strategy / validation
                    rej_strategy += 1
                    logger.debug(
                        "[EvolutionEngine] Program {} rejected by strategy (metrics={})",
                        prog.short_id,
                        prog.metrics,
                    )
                    await self._notify_hook(prog, MutationOutcome.REJECTED_STRATEGY)
                    reject_ids.append(prog.id)
            except Exception as e:
                # Isolate per-program failures: log and discard the offending program
                # so the remaining programs in this batch are still processed.
                logger.error(
                    "[EvolutionEngine] Ingestion failed for program {}: {} — discarding",
                    prog.short_id,
                    e,
                )
                reject_ids.append(prog.id)

        # Batch DONE → DISCARDED (raw JSON patch, no Pydantic serialization).
        # Also update in-memory state so any downstream code sees DISCARDED.
        if reject_ids:
            reject_set = set(reject_ids)
            for prog in completed:
                if prog.id in reject_set:
                    prog.state = ProgramState.DISCARDED
            try:
                await self.storage.batch_transition_by_ids(
                    reject_ids,
                    ProgramState.DONE.value,
                    ProgramState.DISCARDED.value,
                )
            except Exception as e:
                logger.error(
                    "[EvolutionEngine] batch discard failed for {} programs: {}",
                    len(reject_ids),
                    e,
                )

        self.metrics.programs_processed += added
        self.metrics.record_ingestion_metrics(added, rej_valid, rej_strategy)
        logger.info(
            "[EvolutionEngine] gen={} Ingest done | added={}, rejected_validation={}, rejected_strategy={}",
            self.metrics.total_generations,
            added,
            rej_valid,
            rej_strategy,
        )

    async def _refresh_archive_programs(self) -> int:
        """Flip all archive programs from DONE to QUEUED so lineage/descendant-aware stages re-run."""
        program_ids_to_refresh = await self.strategy.get_program_ids()

        if not program_ids_to_refresh:
            return 0

        try:
            count = await self.storage.batch_transition_by_ids(
                program_ids_to_refresh,
                ProgramState.DONE.value,
                ProgramState.QUEUED.value,
            )
        except Exception as e:
            logger.error(
                "[EvolutionEngine] gen={} batch_transition_by_ids failed: {}",
                self.metrics.total_generations,
                e,
            )
            return 0

        if count:
            logger.info(
                "[EvolutionEngine] gen={} Submitted {} program(s) for refresh",
                self.metrics.total_generations,
                count,
            )
            self.metrics.record_reprocess_metrics(count)
        return count

    async def _has_active_dags(self) -> bool:
        """True if any programs are QUEUED or RUNNING (i.e., engine not idle).

        Uses count_by_status (SCARD, O(1)) for the fast path.  Falls back to
        the expensive get_all_by_status after 30s of continuous waiting to
        detect ghost IDs that would otherwise stall _await_idle forever.
        """
        queued, running = await asyncio.gather(
            self.storage.count_by_status(ProgramState.QUEUED.value),
            self.storage.count_by_status(ProgramState.RUNNING.value),
        )

        if queued or running:
            current_counts = (queued, running)
            if self._last_pending_dags_counts != current_counts:
                logger.debug(
                    "[EvolutionEngine] Pending DAGs: queued={}, running={}",
                    queued,
                    running,
                )
                self._last_pending_dags_counts = current_counts
            return True

        self._last_pending_dags_counts = None
        return False

    async def _set_state(self, program: Program, state: ProgramState) -> None:
        await self.state.set_program_state(program, state)

    async def _notify_hook(self, prog: Program, outcome: MutationOutcome) -> None:
        """Call on_program_ingested with fault isolation.

        Hook failures are non-fatal: they must never cause a program that was
        already accepted by the strategy to be discarded (which would create a
        ghost entry in the archive).
        """
        try:
            await self.mutation_operator.on_program_ingested(
                prog, self.storage, outcome=outcome
            )
        except Exception as exc:
            logger.warning(
                "[EvolutionEngine] on_program_ingested hook failed for {}: {} "
                "(non-fatal, program state unchanged)",
                prog.short_id,
                exc,
            )

    async def restore_state(self) -> None:
        """Restore total_generations from storage after a resume."""
        gen = await self.storage.load_run_state(_RUN_STATE_TOTAL_GENERATIONS)
        if gen is not None:
            self.metrics.total_generations = gen
            logger.info("[EvolutionEngine] Restored total_generations={}", gen)

    def _reached_generation_cap(self) -> bool:
        cap = self.config.max_generations
        return cap is not None and self.metrics.total_generations >= cap

    async def _select_top_programs_by_fitness(self, total: int) -> list[Program]:
        if total <= 0:
            return []
        fitness_key = self.config.fitness_key
        if not fitness_key:
            logger.warning(
                "[EvolutionEngine] Memory enabled but fitness_key is not configured"
            )
            return []

        program_ids = await self.strategy.get_program_ids()
        if not program_ids:
            return []

        programs = await self.storage.mget(program_ids)
        scored: list[tuple[Program, float]] = []
        for prog in programs:
            if not prog:
                continue
            metrics = prog.metrics or {}
            fitness_val = metrics.get(fitness_key)
            if fitness_val is None:
                continue
            validity = metrics.get(VALIDITY_KEY)
            if validity is not None and validity < 0.5:
                continue
            scored.append((prog, float(fitness_val)))

        if not scored:
            return []

        scored.sort(
            key=lambda item: item[1],
            reverse=bool(self.config.fitness_key_higher_is_better),
        )
        return [prog for prog, _ in scored[:total]]

    def _read_memory_instructions(self) -> str | None:
        path = Path(self.config.memory_path)
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            logger.warning("[EvolutionEngine] Memory file not found: {}", path)
            return None
        except Exception as exc:
            logger.warning("[EvolutionEngine] Failed to read memory file {}: {}", path, exc)
            return None

        text = text.strip()
        if not text:
            return None

        if path.suffix.lower() == ".json":
            formatted = self._format_memory_cards_json(text, path)
            if not formatted:
                return None
            return formatted

        return text

    @staticmethod
    def _format_memory_cards_json(raw: str, path: Path) -> str | None:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning(
                "[EvolutionEngine] Failed to parse memory JSON {}: {}", path, exc
            )
            return None

        cards = payload.get("memory_cards")
        if not isinstance(cards, list) or not cards:
            logger.warning(
                "[EvolutionEngine] Memory JSON {} missing non-empty memory_cards list",
                path,
            )
            return None

        formatted_cards: list[str] = []
        for idx, card in enumerate(cards, start=1):
            if not isinstance(card, dict):
                continue
            formatted = EvolutionEngine._format_memory_card(card, idx)
            if formatted:
                formatted_cards.append(formatted)

        if not formatted_cards:
            logger.warning(
                "[EvolutionEngine] Memory JSON {} contained no usable cards", path
            )
            return None

        return "\n\n---\n\n".join(formatted_cards)

    @staticmethod
    def _format_memory_card(card: dict[str, Any], index: int) -> str:
        card_id = str(card.get("card_id") or f"card_{index}")
        lines = [f"CARD_ID: {card_id}"]

        card_type = card.get("type")
        if card_type:
            lines.append(f"TYPE: {card_type}")

        EvolutionEngine._append_card_section(lines, "WHEN_TO_USE", card.get("when_to_use"))
        EvolutionEngine._append_card_section(
            lines, "MUTATION_ACTIONS", card.get("mutation_actions")
        )
        EvolutionEngine._append_card_section(
            lines, "WORKS_BEST_TOGETHER", card.get("works_best_together")
        )
        EvolutionEngine._append_card_section(lines, "NOTES", card.get("notes"))

        return "\n".join(lines)

    @staticmethod
    def _append_card_section(
        lines: list[str], label: str, items: object
    ) -> None:
        if not items:
            return
        if not isinstance(items, list):
            items = [items]
        cleaned = [str(item).strip() for item in items if item and str(item).strip()]
        if not cleaned:
            return
        lines.append(f"{label}:")
        lines.extend(f"- {item}" for item in cleaned)
