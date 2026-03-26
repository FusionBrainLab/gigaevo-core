"""SteadyStateEvolutionEngine — continuous mutation/evaluation interleaving.

Instead of the generational barrier (produce N mutants -> wait for ALL DAGs ->
ingest -> refresh -> repeat), this engine runs two concurrent async loops:

* **Mutation loop** — spawns up to ``max_in_flight`` concurrent mutation tasks.
  Each task acquires one semaphore slot, calls the LLM, and deposits the result.
  Backpressure is enforced by ``asyncio.Semaphore(max_in_flight)``.

* **Ingestion loop** — polls for DONE programs, ingests each immediately, and
  releases a semaphore slot so mutation tasks can proceed.  Triggers an
  *epoch refresh* every ``max_mutations_per_generation`` processed programs.

An **epoch refresh** is the only synchronization point: new mutation tasks are
blocked, all in-flight programs are drained, the archive is refreshed (so
NO_CACHE stages see a consistent population snapshot), and
``total_generations`` is incremented.

See ``SteadyStateEngineConfig`` for tunables.
"""

from __future__ import annotations

import asyncio
import contextlib
import time

from loguru import logger

from gigaevo.evolution.engine.config import SteadyStateEngineConfig
from gigaevo.evolution.engine.core import (
    _RUN_STATE_TOTAL_GENERATIONS,
    EvolutionEngine,
)
from gigaevo.evolution.engine.mutation import generate_mutations
from gigaevo.llm.bandit import MutationOutcome
from gigaevo.programs.program import EXCLUDE_STAGE_RESULTS, Program
from gigaevo.programs.program_state import ProgramState

# Minimum interval between _sweep_discarded calls (seconds)
_SWEEP_INTERVAL = 5.0


class SteadyStateEvolutionEngine(EvolutionEngine):
    """Evolution engine with continuous mutation/evaluation interleaving.

    Replaces the generational ``step()`` with two concurrent loops governed by
    a backpressure semaphore.  At most ``max_in_flight`` mutant programs exist
    between "produced" and "ingested/discarded" at any instant.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        cfg: SteadyStateEngineConfig = self.config  # type: ignore[assignment]
        if not isinstance(cfg, SteadyStateEngineConfig):
            raise TypeError(
                f"SteadyStateEvolutionEngine requires SteadyStateEngineConfig, "
                f"got {type(cfg).__name__}"
            )
        self._ss_config = cfg

        # Backpressure
        self._in_flight: set[str] = set()
        self._in_flight_sema = asyncio.Semaphore(cfg.max_in_flight)
        self._in_flight_lock = asyncio.Lock()

        # Epoch refresh gating
        self._mutation_gate = asyncio.Event()
        self._mutation_gate.set()  # open by default

        # Epoch bookkeeping
        self._processed_since_epoch = 0
        self._epoch_mutants = 0  # mutants produced in current epoch (for logging)

        # Cached elites (refreshed at epoch boundaries)
        self._cached_elites: list[Program] | None = None
        self._elite_cache_lock = asyncio.Lock()

        # Sweep throttle
        self._last_sweep_time = 0.0

        # Child tasks
        self._mutation_task: asyncio.Task | None = None
        self._ingestion_task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # Public API overrides
    # ------------------------------------------------------------------

    async def step(self) -> None:
        raise NotImplementedError(
            "SteadyStateEvolutionEngine uses run() directly. "
            "step() is not meaningful in steady-state mode."
        )

    async def run(self) -> None:
        logger.info(
            "[SteadyState] Start | max_in_flight={} epoch_size={} max_generations={}",
            self._ss_config.max_in_flight,
            self._ss_config.epoch_trigger_count,
            self._ss_config.max_generations,
        )
        self._running = True
        self._run_start_time = time.monotonic()
        self._run_start_gen = self.metrics.total_generations

        try:
            # Phase 0: drain initial population (seed programs already queued)
            await self._await_idle()
            await self._ingest_completed_programs(mutation_ids=None)
            self.storage.snapshot.bump(incremental=True)

            # Call pre_step_hook once at startup (mirrors parent's per-step call)
            if self._pre_step_hook:
                await self._pre_step_hook()

            # Launch concurrent loops
            self._mutation_task = asyncio.create_task(
                self._mutation_loop(), name="ss-mutation"
            )
            self._ingestion_task = asyncio.create_task(
                self._ingestion_loop(), name="ss-ingestion"
            )

            done, pending = await asyncio.wait(
                [self._mutation_task, self._ingestion_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending:
                t.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await t

            # Re-raise exceptions from completed tasks so callers see them
            for t in done:
                if not t.cancelled():
                    exc = t.exception()
                    if exc and not isinstance(exc, asyncio.CancelledError):
                        logger.error("[SteadyState] Loop failed: {}", exc)
                        raise exc

            # Final epoch to capture any stragglers (120s timeout for shutdown)
            if self._in_flight:
                try:
                    await asyncio.wait_for(self._epoch_refresh(), timeout=120.0)
                except TimeoutError:
                    logger.warning(
                        "[SteadyState] Final drain timed out with {} in-flight",
                        len(self._in_flight),
                    )

        except asyncio.CancelledError:
            logger.debug("[SteadyState] run() cancelled")
            raise
        finally:
            self._running = False
            logger.info("[SteadyState] Stopped")

    # ------------------------------------------------------------------
    # Mutation loop (producer) — concurrent task spawner
    # ------------------------------------------------------------------

    async def _mutation_loop(self) -> None:
        """Spawn concurrent mutation tasks, gated by the semaphore.

        Each spawned task owns exactly one semaphore slot from creation
        to either ``_in_flight.add()`` or explicit ``release()``.
        """
        logger.info("[SteadyState] Mutation loop started")
        active_tasks: set[asyncio.Task] = set()
        try:
            while self._running and not self._reached_generation_cap():
                # Respect epoch refresh pause
                await self._mutation_gate.wait()

                # Backpressure: block until a slot opens
                await self._in_flight_sema.acquire()

                if not self._running or self._reached_generation_cap():
                    self._in_flight_sema.release()
                    break

                # Re-check gate after semaphore acquisition — the gate may
                # have closed while we were waiting for a slot.
                if not self._mutation_gate.is_set():
                    self._in_flight_sema.release()
                    continue

                # Spawn a concurrent mutation task (owns the acquired slot)
                task = asyncio.create_task(
                    self._produce_one_mutant(),
                    name=f"ss-mutate-{self._epoch_mutants}",
                )
                active_tasks.add(task)
                task.add_done_callback(active_tasks.discard)

        except asyncio.CancelledError:
            raise
        finally:
            # Cancel any still-running mutation tasks on shutdown
            for t in active_tasks:
                t.cancel()
            if active_tasks:
                await asyncio.gather(*active_tasks, return_exceptions=True)
            logger.info("[SteadyState] Mutation loop stopped")

    async def _produce_one_mutant(self) -> None:
        """Single mutation task. Owns one semaphore slot on entry.

        Invariant: every exit path either adds the program to ``_in_flight``
        (transferring slot ownership) or releases the semaphore slot.
        """
        try:
            elites = await self._get_cached_elites()
            if not elites:
                self._in_flight_sema.release()
                return

            mutation_ids = await self._create_single_mutant(elites)
            if mutation_ids:
                async with self._in_flight_lock:
                    self._in_flight.update(mutation_ids)
                self._epoch_mutants += len(mutation_ids)
            else:
                self._in_flight_sema.release()

        except asyncio.CancelledError:
            self._in_flight_sema.release()
            raise
        except Exception as e:
            logger.exception("[SteadyState] Mutation task failed: {}", e)
            self._in_flight_sema.release()

    async def _get_cached_elites(self) -> list[Program]:
        """Return cached elites, refreshing on cache miss (epoch boundary).

        Uses a lock to prevent thundering herd: after epoch refresh clears
        the cache, only one task fetches fresh elites; others wait and reuse.
        """
        if self._cached_elites is not None:
            return self._cached_elites
        async with self._elite_cache_lock:
            # Double-check after acquiring lock
            if self._cached_elites is None:
                self._cached_elites = await self._select_elites_for_mutation()
            return self._cached_elites

    async def _create_single_mutant(self, elites: list[Program]) -> list[str]:
        """Create a single mutant from *elites*. Returns 0 or 1 IDs."""
        mutation_ids = await generate_mutations(
            elites,
            mutator=self.mutation_operator,
            storage=self.storage,
            state_manager=self.state,
            parent_selector=self.config.parent_selector,
            limit=1,
            iteration=self.metrics.total_generations,  # current epoch
        )
        if mutation_ids:
            self.metrics.record_mutation_metrics(len(mutation_ids), 0)
        return mutation_ids

    # ------------------------------------------------------------------
    # Ingestion loop (consumer)
    # ------------------------------------------------------------------

    async def _ingestion_loop(self) -> None:
        logger.info("[SteadyState] Ingestion loop started")
        try:
            while self._running:
                ingested = await self._poll_and_ingest()

                # Sweep for leaked slots — throttled to avoid excessive mget
                now = time.monotonic()
                if now - self._last_sweep_time >= _SWEEP_INTERVAL:
                    await self._sweep_discarded()
                    self._last_sweep_time = now

                # Check epoch trigger
                if self._should_trigger_epoch():
                    await self._epoch_refresh()

                if self._reached_generation_cap():
                    break

                # Adaptive polling: sleep less when there's active work
                interval = (
                    self.config.loop_interval * 0.25
                    if ingested
                    else self.config.loop_interval
                )
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            raise
        finally:
            logger.info("[SteadyState] Ingestion loop stopped")

    async def _poll_and_ingest(self) -> int:
        """Poll for completed in-flight programs and ingest them.

        Returns the number of programs processed (ingested + rejected).
        Only releases semaphore slots for programs that were confirmed DONE
        by ``_ingest_batch`` (avoids over-release if a program's state changed
        between ``get_ids_by_status`` and ``mget``).
        """
        done_ids = await self.storage.get_ids_by_status(ProgramState.DONE.value)
        if not done_ids:
            return 0

        async with self._in_flight_lock:
            in_flight_snapshot = set(self._in_flight)

        ingestable = [pid for pid in done_ids if pid in in_flight_snapshot]
        if not ingestable:
            return 0

        added, handled_ids = await self._ingest_batch(ingestable)

        # Only release slots for programs that were actually processed
        async with self._in_flight_lock:
            for pid in handled_ids:
                if pid in self._in_flight:
                    self._in_flight.discard(pid)
                    self._in_flight_sema.release()

        self._processed_since_epoch += len(handled_ids)
        # Return count of handled programs (accepted + rejected), not just accepted.
        # This ensures adaptive sleep triggers even when all programs were rejected.
        return len(handled_ids)

    async def _ingest_batch(self, program_ids: list[str]) -> tuple[int, list[str]]:
        """Ingest specific completed programs.

        Returns ``(added_count, handled_ids)`` where *handled_ids* is the list
        of program IDs that were confirmed DONE and processed (accepted or
        rejected).  IDs whose state changed between the status query and
        ``mget`` are excluded — their semaphore slots must NOT be released.
        """
        if not program_ids:
            return 0, []

        completed = await self.storage.mget(program_ids, exclude=EXCLUDE_STAGE_RESULTS)
        completed = [p for p in completed if p.state == ProgramState.DONE]

        if not completed:
            return 0, []

        added = 0
        rej_valid = 0
        rej_strategy = 0
        reject_ids: list[str] = []

        for prog in completed:
            try:
                if not self.config.program_acceptor.is_accepted(prog):
                    logger.info(
                        "[SteadyState] Program {} REJECTED by acceptor (metrics={})",
                        prog.short_id,
                        prog.metrics,
                    )
                    await self._notify_hook(prog, MutationOutcome.REJECTED_ACCEPTOR)
                    reject_ids.append(prog.id)
                    rej_valid += 1
                elif await self.strategy.add(prog):
                    added += 1
                    await self._notify_hook(prog, MutationOutcome.ACCEPTED)
                    logger.debug(
                        "[SteadyState] Program {} accepted (metrics={})",
                        prog.short_id,
                        prog.metrics,
                    )
                else:
                    await self._notify_hook(prog, MutationOutcome.REJECTED_STRATEGY)
                    reject_ids.append(prog.id)
                    rej_strategy += 1
                    logger.debug(
                        "[SteadyState] Program {} rejected by strategy",
                        prog.short_id,
                    )
            except Exception as e:
                logger.error(
                    "[SteadyState] Ingestion failed for {}: {}", prog.short_id, e
                )
                reject_ids.append(prog.id)

        # Batch DONE -> DISCARDED for rejects
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
                    "[SteadyState] Batch discard failed for {} programs: {}",
                    len(reject_ids),
                    e,
                )

        self.metrics.programs_processed += added
        self.metrics.record_ingestion_metrics(added, rej_valid, rej_strategy)
        handled = [p.id for p in completed]
        return added, handled

    # ------------------------------------------------------------------
    # Sweep for leaked semaphore slots
    # ------------------------------------------------------------------

    async def _sweep_discarded(self) -> None:
        """Release slots for in-flight programs that DagRunner discarded.

        Uses ``mget`` to fetch the actual program state atomically, avoiding
        the TOCTOU race of checking multiple status sets sequentially.
        """
        async with self._in_flight_lock:
            if not self._in_flight:
                return
            candidates = list(self._in_flight)

        # Atomic check: fetch actual state of each candidate in one MGET
        programs = await self.storage.mget(candidates, exclude=EXCLUDE_STAGE_RESULTS)
        # Build a map of id -> state; programs that don't exist return None
        state_by_id: dict[str, ProgramState | None] = {}
        for prog in programs:
            if prog is not None:
                state_by_id[prog.id] = prog.state

        # A program is leaked if it's DISCARDED or completely gone from Redis
        leaked = [
            pid
            for pid in candidates
            if pid not in state_by_id or state_by_id[pid] == ProgramState.DISCARDED
        ]
        if not leaked:
            return

        logger.warning(
            "[SteadyState] Sweeping {} leaked in-flight programs (DISCARDED or vanished)",
            len(leaked),
        )
        async with self._in_flight_lock:
            for pid in leaked:
                if pid in self._in_flight:
                    self._in_flight.discard(pid)
                    self._in_flight_sema.release()
                    # Do NOT count ghosts in _processed_since_epoch — they were
                    # never ingested, so shouldn't trigger premature epoch refresh

    # ------------------------------------------------------------------
    # Epoch refresh
    # ------------------------------------------------------------------

    def _should_trigger_epoch(self) -> bool:
        return self._processed_since_epoch >= self._ss_config.epoch_trigger_count

    async def _epoch_refresh(self) -> None:
        """Periodic synchronization: drain in-flight, refresh archive, bump epoch."""
        epoch = self.metrics.total_generations
        epoch_t0 = time.monotonic()
        logger.info("[SteadyState] ---- Epoch {} refresh ----", epoch)

        # Gate mutation loop — try/finally ensures it always reopens
        self._mutation_gate.clear()
        try:
            # 1. Drain all in-flight mutants (with timeout to prevent hangs if DagRunner dies)
            # Use a generous 600s (10min) for normal operation; final shutdown uses 120s
            await self._drain_in_flight(timeout_sec=600.0)

            # 2. Pre-step hook (called once per epoch, mirrors parent's per-step call)
            if self._pre_step_hook:
                await self._pre_step_hook()

            # 3. Full snapshot bump
            self.storage.snapshot.bump()

            # 4. Incremental bump (drain already ingested all DONE in-flight programs;
            #    no need to call _ingest_completed_programs which would re-fetch ALL
            #    DONE programs globally, bypassing steady-state bookkeeping)
            self.storage.snapshot.bump(incremental=True)

            # 5. Refresh archive (DONE -> QUEUED for lineage/insights stages)
            refreshed = await self._refresh_archive_programs()
            if refreshed:
                await self._await_idle()
                await self.strategy.reindex_archive()

            # 6. Increment epoch counter
            self.metrics.total_generations += 1
            await self.storage.save_run_state(
                _RUN_STATE_TOTAL_GENERATIONS, self.metrics.total_generations
            )

            # 7. Log epoch summary
            epoch_elapsed = time.monotonic() - epoch_t0
            archive_size = len(await self.strategy.get_program_ids())
            archive_delta = archive_size - self._prev_archive_size
            self._prev_archive_size = archive_size
            if archive_delta == 0:
                self._stagnant_gens += 1
            else:
                self._stagnant_gens = 0

            best_str = self._metrics_tracker.format_best_summary()
            eta_str = self._format_eta()

            logger.info(
                "[SteadyState] epoch={} done | mutants={} refreshed={}"
                " | archive={} ({:+d}){} ({:.1f}s){}",
                epoch,
                self._epoch_mutants,
                refreshed,
                archive_size,
                archive_delta,
                best_str,
                epoch_elapsed,
                eta_str,
            )

            if self._stagnant_gens >= 5:
                logger.warning(
                    "[SteadyState] Archive stagnant for {} consecutive epochs",
                    self._stagnant_gens,
                )
        finally:
            # Reset epoch bookkeeping, invalidate elite cache, resume mutation
            self._processed_since_epoch = 0
            self._epoch_mutants = 0
            self._cached_elites = None  # force fresh selection next epoch
            self._mutation_gate.set()

    # ------------------------------------------------------------------
    # Drain in-flight
    # ------------------------------------------------------------------

    async def _drain_in_flight(self, timeout_sec: float | None = None) -> None:
        """Wait for all in-flight mutants to finish DAG evaluation, then ingest.

        Uses **scoped** state checks (``mget`` on ``_in_flight`` IDs only)
        rather than the global ``_has_active_dags()`` to avoid hanging when
        non-in-flight programs (e.g. archive refresh) are also QUEUED/RUNNING.

        Relies on existing ``dag_timeout``/``stage_timeout`` for stuck programs.
        If timeout_sec is exceeded, force-releases remaining slots with a warning.
        """
        t0 = time.monotonic()

        while True:
            async with self._in_flight_lock:
                if not self._in_flight:
                    break
                candidates = list(self._in_flight)
                remaining = len(candidates)

            # Scoped check: fetch actual state of only OUR in-flight programs
            programs = await self.storage.mget(
                candidates, exclude=EXCLUDE_STAGE_RESULTS
            )
            found_ids = {p.id for p in programs if p is not None}

            done_ids: list[str] = []
            gone_ids: list[str] = []
            still_active = 0

            for prog in programs:
                if prog is None:
                    continue
                if prog.state == ProgramState.DONE:
                    done_ids.append(prog.id)
                elif prog.state in (ProgramState.QUEUED, ProgramState.RUNNING):
                    still_active += 1
                else:
                    gone_ids.append(prog.id)  # DISCARDED or unexpected

            # IDs that vanished entirely from Redis
            for pid in candidates:
                if pid not in found_ids:
                    gone_ids.append(pid)

            # Ingest DONE programs
            if done_ids:
                await self._ingest_batch(done_ids)

            # Release slots for all resolved programs (DONE + gone)
            resolved = set(done_ids) | set(gone_ids)
            if resolved:
                async with self._in_flight_lock:
                    for pid in resolved:
                        if pid in self._in_flight:
                            self._in_flight.discard(pid)
                            self._in_flight_sema.release()

            # If nothing still active among OUR programs, we're done.
            # IMPORTANT: only clean up IDs from the original `candidates` snapshot,
            # NOT anything newly added by a mutation loop that slipped past the gate.
            if still_active == 0:
                async with self._in_flight_lock:
                    for pid in candidates:
                        if pid in self._in_flight:
                            self._in_flight.discard(pid)
                            self._in_flight_sema.release()
                break

            elapsed = time.monotonic() - t0

            # Check timeout
            if timeout_sec is not None and elapsed > timeout_sec:
                logger.warning(
                    "[SteadyState] Drain timeout ({:.0f}s) with {} in-flight; "
                    "force-releasing slots",
                    elapsed,
                    remaining,
                )
                async with self._in_flight_lock:
                    for pid in list(self._in_flight):
                        self._in_flight.discard(pid)
                        self._in_flight_sema.release()
                break

            if elapsed > 30 and int(elapsed) % 60 < self.config.loop_interval:
                logger.info(
                    "[SteadyState] Draining: {} in-flight ({:.0f}s)",
                    remaining,
                    elapsed,
                )
            await asyncio.sleep(self.config.loop_interval)
