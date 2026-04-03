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
from typing import cast

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
        cfg = cast(SteadyStateEngineConfig, self.config)
        if not isinstance(cfg, SteadyStateEngineConfig):
            raise TypeError(
                f"SteadyStateEvolutionEngine requires SteadyStateEngineConfig, "
                f"got {type(self.config).__name__}"
            )
        self._ss_config: SteadyStateEngineConfig = self.config

        # Backpressure
        self._in_flight: set[str] = set()
        self._in_flight_sema = asyncio.Semaphore(self._ss_config.max_in_flight)
        self._in_flight_lock = asyncio.Lock()

        # Epoch refresh gating
        self._mutation_gate = asyncio.Event()
        self._mutation_gate.set()  # open by default
        self._draining = False  # True during scoped drain (suppress epoch trigger)

        # Epoch bookkeeping
        self._processed_since_epoch = 0
        self._epoch_mutants = 0  # mutants produced in current epoch (for logging)
        self._epoch_eligible_since: float | None = None  # low-watermark fallback timer

        # Cached elites (refreshed at epoch boundaries)
        self._cached_elites: list[Program] | None = None
        self._elite_cache_lock = asyncio.Lock()

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

        # Persist initial epoch counter so status tools can read it immediately
        await self.storage.save_run_state(
            _RUN_STATE_TOTAL_GENERATIONS, self.metrics.total_generations
        )

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

            # Capture exception from completed tasks (will re-raise after drain)
            loop_exc = None
            for t in done:
                if not t.cancelled():
                    exc = t.exception()
                    if exc and not isinstance(exc, asyncio.CancelledError):
                        logger.error("[SteadyState] Loop failed: {}", exc)
                        loop_exc = exc

            # Final epoch to capture any stragglers.
            # No timeout: DAG eval duration is problem-dependent; a fixed timeout
            # would silently drop programs still mid-evaluation.
            # Stuck programs are handled by dag_timeout/stage_timeout in DagRunner.
            if self._in_flight:
                await self._epoch_refresh()

            if loop_exc is not None:
                raise loop_exc

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
                if len(mutation_ids) > 1:
                    # Defensive: 1 semaphore slot acquired but >1 ID returned.
                    # Only track the first; release would over-release otherwise.
                    logger.warning(
                        "[SteadyState] generate_mutations(limit=1) returned {} IDs; "
                        "tracking only the first",
                        len(mutation_ids),
                    )
                    mutation_ids = mutation_ids[:1]
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

                # Check epoch trigger
                if self._should_trigger_epoch():
                    await self._epoch_refresh()

                if self._reached_generation_cap():
                    break

                # Adaptive polling: tighter when pipeline is saturated
                # (slot release is the critical path when all slots are full)
                if len(self._in_flight) >= self._ss_config.max_in_flight:
                    interval = self.config.loop_interval * 0.25
                elif ingested:
                    interval = self.config.loop_interval * 0.25
                else:
                    interval = self.config.loop_interval
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            raise
        finally:
            logger.info("[SteadyState] Ingestion loop stopped")

    async def _poll_and_ingest(self, *, exclude_ids: set[str] | None = None) -> int:
        """Poll for completed in-flight programs, ingest them, and sweep leaks.

        Returns the number of programs processed (ingested + rejected + swept).
        Combines ingestion and leak detection in a single pass: fetches DONE IDs,
        then mgets ALL in-flight programs to find both completions and leaks.

        *exclude_ids*: IDs to skip (used by ``_drain_scoped`` to prevent
        double-ingestion of drain-set programs).
        """
        async with self._in_flight_lock:
            if not self._in_flight:
                return 0
            candidates = set(self._in_flight)
            if exclude_ids:
                candidates -= exclude_ids
            if not candidates:
                return 0
            in_flight_snapshot = list(candidates)

        # Single mget for ALL in-flight programs: detect DONE + leaked in one pass
        programs = await self.storage.mget(
            in_flight_snapshot, exclude=EXCLUDE_STAGE_RESULTS
        )
        found_ids = {p.id for p in programs if p is not None}

        done_ids: list[str] = []
        leaked_ids: list[str] = []

        for prog in programs:
            if prog is None:
                continue
            if prog.state == ProgramState.DONE:
                done_ids.append(prog.id)
            elif prog.state == ProgramState.DISCARDED:
                leaked_ids.append(prog.id)
            # QUEUED/RUNNING → still active, skip

        # Programs that vanished from Redis entirely
        for pid in in_flight_snapshot:
            if pid not in found_ids:
                leaked_ids.append(pid)

        # Ingest DONE programs
        handled_ids: list[str] = []
        if done_ids:
            _, handled_ids = await self._ingest_batch(done_ids)

        # Release slots for ingested + leaked programs
        released = set(handled_ids) | set(leaked_ids)
        if released:
            if leaked_ids:
                logger.warning(
                    "[SteadyState] Sweeping {} leaked in-flight programs",
                    len(leaked_ids),
                )
            async with self._in_flight_lock:
                for pid in released:
                    if pid in self._in_flight:
                        self._in_flight.discard(pid)
                        self._in_flight_sema.release()

        self._processed_since_epoch += len(handled_ids)
        return len(handled_ids) + len(leaked_ids)

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
    # Epoch refresh
    # ------------------------------------------------------------------

    # Fallback: if in-flight stays above watermark for this long, trigger anyway
    _EPOCH_WATERMARK_FALLBACK_S = 15.0

    def _should_trigger_epoch(self) -> bool:
        if self._draining:
            return False  # suppress epoch trigger during scoped drain
        if self._processed_since_epoch < self._ss_config.epoch_trigger_count:
            self._epoch_eligible_since = None  # reset if not yet eligible
            return False

        # Count threshold met.  Opportunistic: wait for a natural valley in
        # in-flight count so the subsequent drain is fast.
        # Skip watermark for small max_in_flight (drain is already fast).
        mif = self._ss_config.max_in_flight
        if mif <= 3:
            return True
        watermark = mif // 4
        if len(self._in_flight) <= watermark:
            return True

        # Start fallback timer on first poll where count is met but in-flight is high
        if self._epoch_eligible_since is None:
            self._epoch_eligible_since = time.monotonic()

        # Fallback: don't wait forever (15s is ~1% of a 25-min eval)
        if (
            time.monotonic() - self._epoch_eligible_since
            > self._EPOCH_WATERMARK_FALLBACK_S
        ):
            return True

        return False

    async def _epoch_refresh(self) -> None:
        """Periodic synchronization: drain in-flight, refresh archive, bump epoch.

        Uses **scoped drain**: both mutation and ingestion continue during the
        drain phase.  Only the programs in-flight when the epoch triggered need
        to finish before the archive is refreshed.  New mutations produced and
        ingested during drain are counted toward the *next* epoch.

        ``_draining`` flag suppresses ``_should_trigger_epoch`` so the
        ingestion loop can keep running without triggering a nested refresh.
        The mutation gate is only closed for the brief refresh window.
        """
        epoch = self.metrics.total_generations
        epoch_t0 = time.monotonic()
        logger.info("[SteadyState] ---- Epoch {} refresh ----", epoch)

        try:
            # 1. Enter drain mode — suppresses _should_trigger_epoch so the
            #    ingestion loop (which is our caller) won't re-enter here.
            self._draining = True

            # 2. Snapshot the drain set and processed count (for carry-forward).
            pre_drain_count = self._processed_since_epoch
            async with self._in_flight_lock:
                drain_set = set(self._in_flight)

            if drain_set:
                logger.info(
                    "[SteadyState] Draining {} in-flight programs "
                    "(mutations + ingestion continue)",
                    len(drain_set),
                )
                # No timeout: DAG eval duration is problem-dependent and can
                # exceed any fixed limit.  Stuck programs are handled by
                # dag_timeout / stage_timeout in DagRunner.
                await self._drain_scoped(drain_set, timeout_sec=None)

            # 3. Gate mutation loop for the brief refresh window
            self._mutation_gate.clear()

            # 4. Pre-step hook (called once per epoch, mirrors parent's per-step call)
            if self._pre_step_hook:
                await self._pre_step_hook()

            # 5. Snapshot bump + incremental bump
            self.storage.snapshot.bump()
            self.storage.snapshot.bump(incremental=True)

            # 6. Refresh archive (DONE -> QUEUED for lineage/insights stages)
            refreshed = await self._refresh_archive_programs()

            # 7. Reopen mutation gate BEFORE waiting for refresh DAGs.
            #    First few mutations may see slightly stale population stats
            #    in their mutation context (from previous epoch's collector).
            #    This is acceptable: stats change slowly between epochs.
            self._draining = False
            # Carry forward programs ingested during drain (by _poll_and_ingest
            # inside _drain_scoped).  Without this, those programs "don't count"
            # toward the next epoch trigger, systematically delaying it.
            drain_phase_count = self._processed_since_epoch - pre_drain_count
            self._processed_since_epoch = max(0, drain_phase_count)
            self._epoch_mutants = 0
            self._epoch_eligible_since = None  # reset watermark timer
            # Pre-warm elite cache so mutation tasks don't thundering-herd
            # behind _elite_cache_lock when the gate opens.
            self._cached_elites = await self._select_elites_for_mutation()
            self._mutation_gate.set()

            # 8. Wait for refresh DAGs + reindex (mutations continue)
            if refreshed:
                await self._await_idle()
                await self.strategy.reindex_archive()

            # 9. Increment epoch counter
            self.metrics.total_generations += 1
            await self.storage.save_run_state(
                _RUN_STATE_TOTAL_GENERATIONS, self.metrics.total_generations
            )

            # 10. Log epoch summary
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
            # Safety net: ensure gate is always reopened on exception.
            # Normal path reopens gate at step 7 above.
            self._draining = False
            self._epoch_eligible_since = None
            if not self._mutation_gate.is_set():
                self._processed_since_epoch = 0
                self._epoch_mutants = 0
                self._cached_elites = None
                self._mutation_gate.set()

    # ------------------------------------------------------------------
    # Drain in-flight
    # ------------------------------------------------------------------

    async def _drain_in_flight(self, timeout_sec: float | None = None) -> None:
        """Wait for all in-flight mutants to finish DAG evaluation, then ingest.

        Closes the mutation gate and drains everything.  Used for final shutdown.
        For epoch refresh, prefer :meth:`_drain_scoped` which allows mutations
        to continue during the drain.
        """
        self._mutation_gate.clear()
        async with self._in_flight_lock:
            drain_set = set(self._in_flight)
        if drain_set:
            await self._drain_scoped(drain_set, timeout_sec=timeout_sec)

    async def _drain_scoped(
        self, drain_set: set[str], timeout_sec: float | None = None
    ) -> None:
        """Wait for a specific set of program IDs to finish evaluation, then ingest.

        Unlike :meth:`_drain_in_flight`, this does NOT close the mutation gate.
        New mutations can continue in parallel — their IDs are tracked in
        ``_in_flight`` but are not part of *drain_set* and do not block this
        method.

        Programs in *drain_set* that reach DONE are ingested and their semaphore
        slots released.  Programs that are DISCARDED or vanish have their slots
        force-released.
        """
        # Drain polling interval: tighter than ingestion loop for faster drain
        _DRAIN_POLL_S = 0.5

        t0 = time.monotonic()
        remaining_ids = set(drain_set)

        while remaining_ids:
            candidates = list(remaining_ids)

            # Scoped check: fetch actual state of only drain-set programs
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

            # Ingest DONE programs — use handled_ids (not done_ids) to avoid
            # losing programs whose state changed between drain mget and ingest mget
            handled: list[str] = []
            if done_ids:
                _, handled = await self._ingest_batch(done_ids)

            # Release slots for actually-handled programs + gone programs
            resolved = set(handled) | set(gone_ids)
            if resolved:
                async with self._in_flight_lock:
                    for pid in resolved:
                        if pid in self._in_flight:
                            self._in_flight.discard(pid)
                            self._in_flight_sema.release()
                remaining_ids -= resolved

            # All drain-set programs resolved?
            if still_active == 0:
                # Retry ingestion for any remaining DONE programs that weren't
                # handled due to TOCTOU (state changed between mgets).
                retry_done = [pid for pid in remaining_ids if pid in set(done_ids)]
                if retry_done:
                    _, retry_handled = await self._ingest_batch(retry_done)
                    async with self._in_flight_lock:
                        for pid in retry_handled:
                            if pid in self._in_flight:
                                self._in_flight.discard(pid)
                                self._in_flight_sema.release()
                    remaining_ids -= set(retry_handled)

                # Force-release anything truly unresolvable (vanished between mgets)
                if remaining_ids:
                    logger.warning(
                        "[SteadyState] {} drain-set programs unresolvable, "
                        "force-releasing slots",
                        len(remaining_ids),
                    )
                    async with self._in_flight_lock:
                        for pid in list(remaining_ids):
                            if pid in self._in_flight:
                                self._in_flight.discard(pid)
                                self._in_flight_sema.release()
                break

            elapsed = time.monotonic() - t0

            # Check timeout
            if timeout_sec is not None and elapsed > timeout_sec:
                logger.warning(
                    "[SteadyState] Drain timeout ({:.0f}s) with {} drain-set remaining; "
                    "force-releasing slots",
                    elapsed,
                    len(remaining_ids),
                )
                async with self._in_flight_lock:
                    for pid in list(remaining_ids):
                        if pid in self._in_flight:
                            self._in_flight.discard(pid)
                            self._in_flight_sema.release()
                break

            # Also ingest non-drain-set DONE programs to free semaphore slots.
            # Exclude drain-set IDs to prevent double-ingestion.
            await self._poll_and_ingest(exclude_ids=drain_set)

            if elapsed > 30 and int(elapsed) % 60 < _DRAIN_POLL_S:
                logger.info(
                    "[SteadyState] Draining: {}/{} remaining ({:.0f}s)",
                    len(remaining_ids),
                    len(drain_set),
                    elapsed,
                )
            await asyncio.sleep(_DRAIN_POLL_S)
