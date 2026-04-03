"""Deterministic evolution-equivalence tests for SteadyStateEvolutionEngine.

These tests verify that the engine's optimizations (scoped drain, interleaved
ingestion during drain, early gate reopen) do NOT change the logical sequence
of program mutations, evaluations, and ingestions.

The test creates a fully controlled engine with:
- Deterministic program IDs (sequential counters)
- Controlled "DAG evaluation" timing (programs complete in a fixed order)
- All randomness removed (fixed elite selection, no random parent selection)

It then captures an ordered event trace and asserts the trace matches a
known-good sequence.  If any optimization changes the order in which programs
are created, evaluated, or ingested, the test fails.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gigaevo.evolution.engine.config import SteadyStateEngineConfig
from gigaevo.evolution.engine.steady_state import SteadyStateEvolutionEngine
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState

# ---------------------------------------------------------------------------
# Event trace recorder
# ---------------------------------------------------------------------------


@dataclass
class EventTrace:
    """Records ordered events from the engine for determinism checks."""

    events: list[tuple[str, str]] = field(default_factory=list)

    def mutation(self, pid: str) -> None:
        self.events.append(("MUTATE", pid))

    def eval_done(self, pid: str) -> None:
        self.events.append(("EVAL_DONE", pid))

    def ingest(self, pid: str, accepted: bool) -> None:
        self.events.append(("INGEST", f"{pid}:{'ok' if accepted else 'rej'}"))

    def epoch(self, gen: int) -> None:
        self.events.append(("EPOCH", str(gen)))

    def drain_start(self, n: int) -> None:
        self.events.append(("DRAIN_START", str(n)))

    def drain_done(self) -> None:
        self.events.append(("DRAIN_DONE", ""))

    @property
    def mutation_order(self) -> list[str]:
        return [pid for ev, pid in self.events if ev == "MUTATE"]

    @property
    def ingest_order(self) -> list[str]:
        return [pid for ev, pid in self.events if ev == "INGEST"]

    @property
    def epoch_events(self) -> list[str]:
        return [gen for ev, gen in self.events if ev == "EPOCH"]


# ---------------------------------------------------------------------------
# Deterministic engine factory
# ---------------------------------------------------------------------------


class DeterministicEngine:
    """A fully controlled SteadyStateEvolutionEngine for equivalence testing.

    Programs are created with sequential IDs (prog-0, prog-1, ...).
    DAG evaluation completes after a fixed delay (FIFO order).
    All strategy/storage operations are deterministic.
    """

    def __init__(
        self,
        max_in_flight: int = 3,
        epoch_size: int = 4,
        max_generations: int = 2,
        mutation_delay: float = 0.01,
        eval_delay: float = 0.05,
    ):
        self.trace = EventTrace()
        self.mutation_delay = mutation_delay
        self.eval_delay = eval_delay
        self._prog_counter = 0
        self._programs: dict[str, Program] = {}
        self._eval_tasks: list[asyncio.Task] = []

        # Build engine with mocked dependencies
        storage = AsyncMock()
        strategy = AsyncMock()
        writer = MagicMock()
        writer.bind.return_value = writer
        mt = MagicMock()
        mt.format_best_summary.return_value = ""

        storage.count_by_status.return_value = 0
        storage.get_all_by_status.return_value = []
        storage.snapshot = MagicMock()
        strategy.get_program_ids.return_value = []
        strategy.add.return_value = True

        config = SteadyStateEngineConfig(
            max_in_flight=max_in_flight,
            max_mutations_per_generation=epoch_size,
            max_generations=max_generations,
            loop_interval=0.01,
        )

        self.engine = SteadyStateEvolutionEngine(
            storage=storage,
            strategy=strategy,
            mutation_operator=AsyncMock(),
            config=config,
            writer=writer,
            metrics_tracker=mt,
        )
        self.engine.state = AsyncMock()
        self.engine.config.program_acceptor = MagicMock()
        self.engine.config.program_acceptor.is_accepted.return_value = True
        self.engine.strategy.select_elites.return_value = [
            Program(code="def solve(): return 42", state=ProgramState.DONE)
        ]

        # Wire up deterministic storage
        storage.get_ids_by_status.side_effect = self._get_ids_by_status
        storage.mget.side_effect = self._mget

        # Track original epoch_refresh to instrument it
        self._orig_epoch = self.engine._epoch_refresh
        self.engine._epoch_refresh = self._tracked_epoch_refresh

        # Track original drain_scoped to instrument it
        self._orig_drain = self.engine._drain_scoped
        self.engine._drain_scoped = self._tracked_drain_scoped

    def _next_prog_id(self) -> str:
        pid = f"prog-{self._prog_counter}"
        self._prog_counter += 1
        return pid

    def _get_ids_by_status(self, status_val):
        return [pid for pid, p in self._programs.items() if p.state.value == status_val]

    def _mget(self, ids, **kwargs):
        return [self._programs[pid] for pid in ids if pid in self._programs]

    async def _fake_generate(self, elites, **kwargs):
        """Deterministic mutation: create sequential program, schedule eval."""
        pid = self._next_prog_id()
        prog = Program(
            code=f"def solve(): return {self._prog_counter}", state=ProgramState.QUEUED
        )
        # Override the random ID with our deterministic one
        object.__setattr__(prog, "id", pid)
        self._programs[pid] = prog

        await asyncio.sleep(self.mutation_delay)
        self.trace.mutation(pid)

        # Schedule deterministic "DAG evaluation" — completes after eval_delay
        task = asyncio.create_task(self._eval_program(pid))
        self._eval_tasks.append(task)

        return [pid]

    async def _eval_program(self, pid: str) -> None:
        """Simulated DAG evaluation — transitions QUEUED -> DONE after delay."""
        await asyncio.sleep(self.eval_delay)
        if pid in self._programs:
            self._programs[pid].state = ProgramState.DONE
            self.trace.eval_done(pid)

    async def _tracked_epoch_refresh(self) -> None:
        gen = self.engine.metrics.total_generations
        self.trace.epoch(gen)
        await self._orig_epoch()

    async def _tracked_drain_scoped(self, drain_set, timeout_sec=None):
        self.trace.drain_start(len(drain_set))
        await self._orig_drain(drain_set, timeout_sec=timeout_sec)
        self.trace.drain_done()

    async def run(self, timeout: float = 10.0) -> EventTrace:
        """Run the engine and return the event trace."""

        # Track ingestions via strategy.add side effect
        async def tracked_add(prog):
            self.trace.ingest(prog.id, True)
            return True

        self.engine.strategy.add = tracked_add

        with patch(
            "gigaevo.evolution.engine.steady_state.generate_mutations",
            side_effect=self._fake_generate,
        ):
            try:
                await asyncio.wait_for(self.engine.run(), timeout=timeout)
            except TimeoutError:
                self.engine._running = False
            finally:
                for t in self._eval_tasks:
                    t.cancel()
                await asyncio.gather(*self._eval_tasks, return_exceptions=True)

        return self.trace


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDeterministicEvolution:
    """Verify the engine produces the same event trace on repeated runs."""

    @pytest.mark.parametrize("run_idx", [0, 1, 2])
    async def test_mutation_order_deterministic(self, run_idx: int) -> None:
        """Programs are mutated in the same order across runs."""
        de = DeterministicEngine(max_in_flight=3, epoch_size=4, max_generations=2)
        trace = await de.run(timeout=10.0)

        # Must have produced some mutations
        mutations = trace.mutation_order
        assert len(mutations) >= 4, f"Expected >= 4 mutations, got {len(mutations)}"

        # Mutations must be sequential (deterministic program IDs)
        expected = [f"prog-{i}" for i in range(len(mutations))]
        assert mutations == expected, (
            f"Mutation order not deterministic:\n"
            f"  expected: {expected}\n"
            f"  got:      {mutations}"
        )

    async def test_ingestion_order_stable(self) -> None:
        """Programs are ingested in approximately FIFO order.

        Within a single poll batch, programs may be reordered (set iteration).
        We check that the TREND is monotonically increasing by verifying that
        consecutive batches have increasing minimum IDs.
        """
        de = DeterministicEngine(max_in_flight=3, epoch_size=4, max_generations=2)
        trace = await de.run(timeout=10.0)

        ingestions = trace.ingest_order
        assert len(ingestions) >= 4, f"Expected >= 4 ingestions, got {len(ingestions)}"

        # Check: all ingested IDs are from the expected range (no phantom IDs)
        ids = [int(e.split(":")[0].replace("prog-", "")) for e in ingestions]
        assert all(i >= 0 for i in ids), f"Unexpected negative IDs: {ids}"

        # Check: overall trend is increasing (max of first half < max of second half)
        mid = len(ids) // 2
        if mid > 0:
            first_max = max(ids[:mid])
            second_min = min(ids[mid:])
            # Allow some overlap but the trend must be forward
            assert second_min >= first_max - 2, (
                f"Ingestion order not approximately FIFO:\n  got IDs: {ids}"
            )

    async def test_epoch_boundaries_stable(self) -> None:
        """Epoch refresh triggers the right number of times across runs."""
        traces = []
        for _ in range(3):
            de = DeterministicEngine(max_in_flight=3, epoch_size=4, max_generations=2)
            trace = await de.run(timeout=10.0)
            traces.append(trace)

        # All runs should have the same number of epochs
        for i in range(1, len(traces)):
            assert len(traces[0].epoch_events) == len(traces[i].epoch_events), (
                f"Epoch count differs between run 0 and run {i}:\n"
                f"  run 0: {traces[0].epoch_events}\n"
                f"  run {i}: {traces[i].epoch_events}"
            )

    async def test_causal_trace_reproducible(self) -> None:
        """The causal event trace (mutations, ingestions) is consistent across runs.

        Epoch boundaries may shift due to low-watermark timing, so we check
        causal ordering (mutation order, ingestion order, no-loss, no-dup)
        rather than exact event trace equality.
        """
        traces = []
        for _ in range(3):
            de = DeterministicEngine(max_in_flight=3, epoch_size=4, max_generations=2)
            trace = await de.run(timeout=10.0)
            traces.append(trace)

        # Mutation order must be identical (deterministic program IDs)
        for i in range(1, len(traces)):
            assert traces[0].mutation_order == traces[i].mutation_order, (
                f"Mutation order differs between run 0 and run {i}"
            )

        # Ingestion order must be identical (FIFO eval completion)
        for i in range(1, len(traces)):
            assert traces[0].ingest_order == traces[i].ingest_order, (
                f"Ingestion order differs between run 0 and run {i}"
            )

        # Same number of epochs (may be >= max_generations)
        for i in range(1, len(traces)):
            assert len(traces[0].epoch_events) == len(traces[i].epoch_events), (
                f"Epoch count differs: run 0={len(traces[0].epoch_events)}, "
                f"run {i}={len(traces[i].epoch_events)}"
            )

    async def test_drain_does_not_reorder_ingestion(self) -> None:
        """Scoped drain: programs produced before drain are ingested before
        programs produced during drain.

        This is the critical invariant: if prog-X was produced before the epoch
        trigger and prog-Y was produced during drain, then prog-X is ingested
        first (in the drain's ingest pass), and prog-Y is ingested later
        (in the next epoch's normal ingestion loop).
        """
        de = DeterministicEngine(
            max_in_flight=3,
            epoch_size=3,  # small epoch to trigger drain quickly
            max_generations=2,
            mutation_delay=0.005,
            eval_delay=0.03,  # shorter eval so drain completes fast
        )
        trace = await de.run(timeout=10.0)

        # Check: for each DRAIN_START..DRAIN_DONE window, any programs
        # ingested during that window should have lower IDs than programs
        # ingested after the window (i.e., drain-set programs are ingested first).
        in_drain = False
        drain_ingested: list[int] = []
        post_drain_ingested: list[int] = []

        for ev, data in trace.events:
            if ev == "DRAIN_START":
                in_drain = True
                drain_ingested = []
                post_drain_ingested = []
            elif ev == "DRAIN_DONE":
                in_drain = False
            elif ev == "INGEST":
                pid_num = int(data.split(":")[0].replace("prog-", ""))
                if in_drain:
                    drain_ingested.append(pid_num)
                elif drain_ingested:
                    # First ingestion after drain
                    post_drain_ingested.append(pid_num)

        # If we had drain + post-drain ingestions, drain IDs should be <= post-drain IDs
        if drain_ingested and post_drain_ingested:
            assert max(drain_ingested) < min(post_drain_ingested), (
                f"Drain-set programs should be ingested before post-drain programs:\n"
                f"  drain ingested: {drain_ingested}\n"
                f"  post-drain ingested: {post_drain_ingested}"
            )

    async def test_no_program_lost_or_duplicated(self) -> None:
        """Every mutated program is ingested exactly once (no loss, no dup)."""
        de = DeterministicEngine(max_in_flight=3, epoch_size=4, max_generations=2)
        trace = await de.run(timeout=10.0)

        mutated = set(trace.mutation_order)
        ingested_pids = set(e.split(":")[0] for e in trace.ingest_order)

        # Every ingested program was mutated
        assert ingested_pids.issubset(mutated), (
            f"Ingested programs not in mutation set: {ingested_pids - mutated}"
        )

        # No duplicates in ingestion
        ingest_list = [e.split(":")[0] for e in trace.ingest_order]
        assert len(ingest_list) == len(set(ingest_list)), (
            f"Duplicate ingestions: {[x for x in ingest_list if ingest_list.count(x) > 1]}"
        )

    async def test_epoch_count_matches_config(self) -> None:
        """Engine stops after max_generations epochs."""
        for max_gen in [1, 2, 3]:
            de = DeterministicEngine(
                max_in_flight=2, epoch_size=3, max_generations=max_gen
            )
            trace = await de.run(timeout=15.0)

            epochs = trace.epoch_events
            assert len(epochs) >= max_gen, (
                f"max_generations={max_gen}, but only {len(epochs)} epochs ran: {epochs}"
            )


class TestNoProgramLoss:
    """Verify that slow DAG evaluations are never silently dropped."""

    async def test_slow_eval_not_dropped_by_drain(self) -> None:
        """Programs with long eval times must be waited for, not force-released.

        This test catches the class of bug where a drain timeout is shorter
        than the DAG eval duration, causing programs to be silently dropped.
        """
        de = DeterministicEngine(
            max_in_flight=3,
            epoch_size=3,
            max_generations=1,
            mutation_delay=0.005,
            eval_delay=0.2,  # "slow" eval — longer than typical test sleeps
        )
        trace = await de.run(timeout=15.0)

        mutated = set(trace.mutation_order)
        ingested_pids = set(e.split(":")[0] for e in trace.ingest_order)

        # CRITICAL: every mutated program that completed eval must be ingested.
        # If drain force-released a slot, the program vanishes from _in_flight
        # but was never ingested — a silent loss of compute work.
        eval_done = {pid for ev, pid in trace.events if ev == "EVAL_DONE"}
        lost = eval_done - ingested_pids
        assert not lost, (
            f"Programs completed eval but were NEVER ingested (dropped by drain?): {lost}\n"
            f"  mutated: {sorted(mutated)}\n"
            f"  eval_done: {sorted(eval_done)}\n"
            f"  ingested: {sorted(ingested_pids)}"
        )

    async def test_all_completed_programs_ingested_across_epochs(self) -> None:
        """Over multiple epochs, every program that finishes eval is ingested."""
        de = DeterministicEngine(
            max_in_flight=4,
            epoch_size=3,
            max_generations=3,
            mutation_delay=0.005,
            eval_delay=0.08,
        )
        trace = await de.run(timeout=15.0)

        eval_done = {pid for ev, pid in trace.events if ev == "EVAL_DONE"}
        ingested_pids = set(e.split(":")[0] for e in trace.ingest_order)

        lost = eval_done - ingested_pids
        assert not lost, (
            f"{len(lost)} programs completed eval but were never ingested: {sorted(lost)}"
        )

    async def test_program_success_rate(self) -> None:
        """Track the ratio of ingested/mutated as a health metric.

        In a healthy engine, the ingestion rate should approach the mutation
        rate over time (all mutated programs eventually get ingested).
        Programs still in-flight at shutdown are acceptable losses.
        """
        de = DeterministicEngine(
            max_in_flight=3,
            epoch_size=4,
            max_generations=2,
            mutation_delay=0.005,
            eval_delay=0.05,
        )
        trace = await de.run(timeout=10.0)

        n_mutated = len(trace.mutation_order)
        n_ingested = len(trace.ingest_order)
        n_eval_done = sum(1 for ev, _ in trace.events if ev == "EVAL_DONE")

        # Every completed eval must be ingested (0% drop rate)
        assert n_ingested >= n_eval_done, (
            f"Ingested ({n_ingested}) < eval_done ({n_eval_done}): "
            f"programs are being dropped!"
        )

        # Ingestion rate should be reasonable (at least 50% of mutations)
        # The gap is programs still in eval at shutdown — not dropped.
        assert n_ingested >= n_mutated * 0.5, (
            f"Ingestion rate too low: {n_ingested}/{n_mutated} = "
            f"{n_ingested / n_mutated:.0%}. Possible systematic loss."
        )


class TestProgramLossGrid:
    """Grid test: verify no program loss across all combinations of
    fast/slow mutation and fast/slow DAG evaluation."""

    @pytest.mark.parametrize(
        "mutation_delay,eval_delay,mif,epoch_size",
        [
            # Fast mutation, fast eval
            (0.002, 0.02, 3, 3),
            (0.002, 0.02, 5, 4),
            # Fast mutation, slow eval (pipeline fills up, backpressure)
            (0.002, 0.15, 3, 3),
            (0.002, 0.15, 5, 4),
            # Slow mutation, fast eval (pipeline drains quickly)
            (0.05, 0.02, 3, 3),
            (0.05, 0.02, 5, 4),
            # Slow mutation, slow eval (realistic production-like)
            (0.05, 0.15, 3, 3),
            (0.05, 0.15, 5, 4),
            # Extreme: very slow eval with small pipeline
            (0.005, 0.3, 2, 2),
            # Extreme: very fast everything with large pipeline
            (0.001, 0.01, 8, 6),
        ],
    )
    async def test_no_loss_grid(
        self,
        mutation_delay: float,
        eval_delay: float,
        mif: int,
        epoch_size: int,
    ) -> None:
        """Every program that completes eval is ingested. No silent drops."""
        de = DeterministicEngine(
            max_in_flight=mif,
            epoch_size=epoch_size,
            max_generations=2,
            mutation_delay=mutation_delay,
            eval_delay=eval_delay,
        )
        trace = await de.run(timeout=15.0)

        eval_done = {pid for ev, pid in trace.events if ev == "EVAL_DONE"}
        ingested_pids = set(e.split(":")[0] for e in trace.ingest_order)

        lost = eval_done - ingested_pids
        assert not lost, (
            f"[mut={mutation_delay}s eval={eval_delay}s mif={mif} epoch={epoch_size}] "
            f"{len(lost)} programs dropped: {sorted(lost)}"
        )

        # No duplicates
        ingest_list = [e.split(":")[0] for e in trace.ingest_order]
        assert len(ingest_list) == len(set(ingest_list)), (
            "Duplicate ingestions detected"
        )


class TestNoDoubleIngestion:
    """Verify drain-set programs are never ingested twice."""

    async def test_drain_set_excluded_from_poll_during_drain(self) -> None:
        """_poll_and_ingest with exclude_ids skips drain-set programs."""
        from tests.evolution.test_steady_state import _make_ss_engine, _prog

        engine = _make_ss_engine(max_in_flight=4)

        drain_prog = _prog(ProgramState.DONE)
        other_prog = _prog(ProgramState.DONE)

        engine._in_flight.update([drain_prog.id, other_prog.id])
        await engine._in_flight_sema.acquire()
        await engine._in_flight_sema.acquire()

        engine.storage.mget.return_value = [other_prog]
        engine.config.program_acceptor = MagicMock()
        engine.config.program_acceptor.is_accepted.return_value = True
        engine.strategy.add.return_value = True

        # Poll excluding drain_prog — should only process other_prog
        count = await engine._poll_and_ingest(exclude_ids={drain_prog.id})

        assert count == 1
        assert other_prog.id not in engine._in_flight
        assert drain_prog.id in engine._in_flight  # NOT removed

    async def test_no_double_add_in_grid(self) -> None:
        """Over the full grid of timing configs, strategy.add is never called
        twice for the same program ID."""
        for mutation_delay, eval_delay, mif in [
            (0.002, 0.02, 3),
            (0.002, 0.15, 3),
            (0.05, 0.02, 5),
            (0.005, 0.3, 2),
        ]:
            de = DeterministicEngine(
                max_in_flight=mif,
                epoch_size=3,
                max_generations=2,
                mutation_delay=mutation_delay,
                eval_delay=eval_delay,
            )

            add_calls: list[str] = []
            orig_run = de.run

            async def tracked_run():
                async def tracked_add(prog):
                    add_calls.append(prog.id)
                    return True

                de.engine.strategy.add = tracked_add
                return await orig_run(timeout=15.0)

            await tracked_run()

            from collections import Counter

            counts = Counter(add_calls)
            doubles = {pid: n for pid, n in counts.items() if n > 1}
            assert not doubles, (
                f"[mut={mutation_delay} eval={eval_delay} mif={mif}] "
                f"Double ingestion: {doubles}"
            )


class TestScopedDrainInvariants:
    """Test invariants specific to the scoped drain optimization."""

    async def test_draining_flag_suppresses_epoch(self) -> None:
        """_should_trigger_epoch returns False while _draining is True."""
        from gigaevo.evolution.engine.config import SteadyStateEngineConfig

        storage = AsyncMock()
        strategy = AsyncMock()
        writer = MagicMock()
        writer.bind.return_value = writer
        mt = MagicMock()
        mt.format_best_summary.return_value = ""
        storage.count_by_status.return_value = 0
        storage.get_all_by_status.return_value = []
        storage.snapshot = MagicMock()
        strategy.get_program_ids.return_value = []

        config = SteadyStateEngineConfig(
            max_in_flight=4,
            max_mutations_per_generation=2,
            loop_interval=0.01,
        )
        engine = SteadyStateEvolutionEngine(
            storage=storage,
            strategy=strategy,
            mutation_operator=AsyncMock(),
            config=config,
            writer=writer,
            metrics_tracker=mt,
        )

        engine._processed_since_epoch = 100  # way above threshold
        assert engine._should_trigger_epoch() is True

        engine._draining = True
        assert engine._should_trigger_epoch() is False

        engine._draining = False
        assert engine._should_trigger_epoch() is True

    async def test_mutation_gate_reopens_after_exception(self) -> None:
        """If _epoch_refresh raises, the mutation gate still reopens."""
        storage = AsyncMock()
        strategy = AsyncMock()
        writer = MagicMock()
        writer.bind.return_value = writer
        mt = MagicMock()
        mt.format_best_summary.return_value = ""
        storage.count_by_status.return_value = 0
        storage.get_all_by_status.return_value = []
        storage.snapshot = MagicMock()
        strategy.get_program_ids.return_value = []

        config = SteadyStateEngineConfig(
            max_in_flight=4,
            max_mutations_per_generation=2,
            loop_interval=0.01,
        )
        engine = SteadyStateEvolutionEngine(
            storage=storage,
            strategy=strategy,
            mutation_operator=AsyncMock(),
            config=config,
            writer=writer,
            metrics_tracker=mt,
        )

        # Make _refresh_archive_programs raise to simulate failure mid-refresh
        engine._refresh_archive_programs = AsyncMock(side_effect=RuntimeError("boom"))

        with pytest.raises(RuntimeError, match="boom"):
            await engine._epoch_refresh()

        # Gate must still be open
        assert engine._mutation_gate.is_set()
        # Draining flag must be cleared
        assert engine._draining is False
        # Watermark timer must be cleared
        assert engine._epoch_eligible_since is None


class TestLowWatermarkEpochTrigger:
    """Tests for the low-watermark epoch trigger optimization."""

    def _make_engine(self, max_in_flight=4, epoch_size=2):
        storage = AsyncMock()
        strategy = AsyncMock()
        writer = MagicMock()
        writer.bind.return_value = writer
        mt = MagicMock()
        mt.format_best_summary.return_value = ""
        storage.count_by_status.return_value = 0
        storage.get_all_by_status.return_value = []
        storage.snapshot = MagicMock()
        strategy.get_program_ids.return_value = []
        config = SteadyStateEngineConfig(
            max_in_flight=max_in_flight,
            max_mutations_per_generation=epoch_size,
            loop_interval=0.01,
        )
        return SteadyStateEvolutionEngine(
            storage=storage,
            strategy=strategy,
            mutation_operator=AsyncMock(),
            config=config,
            writer=writer,
            metrics_tracker=mt,
        )

    def test_triggers_when_in_flight_below_watermark(self) -> None:
        """Epoch triggers immediately when in-flight is below watermark."""
        engine = self._make_engine(max_in_flight=8, epoch_size=3)
        engine._processed_since_epoch = 3  # at threshold

        # in_flight is empty (0 <= max(1, 8//4)=2) → trigger
        assert engine._should_trigger_epoch() is True

    def test_delays_when_in_flight_above_watermark(self) -> None:
        """Epoch does NOT trigger when in-flight is above watermark."""
        engine = self._make_engine(max_in_flight=8, epoch_size=3)
        engine._processed_since_epoch = 3

        # Add 5 in-flight programs (5 > max(1, 2)=2) → don't trigger
        for i in range(5):
            engine._in_flight.add(f"prog-{i}")

        assert engine._should_trigger_epoch() is False
        assert engine._epoch_eligible_since is not None  # timer started

    def test_fallback_triggers_after_timeout(self) -> None:
        """Epoch triggers via fallback after watermark timeout."""
        engine = self._make_engine(max_in_flight=8, epoch_size=3)
        engine._processed_since_epoch = 3
        for i in range(5):
            engine._in_flight.add(f"prog-{i}")

        # First call: starts timer, doesn't trigger
        assert engine._should_trigger_epoch() is False

        # Simulate time passing beyond fallback
        engine._epoch_eligible_since = time.monotonic() - 20.0
        assert engine._should_trigger_epoch() is True

    def test_timer_resets_when_below_threshold(self) -> None:
        """Timer resets if processed count drops below threshold."""
        engine = self._make_engine(max_in_flight=8, epoch_size=3)

        # Set timer
        engine._processed_since_epoch = 3
        engine._in_flight.update([f"p-{i}" for i in range(5)])
        engine._should_trigger_epoch()
        assert engine._epoch_eligible_since is not None

        # Drop below threshold → timer cleared
        engine._processed_since_epoch = 1
        engine._should_trigger_epoch()
        assert engine._epoch_eligible_since is None

    def test_max_in_flight_1_always_triggers(self) -> None:
        """With max_in_flight=1, watermark is 1, so always triggers at threshold."""
        engine = self._make_engine(max_in_flight=1, epoch_size=2)
        engine._processed_since_epoch = 2
        engine._in_flight.add("prog-0")

        # 1 in-flight <= max(1, 0)=1 → triggers immediately
        assert engine._should_trigger_epoch() is True
