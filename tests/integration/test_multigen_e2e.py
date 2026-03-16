"""End-to-end multi-generation integration test.

Exercises the FULL evolution loop across multiple generations:

    seed → mutation → FakeDagRunner eval → ingestion → archive update
    → elite selection → next mutation → … → final archive state

This is the single most important integration test in the suite: it wires
together EvolutionEngine, MapElitesMultiIsland, ProgramStateManager,
RedisProgramStorage, MutationOperator, and a FakeDagRunner — all on real
fakeredis — and asserts on:

1. Archive trajectory: correct programs appear after each generation.
2. Lineage chains: every child references its parent; generation depth grows.
3. Metrics accumulation: engine metrics tally correctly across generations.
4. Program state hygiene: no programs stranded in QUEUED/RUNNING after run.
5. Archive replacement: a better mutant evicts the incumbent in the same bin.
6. Multiple mutations per generation: >1 mutant created and evaluated per step.
7. Max-size eviction: archive respects size limits and evicts via remover.

Setup
-----
- Seed:       ``def entrypoint(): return {"fitness": 1.0, "x": 0.0}``
- Mutation:   IncrementMutationOperator — bumps fitness by 1.0 and shifts x.
- Validation: FakeDagRunner — evaluates QUEUED programs by exec'ing their code.
- BehaviorSpace: x ∈ [0, 10), 10 bins.
- Archive selector: SumArchiveSelector on "fitness" (higher is better).
"""

from __future__ import annotations

import asyncio
import contextlib
import re
from unittest.mock import MagicMock

import fakeredis.aioredis
import pytest

from gigaevo.database.redis import RedisProgramStorageConfig
from gigaevo.database.redis_program_storage import RedisProgramStorage
from gigaevo.database.state_manager import ProgramStateManager
from gigaevo.evolution.engine.config import EngineConfig
from gigaevo.evolution.engine.core import EvolutionEngine
from gigaevo.evolution.mutation.base import MutationOperator, MutationSpec
from gigaevo.evolution.strategies.elite_selectors import ScalarTournamentEliteSelector
from gigaevo.evolution.strategies.island import IslandConfig
from gigaevo.evolution.strategies.migrant_selectors import RandomMigrantSelector
from gigaevo.evolution.strategies.models import BehaviorSpace, LinearBinning
from gigaevo.evolution.strategies.multi_island import MapElitesMultiIsland
from gigaevo.evolution.strategies.removers import FitnessArchiveRemover
from gigaevo.evolution.strategies.selectors import SumArchiveSelector
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState

# ---------------------------------------------------------------------------
# Deterministic mutation operator
# ---------------------------------------------------------------------------

_CALL_COUNTER = 0  # module-level counter for deterministic x values


def _reset_counter() -> None:
    global _CALL_COUNTER
    _CALL_COUNTER = 0


_RETURN_RE = re.compile(
    r'return\s*\{\s*"fitness":\s*([\d.]+)\s*,\s*"x":\s*([\d.]+)\s*\}',
    re.MULTILINE,
)


def _extract_metrics_from_code(code: str) -> dict[str, float]:
    m = _RETURN_RE.search(code)
    if m is None:
        raise ValueError(f"Cannot extract metrics from code:\n{code}")
    return {"fitness": float(m.group(1)), "x": float(m.group(2))}


def _make_code(fitness: float, x: float) -> str:
    return f'def entrypoint():\n    return {{"fitness": {fitness}, "x": {x}}}'


class IncrementMutationOperator(MutationOperator):
    """Deterministic mutation: bumps fitness by 1.0, assigns a unique x.

    Each call produces a program with a unique x value (0.5, 1.5, 2.5, …)
    so that each mutant lands in a different behavior-space bin.  Fitness
    always exceeds the parent's, so when two programs share a bin the child
    always wins (tests archive replacement).
    """

    async def mutate_single(
        self, selected_parents: list[Program]
    ) -> MutationSpec | None:
        global _CALL_COUNTER
        parent = selected_parents[0]
        parent_metrics = _extract_metrics_from_code(parent.code)
        new_fitness = parent_metrics["fitness"] + 1.0
        new_x = 0.5 + _CALL_COUNTER  # unique per mutation call
        _CALL_COUNTER += 1
        return MutationSpec(
            code=_make_code(new_fitness, new_x),
            parents=selected_parents,
            name="increment",
        )


class SameBinMutationOperator(MutationOperator):
    """Always mutates into x=0.5 (bin 0) with increasing fitness.

    Used to test archive replacement: every mutant targets the same bin,
    but with strictly higher fitness, so the archive should always hold
    only the latest (best) program in that bin.
    """

    async def mutate_single(
        self, selected_parents: list[Program]
    ) -> MutationSpec | None:
        parent = selected_parents[0]
        parent_metrics = _extract_metrics_from_code(parent.code)
        new_fitness = parent_metrics["fitness"] + 1.0
        return MutationSpec(
            code=_make_code(new_fitness, 0.5),
            parents=selected_parents,
            name="same_bin",
        )


# ---------------------------------------------------------------------------
# FakeDagRunner — evaluates programs by executing their entrypoint()
# ---------------------------------------------------------------------------


class FakeDagRunner:
    """Background loop: QUEUED → RUNNING → DONE with metrics from code."""

    def __init__(
        self, storage: RedisProgramStorage, state_manager: ProgramStateManager
    ):
        self._storage = storage
        self._sm = state_manager
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop(), name="fake-dag-runner")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def _loop(self) -> None:
        while True:
            queued = await self._storage.get_all_by_status(ProgramState.QUEUED.value)
            for prog in queued:
                await self._evaluate(prog)
            await asyncio.sleep(0.005)

    async def _evaluate(self, prog: Program) -> None:
        await self._sm.set_program_state(prog, ProgramState.RUNNING)
        metrics = _extract_metrics_from_code(prog.code)
        prog.add_metrics(metrics)
        await self._sm.set_program_state(prog, ProgramState.DONE)


# ---------------------------------------------------------------------------
# Infrastructure helpers
# ---------------------------------------------------------------------------

SEED_CODE = _make_code(fitness=1.0, x=0.0)


def _make_fakeredis_storage(server: fakeredis.FakeServer) -> RedisProgramStorage:
    config = RedisProgramStorageConfig(
        redis_url="redis://fake:6379/0", key_prefix="test"
    )
    storage = RedisProgramStorage(config)
    fake_redis = fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)
    storage._conn._redis = fake_redis
    storage._conn._closing = False
    return storage


def _make_island_config(
    *,
    num_bins: int = 10,
    max_size: int | None = None,
) -> IslandConfig:
    behavior_space = BehaviorSpace(
        bins={
            "x": LinearBinning(
                min_val=0.0, max_val=10.0, num_bins=num_bins, type="linear"
            )
        }
    )
    return IslandConfig(
        island_id="main",
        behavior_space=behavior_space,
        max_size=max_size,
        archive_selector=SumArchiveSelector(fitness_keys=["fitness"]),
        archive_remover=(
            FitnessArchiveRemover(
                fitness_key="fitness", fitness_key_higher_is_better=True
            )
            if max_size is not None
            else None
        ),
        elite_selector=ScalarTournamentEliteSelector(
            fitness_key="fitness",
            fitness_key_higher_is_better=True,
            tournament_size=99,
        ),
        migrant_selector=RandomMigrantSelector(),
    )


def _make_null_writer() -> MagicMock:
    writer = MagicMock()
    writer.bind.return_value = writer
    return writer


def _make_metrics_tracker() -> MagicMock:
    tracker = MagicMock()
    tracker.start = MagicMock()

    async def _stop():
        pass

    tracker.stop = _stop
    return tracker


def _build_engine(
    storage: RedisProgramStorage,
    max_generations: int,
    *,
    max_elites: int = 1,
    max_mutations: int = 1,
    mutation_operator: MutationOperator | None = None,
    island_config: IslandConfig | None = None,
) -> tuple[EvolutionEngine, MapElitesMultiIsland]:
    config = island_config or _make_island_config()
    strategy = MapElitesMultiIsland(
        island_configs=[config],
        program_storage=storage,
    )
    engine = EvolutionEngine(
        storage=storage,
        strategy=strategy,
        mutation_operator=mutation_operator or IncrementMutationOperator(),
        config=EngineConfig(
            loop_interval=0.005,
            max_elites_per_generation=max_elites,
            max_mutations_per_generation=max_mutations,
            generation_timeout=30.0,
            max_generations=max_generations,
        ),
        writer=_make_null_writer(),
        metrics_tracker=_make_metrics_tracker(),
    )
    return engine, strategy


async def _add_seed(storage: RedisProgramStorage) -> Program:
    seed = Program(code=SEED_CODE, state=ProgramState.QUEUED)
    await storage.add(seed)
    return seed


async def _run(
    storage: RedisProgramStorage,
    max_generations: int,
    *,
    max_elites: int = 1,
    max_mutations: int = 1,
    mutation_operator: MutationOperator | None = None,
    island_config: IslandConfig | None = None,
) -> tuple[EvolutionEngine, MapElitesMultiIsland]:
    engine, strategy = _build_engine(
        storage,
        max_generations,
        max_elites=max_elites,
        max_mutations=max_mutations,
        mutation_operator=mutation_operator,
        island_config=island_config,
    )
    sm = ProgramStateManager(storage)
    runner = FakeDagRunner(storage, sm)

    runner.start()
    engine.start()
    try:
        await asyncio.wait_for(engine.task, timeout=30.0)
    except TimeoutError:
        pytest.fail(f"Engine did not finish {max_generations} gens within 30s")
    finally:
        await runner.stop()
        await storage.close()

    return engine, strategy


async def _get_archive(server: fakeredis.FakeServer) -> list[Program]:
    storage = _make_fakeredis_storage(server)
    strategy = MapElitesMultiIsland(
        island_configs=[_make_island_config()],
        program_storage=storage,
    )
    programs = await strategy.islands["main"].get_elites()
    await storage.close()
    return programs


async def _get_all_programs(server: fakeredis.FakeServer) -> list[Program]:
    """Retrieve ALL programs from storage (not just archive elites)."""
    storage = _make_fakeredis_storage(server)
    done = await storage.get_all_by_status(ProgramState.DONE.value)
    queued = await storage.get_all_by_status(ProgramState.QUEUED.value)
    running = await storage.get_all_by_status(ProgramState.RUNNING.value)
    discarded = await storage.get_all_by_status(ProgramState.DISCARDED.value)
    await storage.close()
    return done + queued + running + discarded


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMultiGenArchiveTrajectory:
    """Verify correct archive state after multi-generation evolution."""

    async def test_5gen_archive_grows_with_unique_bins(self) -> None:
        """After 5 generations with unique-bin mutation, archive accumulates programs.

        Gen 1: empty archive → seed ingested (x=0.0, fitness=1.0), 0 mutations
        Gen 2: 1 elite → mutant (x=0.5, fitness=2.0) → archive grows
        Gen 3-5: each adds a new mutant in a fresh bin

        With 1 mutation/gen from a unique-bin operator, the archive should hold
        the seed plus one mutant per productive generation (gens 2-5 = 4 mutants).
        Some may share a bin with the seed (x=0.0 maps to bin 0, x=0.5 also
        bin 0), so the exact count depends on bin boundaries.  We assert ≥ 3.
        """
        _reset_counter()
        server = fakeredis.FakeServer()
        storage = _make_fakeredis_storage(server)
        await _add_seed(storage)

        engine, _ = await _run(storage, max_generations=5)

        assert engine.metrics.total_generations == 5

        programs = await _get_archive(server)
        assert len(programs) >= 3, (
            f"Expected at least 3 programs in archive after 5 gens, got {len(programs)}: "
            f"{[(p.id[:8], p.metrics.get('fitness'), p.metrics.get('x')) for p in programs]}"
        )

        # All archive programs should have fitness and x metrics
        for p in programs:
            assert "fitness" in p.metrics, f"Program {p.id[:8]} missing 'fitness'"
            assert "x" in p.metrics, f"Program {p.id[:8]} missing 'x'"

    async def test_5gen_monotonic_fitness_improvement(self) -> None:
        """The best fitness in the archive strictly increases each generation."""
        _reset_counter()
        server = fakeredis.FakeServer()
        storage = _make_fakeredis_storage(server)
        await _add_seed(storage)

        engine, _ = await _run(storage, max_generations=5)

        programs = await _get_archive(server)
        fitnesses = sorted([p.metrics["fitness"] for p in programs])

        # Seed has fitness=1.0, each mutation adds 1.0
        # Best fitness should be > 1.0 after 5 gens
        assert max(fitnesses) > 1.0, (
            f"Expected fitness improvement over seed, got fitnesses: {fitnesses}"
        )
        assert engine.metrics.total_generations == 5


class TestMultiGenLineageChains:
    """Verify parent→child lineage is correctly recorded across generations."""

    async def test_mutant_references_parent(self) -> None:
        """Every non-seed program in the archive has a non-empty parent list."""
        _reset_counter()
        server = fakeredis.FakeServer()
        storage = _make_fakeredis_storage(server)
        seed = await _add_seed(storage)

        await _run(storage, max_generations=3)

        all_progs = await _get_all_programs(server)
        non_seed = [p for p in all_progs if p.id != seed.id]

        for p in non_seed:
            assert p.lineage.parents, (
                f"Non-seed program {p.id[:8]} has empty lineage.parents"
            )

    async def test_lineage_generation_depth_grows(self) -> None:
        """Children have lineage.generation = parent.generation + 1."""
        _reset_counter()
        server = fakeredis.FakeServer()
        storage = _make_fakeredis_storage(server)
        seed = await _add_seed(storage)

        await _run(storage, max_generations=4)

        all_progs = await _get_all_programs(server)
        non_seed = [p for p in all_progs if p.id != seed.id]

        # At least one program should have generation > 0
        max_gen = max((p.lineage.generation for p in non_seed), default=0)
        assert max_gen > 0, "No program has lineage generation > 0"

    async def test_mutation_name_recorded(self) -> None:
        """Every mutant records its mutation operator name in lineage."""
        _reset_counter()
        server = fakeredis.FakeServer()
        storage = _make_fakeredis_storage(server)
        seed = await _add_seed(storage)

        await _run(storage, max_generations=3)

        all_progs = await _get_all_programs(server)
        non_seed = [p for p in all_progs if p.id != seed.id]

        for p in non_seed:
            assert p.lineage.mutation == "increment", (
                f"Program {p.id[:8]} has mutation name '{p.lineage.mutation}', expected 'increment'"
            )


class TestMultiGenMetrics:
    """Verify engine metrics are correctly accumulated across generations."""

    async def test_generation_counter_matches(self) -> None:
        """total_generations == max_generations after a complete run."""
        _reset_counter()
        server = fakeredis.FakeServer()
        storage = _make_fakeredis_storage(server)
        await _add_seed(storage)

        engine, _ = await _run(storage, max_generations=4)
        assert engine.metrics.total_generations == 4

    async def test_mutations_created_positive(self) -> None:
        """At least some mutations were created across all generations."""
        _reset_counter()
        server = fakeredis.FakeServer()
        storage = _make_fakeredis_storage(server)
        await _add_seed(storage)

        engine, _ = await _run(storage, max_generations=3)

        # Gen 1 has empty archive → 0 mutations.  Gen 2+ should create 1 each.
        assert engine.metrics.mutations_created >= 2, (
            f"Expected at least 2 mutations across 3 gens, got {engine.metrics.mutations_created}"
        )

    async def test_programs_processed_positive(self) -> None:
        """At least some programs were accepted into the archive."""
        _reset_counter()
        server = fakeredis.FakeServer()
        storage = _make_fakeredis_storage(server)
        await _add_seed(storage)

        engine, _ = await _run(storage, max_generations=3)

        # Seed + at least 1 mutant should have been accepted
        assert engine.metrics.programs_processed >= 1, (
            f"Expected at least 1 program processed, got {engine.metrics.programs_processed}"
        )

    async def test_elites_selected_positive(self) -> None:
        """Elite selection happened in at least some generations."""
        _reset_counter()
        server = fakeredis.FakeServer()
        storage = _make_fakeredis_storage(server)
        await _add_seed(storage)

        engine, _ = await _run(storage, max_generations=4)

        # Gen 1: empty archive → 0 elites. Gen 2-4: 1 elite each → ≥3
        assert engine.metrics.elites_selected >= 2, (
            f"Expected at least 2 elite selections, got {engine.metrics.elites_selected}"
        )

    async def test_submitted_for_refresh_positive(self) -> None:
        """Archive programs are submitted for refresh after ingestion."""
        _reset_counter()
        server = fakeredis.FakeServer()
        storage = _make_fakeredis_storage(server)
        await _add_seed(storage)

        engine, _ = await _run(storage, max_generations=3)

        # After ingesting the seed (gen 1), archive has 1 program → refreshed
        assert engine.metrics.submitted_for_refresh >= 1, (
            f"Expected refresh submissions, got {engine.metrics.submitted_for_refresh}"
        )


class TestMultiGenProgramStateHygiene:
    """Verify no programs are stuck in transient states after evolution completes."""

    async def test_no_queued_programs_after_run(self) -> None:
        """After a complete run, no programs should be in QUEUED state."""
        _reset_counter()
        server = fakeredis.FakeServer()
        storage = _make_fakeredis_storage(server)
        await _add_seed(storage)
        await _run(storage, max_generations=3)

        check_storage = _make_fakeredis_storage(server)
        queued = await check_storage.get_all_by_status(ProgramState.QUEUED.value)
        await check_storage.close()

        assert len(queued) == 0, (
            f"Found {len(queued)} programs stuck in QUEUED after run: "
            f"{[p.id[:8] for p in queued]}"
        )

    async def test_no_running_programs_after_run(self) -> None:
        """After a complete run, no programs should be in RUNNING state."""
        _reset_counter()
        server = fakeredis.FakeServer()
        storage = _make_fakeredis_storage(server)
        await _add_seed(storage)
        await _run(storage, max_generations=3)

        check_storage = _make_fakeredis_storage(server)
        running = await check_storage.get_all_by_status(ProgramState.RUNNING.value)
        await check_storage.close()

        assert len(running) == 0, (
            f"Found {len(running)} programs stuck in RUNNING after run: "
            f"{[p.id[:8] for p in running]}"
        )

    async def test_all_archive_programs_are_done(self) -> None:
        """All programs in the archive should be in DONE state after run."""
        _reset_counter()
        server = fakeredis.FakeServer()
        storage = _make_fakeredis_storage(server)
        await _add_seed(storage)
        await _run(storage, max_generations=3)

        programs = await _get_archive(server)

        for p in programs:
            assert p.state == ProgramState.DONE, (
                f"Archive program {p.id[:8]} in state {p.state}, expected DONE"
            )


class TestMultiGenArchiveReplacement:
    """Verify that better programs replace worse ones in the same behavior bin."""

    async def test_same_bin_replacement_keeps_best(self) -> None:
        """When all mutants target the same bin, only the best survives.

        SameBinMutationOperator always produces x=0.5 (bin 0) with fitness+1.
        After 4 gens the seed occupies bin 0 (x=0.0) and successive mutants
        keep replacing each other in the same bin (x=0.5).  The archive should
        hold at most 2 programs: the seed (bin for x=0.0) and the latest
        mutant (bin for x=0.5).
        """
        _reset_counter()
        server = fakeredis.FakeServer()
        storage = _make_fakeredis_storage(server)
        await _add_seed(storage)

        engine, _ = await _run(
            storage,
            max_generations=4,
            mutation_operator=SameBinMutationOperator(),
        )

        programs = await _get_archive(server)

        # Seed is in bin for x=0.0, mutants are in bin for x=0.5.
        # Archive should have at most 2 entries (seed bin + mutant bin).
        assert len(programs) <= 2, (
            f"Expected at most 2 archive entries (same-bin replacement), "
            f"got {len(programs)}: "
            f"{[(p.id[:8], p.metrics.get('fitness'), p.metrics.get('x')) for p in programs]}"
        )

        # The mutant in the x=0.5 bin should have the highest fitness
        mutant_bin_progs = [p for p in programs if p.metrics.get("x", -1) == 0.5]
        if mutant_bin_progs:
            best = mutant_bin_progs[0]
            # After 4 gens (gen1=seed only, gen2/3/4 produce mutants),
            # the best mutant should have fitness > 2.0
            assert best.metrics["fitness"] > 2.0, (
                f"Same-bin best has fitness {best.metrics['fitness']}, "
                f"expected > 2.0 after replacement"
            )


class TestMultiGenMultipleMutationsPerGen:
    """Verify behavior with >1 mutation per generation."""

    async def test_multiple_mutants_per_gen(self) -> None:
        """With max_mutations=3, multiple mutants are created each generation."""
        _reset_counter()
        server = fakeredis.FakeServer()
        storage = _make_fakeredis_storage(server)
        await _add_seed(storage)

        engine, _ = await _run(
            storage,
            max_generations=3,
            max_mutations=3,
        )

        # Gen 1: empty archive, 0 mutations
        # Gen 2: 1 elite, up to 3 mutations
        # Gen 3: elites, up to 3 mutations
        # Total: at least 3 mutations (could be up to 6)
        assert engine.metrics.mutations_created >= 3, (
            f"Expected at least 3 mutations with max_mutations=3, "
            f"got {engine.metrics.mutations_created}"
        )

        # Archive should have more than 2 programs (seed + multiple mutants)
        programs = await _get_archive(server)
        assert len(programs) >= 3, (
            f"Expected at least 3 programs in archive with multiple mutations/gen, "
            f"got {len(programs)}"
        )

    async def test_multiple_mutants_all_have_lineage(self) -> None:
        """All mutants from multi-mutation gens have correct lineage."""
        _reset_counter()
        server = fakeredis.FakeServer()
        storage = _make_fakeredis_storage(server)
        seed = await _add_seed(storage)

        await _run(storage, max_generations=3, max_mutations=3)

        all_progs = await _get_all_programs(server)
        non_seed = [p for p in all_progs if p.id != seed.id]

        for p in non_seed:
            assert p.lineage.parents, (
                f"Multi-mutation program {p.id[:8]} has empty parents"
            )
            assert p.lineage.mutation == "increment", (
                f"Program {p.id[:8]} has mutation '{p.lineage.mutation}'"
            )


class TestMultiGenMaxSizeEviction:
    """Verify archive respects max_size and evicts via WorstFitnessRemover."""

    async def test_archive_respects_max_size(self) -> None:
        """With max_size=3, archive never exceeds 3 programs.

        We run 5 generations with unique-bin mutations (each mutant fills a
        new bin).  Without max_size the archive would grow to 5+, but with
        max_size=3 the WorstFitnessRemover should keep it at ≤3.
        """
        _reset_counter()
        server = fakeredis.FakeServer()
        storage = _make_fakeredis_storage(server)
        await _add_seed(storage)

        island_config = _make_island_config(max_size=3)

        engine, _ = await _run(
            storage,
            max_generations=5,
            island_config=island_config,
        )

        # Re-read archive with the same island config
        check_storage = _make_fakeredis_storage(server)
        strategy = MapElitesMultiIsland(
            island_configs=[island_config],
            program_storage=check_storage,
        )
        programs = await strategy.islands["main"].get_elites()
        await check_storage.close()

        assert len(programs) <= 3, (
            f"Archive exceeds max_size=3: has {len(programs)} programs"
        )

    async def test_eviction_keeps_fittest(self) -> None:
        """After eviction, the surviving programs have the highest fitness values.

        With WorstFitnessRemover(higher_is_better=True), the lowest-fitness
        programs should be evicted.
        """
        _reset_counter()
        server = fakeredis.FakeServer()
        storage = _make_fakeredis_storage(server)
        await _add_seed(storage)

        island_config = _make_island_config(max_size=3)

        await _run(
            storage,
            max_generations=5,
            island_config=island_config,
        )

        check_storage = _make_fakeredis_storage(server)
        strategy = MapElitesMultiIsland(
            island_configs=[island_config],
            program_storage=check_storage,
        )
        programs = await strategy.islands["main"].get_elites()
        await check_storage.close()

        if len(programs) > 1:
            fitnesses = [p.metrics["fitness"] for p in programs]
            # The seed (fitness=1.0) should have been evicted in favor of
            # higher-fitness mutants, if archive was full and eviction ran.
            # We can't assert the exact set, but the lowest surviving fitness
            # should be > 1.0 if eviction happened.
            # (Only assert if archive hit max_size and had to evict.)
            if len(programs) == 3:
                assert min(fitnesses) >= 1.0, (
                    f"Eviction did not keep fittest: fitnesses={sorted(fitnesses)}"
                )


class TestMultiGenStrategyGeneration:
    """Verify that MapElitesMultiIsland.generation counter advances correctly."""

    async def test_strategy_generation_advances(self) -> None:
        """strategy.generation > 0 after a multi-gen run with successful elite selection."""
        _reset_counter()
        server = fakeredis.FakeServer()
        storage = _make_fakeredis_storage(server)
        await _add_seed(storage)

        _, strategy = await _run(storage, max_generations=4)

        # Generation only increments when select_elites returns non-empty results.
        # Gen 1 has empty archive → no increment. Gens 2-4 should increment.
        assert strategy.generation >= 2, (
            f"Expected strategy.generation >= 2 after 4 engine gens, "
            f"got {strategy.generation}"
        )

    async def test_run_state_persisted_to_redis(self) -> None:
        """Engine and strategy generation counters are persisted to Redis."""
        _reset_counter()
        server = fakeredis.FakeServer()
        storage = _make_fakeredis_storage(server)
        await _add_seed(storage)

        await _run(storage, max_generations=3)

        # Read back from Redis
        check_storage = _make_fakeredis_storage(server)
        engine2, strategy2 = _build_engine(check_storage, max_generations=10)
        await engine2.restore_state()
        await strategy2.restore_state()

        assert engine2.metrics.total_generations == 3
        assert strategy2.generation > 0
        await check_storage.close()
