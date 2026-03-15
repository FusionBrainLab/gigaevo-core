"""End-to-end resume correctness test.

Key property: a split run (gens 1..N, then resume for gens N+1..M) produces
the same archive state as one contiguous run (gens 1..M), given deterministic
mutation and deterministic validation.

Setup
-----
- Seed:       ``def entrypoint(): return 1024.0``  (QUEUED, added before engine starts)
- Mutation:   HalvingMutationOperator — ``return N`` → ``return N/2.0``
- Validation: FakeDagRunner — evaluates QUEUED programs immediately:
                fitness = -value           (lower value → higher fitness → better)
                x = log2(1024 / value)     (depth of halving: 0, 1, 2, 3, 4 ...)
- BehaviorSpace: x ∈ [0, 5), 5 bins — each halving depth gets its own bin.
                 Programs NEVER replace each other; archive accumulates.
- EliteSelector: ScalarTournamentEliteSelector (tournament = full pool) — always
                 picks the highest-fitness (most-halved) program deterministically.
- max_elites=1, max_mutations=1

Expected trajectory (each engine step = 1 generation):
  Gen 1: archive empty → seed(1024, x=0) ingested   → archive = {1024}
  Gen 2: best = seed(1024) → mutate → 512 (x=1)     → archive = {1024, 512}
  Gen 3: best = 512        → mutate → 256 (x=2)      → archive = {1024, 512, 256}
  Gen 4: best = 256        → mutate → 128 (x=3)      → archive = {1024, 512, 256, 128}
  Gen 5: best = 128        → mutate →  64 (x=4)      → archive = {1024, 512, 256, 128, 64}

After 5 gens: archive has 5 programs.  The same 5 programs must appear after
a split 3+2 run on the same Redis DB — this is the core assertion.

Sub-tests
---------
1. Contiguous sanity:    5 gens → archive = {1024, 512, 256, 128, 64}
2. Split 3+2 equality:   same 5-program archive as contiguous (core resume test)
3. Counter persistence:  engine + strategy generation counters restored, not reset
4. Stranded recovery:    RUNNING programs are reset to QUEUED on resume
"""

from __future__ import annotations

import asyncio
import contextlib
import math
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
from gigaevo.evolution.strategies.selectors import SumArchiveSelector
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED_VALUE = 1024.0
SEED_CODE = f"def entrypoint():\n    return {SEED_VALUE}"

_VALUE_RE = re.compile(r"return\s+([\d.]+)", re.MULTILINE)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_value(code: str) -> float:
    m = _VALUE_RE.search(code)
    if m is None:
        raise ValueError(f"Cannot extract return value from code:\n{code}")
    return float(m.group(1))


def _halved_code(value: float) -> str:
    return f"def entrypoint():\n    return {value / 2.0}"


def _compute_metrics(code: str) -> dict[str, float]:
    """Deterministic validation.

    fitness = -value  (higher = better → lower value wins)
    x = log2(SEED_VALUE / value)  (halving depth: 0 for seed, 1 for first child …)

    Each halving depth is mapped to its own bin, so programs NEVER evict each
    other — the archive accumulates one program per generation.
    """
    value = _extract_value(code)
    depth = math.log2(SEED_VALUE / value) if value > 0 else 0.0
    return {"fitness": -value, "x": depth}


def _archive_values(programs: list[Program]) -> set[float]:
    """Return the set of return-values present in *programs*."""
    return {_extract_value(p.code) for p in programs}


# ---------------------------------------------------------------------------
# Deterministic mutation operator: always halves the parent's return value
# ---------------------------------------------------------------------------


class HalvingMutationOperator(MutationOperator):
    async def mutate_single(
        self, selected_parents: list[Program]
    ) -> MutationSpec | None:
        parent = selected_parents[0]
        value = _extract_value(parent.code)
        return MutationSpec(
            code=_halved_code(value),
            parents=selected_parents,
            name="halving",
        )


# ---------------------------------------------------------------------------
# Fake DAG runner: background task that processes QUEUED → RUNNING → DONE
# ---------------------------------------------------------------------------


class FakeDagRunner:
    """Simulates exec_runner workers: evaluates QUEUED programs immediately."""

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
        # QUEUED → RUNNING
        await self._sm.set_program_state(prog, ProgramState.RUNNING)
        # Compute deterministic metrics and add them to the program object.
        # New metric keys win the merge even when prog.atomic_counter < stored
        # counter, because _merge_dict_by_prog_ts always adds NEW keys from
        # the incoming side regardless of counter comparison.
        prog.add_metrics(_compute_metrics(prog.code))
        # RUNNING → DONE (metrics persisted via merge)
        await self._sm.set_program_state(prog, ProgramState.DONE)


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def _make_fakeredis_storage(server: fakeredis.FakeServer) -> RedisProgramStorage:
    config = RedisProgramStorageConfig(
        redis_url="redis://fake:6379/0", key_prefix="test"
    )
    storage = RedisProgramStorage(config)
    fake_redis = fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)
    storage._conn._redis = fake_redis
    storage._conn._closing = False
    return storage


def _make_island_config() -> IslandConfig:
    """5 bins for halving depths 0-4; programs never replace each other."""
    behavior_space = BehaviorSpace(
        bins={"x": LinearBinning(min_val=0.0, max_val=5.0, num_bins=5, type="linear")}
    )
    return IslandConfig(
        island_id="test",
        behavior_space=behavior_space,
        max_size=None,
        archive_selector=SumArchiveSelector(fitness_keys=["fitness"]),
        archive_remover=None,
        # Always pick the highest-fitness (most-halved) program → deterministic.
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


def _build_engine(
    storage: RedisProgramStorage, max_generations: int
) -> tuple[EvolutionEngine, MapElitesMultiIsland]:
    strategy = MapElitesMultiIsland(
        island_configs=[_make_island_config()],
        program_storage=storage,
    )
    tracker = MagicMock()
    tracker.start = MagicMock()

    async def _stop():
        pass

    tracker.stop = _stop

    engine = EvolutionEngine(
        storage=storage,
        strategy=strategy,
        mutation_operator=HalvingMutationOperator(),
        config=EngineConfig(
            loop_interval=0.005,
            max_elites_per_generation=1,
            max_mutations_per_generation=1,
            generation_timeout=30.0,
            max_generations=max_generations,
        ),
        writer=_make_null_writer(),
        metrics_tracker=tracker,
    )
    return engine, strategy


async def _add_seed(storage: RedisProgramStorage) -> Program:
    """Add the seed program in QUEUED state — FakeDagRunner will evaluate it."""
    seed = Program(code=SEED_CODE, state=ProgramState.QUEUED)
    await storage.add(seed)
    return seed


async def _run(storage: RedisProgramStorage, max_generations: int) -> EvolutionEngine:
    """Run a fresh engine on *storage* for *max_generations* gens, then stop."""
    engine, _ = _build_engine(storage, max_generations)
    state_manager = ProgramStateManager(storage)
    dag_runner = FakeDagRunner(storage, state_manager)

    dag_runner.start()
    engine.start()
    try:
        await asyncio.wait_for(engine.task, timeout=30.0)
    except asyncio.TimeoutError:
        pytest.fail(f"Engine did not finish {max_generations} gens within 30 s")
    finally:
        await dag_runner.stop()
        await storage.close()

    return engine


async def _resume(storage: RedisProgramStorage, total_cap: int) -> EvolutionEngine:
    """Resume on existing *storage* up to *total_cap* total generations."""
    engine, strategy = _build_engine(storage, total_cap)
    await engine.restore_state()
    await engine.strategy.restore_state()

    state_manager = ProgramStateManager(storage)
    dag_runner = FakeDagRunner(storage, state_manager)

    dag_runner.start()
    engine.start()
    try:
        await asyncio.wait_for(engine.task, timeout=30.0)
    except asyncio.TimeoutError:
        pytest.fail("Resumed engine did not finish within 30 s")
    finally:
        await dag_runner.stop()
        await storage.close()

    return engine


async def _get_archive_programs(server: fakeredis.FakeServer) -> list[Program]:
    """Read all programs currently in the island archive from Redis."""
    storage = _make_fakeredis_storage(server)
    strategy = MapElitesMultiIsland(
        island_configs=[_make_island_config()],
        program_storage=storage,
    )
    programs = await strategy.islands["test"].get_elites()
    await storage.close()
    return programs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestResumeEndToEnd:
    """Verify that a split run == contiguous run after the same total generations."""

    async def test_contiguous_builds_archive_with_multiple_programs(self) -> None:
        """5 gens of halving from 1024 fills 5 distinct bins: values {1024,512,256,128,64}."""
        server = fakeredis.FakeServer()
        storage = _make_fakeredis_storage(server)
        await _add_seed(storage)

        engine = await _run(storage, max_generations=5)
        assert engine.metrics.total_generations == 5

        programs = await _get_archive_programs(server)
        values = _archive_values(programs)

        assert values == {1024.0, 512.0, 256.0, 128.0, 64.0}, (
            f"Expected 5 distinct programs in archive, got values: {sorted(values)}"
        )

    async def test_split_3_2_matches_contiguous_5(self) -> None:
        """Split run (3 + 2 gens) produces the identical 5-program archive as 5 contiguous gens."""
        # --- Contiguous run ---
        server_c = fakeredis.FakeServer()
        storage_c = _make_fakeredis_storage(server_c)
        await _add_seed(storage_c)
        engine_c = await _run(storage_c, max_generations=5)
        assert engine_c.metrics.total_generations == 5

        programs_c = await _get_archive_programs(server_c)
        values_c = _archive_values(programs_c)

        # --- Split run: part 1 (3 gens) ---
        server_s = fakeredis.FakeServer()
        storage_s1 = _make_fakeredis_storage(server_s)
        await _add_seed(storage_s1)
        engine_p1 = await _run(storage_s1, max_generations=3)
        assert engine_p1.metrics.total_generations == 3

        programs_after_p1 = await _get_archive_programs(server_s)
        assert len(programs_after_p1) == 3, (
            f"Expected 3 programs after part 1, got {len(programs_after_p1)}"
        )

        # --- Split run: part 2 (resume → 2 more gens, total cap = 5) ---
        storage_s2 = _make_fakeredis_storage(server_s)
        engine_p2 = await _resume(storage_s2, total_cap=5)
        assert engine_p2.metrics.total_generations == 5

        programs_s = await _get_archive_programs(server_s)
        values_s = _archive_values(programs_s)

        # Core assertion: identical archive regardless of how we got there
        assert values_s == values_c, (
            f"Archive mismatch after split run.\n"
            f"  Contiguous: {sorted(values_c)}\n"
            f"  Split 3+2:  {sorted(values_s)}"
        )

    async def test_part1_programs_all_survive_resume(self) -> None:
        """Every program in the archive before a stop is still there after resume.

        This directly tests that resume does not lose or corrupt archive state.
        With 5 bins and no eviction, all 3 programs from part 1 must remain
        in the archive after part 2 adds 2 more programs.
        """
        server = fakeredis.FakeServer()

        # Part 1: 3 gens → archive = {1024, 512, 256}
        storage1 = _make_fakeredis_storage(server)
        await _add_seed(storage1)
        await _run(storage1, max_generations=3)

        programs_before = await _get_archive_programs(server)
        ids_before = {p.id for p in programs_before}
        assert len(ids_before) == 3

        # Part 2: resume → 2 more gens → archive = {1024, 512, 256, 128, 64}
        storage2 = _make_fakeredis_storage(server)
        await _resume(storage2, total_cap=5)

        programs_after = await _get_archive_programs(server)
        ids_after = {p.id for p in programs_after}

        # All 3 original program IDs must still be present
        lost = ids_before - ids_after
        assert not lost, (
            f"{len(lost)} archive program(s) were lost during resume: {lost}"
        )
        assert len(ids_after) == 5, (
            f"Expected 5 programs after resume, got {len(ids_after)}"
        )

    async def test_generation_counter_survives_resume(self) -> None:
        """Engine and strategy generation counters are restored from Redis, not reset."""
        server = fakeredis.FakeServer()

        # Part 1: run 3 gens
        storage1 = _make_fakeredis_storage(server)
        await _add_seed(storage1)
        engine1 = await _run(storage1, max_generations=3)
        assert engine1.metrics.total_generations == 3

        # Inspect restored counters without running any more gens
        storage2 = _make_fakeredis_storage(server)
        engine2, strategy2 = _build_engine(storage2, max_generations=10)
        await engine2.restore_state()
        await engine2.strategy.restore_state()

        assert engine2.metrics.total_generations == 3, (
            f"Engine counter must be restored to 3, got {engine2.metrics.total_generations}"
        )
        # Strategy generation counts only gens where select_elites returned non-empty
        # results (gen 1 has empty archive). It is strictly < engine total_generations
        # and strictly > 0 after a successful run.
        assert 0 < engine2.strategy.generation < engine2.metrics.total_generations, (
            f"Strategy generation {engine2.strategy.generation} out of expected range "
            f"(0, {engine2.metrics.total_generations})"
        )
        await storage2.close()

    async def test_resume_at_max_generations_takes_zero_steps(self) -> None:
        """Resuming when total_generations already equals max_generations is a no-op.

        The engine must exit immediately without running additional mutations,
        leaving the archive unchanged.
        """
        server = fakeredis.FakeServer()

        # Part 1: run exactly 5 gens
        storage1 = _make_fakeredis_storage(server)
        await _add_seed(storage1)
        engine1 = await _run(storage1, max_generations=5)
        assert engine1.metrics.total_generations == 5

        values_before = _archive_values(await _get_archive_programs(server))

        # Part 2: resume with the same cap — engine should do nothing
        storage2 = _make_fakeredis_storage(server)
        engine2 = await _resume(storage2, total_cap=5)

        # Counter must remain at 5 (zero additional steps taken)
        assert engine2.metrics.total_generations == 5

        # Archive must be identical to what it was before the no-op resume
        values_after = _archive_values(await _get_archive_programs(server))
        assert values_after == values_before

    async def test_stranded_running_programs_recovered_on_resume(self) -> None:
        """Programs stuck in RUNNING state (crash mid-eval) are reset to QUEUED on resume."""
        server = fakeredis.FakeServer()
        storage = _make_fakeredis_storage(server)

        # Simulate two programs mid-evaluation when the engine crashed
        for val in [512.0, 256.0]:
            prog = Program(
                code=f"def entrypoint(): return {val}", state=ProgramState.QUEUED
            )
            await storage.add(prog)
            sm = ProgramStateManager(storage)
            await sm.set_program_state(prog, ProgramState.RUNNING)

        assert await storage.count_by_status(ProgramState.RUNNING.value) == 2
        assert await storage.count_by_status(ProgramState.QUEUED.value) == 0

        recovered = await storage.recover_stranded_programs()

        assert recovered == 2
        assert await storage.count_by_status(ProgramState.RUNNING.value) == 0
        assert await storage.count_by_status(ProgramState.QUEUED.value) == 2

        await storage.close()
