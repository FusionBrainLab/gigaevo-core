"""Tests for redis.resume correctness.

Covers three failure modes that would make a resumed run diverge from a
contiguous run:
  1. RUNNING programs stuck forever  →  recover_stranded_programs()
  2. EngineMetrics.iteration reset to 0  →  EvolutionEngine.restore_state()
  3. MapElitesMultiIsland.generation / last_migration reset to 0
       →  MapElitesMultiIsland.restore_state()
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from gigaevo.evolution.engine.config import SteadyStateEngineConfig
from gigaevo.evolution.engine.snapshot import (
    ENGINE_SNAPSHOT_KEY,
    EngineSnapshot,
)
from gigaevo.evolution.engine.steady_state import SteadyStateEvolutionEngine
from gigaevo.evolution.strategies.elite_selectors import RandomEliteSelector
from gigaevo.evolution.strategies.island import IslandConfig
from gigaevo.evolution.strategies.migrant_selectors import RandomMigrantSelector
from gigaevo.evolution.strategies.models import BehaviorSpace, LinearBinning
from gigaevo.evolution.strategies.multi_island import (
    _RUN_STATE_GENERATION,
    _RUN_STATE_LAST_MIGRATION,
    MapElitesMultiIsland,
)
from gigaevo.evolution.strategies.selectors import SumArchiveSelector
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prog(state: ProgramState = ProgramState.RUNNING) -> Program:
    p = Program(code="def solve(): return 42", state=state, atomic_counter=999_999)
    p.add_metrics({"score": 50.0, "x": 5.0})
    return p


def _make_behavior_space() -> BehaviorSpace:
    return BehaviorSpace(
        bins={"x": LinearBinning(min_val=0, max_val=10, num_bins=5, type="linear")}
    )


def _make_island_config(island_id: str = "test") -> IslandConfig:
    return IslandConfig(
        island_id=island_id,
        behavior_space=_make_behavior_space(),
        max_size=None,
        archive_selector=SumArchiveSelector(fitness_keys=["score"]),
        archive_remover=None,
        elite_selector=RandomEliteSelector(),
        migrant_selector=RandomMigrantSelector(),
    )


def _make_engine(storage=None) -> SteadyStateEvolutionEngine:
    if storage is None:
        storage = AsyncMock()
        storage.load_run_state = AsyncMock(return_value=None)
        storage.save_run_state = AsyncMock()
    strategy = AsyncMock()
    writer = MagicMock()
    writer.bind.return_value = writer
    metrics_tracker = MagicMock()

    engine = SteadyStateEvolutionEngine(
        storage=storage,
        strategy=strategy,
        mutation_operator=AsyncMock(),
        config=SteadyStateEngineConfig(),
        writer=writer,
        metrics_tracker=metrics_tracker,
    )
    engine.state = AsyncMock()
    return engine


# ---------------------------------------------------------------------------
# recover_stranded_programs
# ---------------------------------------------------------------------------


class TestRecoverStrandedPrograms:
    async def test_running_programs_become_queued(self, fakeredis_storage) -> None:
        """RUNNING programs are reset to QUEUED on recovery."""
        # add() automatically places the program in the RUNNING status set
        p1 = _prog(ProgramState.RUNNING)
        p2 = _prog(ProgramState.RUNNING)
        await fakeredis_storage.add(p1)
        await fakeredis_storage.add(p2)

        recovered = await fakeredis_storage.recover_stranded_programs()

        assert recovered == 2
        assert await fakeredis_storage.count_by_status(ProgramState.RUNNING.value) == 0
        assert await fakeredis_storage.count_by_status(ProgramState.QUEUED.value) == 2

        restored = await fakeredis_storage.get(p1.id)
        assert restored is not None
        assert restored.state == ProgramState.QUEUED

    async def test_no_running_programs_returns_zero(self, fakeredis_storage) -> None:
        """Returns 0 when there are no RUNNING programs."""
        p = _prog(ProgramState.DONE)
        await fakeredis_storage.add(p)
        await fakeredis_storage.transition_status(p.id, None, ProgramState.DONE.value)

        recovered = await fakeredis_storage.recover_stranded_programs()

        assert recovered == 0

    async def test_only_running_programs_are_affected(self, fakeredis_storage) -> None:
        """DONE/QUEUED programs are not touched."""
        running = _prog(ProgramState.RUNNING)
        done = _prog(ProgramState.DONE)
        done.add_metrics({"score": 1.0})

        # add() places each program into its initial status set automatically
        await fakeredis_storage.add(running)
        await fakeredis_storage.add(done)

        await fakeredis_storage.recover_stranded_programs()

        assert await fakeredis_storage.count_by_status(ProgramState.RUNNING.value) == 0
        assert await fakeredis_storage.count_by_status(ProgramState.QUEUED.value) == 1
        assert await fakeredis_storage.count_by_status(ProgramState.DONE.value) == 1

    async def test_empty_database_returns_zero(self, fakeredis_storage) -> None:
        """Empty database returns 0."""
        assert await fakeredis_storage.recover_stranded_programs() == 0


# ---------------------------------------------------------------------------
# EvolutionEngine.restore_state
# ---------------------------------------------------------------------------


class TestEvolutionEngineRestoreState:
    async def test_restores_total_mutants(self, fakeredis_storage) -> None:
        """restore_state() loads total_mutants from Redis."""
        snap = EngineSnapshot(total_mutants=17)
        await fakeredis_storage.save_run_state(
            ENGINE_SNAPSHOT_KEY, snap.model_dump_json()
        )

        engine = _make_engine(storage=fakeredis_storage)
        assert engine.metrics.iteration == 0  # starts at 0

        await engine.restore_state()

        assert engine.metrics.iteration == 17

    async def test_no_saved_state_keeps_zero(self, fakeredis_storage) -> None:
        """When no state is persisted, total_mutants stays at 0."""
        engine = _make_engine(storage=fakeredis_storage)
        await engine.restore_state()
        assert engine.metrics.iteration == 0

    async def test_restores_programs_processed(self, fakeredis_storage) -> None:
        """restore_state() loads programs_processed from Redis."""
        snap = EngineSnapshot(programs_processed=42)
        await fakeredis_storage.save_run_state(
            ENGINE_SNAPSHOT_KEY, snap.model_dump_json()
        )

        engine = _make_engine(storage=fakeredis_storage)
        assert engine.metrics.programs_processed == 0

        await engine.restore_state()

        assert engine.metrics.programs_processed == 42

    async def test_no_saved_programs_processed_keeps_zero(
        self, fakeredis_storage
    ) -> None:
        """When no programs_processed is persisted, it stays at 0."""
        engine = _make_engine(storage=fakeredis_storage)
        await engine.restore_state()
        assert engine.metrics.programs_processed == 0


# ---------------------------------------------------------------------------
# MapElitesMultiIsland.restore_state
# ---------------------------------------------------------------------------


class TestMapElitesMultiIslandRestoreState:
    async def test_restores_generation_and_last_migration(
        self, fakeredis_storage
    ) -> None:
        """restore_state() loads generation and last_migration from Redis."""
        await fakeredis_storage.save_run_state(_RUN_STATE_GENERATION, 42)
        await fakeredis_storage.save_run_state(_RUN_STATE_LAST_MIGRATION, 40)

        strategy = MapElitesMultiIsland(
            island_configs=[_make_island_config()],
            program_storage=fakeredis_storage,
            migration_interval=50,
        )
        assert strategy.generation == 0
        assert strategy.last_migration == 0

        await strategy.restore_state()

        assert strategy.generation == 42
        assert strategy.last_migration == 40

    async def test_no_saved_state_keeps_defaults(self, fakeredis_storage) -> None:
        """When nothing is persisted, counters default to 0."""
        strategy = MapElitesMultiIsland(
            island_configs=[_make_island_config()],
            program_storage=fakeredis_storage,
        )
        await strategy.restore_state()
        assert strategy.generation == 0
        assert strategy.last_migration == 0

    async def test_generation_is_saved_after_select_elites(
        self, fakeredis_storage
    ) -> None:
        """After select_elites returns results, generation is persisted."""
        p = _prog(ProgramState.DONE)
        await fakeredis_storage.add(p)
        await fakeredis_storage.transition_status(p.id, None, ProgramState.DONE.value)

        strategy = MapElitesMultiIsland(
            island_configs=[_make_island_config()],
            program_storage=fakeredis_storage,
        )
        # Populate the island archive so select_elites returns something
        added = await strategy.islands["test"].add(p)
        assert added

        elites = await strategy.select_elites(total=8)
        assert len(elites) > 0

        saved = await fakeredis_storage.load_run_state(_RUN_STATE_GENERATION)
        assert saved == 1

    async def test_generation_not_saved_when_no_elites(self, fakeredis_storage) -> None:
        """When select_elites returns nothing, generation is not incremented or saved."""
        strategy = MapElitesMultiIsland(
            island_configs=[_make_island_config()],
            program_storage=fakeredis_storage,
        )
        await strategy.select_elites(total=8)

        saved = await fakeredis_storage.load_run_state(_RUN_STATE_GENERATION)
        assert saved is None  # never written

    async def test_generation_continues_after_restore(self, fakeredis_storage) -> None:
        """A resumed strategy increments from the restored generation value."""
        await fakeredis_storage.save_run_state(_RUN_STATE_GENERATION, 7)

        p = _prog(ProgramState.DONE)
        await fakeredis_storage.add(p)
        await fakeredis_storage.transition_status(p.id, None, ProgramState.DONE.value)

        strategy = MapElitesMultiIsland(
            island_configs=[_make_island_config()],
            program_storage=fakeredis_storage,
        )
        await strategy.restore_state()
        assert strategy.generation == 7

        added = await strategy.islands["test"].add(p)
        assert added

        await strategy.select_elites(total=8)

        assert strategy.generation == 8
        saved = await fakeredis_storage.load_run_state(_RUN_STATE_GENERATION)
        assert saved == 8
