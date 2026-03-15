"""Regression tests for EvolutionEngine and MapElitesMultiIsland bugs.

Ghost IDs stalling _has_active_dags (fixed):
    _has_active_dags() uses get_all_by_status() (SMEMBERS + MGET, Nones filtered)
    instead of count_by_status() (raw SCARD).  Ghost IDs — Redis set members with
    no backing program hash — are filtered out and no longer cause _await_idle()
    to spin forever.

Island migration KeyError when current_island is None (fixed):
    _perform_migration() raised KeyError: None when a migrant's current_island
    metadata was None (set by remove_program_by_id after eviction).  Fix: guard
    skips the remove-from-source step when source_island_id is None.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import fakeredis.aioredis

from gigaevo.database.redis import RedisProgramStorageConfig
from gigaevo.database.redis_program_storage import RedisProgramStorage
from gigaevo.database.state_manager import ProgramStateManager
from gigaevo.evolution.engine.config import EngineConfig
from gigaevo.evolution.engine.core import EvolutionEngine
from gigaevo.evolution.mutation.base import MutationOperator, MutationSpec
from gigaevo.evolution.strategies.elite_selectors import (
    RandomEliteSelector,
    ScalarTournamentEliteSelector,
)
from gigaevo.evolution.strategies.island import (
    METADATA_KEY_CURRENT_ISLAND,
    IslandConfig,
    MapElitesIsland,
)
from gigaevo.evolution.strategies.migrant_selectors import RandomMigrantSelector
from gigaevo.evolution.strategies.models import BehaviorSpace, LinearBinning
from gigaevo.evolution.strategies.multi_island import MapElitesMultiIsland
from gigaevo.evolution.strategies.selectors import SumArchiveSelector
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_storage(key_prefix: str = "test") -> tuple[RedisProgramStorage, object]:
    server = fakeredis.FakeServer()
    config = RedisProgramStorageConfig(
        redis_url="redis://fake:6379/0",
        key_prefix=key_prefix,
    )
    storage = RedisProgramStorage(config)
    fake_redis = fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)
    storage._conn._redis = fake_redis
    storage._conn._closing = False
    return storage, fake_redis


def _make_island_config(island_id: str) -> IslandConfig:
    return IslandConfig(
        island_id=island_id,
        behavior_space=BehaviorSpace(
            bins={"fitness": LinearBinning(min_val=0.0, max_val=1.0, num_bins=10)}
        ),
        archive_selector=SumArchiveSelector(fitness_keys=["fitness"]),
        archive_remover=None,
        elite_selector=RandomEliteSelector(),
        migrant_selector=RandomMigrantSelector(),
    )


def _make_engine(storage: RedisProgramStorage) -> EvolutionEngine:
    class _NullMutator(MutationOperator):
        async def mutate_single(
            self, selected_parents: list[Program]
        ) -> MutationSpec | None:
            return None

    strategy = MapElitesMultiIsland(
        island_configs=[
            IslandConfig(
                island_id="test",
                behavior_space=BehaviorSpace(
                    bins={"x": LinearBinning(min_val=0.0, max_val=1.0, num_bins=2)}
                ),
                archive_selector=SumArchiveSelector(fitness_keys=["fitness"]),
                archive_remover=None,
                elite_selector=ScalarTournamentEliteSelector(
                    fitness_key="fitness",
                    fitness_key_higher_is_better=True,
                    tournament_size=2,
                ),
                migrant_selector=RandomMigrantSelector(),
            )
        ],
        program_storage=storage,
    )
    tracker = MagicMock()
    tracker.start = MagicMock()

    async def _stop():
        pass

    tracker.stop = _stop

    writer = MagicMock()
    writer.bind.return_value = writer

    return EvolutionEngine(
        storage=storage,
        strategy=strategy,
        mutation_operator=_NullMutator(),
        config=EngineConfig(loop_interval=0.005, max_generations=1),
        writer=writer,
        metrics_tracker=tracker,
    )


# ---------------------------------------------------------------------------
# Ghost IDs: _has_active_dags() must return False for IDs with no backing data
# ---------------------------------------------------------------------------


async def test_has_active_dags_returns_false_with_ghost_id() -> None:
    """_has_active_dags() must return False for ghost IDs (no backing program hash).

    Before the fix, count_by_status() used raw SCARD and counted the ghost,
    causing _await_idle() to spin forever.  After the fix, get_all_by_status()
    filters ghosts out via MGET → Nones dropped → real count == 0 → False.
    """
    storage, _ = _make_storage()
    try:
        engine = _make_engine(storage)

        ghost_id = "ghost-id-no-backing-data-99999"
        ghost_set_key = "test:status:queued"

        r = await storage._conn.get()
        inserted = await r.sadd(ghost_set_key, ghost_id)
        assert inserted == 1, "Ghost must be inserted into the status set"

        program_key = f"test:program:{ghost_id}"
        assert await r.exists(program_key) == 0, "Ghost must have no backing hash"

        result = await engine._has_active_dags()
        assert result is False, (
            "_has_active_dags() returned True for a ghost ID.  "
            "Fix: use get_all_by_status() (MGET-filtered) instead of "
            "count_by_status() (raw SCARD) to exclude ghost IDs."
        )

    finally:
        await storage.close()


async def test_has_active_dags_returns_true_for_real_queued_program() -> None:
    """Sanity: _has_active_dags() returns True for a real QUEUED program.

    Ensures the ghost-ID fix did not over-correct — real programs with backing
    data must still be counted and must make _has_active_dags() return True.
    """
    storage, _ = _make_storage()
    try:
        state_manager = ProgramStateManager(storage)
        engine = _make_engine(storage)

        program = Program(
            code="def solve(): return 1",
            state=ProgramState.QUEUED,
            atomic_counter=999_999_999,
        )
        await storage.add(program)

        assert await engine._has_active_dags() is True, (
            "_has_active_dags() returned False for a real QUEUED program."
        )

        await state_manager.set_program_state(program, ProgramState.RUNNING)
        await state_manager.set_program_state(program, ProgramState.DONE)

        assert await engine._has_active_dags() is False, (
            "_has_active_dags() still True after all programs reached DONE."
        )

    finally:
        await storage.close()


# ---------------------------------------------------------------------------
# Island migration: _perform_migration must not raise KeyError when
# a migrant's current_island metadata is None
# ---------------------------------------------------------------------------


async def test_migration_no_keyerror_when_current_island_is_none() -> None:
    """_perform_migration() must not raise KeyError when current_island=None.

    A program whose current_island metadata is None (set by remove_program_by_id
    after eviction, or by a partial restore/reindex flow) should be migrated
    without attempting to remove it from its source island.
    """
    storage, _ = _make_storage(key_prefix="test_cf")
    try:
        multi = MapElitesMultiIsland(
            island_configs=[
                _make_island_config("island_A"),
                _make_island_config("island_B"),
            ],
            program_storage=storage,
            migration_interval=1,
            enable_migration=True,
            max_migrants_per_island=5,
        )

        island_a: MapElitesIsland = multi.islands["island_A"]

        prog = Program(code="def solve(): pass", state=ProgramState.RUNNING)
        prog.add_metrics({"fitness": 0.8})
        prog.set_metadata(METADATA_KEY_CURRENT_ISLAND, None)
        await storage.add(prog)

        stored = await storage.get(prog.id)
        assert stored is not None
        assert stored.get_metadata(METADATA_KEY_CURRENT_ISLAND) is None, (
            "Precondition: program must be in storage with current_island=None"
        )

        cell = island_a.config.behavior_space.get_cell(prog.metrics)
        accepted = await island_a.archive_storage.add_elite(
            cell, prog, lambda new, cur: True
        )
        assert accepted, "add_elite must accept the program (precondition)"

        elites = await island_a.get_elites()
        assert len(elites) == 1
        assert elites[0].get_metadata(METADATA_KEY_CURRENT_ISLAND) is None

        # Must complete without KeyError
        await multi._perform_migration()

    finally:
        await storage.close()
