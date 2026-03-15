"""Round-2 regression tests: two HIGH bugs found and fixed.

Bug 1: _perform_migration() raised KeyError: None when source_island_id=None
       (fixed: guard added before self.islands[source_island_id] lookup)

Bug 2: DAG(nodes={}) raised ValueError from max() on empty sequence
       (fixed: explicit guard raises clear ValueError with descriptive message)
"""

from __future__ import annotations

import fakeredis.aioredis
import pytest

from gigaevo.database.redis import RedisProgramStorageConfig
from gigaevo.database.redis_program_storage import RedisProgramStorage
from gigaevo.evolution.strategies.elite_selectors import RandomEliteSelector
from gigaevo.evolution.strategies.island import (
    METADATA_KEY_CURRENT_ISLAND,
    IslandConfig,
    MapElitesIsland,
)
from gigaevo.evolution.strategies.migrant_selectors import RandomMigrantSelector
from gigaevo.evolution.strategies.models import BehaviorSpace, LinearBinning
from gigaevo.evolution.strategies.multi_island import MapElitesMultiIsland
from gigaevo.evolution.strategies.selectors import SumArchiveSelector
from gigaevo.programs.dag.dag import DAG
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState
from tests.conftest import NullWriter

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_fakeredis_storage() -> RedisProgramStorage:
    """RedisProgramStorage backed by an isolated fakeredis server."""
    server = fakeredis.FakeServer()
    config = RedisProgramStorageConfig(
        redis_url="redis://fake:6379/0",
        key_prefix="test_cf",
    )
    storage = RedisProgramStorage(config)
    fake_redis = fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)
    storage._conn._redis = fake_redis
    storage._conn._closing = False
    return storage


def _make_island_config(island_id: str) -> IslandConfig:
    """Minimal 1-D IslandConfig; behavior key is 'fitness' in [0, 1]."""
    return IslandConfig(
        island_id=island_id,
        behavior_space=BehaviorSpace(
            bins={
                "fitness": LinearBinning(min_val=0.0, max_val=1.0, num_bins=10),
            }
        ),
        archive_selector=SumArchiveSelector(fitness_keys=["fitness"]),
        archive_remover=None,
        elite_selector=RandomEliteSelector(),
        migrant_selector=RandomMigrantSelector(),
    )


def _make_program(fitness: float) -> Program:
    """Program with the 'fitness' metric set so it maps into a behavior cell."""
    p = Program(code="def solve(): pass", state=ProgramState.RUNNING)
    p.add_metrics({"fitness": fitness})
    return p


# ---------------------------------------------------------------------------
# Bug 1 – Migration KeyError when current_island is None
#
# File:  gigaevo/evolution/strategies/multi_island.py  ~line 362
#
# Broken code (inside _perform_migration, after destination.add(migrant) succeeds):
#
#     source_island_id = migrant.get_metadata("current_island")   # returns None
#     ...
#     removed = await self.islands[source_island_id]...            # KeyError: None
#
# How current_island becomes None:
#   remove_program_by_id() explicitly sets it:
#       prog.metadata[METADATA_KEY_CURRENT_ISLAND] = None
#       await island.state_manager.update_program(prog)
#
# Reproduction:
#   1. Add a program to island_A  -> current_island="island_A" persisted in storage.
#   2. Evict via remove_program_by_id()  -> current_island=None persisted in storage.
#      The archive entry is also removed.
#   3. Re-inject the program back into island_A's archive using add_elite() directly
#      (add_elite only checks storage.exists(), not metadata, so it accepts it).
#      The program in storage still has current_island=None.
#   4. Call _perform_migration().
#      select_migrants() materialises the program from storage (current_island=None).
#      destination.add(migrant) succeeds (island_B accepts it).
#      self.islands[None]  ->  KeyError: None  <-- BUG
# ---------------------------------------------------------------------------


async def test_migration_no_keyerror_when_current_island_is_none() -> None:
    """Regression: _perform_migration() must NOT raise KeyError when current_island=None.

    Previously: self.islands[source_island_id] with source_island_id=None -> KeyError.
    Fixed: guard skips the remove-from-source step when source island is unknown.
    """
    storage = _make_fakeredis_storage()
    try:
        cfg_a = _make_island_config("island_A")
        cfg_b = _make_island_config("island_B")

        multi = MapElitesMultiIsland(
            island_configs=[cfg_a, cfg_b],
            program_storage=storage,
            migration_interval=1,
            enable_migration=True,
            max_migrants_per_island=5,
        )

        island_a: MapElitesIsland = multi.islands["island_A"]

        # Step 1: Construct a program whose current_island metadata is explicitly
        # None, save it to storage, and inject it directly into island_A's archive.
        #
        # We do NOT use the add->evict path because the merge strategy can
        # preserve old metadata values depending on atomic_counter ordering.
        # Instead we manufacture the broken state directly — the same broken
        # state that _enforce_size_limit / remove_program_by_id produce when
        # they clear current_island but the archive is repopulated before the
        # metadata write completes (race) or during a restore/reindex flow.
        prog = _make_program(fitness=0.8)
        prog.set_metadata(METADATA_KEY_CURRENT_ISLAND, None)  # ← broken state
        await storage.add(prog)

        # Confirm current_island is None in storage (precondition).
        stored = await storage.get(prog.id)
        assert stored is not None
        assert stored.get_metadata(METADATA_KEY_CURRENT_ISLAND) is None, (
            "Precondition: program must be in storage with current_island=None"
        )

        # Step 2: Inject into island_A's archive via add_elite().
        # add_elite() checks storage.exists() but NOT metadata — it accepts the
        # program regardless of what current_island is set to.
        cell = island_a.config.behavior_space.get_cell(prog.metrics)
        archive_accepted = await island_a.archive_storage.add_elite(
            cell,
            prog,
            lambda new, cur: True,  # cell is empty; always accepted
        )
        assert archive_accepted, (
            "add_elite must accept the program (precondition for triggering the bug)"
        )

        # Sanity: island_A's archive now has one program with current_island=None.
        elites = await island_a.get_elites()
        assert len(elites) == 1
        assert elites[0].get_metadata(METADATA_KEY_CURRENT_ISLAND) is None

        # Step 3: Trigger _perform_migration() — must complete without KeyError.
        # The guard now handles source_island_id=None gracefully by treating
        # it as a one-way migration (no remove-from-source step).
        await multi._perform_migration()  # Must NOT raise KeyError

    finally:
        await storage.close()


# ---------------------------------------------------------------------------
# Bug 2 – DAG constructor crashes on empty nodes dict
#
# File:  gigaevo/programs/dag/dag.py  ~line 56
#
# Broken code (inside DAG.__init__):
#
#     max_stage_timeout = max((s.timeout for s in nodes.values()))
#
# When nodes={} this is max([]), which raises:
#     ValueError: max() arg is an empty sequence
#
# Note: DAGAutomata.build({}, [], None) succeeds without error — the crash
# happens on the very next statement in DAG.__init__ after the automata is built.
# ---------------------------------------------------------------------------


async def test_dag_constructor_raises_clear_error_on_empty_nodes(state_manager) -> None:
    """Regression: DAG(nodes={}) raises ValueError with a descriptive message.

    Previously: max() on empty generator gave "max() arg is an empty sequence".
    Fixed: explicit guard raises ValueError("DAG requires at least one stage...").
    """
    with pytest.raises(ValueError, match="at least one stage"):
        DAG(
            nodes={},
            data_flow_edges=[],
            execution_order_deps=None,
            state_manager=state_manager,
            writer=NullWriter(),
        )
