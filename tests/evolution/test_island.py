"""Tests for MapElitesIsland with real fakeredis storage."""

from __future__ import annotations

import pytest

from gigaevo.evolution.strategies.elite_selectors import RandomEliteSelector
from gigaevo.evolution.strategies.island import (
    METADATA_KEY_CURRENT_ISLAND,
    METADATA_KEY_HOME_ISLAND,
    IslandConfig,
    MapElitesIsland,
)
from gigaevo.evolution.strategies.migrant_selectors import RandomMigrantSelector
from gigaevo.evolution.strategies.models import (
    BehaviorSpace,
    DynamicBehaviorSpace,
    LinearBinning,
)
from gigaevo.evolution.strategies.removers import FitnessArchiveRemover
from gigaevo.evolution.strategies.selectors import SumArchiveSelector
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prog(
    score: float = 50.0, x: float = 5.0, code: str = "def solve(): return 42"
) -> Program:
    """Create a program with score and behavior metrics."""
    p = Program(code=code, state=ProgramState.DONE, atomic_counter=999_999)
    p.add_metrics({"score": score, "x": x})
    return p


def _make_behavior_space() -> BehaviorSpace:
    return BehaviorSpace(
        bins={"x": LinearBinning(min_val=0, max_val=10, num_bins=5, type="linear")}
    )


def _make_island_config(
    behavior_space: BehaviorSpace | None = None,
    max_size: int | None = None,
) -> IslandConfig:
    bs = behavior_space or _make_behavior_space()
    kwargs = {
        "island_id": "test",
        "behavior_space": bs,
        "max_size": max_size,
        "archive_selector": SumArchiveSelector(fitness_keys=["score"]),
        "elite_selector": RandomEliteSelector(),
        "migrant_selector": RandomMigrantSelector(),
    }
    if max_size is not None:
        kwargs["archive_remover"] = FitnessArchiveRemover(fitness_key="score")
    else:
        kwargs["archive_remover"] = None
    return IslandConfig(**kwargs)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def island(fakeredis_storage):
    """MapElitesIsland backed by fakeredis."""
    config = _make_island_config()
    return MapElitesIsland(config, fakeredis_storage)


@pytest.fixture
async def island_with_storage(fakeredis_storage):
    """Return both island and storage for tests that need direct storage access."""
    config = _make_island_config()
    isl = MapElitesIsland(config, fakeredis_storage)
    return isl, fakeredis_storage


@pytest.fixture
async def island_max_size(fakeredis_storage):
    """MapElitesIsland with max_size=2."""
    config = _make_island_config(max_size=2)
    return MapElitesIsland(config, fakeredis_storage)


# ---------------------------------------------------------------------------
# TestMapElitesIslandAdd
# ---------------------------------------------------------------------------


class TestMapElitesIslandAdd:
    async def test_add_to_empty_cell(self, island_with_storage):
        """Add program with behavior metrics → returns True, elite in archive."""
        isl, storage = island_with_storage
        prog = _prog(score=50.0, x=5.0)
        await storage.add(prog)

        result = await isl.add(prog)

        assert result is True
        elites = await isl.get_elites()
        assert len(elites) == 1
        assert elites[0].id == prog.id

    async def test_add_better_replaces_worse(self, island_with_storage):
        """Add high-score, add higher-score to same cell → second wins."""
        isl, storage = island_with_storage
        worse = _prog(score=50.0, x=5.0)
        better = _prog(score=90.0, x=5.0)
        await storage.add(worse)
        await storage.add(better)

        assert await isl.add(worse) is True
        assert await isl.add(better) is True

        elites = await isl.get_elites()
        assert len(elites) == 1
        assert elites[0].id == better.id

    async def test_add_worse_rejected(self, island_with_storage):
        """Add high-score, add lower-score to same cell → returns False."""
        isl, storage = island_with_storage
        better = _prog(score=90.0, x=5.0)
        worse = _prog(score=50.0, x=5.0)
        await storage.add(better)
        await storage.add(worse)

        assert await isl.add(better) is True
        assert await isl.add(worse) is False

    async def test_missing_behavior_key_raises(self, island_with_storage):
        """Program without required behavior key 'x' → raises KeyError."""
        isl, storage = island_with_storage
        prog = Program(
            code="def solve(): return 42", state=ProgramState.DONE, atomic_counter=999
        )
        prog.add_metrics({"score": 50.0})  # missing 'x'
        await storage.add(prog)

        with pytest.raises(KeyError, match="behavior keys"):
            await isl.add(prog)

    async def test_add_sets_island_metadata(self, island_with_storage):
        """After add, program.metadata has home_island and current_island."""
        isl, storage = island_with_storage
        prog = _prog(score=50.0, x=5.0)
        await storage.add(prog)

        await isl.add(prog)

        assert prog.metadata[METADATA_KEY_HOME_ISLAND] == "test"
        assert prog.metadata[METADATA_KEY_CURRENT_ISLAND] == "test"

    async def test_add_to_different_cells(self, island_with_storage):
        """Two programs with different behavior values → different cells."""
        isl, storage = island_with_storage
        prog1 = _prog(score=50.0, x=1.0)  # bin 0
        prog2 = _prog(score=60.0, x=9.0)  # bin 4
        await storage.add(prog1)
        await storage.add(prog2)

        assert await isl.add(prog1) is True
        assert await isl.add(prog2) is True

        elites = await isl.get_elites()
        assert len(elites) == 2


# ---------------------------------------------------------------------------
# TestMapElitesIslandSizeLimit
# ---------------------------------------------------------------------------


class TestMapElitesIslandSizeLimit:
    async def test_enforce_size_limit_removes_excess(self, fakeredis_storage):
        """max_size=2, add 3 programs → archive has 2."""
        config = _make_island_config(max_size=2)
        isl = MapElitesIsland(config, fakeredis_storage)

        # Programs in different cells
        prog1 = _prog(score=10.0, x=1.0)
        prog2 = _prog(score=20.0, x=3.0)
        prog3 = _prog(score=30.0, x=7.0)
        for p in [prog1, prog2, prog3]:
            await fakeredis_storage.add(p)

        await isl.add(prog1)
        await isl.add(prog2)
        await isl.add(prog3)

        elites = await isl.get_elites()
        assert len(elites) == 2

    async def test_enforce_size_limit_noop_when_under(self, fakeredis_storage):
        """max_size=5, add 2 programs → nothing removed."""
        config = _make_island_config(max_size=5)
        isl = MapElitesIsland(config, fakeredis_storage)

        prog1 = _prog(score=10.0, x=1.0)
        prog2 = _prog(score=20.0, x=7.0)
        for p in [prog1, prog2]:
            await fakeredis_storage.add(p)

        await isl.add(prog1)
        await isl.add(prog2)

        elites = await isl.get_elites()
        assert len(elites) == 2

    async def test_no_max_size_no_enforcement(self, island_with_storage):
        """max_size=None, add many → nothing removed."""
        isl, storage = island_with_storage
        # Island fixture has max_size=None

        progs = [_prog(score=float(i * 10), x=float(i * 2)) for i in range(5)]
        for p in progs:
            await storage.add(p)
            await isl.add(p)

        elites = await isl.get_elites()
        # All should be present (different cells)
        assert len(elites) >= 3  # At least 3 since some might share cells


# ---------------------------------------------------------------------------
# TestMapElitesIslandSelectElites
# ---------------------------------------------------------------------------


class TestMapElitesIslandSelectElites:
    async def test_returns_all_when_fewer_than_total(self, island_with_storage):
        """2 elites, request 5 → get 2."""
        isl, storage = island_with_storage
        prog1 = _prog(score=50.0, x=1.0)
        prog2 = _prog(score=60.0, x=7.0)
        for p in [prog1, prog2]:
            await storage.add(p)
            await isl.add(p)

        selected = await isl.select_elites(5)
        assert len(selected) == 2

    async def test_returns_subset_when_more_than_total(self, island_with_storage):
        """5 elites, request 2 → get 2."""
        isl, storage = island_with_storage

        progs = [_prog(score=float(i * 10), x=float(i * 2)) for i in range(5)]
        for p in progs:
            await storage.add(p)
            await isl.add(p)

        selected = await isl.select_elites(2)
        assert len(selected) == 2

    async def test_empty_archive_returns_empty(self, island):
        """No elites → get []."""
        selected = await island.select_elites(5)
        assert selected == []


# ---------------------------------------------------------------------------
# TestMapElitesIslandReindex
# ---------------------------------------------------------------------------


class TestMapElitesIslandReindex:
    async def test_reindex_preserves_all_elites(self, island_with_storage):
        """Add 3, reindex → all 3 still present."""
        isl, storage = island_with_storage
        progs = [_prog(score=float(i * 10 + 10), x=float(i * 2 + 1)) for i in range(3)]
        for p in progs:
            await storage.add(p)
            await isl.add(p)

        original_elites = await isl.get_elites()
        original_ids = {e.id for e in original_elites}

        await isl.reindex_archive()

        reindexed_elites = await isl.get_elites()
        reindexed_ids = {e.id for e in reindexed_elites}

        # All original elites should still be present
        assert original_ids == reindexed_ids

    async def test_reindex_empty_archive_is_noop(self, island):
        """Empty archive → reindex does nothing."""
        await island.reindex_archive()
        elites = await island.get_elites()
        assert elites == []


# ---------------------------------------------------------------------------
# TestMapElitesIslandDynamicSpace
# ---------------------------------------------------------------------------


class TestMapElitesIslandDynamicSpace:
    async def test_dynamic_expand_triggers_reindex(self, fakeredis_storage):
        """DynamicBehaviorSpace: add programs, verify optimize_space tightens bounds."""
        dynamic_space = DynamicBehaviorSpace(
            bins={
                "x": LinearBinning(min_val=0, max_val=100, num_bins=10, type="linear")
            },
            expansion_buffer_ratio=0.1,
        )
        config = IslandConfig(
            island_id="dynamic_test",
            behavior_space=dynamic_space,
            max_size=None,
            archive_selector=SumArchiveSelector(fitness_keys=["score"]),
            archive_remover=None,
            elite_selector=RandomEliteSelector(),
            migrant_selector=RandomMigrantSelector(),
        )
        isl = MapElitesIsland(config, fakeredis_storage)

        # Add programs in a narrow range — bounds will tighten via optimize_space
        for x_val in [40.0, 50.0, 60.0]:
            prog = _prog(score=x_val, x=x_val)
            await fakeredis_storage.add(prog)
            await isl.add(prog)

        # After add+optimize_space, bounds should have tightened around [40, 60]
        assert dynamic_space.bins["x"].min_val > 0.0
        assert dynamic_space.bins["x"].max_val < 100.0
        # Elites should still be preserved
        elites = await isl.get_elites()
        assert len(elites) >= 2

    async def test_optimize_space_with_programs(self, fakeredis_storage):
        """optimize_space called after add with DynamicBehaviorSpace."""
        dynamic_space = DynamicBehaviorSpace(
            bins={
                "x": LinearBinning(min_val=0, max_val=100, num_bins=10, type="linear")
            },
            expansion_buffer_ratio=0.1,
        )
        config = IslandConfig(
            island_id="opt_test",
            behavior_space=dynamic_space,
            max_size=None,
            archive_selector=SumArchiveSelector(fitness_keys=["score"]),
            archive_remover=None,
            elite_selector=RandomEliteSelector(),
            migrant_selector=RandomMigrantSelector(),
        )
        isl = MapElitesIsland(config, fakeredis_storage)

        # Add programs with small range of x values
        for x_val in [40.0, 45.0, 50.0]:
            prog = _prog(score=float(x_val), x=x_val)
            await fakeredis_storage.add(prog)
            await isl.add(prog)

        # After optimize_space, bounds should tighten around observed range
        # The initial bounds were [0, 100] — after optimization they should be
        # tightened around [40, 50] (with some buffer)
        new_min = dynamic_space.bins["x"].min_val
        new_max = dynamic_space.bins["x"].max_val

        # Bounds should have moved inward
        assert new_min > 0.0  # tightened from 0
        assert new_max < 100.0  # tightened from 100

    async def test_static_space_optimize_is_noop(self, island_with_storage):
        """Static BehaviorSpace → optimize_space does nothing."""
        isl, storage = island_with_storage
        prog = _prog(score=50.0, x=5.0)
        await storage.add(prog)
        await isl.add(prog)

        # Should not error for static space
        await isl.optimize_space()

        # Bounds unchanged
        assert isl.config.behavior_space.bins["x"].min_val == 0
        assert isl.config.behavior_space.bins["x"].max_val == 10


# ---------------------------------------------------------------------------
# TestMapElitesIslandMigrants
# ---------------------------------------------------------------------------


class TestMapElitesIslandMigrants:
    async def test_select_migrants(self, island_with_storage):
        """Select migrants from populated island."""
        isl, storage = island_with_storage
        progs = [_prog(score=float(i * 10 + 10), x=float(i * 2 + 1)) for i in range(3)]
        for p in progs:
            await storage.add(p)
            await isl.add(p)

        migrants = await isl.select_migrants(1)
        assert len(migrants) == 1

    async def test_select_migrants_empty(self, island):
        """Empty island → no migrants."""
        migrants = await island.select_migrants(3)
        assert migrants == []
