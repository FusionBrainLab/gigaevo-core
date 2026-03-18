"""Edge-case and boundary tests for MapElitesIsland and BehaviorSpace / DynamicBehaviorSpace.

Focus areas:

- island.py: setdefault None-preserving semantics, missing/extra behavior keys,
  return_exceptions silently swallowing discard failures, reindex collisions,
  select_elites with zero/negative total, migrant_selector count pass-through,
  async __len__ semantics.
- models.py: LinearBinning degenerate min==max, DynamicBehaviorSpace zero-range
  margin, check_and_expand skipping missing keys but get_cell raising,
  BinningStrategy.update_bounds shrink boundaries, NaN/Inf metric handling.
"""

from __future__ import annotations

from unittest.mock import patch

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
    score: float = 50.0,
    x: float = 5.0,
    y: float | None = None,
    code: str = "def solve(): return 42",
    metadata: dict | None = None,
) -> Program:
    """Create a DONE program with score and behavior metrics."""
    p = Program(code=code, state=ProgramState.DONE, atomic_counter=999_999)
    metrics: dict[str, float] = {"score": score, "x": x}
    if y is not None:
        metrics["y"] = y
    p.add_metrics(metrics)
    if metadata is not None:
        p.metadata = metadata
    return p


def _make_behavior_space(**overrides) -> BehaviorSpace:
    bins = overrides.pop(
        "bins",
        {"x": LinearBinning(min_val=0, max_val=10, num_bins=5, type="linear")},
    )
    return BehaviorSpace(bins=bins)


def _make_island_config(
    behavior_space: BehaviorSpace | DynamicBehaviorSpace | None = None,
    max_size: int | None = None,
    island_id: str = "test",
) -> IslandConfig:
    bs = behavior_space or _make_behavior_space()
    kwargs: dict = {
        "island_id": island_id,
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


# ===================================================================
# 1. add() setdefault preserves existing None (island.py line 154)
# Target: program.metadata.setdefault(METADATA_KEY_HOME_ISLAND, self.config.island_id)
# setdefault does NOT overwrite an existing key, even if the value is None.
# ===================================================================


class TestSetdefaultPreservesNone:
    """island.py L154: setdefault does NOT overwrite when key exists with None."""

    async def test_home_island_none_stays_none_after_add(self, fakeredis_storage):
        """If metadata already has home_island=None, add() does NOT overwrite it."""
        config = _make_island_config()
        isl = MapElitesIsland(config, fakeredis_storage)

        prog = _prog(score=50.0, x=5.0, metadata={METADATA_KEY_HOME_ISLAND: None})
        await fakeredis_storage.add(prog)
        await isl.add(prog)

        # setdefault won't overwrite existing None — this tests the real behavior
        assert prog.metadata[METADATA_KEY_HOME_ISLAND] is None

    async def test_home_island_absent_gets_set(self, fakeredis_storage):
        """If metadata has no home_island key, add() sets it to island_id."""
        config = _make_island_config()
        isl = MapElitesIsland(config, fakeredis_storage)

        prog = _prog(score=50.0, x=5.0)
        assert METADATA_KEY_HOME_ISLAND not in prog.metadata
        await fakeredis_storage.add(prog)
        await isl.add(prog)

        assert prog.metadata[METADATA_KEY_HOME_ISLAND] == "test"

    async def test_home_island_preset_to_other_island_not_overwritten(
        self, fakeredis_storage
    ):
        """If home_island is already set to another island, setdefault keeps it."""
        config = _make_island_config(island_id="island-B")
        isl = MapElitesIsland(config, fakeredis_storage)

        prog = _prog(score=50.0, x=5.0, metadata={METADATA_KEY_HOME_ISLAND: "island-A"})
        await fakeredis_storage.add(prog)
        await isl.add(prog)

        # home_island stays as "island-A" (original), NOT "island-B"
        assert prog.metadata[METADATA_KEY_HOME_ISLAND] == "island-A"
        # But current_island IS updated (assignment, not setdefault)
        assert prog.metadata[METADATA_KEY_CURRENT_ISLAND] == "island-B"


# ===================================================================
# 2. add() missing behavior keys raises KeyError, extra keys are OK
# Target: island.py lines 82-90
# ===================================================================


class TestBehaviorKeyValidation:
    """island.py L82-90: missing keys raise KeyError; extra keys are fine."""

    async def test_missing_behavior_key_raises_key_error(self, fakeredis_storage):
        """Program with score but missing 'x' -> KeyError."""
        config = _make_island_config()
        isl = MapElitesIsland(config, fakeredis_storage)

        prog = Program(code="x=1", state=ProgramState.DONE, atomic_counter=999_999)
        prog.add_metrics({"score": 50.0})  # missing 'x'
        await fakeredis_storage.add(prog)

        with pytest.raises(KeyError, match="behavior keys"):
            await isl.add(prog)

    async def test_extra_metrics_beyond_behavior_keys_ok(self, fakeredis_storage):
        """Program with extra metrics beyond behavior_keys works fine."""
        config = _make_island_config()
        isl = MapElitesIsland(config, fakeredis_storage)

        prog = _prog(score=50.0, x=5.0)
        prog.add_metrics({"extra1": 99.0, "extra2": 77.0})
        await fakeredis_storage.add(prog)

        result = await isl.add(prog)
        assert result is True

    async def test_all_behavior_keys_missing_raises(self, fakeredis_storage):
        """Program with zero matching keys raises listing all missing."""
        bs = BehaviorSpace(
            bins={
                "a": LinearBinning(min_val=0, max_val=10, num_bins=5, type="linear"),
                "b": LinearBinning(min_val=0, max_val=10, num_bins=5, type="linear"),
            }
        )
        config = _make_island_config(behavior_space=bs)
        isl = MapElitesIsland(config, fakeredis_storage)

        prog = Program(code="x=1", state=ProgramState.DONE, atomic_counter=999_999)
        prog.add_metrics({"score": 50.0, "x": 5.0})  # has neither 'a' nor 'b'
        await fakeredis_storage.add(prog)

        with pytest.raises(KeyError):
            await isl.add(prog)


# ===================================================================
# 3. _enforce_size_limit return_exceptions=True swallows discard errors
# Target: island.py lines 277-280
# ===================================================================


class TestEnforceSizeLimitDiscardFailure:
    """island.py L277-279: return_exceptions=True silently swallows discard errors.
    The removed count (L280) still reflects len(to_remove), not actual successes."""

    async def test_size_enforcement_continues_when_discard_raises(
        self, fakeredis_storage
    ):
        """Even if _discard_one raises for a program, the archive is still trimmed."""
        config = _make_island_config(max_size=2)
        isl = MapElitesIsland(config, fakeredis_storage)

        # Add 3 programs in different cells
        progs = [_prog(score=float(i * 10 + 10), x=float(i * 2 + 1)) for i in range(3)]
        for p in progs:
            await fakeredis_storage.add(p)

        # Patch set_program_state to raise on the lowest-score program
        original_set_state = isl.state_manager.set_program_state
        lowest_id = progs[0].id  # score=10

        async def flaky_set_state(prog, state):
            if prog.id == lowest_id:
                raise RuntimeError("Simulated discard failure")
            return await original_set_state(prog, state)

        with patch.object(
            isl.state_manager, "set_program_state", side_effect=flaky_set_state
        ):
            await isl.add(progs[0])
            await isl.add(progs[1])
            await isl.add(progs[2])

        # Archive should still have been trimmed to max_size
        # (bulk_remove_elites_by_id happens BEFORE the per-program _discard_one)
        final_size = await isl.__len__()
        assert final_size <= 2


# ===================================================================
# 4. reindex_archive: two elites collide into same cell after re-indexing
# Target: island.py lines 296-339
# ===================================================================


class TestReindexArchiveCollision:
    """island.py L296-339: reindex clears all then re-adds.
    If two elites now map to the same cell, only one survives."""

    async def test_reindex_collision_fewer_elites(self, fakeredis_storage):
        """After changing binning so two programs map to same cell,
        reindex drops one of them."""
        # Wide bins so initially programs are in different cells
        wide_space = DynamicBehaviorSpace(
            bins={
                "x": LinearBinning(min_val=0, max_val=100, num_bins=10, type="linear")
            },
            expansion_buffer_ratio=0.0,  # no buffer to make it predictable
        )
        config = _make_island_config(behavior_space=wide_space, island_id="reindex")
        isl = MapElitesIsland(config, fakeredis_storage)

        # Two programs at x=15 and x=19 — in 10 bins over [0,100], both in bin 1
        # Actually bins: [0,10), [10,20), [20,30)...
        # x=15 -> bin 1, x=19 -> bin 1. Same cell!
        # Use values in different bins first, then shrink so they collide.
        p1 = _prog(score=30.0, x=15.0)
        p2 = _prog(score=80.0, x=85.0)
        await fakeredis_storage.add(p1)
        await fakeredis_storage.add(p2)
        await isl.add(p1)
        await isl.add(p2)

        # Now both should be in archive (different cells)
        elites_before = await isl.get_elites()
        assert len(elites_before) == 2

        # Shrink the space so both map to same cell
        # Range [14, 20] with 10 bins => each bin width = 0.6
        # x=15 -> bin 1, x=85 is clamped to 20 -> bin 9
        # Instead: let's make the range tiny: [0, 1] with 1 bin => everything bin 0
        wide_space.bins["x"].min_val = 0.0
        wide_space.bins["x"].max_val = 1.0
        wide_space.bins["x"].num_bins = 1

        await isl.reindex_archive()

        # Now both are clamped into the single bin — only better one survives
        elites_after = await isl.get_elites()
        assert len(elites_after) == 1
        # The higher-score program should win
        assert elites_after[0].id == p2.id

    async def test_reindex_empty_is_noop(self, fakeredis_storage):
        """Empty archive -> reindex does nothing, no error."""
        config = _make_island_config()
        isl = MapElitesIsland(config, fakeredis_storage)
        await isl.reindex_archive()
        assert await isl.__len__() == 0


# ===================================================================
# 5. select_elites with total <= 0 (island.py line 177)
# ===================================================================


class TestSelectElitesZeroNegative:
    """island.py L177: `if not elites or total <= 0: return []`"""

    async def test_select_elites_total_zero_returns_empty(self, fakeredis_storage):
        """total=0 should return empty list even with elites."""
        config = _make_island_config()
        isl = MapElitesIsland(config, fakeredis_storage)

        prog = _prog(score=50.0, x=5.0)
        await fakeredis_storage.add(prog)
        await isl.add(prog)

        result = await isl.select_elites(0)
        assert result == []

    async def test_select_elites_negative_total_returns_empty(self, fakeredis_storage):
        """Negative total also returns empty list."""
        config = _make_island_config()
        isl = MapElitesIsland(config, fakeredis_storage)

        prog = _prog(score=50.0, x=5.0)
        await fakeredis_storage.add(prog)
        await isl.add(prog)

        result = await isl.select_elites(-5)
        assert result == []

    async def test_select_elites_zero_with_empty_archive(self, fakeredis_storage):
        """total=0 with empty archive -> empty list."""
        config = _make_island_config()
        isl = MapElitesIsland(config, fakeredis_storage)

        result = await isl.select_elites(0)
        assert result == []


# ===================================================================
# 6. select_migrants: migrant_selector count passthrough (island.py L206)
# ===================================================================


class TestSelectMigrantsPassthrough:
    """island.py L206: island does not clamp migrant_selector result."""

    async def test_select_migrants_empty_archive(self, fakeredis_storage):
        """No elites -> empty list regardless of count."""
        config = _make_island_config()
        isl = MapElitesIsland(config, fakeredis_storage)

        result = await isl.select_migrants(10)
        assert result == []

    async def test_select_migrants_count_exceeds_elites(self, fakeredis_storage):
        """RandomMigrantSelector returns all when count > len(elites)."""
        config = _make_island_config()
        isl = MapElitesIsland(config, fakeredis_storage)

        prog = _prog(score=50.0, x=5.0)
        await fakeredis_storage.add(prog)
        await isl.add(prog)

        # Ask for 100 migrants when there's only 1 elite
        result = await isl.select_migrants(100)
        # RandomMigrantSelector returns all programs when count >= len(programs)
        assert len(result) == 1

    async def test_select_migrants_delegates_to_selector(self, fakeredis_storage):
        """Verify island passes count directly without clamping."""
        config = _make_island_config()
        isl = MapElitesIsland(config, fakeredis_storage)

        progs = [_prog(score=float(i * 10 + 10), x=float(i * 2 + 1)) for i in range(5)]
        for p in progs:
            await fakeredis_storage.add(p)
            await isl.add(p)

        result = await isl.select_migrants(2)
        assert len(result) == 2


# ===================================================================
# 7. __len__ is async (island.py L220)
# ===================================================================


class TestAsyncLen:
    """island.py L220: __len__ is async, so builtin len() cannot be used."""

    async def test_await_len_returns_int(self, fakeredis_storage):
        """await island.__len__() returns correct count."""
        config = _make_island_config()
        isl = MapElitesIsland(config, fakeredis_storage)

        assert await isl.__len__() == 0

        prog = _prog(score=50.0, x=5.0)
        await fakeredis_storage.add(prog)
        await isl.add(prog)

        assert await isl.__len__() == 1

    async def test_builtin_len_raises_type_error(self, fakeredis_storage):
        """Using builtin len() on island should raise TypeError
        because __len__ returns a coroutine, not an int."""
        config = _make_island_config()
        isl = MapElitesIsland(config, fakeredis_storage)

        # Calling len() on an object whose __len__ is async gives TypeError
        # because __len__ returns a coroutine, which is not an int.
        with pytest.raises(TypeError):
            len(isl)


# ===================================================================
# 8. LinearBinning.get_index: degenerate min_val == max_val (models.py L84)
# ===================================================================


class TestLinearBinningDegenerateRange:
    """models.py L84: when max_val == min_val, always returns 0."""

    def test_degenerate_single_bin_returns_zero(self):
        """min==max with 1 bin always returns bin 0."""
        b = LinearBinning(min_val=5.0, max_val=5.0, num_bins=1, type="linear")
        assert b.get_index(5.0) == 0

    def test_degenerate_multi_bin_still_returns_zero(self):
        """min==max with num_bins=10 STILL returns 0 — higher bins never used."""
        b = LinearBinning(min_val=5.0, max_val=5.0, num_bins=10, type="linear")
        assert b.get_index(5.0) == 0
        # Even values that differ should be clamped and return 0
        assert b.get_index(0.0) == 0
        assert b.get_index(100.0) == 0

    def test_degenerate_bin_width_is_zero(self):
        """Bin width with degenerate range is 0."""
        b = LinearBinning(min_val=5.0, max_val=5.0, num_bins=3, type="linear")
        assert b.get_bin_width(0) == 0.0

    def test_normal_range_boundary_is_last_bin(self):
        """value == max_val should map to last bin (num_bins - 1)."""
        b = LinearBinning(min_val=0.0, max_val=10.0, num_bins=5, type="linear")
        # 10.0 normalized = 1.0 => idx=5 => clamped to 4
        assert b.get_index(10.0) == 4


# ===================================================================
# 9. DynamicBehaviorSpace._calculate_margin: zero range (models.py L200-201)
# ===================================================================


class TestDynamicMarginZeroRange:
    """models.py L200-201: current_range == 0 returns 1e-5."""

    def test_zero_range_margin(self):
        """When current range is 0, margin is 1e-5 (epsilon fallback)."""
        space = DynamicBehaviorSpace(
            bins={
                "x": LinearBinning(min_val=5.0, max_val=5.0, num_bins=3, type="linear")
            },
            expansion_buffer_ratio=0.1,
        )
        margin = space._calculate_margin(0.0)
        assert margin == 1e-5

    def test_nonzero_range_margin(self):
        """Normal case: margin = range * buffer_ratio."""
        space = DynamicBehaviorSpace(
            bins={
                "x": LinearBinning(min_val=0.0, max_val=10.0, num_bins=5, type="linear")
            },
            expansion_buffer_ratio=0.2,
        )
        assert space._calculate_margin(10.0) == pytest.approx(2.0)

    def test_zero_range_expansion_works_within_initial_bounds(self):
        """With zero-range bins, expansion succeeds if initial bounds allow it.
        Since _clamp_to_initial_bounds restricts expansion, the initial bounds
        must be wider than current bounds for expansion to work."""
        space = DynamicBehaviorSpace(
            bins={
                "x": LinearBinning(
                    min_val=0.0, max_val=100.0, num_bins=3, type="linear"
                )
            },
            expansion_buffer_ratio=0.1,
        )
        # Shrink current bounds to degenerate range (zero width)
        space.bins["x"].min_val = 50.0
        space.bins["x"].max_val = 50.0

        # Value below current min=50 should trigger expansion
        # margin = _calculate_margin(0) = 1e-5
        # new_min = clamp(4.0 - 1e-5, initial_min=0) = 3.99999
        metrics = {"x": 4.0}
        expanded = space.check_and_expand(metrics)
        assert expanded
        assert space.bins["x"].min_val < 4.0


# ===================================================================
# 10. DynamicBehaviorSpace.check_and_expand: missing key skipped but get_cell raises
# Target: models.py lines 288-289 (check_and_expand) vs L146 (get_cell)
# ===================================================================


class TestCheckAndExpandMissingKeyInconsistency:
    """models.py L288-289: check_and_expand skips missing keys (continue),
    but get_cell raises KeyError for the same missing key."""

    def test_check_and_expand_missing_key_returns_false(self):
        """check_and_expand silently skips missing keys and returns False."""
        space = DynamicBehaviorSpace(
            bins={
                "x": LinearBinning(
                    min_val=0.0, max_val=10.0, num_bins=5, type="linear"
                ),
                "y": LinearBinning(
                    min_val=0.0, max_val=10.0, num_bins=5, type="linear"
                ),
            },
            expansion_buffer_ratio=0.1,
        )
        # metrics has 'x' but NOT 'y'
        metrics = {"x": 5.0}
        expanded = space.check_and_expand(metrics)
        # No expansion because x=5 is within bounds; y is skipped
        assert expanded is False

    def test_get_cell_raises_for_same_missing_key(self):
        """get_cell raises KeyError for the key that check_and_expand skipped."""
        space = DynamicBehaviorSpace(
            bins={
                "x": LinearBinning(
                    min_val=0.0, max_val=10.0, num_bins=5, type="linear"
                ),
                "y": LinearBinning(
                    min_val=0.0, max_val=10.0, num_bins=5, type="linear"
                ),
            },
            expansion_buffer_ratio=0.1,
        )
        metrics = {"x": 5.0}  # missing 'y'
        # check_and_expand says "fine"
        space.check_and_expand(metrics)
        # But get_cell disagrees
        with pytest.raises(KeyError, match="y"):
            space.get_cell(metrics)

    def test_expand_succeeds_then_get_cell_still_raises_if_missing(self):
        """Expansion can succeed on present keys while missing keys cause
        get_cell to raise."""
        space = DynamicBehaviorSpace(
            bins={
                "x": LinearBinning(
                    min_val=0.0, max_val=100.0, num_bins=5, type="linear"
                ),
                "y": LinearBinning(
                    min_val=0.0, max_val=10.0, num_bins=5, type="linear"
                ),
            },
            expansion_buffer_ratio=0.1,
        )
        # Shrink x bounds so the value triggers expansion
        space.bins["x"].min_val = 0.0
        space.bins["x"].max_val = 10.0

        # x=15 is outside [0, 10] -> expansion needed.
        # Initial bounds for x are [0, 100], so expansion to ~16 is OK.
        # y is missing -> check_and_expand skips it.
        metrics = {"x": 15.0}
        expanded = space.check_and_expand(metrics)
        assert expanded is True
        # But get_cell still fails because y is missing
        with pytest.raises(KeyError, match="y"):
            space.get_cell(metrics)


# ===================================================================
# 11. BinningStrategy.update_bounds: shrink boundary
# Target: models.py lines 60-73
# ===================================================================


class TestBinningStrategyUpdateBounds:
    """models.py L60-73: update_bounds allows shrinking.
    Shrink to equal is OK; shrink past (min > max) raises ValueError."""

    def test_shrink_to_equal_is_ok(self):
        """Setting min=max should not raise."""
        b = LinearBinning(min_val=0.0, max_val=10.0, num_bins=5, type="linear")
        changed = b.update_bounds(new_min=5.0, new_max=5.0)
        assert changed is True
        assert b.min_val == 5.0
        assert b.max_val == 5.0

    def test_shrink_past_raises_value_error(self):
        """Setting new_min > new_max raises ValueError."""
        b = LinearBinning(min_val=0.0, max_val=10.0, num_bins=5, type="linear")
        with pytest.raises(ValueError, match="Invalid bounds"):
            b.update_bounds(new_min=8.0, new_max=3.0)

    def test_expand_returns_true(self):
        """Expanding bounds returns True."""
        b = LinearBinning(min_val=2.0, max_val=8.0, num_bins=5, type="linear")
        changed = b.update_bounds(new_min=0.0, new_max=10.0)
        assert changed is True
        assert b.min_val == 0.0
        assert b.max_val == 10.0

    def test_no_change_returns_false(self):
        """Passing same values returns False."""
        b = LinearBinning(min_val=0.0, max_val=10.0, num_bins=5, type="linear")
        changed = b.update_bounds(new_min=0.0, new_max=10.0)
        assert changed is False

    def test_partial_update_min_only(self):
        """Only updating min."""
        b = LinearBinning(min_val=0.0, max_val=10.0, num_bins=5, type="linear")
        changed = b.update_bounds(new_min=3.0)
        assert changed is True
        assert b.min_val == 3.0
        assert b.max_val == 10.0

    def test_shrink_min_above_max_raises(self):
        """Shrinking just min to above max raises ValueError."""
        b = LinearBinning(min_val=0.0, max_val=5.0, num_bins=5, type="linear")
        with pytest.raises(ValueError, match="Invalid bounds"):
            b.update_bounds(new_min=6.0)

    def test_construction_with_min_gt_max_raises(self):
        """Model validator rejects min > max at construction time."""
        with pytest.raises(ValueError, match="Invalid bounds"):
            LinearBinning(min_val=10.0, max_val=5.0, num_bins=5, type="linear")


# ===================================================================
# 12. BehaviorSpace.get_cell with NaN/Inf (models.py L140-152 + _clamp)
# ===================================================================


class TestGetCellFloatEdgeCases:
    """models.py L140-152: get_cell behavior with NaN and Inf metrics."""

    def test_positive_inf_clamps_to_last_bin(self):
        """float('inf') should clamp to max_val -> last bin."""
        bs = _make_behavior_space()
        cell = bs.get_cell({"x": float("inf")})
        # With 5 bins over [0,10], inf clamps to 10 => bin 4
        assert cell == (4,)

    def test_negative_inf_clamps_to_first_bin(self):
        """float('-inf') should clamp to min_val -> bin 0."""
        bs = _make_behavior_space()
        cell = bs.get_cell({"x": float("-inf")})
        assert cell == (0,)

    def test_nan_produces_unpredictable_result(self):
        """NaN is not caught by _clamp — it propagates through.
        The behavior may vary but we document it does not raise."""
        bs = _make_behavior_space()
        # NaN comparisons are all False: max(0, min(nan, 10)) depends on impl
        # This test documents the behavior: NaN does NOT raise an exception
        # but produces a potentially arbitrary bin index.
        cell = bs.get_cell({"x": float("nan")})
        # The result is a tuple of one int — we just verify structure
        assert isinstance(cell, tuple)
        assert len(cell) == 1
        assert isinstance(cell[0], int)

    def test_value_exactly_at_bin_boundary(self):
        """Value exactly on internal bin boundary maps to higher bin."""
        bs = BehaviorSpace(
            bins={
                "x": LinearBinning(min_val=0.0, max_val=10.0, num_bins=5, type="linear")
            }
        )
        # Bin boundaries: 0, 2, 4, 6, 8, 10
        # x=2.0: normalized=0.2, idx=int(0.2*5)=1 => bin 1
        cell = bs.get_cell({"x": 2.0})
        assert cell == (1,)

    def test_value_just_below_max(self):
        """Value just below max_val maps to last bin."""
        bs = BehaviorSpace(
            bins={
                "x": LinearBinning(min_val=0.0, max_val=10.0, num_bins=5, type="linear")
            }
        )
        cell = bs.get_cell({"x": 9.999999})
        assert cell == (4,)


# ===================================================================
# 13. BehaviorSpace.get_cell: multi-dimensional behavior space
# ===================================================================


class TestMultiDimensionalBehaviorSpace:
    """Verify get_cell works with 2+ dimensions."""

    def test_two_dimensions(self):
        """Two-dimensional space produces tuple of two indices."""
        bs = BehaviorSpace(
            bins={
                "x": LinearBinning(
                    min_val=0.0, max_val=10.0, num_bins=5, type="linear"
                ),
                "y": LinearBinning(
                    min_val=0.0, max_val=100.0, num_bins=10, type="linear"
                ),
            }
        )
        cell = bs.get_cell({"x": 5.0, "y": 50.0})
        assert cell == (2, 5)

    def test_total_cells_product(self):
        """total_cells is product of all bin counts."""
        bs = BehaviorSpace(
            bins={
                "x": LinearBinning(
                    min_val=0.0, max_val=10.0, num_bins=5, type="linear"
                ),
                "y": LinearBinning(
                    min_val=0.0, max_val=100.0, num_bins=10, type="linear"
                ),
            }
        )
        assert bs.total_cells == 50


# ===================================================================
# 14. DynamicBehaviorSpace.update_bounds (batch) — models.py L260-276
# ===================================================================


class TestDynamicBehaviorSpaceUpdateBounds:
    """models.py L260-276: update_bounds dict interface."""

    def test_update_multiple_dimensions(self):
        """Can update bounds for multiple dimensions at once."""
        space = DynamicBehaviorSpace(
            bins={
                "x": LinearBinning(
                    min_val=0.0, max_val=10.0, num_bins=5, type="linear"
                ),
                "y": LinearBinning(
                    min_val=0.0, max_val=10.0, num_bins=5, type="linear"
                ),
            },
            expansion_buffer_ratio=0.1,
        )
        changed = space.update_bounds({"x": (1.0, 9.0), "y": (2.0, 8.0)})
        assert changed is True
        assert space.bins["x"].min_val == 1.0
        assert space.bins["x"].max_val == 9.0
        assert space.bins["y"].min_val == 2.0
        assert space.bins["y"].max_val == 8.0

    def test_update_unknown_key_ignored(self):
        """Keys not in bins are silently ignored."""
        space = DynamicBehaviorSpace(
            bins={
                "x": LinearBinning(min_val=0.0, max_val=10.0, num_bins=5, type="linear")
            },
            expansion_buffer_ratio=0.1,
        )
        changed = space.update_bounds({"z": (0.0, 5.0)})
        assert changed is False

    def test_update_with_none_values_no_change(self):
        """Passing None for both min and max leaves dimension unchanged."""
        space = DynamicBehaviorSpace(
            bins={
                "x": LinearBinning(min_val=0.0, max_val=10.0, num_bins=5, type="linear")
            },
            expansion_buffer_ratio=0.1,
        )
        changed = space.update_bounds({"x": (None, None)})
        assert changed is False


# ===================================================================
# 15. BehaviorSpace validation: empty bins dict
# ===================================================================


class TestBehaviorSpaceValidation:
    """models.py L136-138: empty bins dict raises ValueError."""

    def test_empty_bins_raises(self):
        """Cannot create BehaviorSpace with no dimensions."""
        with pytest.raises(ValueError, match="at least one dimension"):
            BehaviorSpace(bins={})

    def test_single_bin_is_valid(self):
        """One dimension is sufficient."""
        bs = BehaviorSpace(
            bins={
                "x": LinearBinning(min_val=0.0, max_val=10.0, num_bins=1, type="linear")
            }
        )
        assert bs.behavior_keys == ["x"]


# ===================================================================
# 16. IslandConfig validation: max_size set but archive_remover None
# ===================================================================


class TestIslandConfigValidation:
    """island.py L58-63: max_size set but archive_remover=None raises ValueError."""

    def test_max_size_without_remover_raises(self):
        """max_size set without archive_remover raises."""
        with pytest.raises(ValueError, match="archive_remover"):
            IslandConfig(
                island_id="bad",
                behavior_space=_make_behavior_space(),
                max_size=10,
                archive_selector=SumArchiveSelector(fitness_keys=["score"]),
                archive_remover=None,
                elite_selector=RandomEliteSelector(),
                migrant_selector=RandomMigrantSelector(),
            )

    def test_max_size_none_remover_none_is_ok(self):
        """max_size=None with archive_remover=None is valid."""
        config = IslandConfig(
            island_id="ok",
            behavior_space=_make_behavior_space(),
            max_size=None,
            archive_selector=SumArchiveSelector(fitness_keys=["score"]),
            archive_remover=None,
            elite_selector=RandomEliteSelector(),
            migrant_selector=RandomMigrantSelector(),
        )
        assert config.max_size is None


# ===================================================================
# 17. DynamicBehaviorSpace initial bounds clamp
# ===================================================================


class TestDynamicBoundsClamp:
    """models.py L204-209: _clamp_to_initial_bounds prevents expansion past initial."""

    def test_expand_clamped_to_initial_max(self):
        """Expansion cannot go beyond the initial max."""
        space = DynamicBehaviorSpace(
            bins={
                "x": LinearBinning(
                    min_val=0.0, max_val=100.0, num_bins=10, type="linear"
                )
            },
            expansion_buffer_ratio=0.1,
        )
        # Try to expand beyond 100
        new_min, new_max = space._calculate_bounds_for_value(
            "x", 120.0, space.bins["x"]
        )
        # new_max should be clamped to initial max (100.0)
        assert new_max is not None
        assert new_max <= 100.0

    def test_expand_clamped_to_initial_min(self):
        """Expansion cannot go below the initial min."""
        space = DynamicBehaviorSpace(
            bins={
                "x": LinearBinning(
                    min_val=0.0, max_val=100.0, num_bins=10, type="linear"
                )
            },
            expansion_buffer_ratio=0.1,
        )
        new_min, new_max = space._calculate_bounds_for_value(
            "x", -50.0, space.bins["x"]
        )
        assert new_min is not None
        assert new_min >= 0.0


# ===================================================================
# 18. DynamicBehaviorSpace.calculate_optimized_bounds with empty batch
# ===================================================================


class TestCalculateOptimizedBounds:
    """models.py L300-322: edge cases in batch optimization."""

    def test_empty_batch_returns_empty(self):
        """Empty metrics_batch -> empty bounds dict."""
        space = DynamicBehaviorSpace(
            bins={
                "x": LinearBinning(
                    min_val=0.0, max_val=100.0, num_bins=10, type="linear"
                )
            },
            expansion_buffer_ratio=0.1,
        )
        result = space.calculate_optimized_bounds([])
        assert result == {}

    def test_single_point_batch(self):
        """Single metric point creates bounds around that point with margin."""
        space = DynamicBehaviorSpace(
            bins={
                "x": LinearBinning(
                    min_val=0.0, max_val=100.0, num_bins=10, type="linear"
                )
            },
            expansion_buffer_ratio=0.1,
        )
        result = space.calculate_optimized_bounds([{"x": 50.0}])
        assert "x" in result
        new_min, new_max = result["x"]
        # Single point: range=0, margin=1e-5
        assert new_min == pytest.approx(50.0 - 1e-5)
        assert new_max == pytest.approx(50.0 + 1e-5)

    def test_batch_missing_key_excluded(self):
        """If a key is missing from all metrics, it's excluded from bounds."""
        space = DynamicBehaviorSpace(
            bins={
                "x": LinearBinning(
                    min_val=0.0, max_val=100.0, num_bins=10, type="linear"
                ),
                "y": LinearBinning(
                    min_val=0.0, max_val=100.0, num_bins=10, type="linear"
                ),
            },
            expansion_buffer_ratio=0.1,
        )
        # Batch only has 'x', not 'y'
        result = space.calculate_optimized_bounds([{"x": 50.0}, {"x": 60.0}])
        assert "x" in result
        assert "y" not in result


# ===================================================================
# 19. Island add() with DynamicBehaviorSpace triggers expand + reindex
# ===================================================================


class TestIslandDynamicExpandIntegration:
    """island.py L94-101: add() with DynamicBehaviorSpace triggers check_and_expand
    and reindex_archive when expansion occurs."""

    async def test_add_out_of_bounds_triggers_expansion(self, fakeredis_storage):
        """Adding a program outside current bounds expands the dynamic space."""
        # Use wide initial bounds so expansion is not capped by initial hard limits
        space = DynamicBehaviorSpace(
            bins={
                "x": LinearBinning(
                    min_val=0.0, max_val=100.0, num_bins=5, type="linear"
                )
            },
            expansion_buffer_ratio=0.1,
        )
        # Manually shrink current bounds to [10, 20]
        space.bins["x"].min_val = 10.0
        space.bins["x"].max_val = 20.0

        config = _make_island_config(behavior_space=space, island_id="dynamic")
        isl = MapElitesIsland(config, fakeredis_storage)

        # First add: within the shrunk bounds
        p1 = _prog(score=60.0, x=15.0)
        await fakeredis_storage.add(p1)
        await isl.add(p1)

        # Second add: x=25 is outside the shrunk [10, 20] bounds
        p2 = _prog(score=70.0, x=25.0)
        await fakeredis_storage.add(p2)
        await isl.add(p2)

        # Bounds should have expanded to include x=25
        assert space.bins["x"].max_val > 20.0


# ===================================================================
# 20. LinearBinning edge: exact min_val and max_val placement
# ===================================================================


class TestLinearBinningEdgePlacement:
    """models.py L83-90: verify exact min/max placement in bins."""

    def test_exact_min_val_maps_to_bin_zero(self):
        """Value equal to min_val maps to bin 0."""
        b = LinearBinning(min_val=0.0, max_val=10.0, num_bins=5, type="linear")
        assert b.get_index(0.0) == 0

    def test_exact_max_val_maps_to_last_bin(self):
        """Value equal to max_val maps to last bin."""
        b = LinearBinning(min_val=0.0, max_val=10.0, num_bins=5, type="linear")
        assert b.get_index(10.0) == 4

    def test_below_min_clamps_to_bin_zero(self):
        """Value below min_val clamps to bin 0."""
        b = LinearBinning(min_val=0.0, max_val=10.0, num_bins=5, type="linear")
        assert b.get_index(-5.0) == 0

    def test_above_max_clamps_to_last_bin(self):
        """Value above max_val clamps to last bin."""
        b = LinearBinning(min_val=0.0, max_val=10.0, num_bins=5, type="linear")
        assert b.get_index(15.0) == 4

    def test_get_bin_edges_first_bin(self):
        """First bin edges are [min_val, min_val + step)."""
        b = LinearBinning(min_val=0.0, max_val=10.0, num_bins=5, type="linear")
        lower, upper = b.get_bin_edges(0)
        assert lower == pytest.approx(0.0)
        assert upper == pytest.approx(2.0)

    def test_get_bin_center(self):
        """Center of first bin in [0,10] with 5 bins is 1.0."""
        b = LinearBinning(min_val=0.0, max_val=10.0, num_bins=5, type="linear")
        assert b.get_bin_center(0) == pytest.approx(1.0)


# ===================================================================
# 21. DynamicBehaviorSpace.describe() consistency
# ===================================================================


class TestDynamicDescribe:
    """Verify describe() returns current (potentially updated) bounds."""

    def test_describe_reflects_current_bounds(self):
        """After update_bounds, describe() shows new values."""
        space = DynamicBehaviorSpace(
            bins={
                "x": LinearBinning(
                    min_val=0.0, max_val=100.0, num_bins=10, type="linear"
                )
            },
            expansion_buffer_ratio=0.1,
        )
        space.update_bounds({"x": (20.0, 80.0)})
        desc = space.describe()
        assert desc["x"]["min"] == 20.0
        assert desc["x"]["max"] == 80.0


# ===================================================================
# reindex_archive idempotency tests
# ===================================================================


class TestReindexArchiveIdempotency:
    """reindex_archive() must be idempotent: calling it twice with unchanged
    metrics produces the same archive state (same elites, same cells)."""

    async def test_double_reindex_same_result(self, fakeredis_storage):
        """Calling reindex_archive twice yields identical archive both times."""
        config = _make_island_config()
        isl = MapElitesIsland(config, fakeredis_storage)

        progs = [_prog(score=float(i * 20 + 10), x=float(i * 2 + 1)) for i in range(4)]
        for p in progs:
            await fakeredis_storage.add(p)
            await isl.add(p)

        await isl.reindex_archive()
        elites_first = await isl.get_elites()
        ids_first = sorted(e.id for e in elites_first)
        cells_first = sorted(
            (
                isl.config.behavior_space.get_cell(e.metrics),
                e.id,
            )
            for e in elites_first
        )

        await isl.reindex_archive()
        elites_second = await isl.get_elites()
        ids_second = sorted(e.id for e in elites_second)
        cells_second = sorted(
            (
                isl.config.behavior_space.get_cell(e.metrics),
                e.id,
            )
            for e in elites_second
        )

        assert ids_first == ids_second
        assert cells_first == cells_second

    async def test_reindex_after_fitness_change_evicts_worse(self, fakeredis_storage):
        """When a program's fitness degrades, reindex evicts it if a better
        program now maps to the same cell."""
        # Single bin so all programs compete in same cell
        space = BehaviorSpace(
            bins={"x": LinearBinning(min_val=0, max_val=10, num_bins=1, type="linear")}
        )
        config = _make_island_config(behavior_space=space)
        isl = MapElitesIsland(config, fakeredis_storage)

        # p1 starts as the better program
        p1 = _prog(score=80.0, x=5.0)
        p2 = _prog(score=40.0, x=3.0)
        await fakeredis_storage.add(p1)
        await fakeredis_storage.add(p2)
        await isl.add(p1)
        await isl.add(p2)  # rejected: p1 (80) > p2 (40)

        elites = await isl.get_elites()
        assert len(elites) == 1
        assert elites[0].id == p1.id

        # Degrade p1's score and manually add p2 to storage
        # (simulates what happens after PromptFitnessStage re-runs)
        p1.metrics["score"] = 20.0
        await fakeredis_storage.add(p1)

        # Before reindex: p1 still in archive with stale high score
        elites_before = await isl.get_elites()
        assert elites_before[0].id == p1.id

        # Force p2 into archive first so reindex has both candidates
        # We need to manually set up the archive to have both
        await isl.archive_storage.add_elite(
            space.get_cell(p2.metrics), p2, config.archive_selector
        )

        await isl.reindex_archive()

        elites_after = await isl.get_elites()
        assert len(elites_after) == 1
        # p2 (40) should now beat p1 (20)
        assert elites_after[0].id == p2.id

    async def test_reindex_idempotent_with_dynamic_space(self, fakeredis_storage):
        """reindex_archive on DynamicBehaviorSpace is idempotent."""
        space = DynamicBehaviorSpace(
            bins={
                "x": LinearBinning(
                    min_val=0.0, max_val=100.0, num_bins=10, type="linear"
                )
            },
            expansion_buffer_ratio=0.1,
        )
        config = _make_island_config(behavior_space=space, island_id="dynamic")
        isl = MapElitesIsland(config, fakeredis_storage)

        progs = [_prog(score=float(i * 15 + 5), x=float(i * 25 + 5)) for i in range(3)]
        for p in progs:
            await fakeredis_storage.add(p)
            await isl.add(p)

        # Snapshot bounds before
        desc_before = space.describe()

        await isl.reindex_archive()
        elites_first = sorted(e.id for e in await isl.get_elites())
        desc_mid = space.describe()

        await isl.reindex_archive()
        elites_second = sorted(e.id for e in await isl.get_elites())
        desc_after = space.describe()

        assert elites_first == elites_second
        # Bounds should not change from reindex (no check_and_expand called)
        assert desc_before == desc_mid == desc_after

    async def test_reindex_multi_island_delegates(self, fakeredis_storage):
        """MapElitesMultiIsland.reindex_archive delegates to each island."""
        from gigaevo.evolution.strategies.multi_island import MapElitesMultiIsland

        configs = [
            _make_island_config(island_id="a"),
            _make_island_config(island_id="b"),
        ]
        multi = MapElitesMultiIsland(configs, fakeredis_storage)

        # Add one program to island a
        p = _prog(score=50.0, x=5.0)
        await fakeredis_storage.add(p)
        await multi.add(p, island_id="a")

        # Double reindex should not error and be idempotent
        await multi.reindex_archive()
        ids_first = sorted(await multi.get_program_ids())

        await multi.reindex_archive()
        ids_second = sorted(await multi.get_program_ids())

        assert ids_first == ids_second
