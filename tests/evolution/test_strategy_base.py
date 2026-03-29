"""Tests for StrategyMetrics and EvolutionStrategy base class.

Covers computed fields, boundary conditions (zero populations, zero programs),
and the to_dict serialization.
"""

from __future__ import annotations

import pytest

from gigaevo.evolution.strategies.base import EvolutionStrategy, StrategyMetrics
from gigaevo.programs.program import Program


class TestStrategyMetrics:
    def test_default_values(self):
        m = StrategyMetrics()
        assert m.total_programs == 0
        assert m.active_populations == 0
        assert m.strategy_specific_metrics is None

    def test_programs_per_population(self):
        m = StrategyMetrics(total_programs=100, active_populations=4)
        assert m.programs_per_population == 25.0

    def test_programs_per_population_zero_populations(self):
        """Division by zero must return 0.0, not raise."""
        m = StrategyMetrics(total_programs=10, active_populations=0)
        assert m.programs_per_population == 0.0

    def test_has_programs_true(self):
        m = StrategyMetrics(total_programs=1)
        assert m.has_programs is True

    def test_has_programs_false(self):
        m = StrategyMetrics(total_programs=0)
        assert m.has_programs is False

    def test_to_dict_basic(self):
        m = StrategyMetrics(total_programs=10, active_populations=2)
        d = m.to_dict()
        assert d["total_programs"] == 10
        assert d["active_populations"] == 2
        assert d["programs_per_population"] == 5.0
        assert d["has_programs"] is True

    def test_to_dict_with_strategy_specific(self):
        m = StrategyMetrics(
            total_programs=5,
            active_populations=1,
            strategy_specific_metrics={"migration_count": 3, "stale_islands": 0},
        )
        d = m.to_dict()
        assert d["migration_count"] == 3
        assert d["stale_islands"] == 0

    def test_to_dict_without_strategy_specific(self):
        m = StrategyMetrics(total_programs=0, active_populations=0)
        d = m.to_dict()
        assert "migration_count" not in d

    def test_programs_per_population_rounding(self):
        """to_dict rounds programs_per_population to 2 decimal places."""
        m = StrategyMetrics(total_programs=10, active_populations=3)
        d = m.to_dict()
        assert d["programs_per_population"] == 3.33

    def test_negative_values_rejected(self):
        """Pydantic ge=0 constraint should reject negative values."""
        with pytest.raises(Exception):
            StrategyMetrics(total_programs=-1)

    def test_negative_populations_rejected(self):
        with pytest.raises(Exception):
            StrategyMetrics(active_populations=-1)


class TestEvolutionStrategyDefaults:
    """Test default implementations of optional methods on the ABC."""

    async def test_get_metrics_returns_none(self):
        class MinimalStrategy(EvolutionStrategy):
            async def add(self, program):
                return True

            async def select_elites(self, total):
                return []

            async def get_program_ids(self):
                return []

        s = MinimalStrategy()
        assert await s.get_metrics() is None

    async def test_remove_program_raises(self):
        class MinimalStrategy(EvolutionStrategy):
            async def add(self, program):
                return True

            async def select_elites(self, total):
                return []

            async def get_program_ids(self):
                return []

        s = MinimalStrategy()
        with pytest.raises(NotImplementedError):
            await s.remove_program_by_id("some-id")

    async def test_optional_methods_are_noop(self):
        """cleanup, pause, resume, restore_state, reindex_archive are all no-ops by default."""

        class MinimalStrategy(EvolutionStrategy):
            async def add(self, program):
                return True

            async def select_elites(self, total):
                return []

            async def get_program_ids(self):
                return []

        s = MinimalStrategy()
        # These should all complete without error
        await s.cleanup()
        await s.pause()
        await s.resume()
        await s.restore_state()
        await s.reindex_archive()
