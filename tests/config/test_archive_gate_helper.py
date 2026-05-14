"""Hydra-instantiable helper that turns an evolution_strategy into an
ArchiveGateProvider.

Returns None when disabled or when the strategy is not a supported
MAP-Elites strategy. Today only ``MapElitesMultiIsland`` is supported.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from gigaevo.config.helpers import build_archive_gate_provider
from gigaevo.evolution.strategies.base import EvolutionStrategy
from gigaevo.evolution.strategies.island import MapElitesIsland
from gigaevo.evolution.strategies.multi_island import MapElitesMultiIsland
from gigaevo.programs.stages.archive_gate import AllIslandsGateProvider


def _multi_island_mock(island_count: int = 2) -> MapElitesMultiIsland:
    strategy = MagicMock(spec=MapElitesMultiIsland)
    strategy.islands = {
        f"island_{i}": MagicMock(spec=MapElitesIsland) for i in range(island_count)
    }
    return strategy


def test_disabled_returns_none():
    strategy = _multi_island_mock()
    assert build_archive_gate_provider(strategy=strategy, enabled=False) is None


def test_non_mapelites_strategy_returns_none():
    # A different EvolutionStrategy subclass that has no islands — must fail closed.
    strategy = MagicMock(spec=EvolutionStrategy)
    assert build_archive_gate_provider(strategy=strategy, enabled=True) is None


def test_enabled_with_multi_island_returns_provider():
    strategy = _multi_island_mock(island_count=3)
    provider = build_archive_gate_provider(strategy=strategy, enabled=True)
    assert isinstance(provider, AllIslandsGateProvider)


def test_enabled_with_none_strategy_returns_none():
    assert build_archive_gate_provider(strategy=None, enabled=True) is None
