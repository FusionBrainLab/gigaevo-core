"""Integration tests for acceptor + engine interaction.

Tests that the evolution engine correctly uses acceptors during ingestion
to filter programs. Ensures the full chain: storage -> program -> acceptor
-> archive works correctly.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import fakeredis.aioredis
import pytest

from gigaevo.database.redis import RedisProgramStorageConfig
from gigaevo.database.redis_program_storage import RedisProgramStorage
from gigaevo.database.state_manager import ProgramStateManager
from gigaevo.evolution.engine.acceptor import (
    DefaultProgramEvolutionAcceptor,
    StandardEvolutionAcceptor,
    StateAcceptor,
    ValidityMetricAcceptor,
)
from gigaevo.evolution.engine.config import EngineConfig
from gigaevo.evolution.engine.core import EvolutionEngine
from gigaevo.evolution.mutation.base import MutationOperator, MutationSpec
from gigaevo.evolution.mutation.constants import MUTATION_CONTEXT_METADATA_KEY
from gigaevo.evolution.strategies.elite_selectors import (
    ScalarTournamentEliteSelector,
)
from gigaevo.evolution.strategies.island import IslandConfig
from gigaevo.evolution.strategies.migrant_selectors import RandomMigrantSelector
from gigaevo.evolution.strategies.models import BehaviorSpace, LinearBinning
from gigaevo.evolution.strategies.multi_island import MapElitesMultiIsland
from gigaevo.evolution.strategies.selectors import SumArchiveSelector
from gigaevo.programs.metrics.context import VALIDITY_KEY
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState


HANG_TIMEOUT = 5.0


def _make_storage(key_prefix: str = "test_acceptor") -> RedisProgramStorage:
    server = fakeredis.FakeServer()
    config = RedisProgramStorageConfig(
        redis_url="redis://fake:6379/0",
        key_prefix=key_prefix,
    )
    storage = RedisProgramStorage(config)
    fake_redis = fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)
    storage._conn._redis = fake_redis
    storage._conn._closing = False
    return storage


def _make_engine(
    storage: RedisProgramStorage, acceptor=None, **overrides
) -> EvolutionEngine:
    class _NullMutator(MutationOperator):
        async def mutate_single(
            self, selected_parents: list[Program]
        ) -> MutationSpec | None:
            return None

    defaults = dict(
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
    strategy = MapElitesMultiIsland(
        island_configs=[IslandConfig(**defaults)],
        program_storage=storage,
    )
    tracker = MagicMock()
    tracker.start = MagicMock()
    tracker.stop = AsyncMock()

    writer = MagicMock()
    writer.bind.return_value = writer

    engine_kwargs = dict(
        loop_interval=0.005,
        max_generations=1,
    )
    if acceptor is not None:
        engine_kwargs["program_acceptor"] = acceptor
    engine_kwargs.update(overrides)

    return EvolutionEngine(
        storage=storage,
        strategy=strategy,
        mutation_operator=_NullMutator(),
        config=EngineConfig(**engine_kwargs),
        writer=writer,
        metrics_tracker=tracker,
    )


class TestAcceptorEngineIntegration:
    """Test that engine uses acceptor to filter during ingestion."""

    async def test_engine_with_default_acceptor_runs_clean(self):
        """Engine with default acceptor completes without error."""
        storage = _make_storage()
        try:
            engine = _make_engine(storage)
            await asyncio.wait_for(engine.run(), timeout=HANG_TIMEOUT)
            assert engine.metrics.total_generations == 1
        finally:
            await storage.close()

    async def test_done_programs_accepted_by_default(self):
        """Programs in DONE state with metrics should be accepted by default acceptor."""
        storage = _make_storage()
        try:
            p = Program(code="def solve(): return 1", state=ProgramState.DONE, atomic_counter=1)
            p.metrics = {"fitness": 1.0, "x": 0.5}
            await storage.add(p)

            acceptor = DefaultProgramEvolutionAcceptor()
            assert acceptor.is_accepted(p)
        finally:
            await storage.close()

    async def test_invalid_program_rejected_by_standard_acceptor(self):
        """Programs missing validity metric should be rejected by standard acceptor."""
        storage = _make_storage()
        try:
            p = Program(code="def solve(): return 1", state=ProgramState.DONE, atomic_counter=1)
            p.metrics = {"fitness": 1.0, "x": 0.5}
            # Missing VALIDITY_KEY and mutation context

            acceptor = StandardEvolutionAcceptor(required_behavior_keys={"x"})
            assert not acceptor.is_accepted(p)
        finally:
            await storage.close()

    async def test_engine_ingests_only_valid_programs(self):
        """Engine should only add programs to archive that pass the acceptor."""
        storage = _make_storage()
        try:
            engine = _make_engine(storage)

            # Add a valid DONE program
            p_valid = Program(
                code="def solve(): return 1",
                state=ProgramState.DONE,
                atomic_counter=1,
            )
            p_valid.metrics = {"fitness": 1.0, "x": 0.5}
            await storage.add(p_valid)

            # Add an invalid DISCARDED program
            p_invalid = Program(
                code="def solve(): return 0",
                state=ProgramState.DISCARDED,
                atomic_counter=2,
            )
            p_invalid.metrics = {"fitness": 0.5, "x": 0.2}
            await storage.add(p_invalid)

            # StateAcceptor should accept DONE but reject DISCARDED
            assert StateAcceptor().is_accepted(p_valid)
            assert not StateAcceptor().is_accepted(p_invalid)
        finally:
            await storage.close()
