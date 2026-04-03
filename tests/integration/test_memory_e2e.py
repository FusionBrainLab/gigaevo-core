"""End-to-end test: memory flows through real EvolutionEngine via DAG pipeline.

Proves the full cycle:
  seed program
  → FakeDagRunner evaluates (MemoryContextStage sets card IDs on parent)
  → EvolutionEngine ingests → selects elite → generate_mutations
  → child program auto-derives memory_used=True from parent metadata

Uses:
- Real EvolutionEngine + MapElitesMultiIsland (not mocked)
- fakeredis (real program storage, no network)
- MemoryAwareFakeDagRunner (simulates DAG memory stage + validation)
- Deterministic mutation operator (no LLM)

This is the integration test that would have caught the bug where memory was
hardcoded in the engine instead of flowing through the DAG.
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
from gigaevo.evolution.mutation.constants import (
    MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY,
)
from gigaevo.evolution.strategies.elite_selectors import ScalarTournamentEliteSelector
from gigaevo.evolution.strategies.island import IslandConfig
from gigaevo.evolution.strategies.migrant_selectors import RandomMigrantSelector
from gigaevo.evolution.strategies.models import BehaviorSpace, LinearBinning
from gigaevo.evolution.strategies.multi_island import MapElitesMultiIsland
from gigaevo.evolution.strategies.removers import FitnessArchiveRemover
from gigaevo.evolution.strategies.selectors import SumArchiveSelector
from gigaevo.llm.agents.memory_selector import MemorySelection
from gigaevo.memory.provider import MemoryProvider, NullMemoryProvider
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState

# ---------------------------------------------------------------------------
# Deterministic mutation operator
# ---------------------------------------------------------------------------

_RETURN_RE = re.compile(
    r'return\s*\{\s*"fitness":\s*([\d.]+)\s*,\s*"x":\s*([\d.]+)\s*\}',
    re.MULTILINE,
)


def _extract_metrics(code: str) -> dict[str, float]:
    m = _RETURN_RE.search(code)
    if m is None:
        raise ValueError(f"Cannot extract metrics from code:\n{code}")
    return {"fitness": float(m.group(1)), "x": float(m.group(2))}


def _make_code(fitness: float, x: float) -> str:
    return f'def entrypoint():\n    return {{"fitness": {fitness}, "x": {x}}}'


SEED_CODE = _make_code(fitness=1.0, x=0.0)

_CALL_COUNTER = 0


def _reset_counter() -> None:
    global _CALL_COUNTER
    _CALL_COUNTER = 0


class IncrementMutationOperator(MutationOperator):
    """Deterministic: bumps fitness by 1.0, assigns unique x."""

    async def mutate_single(
        self, selected_parents: list[Program], **kwargs
    ) -> MutationSpec | None:
        global _CALL_COUNTER
        parent = selected_parents[0]
        parent_metrics = _extract_metrics(parent.code)
        new_fitness = parent_metrics["fitness"] + 1.0
        new_x = 0.5 + _CALL_COUNTER
        _CALL_COUNTER += 1
        return MutationSpec(
            code=_make_code(new_fitness, new_x),
            parents=selected_parents,
            name="increment",
        )


# ---------------------------------------------------------------------------
# FakeDagRunner that simulates MemoryContextStage behavior
# ---------------------------------------------------------------------------


class ConstantMemoryProvider(MemoryProvider):
    """Always returns the same cards. For deterministic E2E testing."""

    def __init__(self, cards: list[str], card_ids: list[str]) -> None:
        self._cards = cards
        self._card_ids = card_ids
        self.call_count = 0

    async def select_cards(
        self,
        program: Program,
        *,
        task_description: str,
        metrics_description: str,
    ) -> MemorySelection:
        self.call_count += 1
        return MemorySelection(cards=self._cards, card_ids=self._card_ids)


class MemoryAwareFakeDagRunner:
    """FakeDagRunner that simulates the real DAG's memory stage.

    For each QUEUED program:
    1. Calls memory_provider.select_cards() → sets card IDs in metadata
       (mirrors what MemoryContextStage does)
    2. Evaluates the program by extracting metrics from its code
    3. Transitions QUEUED → RUNNING → DONE

    This lets us test the full E2E without needing the real DAG pipeline,
    while verifying that memory metadata persists through the evolution loop.
    """

    def __init__(
        self,
        storage: RedisProgramStorage,
        state_manager: ProgramStateManager,
        memory_provider: MemoryProvider,
    ) -> None:
        self._storage = storage
        self._sm = state_manager
        self._memory_provider = memory_provider
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop(), name="memory-fake-dag")

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

        # Simulate MemoryContextStage: select cards and set metadata
        selection = await self._memory_provider.select_cards(
            prog, task_description="test task", metrics_description="fitness"
        )
        if selection.cards:
            prog.set_metadata(
                MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY, selection.card_ids
            )

        # Evaluate: extract metrics from code
        metrics = _extract_metrics(prog.code)
        prog.add_metrics(metrics)
        await self._sm.set_program_state(prog, ProgramState.DONE)


# ---------------------------------------------------------------------------
# Infrastructure helpers
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
    behavior_space = BehaviorSpace(
        bins={"x": LinearBinning(min_val=0.0, max_val=10.0, num_bins=10, type="linear")}
    )
    return IslandConfig(
        island_id="main",
        behavior_space=behavior_space,
        archive_selector=SumArchiveSelector(fitness_keys=["fitness"]),
        archive_remover=FitnessArchiveRemover(
            fitness_key="fitness", fitness_key_higher_is_better=True
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
) -> tuple[EvolutionEngine, MapElitesMultiIsland]:
    strategy = MapElitesMultiIsland(
        island_configs=[_make_island_config()],
        program_storage=storage,
    )
    engine = EvolutionEngine(
        storage=storage,
        strategy=strategy,
        mutation_operator=IncrementMutationOperator(),
        config=EngineConfig(
            loop_interval=0.005,
            max_elites_per_generation=1,
            max_mutations_per_generation=1,
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


async def _run_evolution(
    storage: RedisProgramStorage,
    max_generations: int,
    memory_provider: MemoryProvider,
) -> tuple[EvolutionEngine, MapElitesMultiIsland]:
    engine, strategy = _build_engine(storage, max_generations)
    sm = ProgramStateManager(storage)
    runner = MemoryAwareFakeDagRunner(storage, sm, memory_provider)

    runner.start()
    engine.start()
    try:
        await asyncio.wait_for(engine.task, timeout=30.0)
    except TimeoutError:
        pytest.fail("Engine did not finish within 30s")
    finally:
        await runner.stop()

    return engine, strategy


# ===========================================================================
# Tests
# ===========================================================================


class TestMemoryE2EWithRealEngine:
    """End-to-end: real EvolutionEngine + memory provider on fakeredis."""

    @pytest.fixture(autouse=True)
    def _reset(self):
        _reset_counter()
        yield

    @pytest.mark.asyncio
    async def test_memory_card_ids_propagate_to_children(self) -> None:
        """When memory provider returns cards, children get memory_used=True.

        Flow: seed → eval (cards set on parent) → mutation → child has memory_used=True
        """
        server = fakeredis.FakeServer()
        storage = _make_fakeredis_storage(server)

        provider = ConstantMemoryProvider(
            cards=["1. Sort evidence by relevance"],
            card_ids=["idea-001"],
        )

        await _add_seed(storage)
        engine, strategy = await _run_evolution(
            storage, max_generations=3, memory_provider=provider
        )

        # Provider was called for every evaluated program
        assert provider.call_count >= 3  # seed + at least 2 mutants

        # Check that children (gen > 0) have memory_used=True
        all_programs = await storage.get_all_by_status(ProgramState.DONE.value)
        children = [p for p in all_programs if p.lineage.parents]

        assert len(children) >= 2, f"Expected >= 2 children, got {len(children)}"
        for child in children:
            assert child.get_metadata("memory_used") is True, (
                f"Child {child.short_id} should have memory_used=True "
                f"(parents: {child.lineage.parents})"
            )

        # Parents should have card IDs in metadata
        for prog in all_programs:
            assert prog.metadata.get(MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY) == [
                "idea-001"
            ]

    @pytest.mark.asyncio
    async def test_null_provider_sets_memory_used_false(self) -> None:
        """With NullMemoryProvider, children get memory_used=False.

        This is the control condition in experiments.
        """
        server = fakeredis.FakeServer()
        storage = _make_fakeredis_storage(server)

        await _add_seed(storage)
        engine, strategy = await _run_evolution(
            storage, max_generations=3, memory_provider=NullMemoryProvider()
        )

        all_programs = await storage.get_all_by_status(ProgramState.DONE.value)
        children = [p for p in all_programs if p.lineage.parents]

        assert len(children) >= 2
        for child in children:
            assert child.get_metadata("memory_used") is False, (
                f"Child {child.short_id} should have memory_used=False with NullMemoryProvider"
            )

        # No program should have memory card IDs
        for prog in all_programs:
            assert MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY not in prog.metadata

    @pytest.mark.asyncio
    async def test_memory_metadata_survives_redis_roundtrip(self) -> None:
        """Card IDs set during evaluation persist through Redis serialization.

        The metadata is set on the program object, then the program is
        serialized to Redis (by set_program_state). When generate_mutations
        reads the parent back, the metadata must still be there.
        """
        server = fakeredis.FakeServer()
        storage = _make_fakeredis_storage(server)

        provider = ConstantMemoryProvider(
            cards=["1. Use BFS", "2. Cache lookups"],
            card_ids=["bfs-card", "cache-card"],
        )

        await _add_seed(storage)
        await _run_evolution(storage, max_generations=2, memory_provider=provider)

        # Fetch programs fresh from Redis (not from in-memory cache)
        all_programs = await storage.get_all_by_status(ProgramState.DONE.value)
        programs_with_cards = [
            p
            for p in all_programs
            if MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY in p.metadata
        ]

        assert len(programs_with_cards) >= 2
        for prog in programs_with_cards:
            card_ids = prog.metadata[MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY]
            assert card_ids == ["bfs-card", "cache-card"]

    @pytest.mark.asyncio
    async def test_evolution_completes_with_memory_provider(self) -> None:
        """Evolution runs to completion with memory enabled — no crashes.

        Verifies that the memory provider integration doesn't break
        the core evolution loop (archive updates, generation counting, etc).
        """
        server = fakeredis.FakeServer()
        storage = _make_fakeredis_storage(server)

        provider = ConstantMemoryProvider(
            cards=["idea A", "idea B", "idea C"],
            card_ids=["a", "b", "c"],
        )

        await _add_seed(storage)
        engine, strategy = await _run_evolution(
            storage, max_generations=5, memory_provider=provider
        )

        # Engine completed all 5 generations
        assert engine.metrics.total_generations == 5

        # Programs were processed
        done = await storage.get_all_by_status(ProgramState.DONE.value)
        assert len(done) >= 2  # seed + at least one mutant

        # All programs are DONE (none stranded)
        queued = await storage.get_all_by_status(ProgramState.QUEUED.value)
        running = await storage.get_all_by_status(ProgramState.RUNNING.value)
        assert len(queued) == 0, f"Stranded QUEUED: {[p.short_id for p in queued]}"
        assert len(running) == 0, f"Stranded RUNNING: {[p.short_id for p in running]}"

    @pytest.mark.asyncio
    async def test_multiple_cards_per_program(self) -> None:
        """Multiple card IDs flow through correctly, not just single cards."""
        server = fakeredis.FakeServer()
        storage = _make_fakeredis_storage(server)

        many_cards = [f"idea-{i}" for i in range(5)]
        many_ids = [f"card-{i}" for i in range(5)]
        provider = ConstantMemoryProvider(cards=many_cards, card_ids=many_ids)

        await _add_seed(storage)
        await _run_evolution(storage, max_generations=2, memory_provider=provider)

        all_programs = await storage.get_all_by_status(ProgramState.DONE.value)
        for prog in all_programs:
            assert prog.metadata[MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY] == many_ids


class TestMemoryE2EControlVsTreatment:
    """Simulate an A/B experiment: control (no memory) vs treatment (with memory)."""

    @pytest.fixture(autouse=True)
    def _reset(self):
        _reset_counter()
        yield

    @pytest.mark.asyncio
    async def test_control_and_treatment_diverge_on_memory_used(self) -> None:
        """Run identical evolution with and without memory provider.

        Control:   NullMemoryProvider → memory_used=False on all children
        Treatment: ConstantMemoryProvider → memory_used=True on all children

        This mirrors how the actual hover/memory experiment works.
        """
        # --- Control ---
        _reset_counter()
        control_server = fakeredis.FakeServer()
        control_storage = _make_fakeredis_storage(control_server)
        await _add_seed(control_storage)
        await _run_evolution(
            control_storage, max_generations=3, memory_provider=NullMemoryProvider()
        )

        # --- Treatment ---
        _reset_counter()
        treatment_server = fakeredis.FakeServer()
        treatment_storage = _make_fakeredis_storage(treatment_server)
        await _add_seed(treatment_storage)
        await _run_evolution(
            treatment_storage,
            max_generations=3,
            memory_provider=ConstantMemoryProvider(
                cards=["Sort by relevance"], card_ids=["idea-1"]
            ),
        )

        # --- Assertions ---
        control_programs = await control_storage.get_all_by_status(
            ProgramState.DONE.value
        )
        treatment_programs = await treatment_storage.get_all_by_status(
            ProgramState.DONE.value
        )

        control_children = [p for p in control_programs if p.lineage.parents]
        treatment_children = [p for p in treatment_programs if p.lineage.parents]

        # Control: all children have memory_used=False
        for child in control_children:
            assert child.get_metadata("memory_used") is False

        # Treatment: all children have memory_used=True
        for child in treatment_children:
            assert child.get_metadata("memory_used") is True

        # Treatment programs have card IDs; control programs don't
        for prog in treatment_programs:
            assert MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY in prog.metadata

        for prog in control_programs:
            assert MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY not in prog.metadata
