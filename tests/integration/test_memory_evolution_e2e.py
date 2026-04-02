"""End-to-end integration: evolution loop WITH memory enabled.

Mirrors test_multigen_e2e.py but adds the memory dimension:
- MemoryAwareMutationOperator: responds to memory_instructions
- Full FakeDagRunner + fakeredis storage
- EvolutionEngine with memory_enabled=True
- Two-phase cycle: fill memory → run with memory

This is the test that will BREAK if memory is refactored incorrectly.
"""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
import re
from unittest.mock import AsyncMock, MagicMock

import fakeredis.aioredis
import pytest

from gigaevo.database.redis import RedisProgramStorageConfig
from gigaevo.database.redis_program_storage import RedisProgramStorage
from gigaevo.database.state_manager import ProgramStateManager
from gigaevo.evolution.engine.config import EngineConfig
from gigaevo.evolution.engine.core import EvolutionEngine
from gigaevo.evolution.engine.mutation import generate_mutations
from gigaevo.evolution.mutation.base import MutationOperator, MutationSpec
from gigaevo.evolution.strategies.elite_selectors import ScalarTournamentEliteSelector
from gigaevo.evolution.strategies.island import IslandConfig
from gigaevo.evolution.strategies.migrant_selectors import RandomMigrantSelector
from gigaevo.evolution.strategies.models import BehaviorSpace, LinearBinning
from gigaevo.evolution.strategies.multi_island import MapElitesMultiIsland
from gigaevo.evolution.strategies.removers import FitnessArchiveRemover
from gigaevo.evolution.strategies.selectors import SumArchiveSelector
from gigaevo.memory.shared_memory.memory import AmemGamMemory
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState

# ---------------------------------------------------------------------------
# Deterministic mutation operators
# ---------------------------------------------------------------------------

_RETURN_RE = re.compile(
    r'return\s*\{\s*"fitness":\s*([\d.]+)\s*,\s*"x":\s*([\d.]+)\s*\}',
    re.MULTILINE,
)

_CALL_COUNTER = 0


def _reset_counter() -> None:
    global _CALL_COUNTER
    _CALL_COUNTER = 0


def _extract_metrics(code: str) -> dict[str, float]:
    m = _RETURN_RE.search(code)
    if m is None:
        raise ValueError(f"Cannot extract metrics:\n{code}")
    return {"fitness": float(m.group(1)), "x": float(m.group(2))}


def _make_code(fitness: float, x: float) -> str:
    return f'def entrypoint():\n    return {{"fitness": {fitness}, "x": {x}}}'


class MemoryAwareMutationOperator(MutationOperator):
    """Deterministic mutation that responds to memory_instructions.

    Without memory: fitness += 1.0
    With memory:    fitness += 2.0

    This makes evolution trajectory MEASURABLY DIFFERENT when memory is
    active, so a refactor that breaks the memory pipeline will cause
    the fitness assertion to fail.
    """

    def __init__(self) -> None:
        self.calls_with_memory: list[str] = []
        self.calls_without_memory: int = 0

    async def mutate_single(
        self,
        selected_parents: list[Program],
        memory_instructions: str | None = None,
    ) -> MutationSpec | None:
        global _CALL_COUNTER
        parent = selected_parents[0]
        parent_metrics = _extract_metrics(parent.code)

        if memory_instructions is not None:
            boost = 2.0
            self.calls_with_memory.append(memory_instructions)
        else:
            boost = 1.0
            self.calls_without_memory += 1

        new_fitness = parent_metrics["fitness"] + boost
        new_x = 0.5 + _CALL_COUNTER
        _CALL_COUNTER += 1

        metadata = {"memory_used": memory_instructions is not None}

        return MutationSpec(
            code=_make_code(new_fitness, new_x),
            parents=selected_parents,
            name="memory_aware",
            metadata=metadata,
        )


class PlainIncrementOperator(MutationOperator):
    """Same as IncrementMutationOperator — no memory support."""

    async def mutate_single(
        self, selected_parents: list[Program]
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
# FakeDagRunner — evaluates by exec'ing code
# ---------------------------------------------------------------------------


class FakeDagRunner:
    def __init__(self, storage: RedisProgramStorage, sm: ProgramStateManager):
        self._storage = storage
        self._sm = sm
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
        await self._sm.set_program_state(prog, ProgramState.RUNNING)
        metrics = _extract_metrics(prog.code)
        prog.add_metrics(metrics)
        await self._sm.set_program_state(prog, ProgramState.DONE)


# ---------------------------------------------------------------------------
# Infrastructure helpers
# ---------------------------------------------------------------------------

SEED_CODE = _make_code(fitness=1.0, x=0.0)


def _make_storage(server: fakeredis.FakeServer) -> RedisProgramStorage:
    config = RedisProgramStorageConfig(
        redis_url="redis://fake:6379/0", key_prefix="test"
    )
    storage = RedisProgramStorage(config)
    storage._conn._redis = fakeredis.aioredis.FakeRedis(
        server=server, decode_responses=True
    )
    storage._conn._closing = False
    return storage


def _make_island_config() -> IslandConfig:
    return IslandConfig(
        island_id="main",
        behavior_space=BehaviorSpace(
            bins={
                "x": LinearBinning(
                    min_val=0.0, max_val=10.0, num_bins=10, type="linear"
                )
            }
        ),
        archive_selector=SumArchiveSelector(fitness_keys=["fitness"]),
        archive_remover=FitnessArchiveRemover(
            fitness_key="fitness",
            fitness_key_higher_is_better=True,
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


def _make_memory(tmp_path: Path, **overrides) -> AmemGamMemory:
    defaults = dict(
        checkpoint_path=str(tmp_path / "mem"),
        use_api=False,
        sync_on_init=False,
        enable_llm_synthesis=False,
        enable_memory_evolution=False,
        enable_llm_card_enrichment=False,
    )
    defaults.update(overrides)
    return AmemGamMemory(**defaults)


async def _run_evolution(
    server: fakeredis.FakeServer,
    max_generations: int,
    *,
    mutation_operator: MutationOperator,
    memory_enabled: bool = False,
    memory_top_n: int = 0,
    memory_path: str = "memory.txt",
) -> tuple[EvolutionEngine, list[Program]]:
    storage = _make_storage(server)
    strategy = MapElitesMultiIsland(
        island_configs=[_make_island_config()],
        program_storage=storage,
    )
    engine = EvolutionEngine(
        storage=storage,
        strategy=strategy,
        mutation_operator=mutation_operator,
        config=EngineConfig(
            loop_interval=0.005,
            max_elites_per_generation=1,
            max_mutations_per_generation=1,
            generation_timeout=30.0,
            max_generations=max_generations,
            memory_enabled=memory_enabled,
            memory_top_n=memory_top_n,
            memory_path=memory_path,
            fitness_key="fitness",
        ),
        writer=_make_null_writer(),
        metrics_tracker=_make_metrics_tracker(),
    )

    seed = Program(code=SEED_CODE, state=ProgramState.QUEUED)
    await storage.add(seed)

    sm = ProgramStateManager(storage)
    runner = FakeDagRunner(storage, sm)

    runner.start()
    engine.start()
    try:
        await asyncio.wait_for(engine.task, timeout=30.0)
    except TimeoutError:
        pytest.fail(f"Engine didn't finish {max_generations} gens in 30s")
    finally:
        await runner.stop()

    # Collect archive programs
    archive_storage = _make_storage(server)
    strategy2 = MapElitesMultiIsland(
        island_configs=[_make_island_config()],
        program_storage=archive_storage,
    )
    programs = await strategy2.islands["main"].get_elites()
    await archive_storage.close()
    await storage.close()

    return engine, programs


# ===========================================================================
# Test Class 1: Memory-enabled evolution vs baseline
# ===========================================================================


class TestMemoryEnabledEvolution:
    """Compare evolution WITH and WITHOUT memory to prove memory changes trajectory."""

    @pytest.mark.asyncio
    async def test_baseline_without_memory(self) -> None:
        """Run without memory: fitness grows by +1.0/gen."""
        _reset_counter()
        server = fakeredis.FakeServer()
        operator = PlainIncrementOperator()

        engine, programs = await _run_evolution(
            server,
            max_generations=5,
            mutation_operator=operator,
        )

        assert engine.metrics.total_generations == 5
        best_fitness = max(p.metrics["fitness"] for p in programs)
        # Seed=1.0, each gen +1.0 → best after 5 gens ≈ 5.0-6.0
        assert best_fitness <= 7.0

    @pytest.mark.asyncio
    async def test_memory_enabled_higher_fitness(self, tmp_path) -> None:
        """Run WITH memory: fitness grows by +2.0/gen (memory boost).

        If memory pipeline is broken, boost = 1.0 and this test fails.
        """
        _reset_counter()
        server = fakeredis.FakeServer()
        operator = MemoryAwareMutationOperator()

        # Create a memory instructions file
        memory_file = tmp_path / "memory.txt"
        memory_file.write_text("Use simulated annealing for local search refinement")

        engine, programs = await _run_evolution(
            server,
            max_generations=5,
            mutation_operator=operator,
            memory_enabled=True,
            memory_top_n=1,
            memory_path=str(memory_file),
        )

        assert engine.metrics.total_generations == 5
        best_fitness = max(p.metrics["fitness"] for p in programs)

        # Verify memory was actually used
        assert len(operator.calls_with_memory) > 0, (
            "Memory instructions never reached the mutation operator"
        )

        # With memory boost (+2.0 per memory-augmented mutation) the best fitness
        # should exceed what's achievable without memory (+1.0/gen → max ~6.0 after 5 gens).
        # If memory pipeline is broken (boost=1.0), best_fitness ≤ 6.0.
        assert best_fitness > 6.0, (
            f"Memory-enabled run should produce fitness > 6.0 (the no-memory ceiling), "
            f"got {best_fitness}. This means memory boost is not applied."
        )

    @pytest.mark.asyncio
    async def test_memory_disabled_uses_no_memory(self) -> None:
        """With memory_enabled=False, operator never gets memory_instructions."""
        _reset_counter()
        server = fakeredis.FakeServer()
        operator = MemoryAwareMutationOperator()

        engine, _ = await _run_evolution(
            server,
            max_generations=3,
            mutation_operator=operator,
            memory_enabled=False,
        )

        assert engine.metrics.total_generations == 3
        assert len(operator.calls_with_memory) == 0
        assert operator.calls_without_memory > 0

    @pytest.mark.asyncio
    async def test_missing_memory_file_falls_back(self) -> None:
        """If memory file doesn't exist, engine falls back to no-memory mutation."""
        _reset_counter()
        server = fakeredis.FakeServer()
        operator = MemoryAwareMutationOperator()

        engine, _ = await _run_evolution(
            server,
            max_generations=3,
            mutation_operator=operator,
            memory_enabled=True,
            memory_top_n=1,
            memory_path="/nonexistent/memory.txt",
        )

        assert engine.metrics.total_generations == 3
        # Memory file missing → _read_memory_instructions returns None → engine
        # passes "" as memory_instructions (line 412: `or ""`)
        # The operator receives "" which is truthy for `is not None` but empty.
        # Either way, the operator should NOT get meaningful memory content.
        for call in operator.calls_with_memory:
            # Empty or whitespace-only: no real memory content
            assert not call or not call.strip(), (
                f"With missing memory file, expected empty instructions, got: {call[:100]}"
            )


# ===========================================================================
# Test Class 2: generate_mutations direct test with memory
# ===========================================================================


class TestGenerateMutationsWithMemory:
    """Test generate_mutations function with memory_instructions kwarg."""

    @pytest.mark.asyncio
    async def test_memory_instructions_reach_operator(self) -> None:
        """Verify memory_instructions flows from generate_mutations to mutate_single."""
        captured = []

        async def mock_mutate(parents, memory_instructions=None):
            captured.append(memory_instructions)
            return MutationSpec(
                code=_make_code(2.0, 0.5),
                parents=parents,
                name="test",
            )

        mock_operator = MagicMock()
        mock_operator.mutate_single = mock_mutate

        parent = Program(code=SEED_CODE, metadata={})
        mock_storage = AsyncMock()
        mock_storage.add = AsyncMock(return_value="prog-id")
        mock_storage.get = AsyncMock(return_value=parent)
        mock_state = AsyncMock()
        mock_selector = MagicMock()
        mock_selector.create_parent_iterator.return_value = iter([[parent]])

        await generate_mutations(
            [parent],
            mutator=mock_operator,
            storage=mock_storage,
            state_manager=mock_state,
            parent_selector=mock_selector,
            limit=1,
            iteration=1,
            memory_instructions="Use annealing",
            memory_used=True,
        )

        assert len(captured) == 1
        assert captured[0] == "Use annealing"

    @pytest.mark.asyncio
    async def test_no_memory_instructions_sends_none(self) -> None:
        captured = []

        async def mock_mutate(parents, memory_instructions=None):
            captured.append(memory_instructions)
            return MutationSpec(
                code=_make_code(2.0, 0.5),
                parents=parents,
                name="test",
            )

        mock_operator = MagicMock()
        mock_operator.mutate_single = mock_mutate

        parent = Program(code=SEED_CODE, metadata={})
        mock_storage = AsyncMock()
        mock_storage.add = AsyncMock(return_value="prog-id")
        mock_storage.get = AsyncMock(return_value=parent)
        mock_state = AsyncMock()
        mock_selector = MagicMock()
        mock_selector.create_parent_iterator.return_value = iter([[parent]])

        await generate_mutations(
            [parent],
            mutator=mock_operator,
            storage=mock_storage,
            state_manager=mock_state,
            parent_selector=mock_selector,
            limit=1,
            iteration=1,
            memory_instructions=None,
        )

        assert len(captured) == 1
        assert captured[0] is None


# ===========================================================================
# Test Class 3: Two-phase cycle (fill memory → use memory)
# ===========================================================================


class TestMemoryFillThenUsePhases:
    """Two-phase cycle: evolve → fill memory → evolve with memory."""

    @pytest.mark.asyncio
    async def test_phase1_evolution_produces_programs(self) -> None:
        """Phase 1: run evolution, collect programs for idea extraction."""
        _reset_counter()
        server = fakeredis.FakeServer()
        operator = PlainIncrementOperator()

        engine, programs = await _run_evolution(
            server,
            max_generations=3,
            mutation_operator=operator,
        )

        assert engine.metrics.total_generations == 3
        assert len(programs) >= 1
        # All programs have fitness
        for p in programs:
            assert "fitness" in p.metrics

    @pytest.mark.asyncio
    async def test_phase2_memory_loaded_and_searchable(self, tmp_path) -> None:
        """Phase 2: fill memory from phase 1 programs, verify searchable."""
        # Simulate phase 1 output: save idea cards to memory
        mem = _make_memory(tmp_path)
        ideas = [
            {
                "id": "idea-sort",
                "description": "Sort evidence by relevance score",
                "keywords": ["sort", "relevance", "evidence"],
            },
            {
                "id": "idea-filter",
                "description": "Filter low-confidence hops",
                "keywords": ["filter", "confidence", "threshold"],
            },
            {
                "id": "idea-depth",
                "description": "Limit retrieval depth to 3 hops",
                "keywords": ["retrieval", "depth", "hops"],
            },
        ]
        for idea in ideas:
            mem.save_card(idea)

        # Reload (new process)
        mem2 = _make_memory(tmp_path)
        assert len(mem2.memory_cards) == 3

        # Search returns relevant cards
        result = mem2.search("sort evidence relevance")
        assert "idea-sort" in result

    @pytest.mark.asyncio
    async def test_full_two_phase_cycle(self, tmp_path) -> None:
        """Complete cycle: evolve → extract ideas → save memory → evolve with memory."""
        # Phase 1: Evolution without memory
        _reset_counter()
        server1 = fakeredis.FakeServer()
        operator1 = PlainIncrementOperator()
        engine1, programs1 = await _run_evolution(
            server1,
            max_generations=3,
            mutation_operator=operator1,
        )
        assert len(programs1) >= 1

        # Extract "ideas" from programs (simulating IdeaTracker)
        mem = _make_memory(tmp_path)
        for i, prog in enumerate(programs1):
            mem.save_card(
                {
                    "id": f"idea-from-gen-{i}",
                    "description": f"Technique from program with fitness {prog.metrics.get('fitness', 0):.1f}",
                    "keywords": ["optimization", f"gen{i}"],
                    "task_description": "Evolution optimization",
                }
            )

        # Phase 2: Evolution with memory
        _reset_counter()
        server2 = fakeredis.FakeServer()
        operator2 = MemoryAwareMutationOperator()

        # Write memory instructions file from ideas
        memory_file = tmp_path / "memory.txt"
        ideas_text = "\n".join(
            f"- {card['description']}" for card in mem.memory_cards.values()
        )
        memory_file.write_text(ideas_text)

        engine2, programs2 = await _run_evolution(
            server2,
            max_generations=3,
            mutation_operator=operator2,
            memory_enabled=True,
            memory_top_n=1,
            memory_path=str(memory_file),
        )

        assert engine2.metrics.total_generations == 3

        # Verify memory was used in phase 2
        assert len(operator2.calls_with_memory) > 0, (
            "Phase 2 should have used memory instructions"
        )

        # Phase 2 should show the ACTUAL content from phase 1 ideas
        non_empty_calls = [c for c in operator2.calls_with_memory if c]
        assert len(non_empty_calls) > 0, "All memory instruction calls were empty"
        for call in non_empty_calls:
            assert "Technique from program" in call, (
                f"Phase 2 memory should contain phase 1 ideas, got: {call[:200]}"
            )


# ===========================================================================
# Test Class 4: Memory selector in the mutation loop
# ===========================================================================


class TestMemorySelectorInMutationLoop:
    """Wire MemorySelectorAgent with real memory into the mutation flow."""

    @pytest.mark.asyncio
    async def test_selector_returns_cards_from_memory(self, tmp_path) -> None:
        """MemorySelectorAgent.select() returns cards from pre-filled memory."""
        from gigaevo.llm.agents.memory_selector import MemorySelectorAgent

        mem = _make_memory(tmp_path)
        mem.save_card(
            {
                "id": "idea-1",
                "description": "Sort evidence by relevance score for better chain quality",
                "keywords": ["sort", "relevance", "evidence", "chain"],
            }
        )
        mem.save_card(
            {
                "id": "idea-2",
                "description": "Filter low-confidence hops using threshold",
                "keywords": ["filter", "confidence", "threshold"],
            }
        )

        # Create selector with injected memory
        selector = MemorySelectorAgent.__new__(MemorySelectorAgent)
        selector._search_lock = asyncio.Lock()
        selector._backend_error = None
        selector.memory = mem

        parent = Program(
            code="def solve(x):\n    return x\n",
            metadata={},
        )

        selection = await selector.select(
            input=[parent],
            mutation_mode="rewrite",
            task_description="Multi-hop fact verification",
            metrics_description="fitness: accuracy on validation set",
            memory_text="",
            max_cards=3,
        )

        # Should find relevant cards
        assert len(selection.cards) > 0, (
            "Selector returned no cards from pre-filled memory"
        )

        # Card IDs should be extractable
        assert isinstance(selection.card_ids, list)

    @pytest.mark.asyncio
    async def test_selector_with_empty_memory_returns_empty(self, tmp_path) -> None:
        """Selector with no cards returns empty selection."""
        from gigaevo.llm.agents.memory_selector import MemorySelectorAgent

        mem = _make_memory(tmp_path)  # Empty

        selector = MemorySelectorAgent.__new__(MemorySelectorAgent)
        selector._search_lock = asyncio.Lock()
        selector._backend_error = None
        selector.memory = mem

        selection = await selector.select(
            input=[Program(code="def f(): pass", metadata={})],
            mutation_mode="rewrite",
            task_description="test",
            metrics_description="fitness",
            memory_text="",
            max_cards=3,
        )

        # Empty memory → "No relevant memories" → no cards parsed
        assert selection.cards == []
