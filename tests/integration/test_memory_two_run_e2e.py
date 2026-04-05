"""End-to-end: Run A extracts ideas → shared memory → Run B consumes them.

Tests the complete two-run memory lifecycle with all components mocked at the
LLM boundary:

  **Run A** (idea extraction):
    Programs evolve → IdeaTracker filters + converts to ProgramRecords
    → Ideas saved to AmemGamMemory as cards

  **Run B** (idea consumption):
    MemorySelectorAgent reads cards from AmemGamMemory
    → MemoryProvider selects cards → MemoryContextStage sets card IDs
    → MutationContextStage includes "## Memory Instructions"
    → generate_mutations auto-derives memory_used=True on children

Real objects: AmemGamMemory, MemorySelectorAgent, EvolutionEngine (fakeredis),
  IdeaTracker (mocked LLM), MemoryContextStage, MutationContextStage.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
from unittest.mock import AsyncMock, MagicMock, patch

import fakeredis.aioredis
import pytest

from gigaevo.database.redis import RedisProgramStorageConfig
from gigaevo.database.redis_program_storage import RedisProgramStorage
from gigaevo.database.state_manager import ProgramStateManager
from gigaevo.evolution.engine.config import EngineConfig
from gigaevo.evolution.engine.core import EvolutionEngine
from gigaevo.evolution.engine.hooks import PostRunHook
from gigaevo.evolution.mutation.base import MutationOperator, MutationSpec
from gigaevo.evolution.mutation.constants import (
    MUTATION_CONTEXT_METADATA_KEY,
    MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY,
)
from gigaevo.evolution.strategies.elite_selectors import ScalarTournamentEliteSelector
from gigaevo.evolution.strategies.island import IslandConfig
from gigaevo.evolution.strategies.migrant_selectors import RandomMigrantSelector
from gigaevo.evolution.strategies.models import BehaviorSpace, LinearBinning
from gigaevo.evolution.strategies.multi_island import MapElitesMultiIsland
from gigaevo.evolution.strategies.removers import FitnessArchiveRemover
from gigaevo.evolution.strategies.selectors import SumArchiveSelector
from gigaevo.llm.agents.memory_selector import MemorySelectorAgent
from gigaevo.memory.ideas_tracker.utils.helpers import (
    build_memory_usage_updates_from_programs,
)
from gigaevo.memory.ideas_tracker.utils.records_converter import (
    programs_to_records,
)
from gigaevo.memory.provider import MemoryProvider, SelectorMemoryProvider
from gigaevo.memory.shared_memory.memory import AmemGamMemory
from gigaevo.programs.metrics.context import MetricsContext, MetricSpec
from gigaevo.programs.program import Lineage, Program
from gigaevo.programs.program_state import ProgramState
from gigaevo.programs.stages.common import StringContainer
from gigaevo.programs.stages.memory_context import MemoryContextStage
from gigaevo.programs.stages.mutation_context import MutationContextStage
from tests.fakes.agentic_memory import make_test_memory

# ---------------------------------------------------------------------------
# Helpers: deterministic mutation + code templates
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
# Helpers: FakeDagRunner with memory awareness
# ---------------------------------------------------------------------------


class MemoryAwareFakeDagRunner:
    """Simulates the real DAG pipeline's memory stage + validation."""

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

        selection = await self._memory_provider.select_cards(
            prog,
            task_description="Multi-hop verification",
            metrics_description="fitness",
        )
        if selection.cards:
            prog.set_metadata(
                MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY, selection.card_ids
            )

        metrics = _extract_metrics(prog.code)
        prog.add_metrics(metrics)
        await self._sm.set_program_state(prog, ProgramState.DONE)


# ---------------------------------------------------------------------------
# Helpers: infrastructure
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
    post_run_hook: PostRunHook | None = None,
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
        post_run_hook=post_run_hook,
    )
    return engine, strategy


async def _add_seed(storage: RedisProgramStorage) -> Program:
    seed = Program(code=_make_code(fitness=1.0, x=0.0), state=ProgramState.QUEUED)
    await storage.add(seed)
    return seed


def _make_memory(tmp_path, **overrides) -> AmemGamMemory:
    return make_test_memory(tmp_path, **overrides)


def _make_selector(memory: AmemGamMemory) -> MemorySelectorAgent:
    """Create a MemorySelectorAgent wired to the given memory (no __init__ side effects)."""
    selector = MemorySelectorAgent.__new__(MemorySelectorAgent)
    selector._search_lock = asyncio.Lock()
    selector._backend_error = None
    selector.memory = memory
    return selector


def _make_run_a_programs() -> list[Program]:
    """Create a population simulating Run A's evolution output.

    Returns programs with realistic lineage and improvements for IdeaTracker.
    """
    seed = Program(
        code="def entrypoint():\n    pass",
        state=ProgramState.DONE,
        metrics={"fitness": 1.0},
        metadata={},
        lineage=Lineage(parents=[], generation=1),
    )

    child1 = Program(
        code="def entrypoint():\n    return sort(evidence)",
        state=ProgramState.DONE,
        metrics={"fitness": 5.0},
        metadata={
            "mutation_output": {
                "archetype": "exploitation",
                "insights_used": ["relevance sorting"],
                "changes": [
                    "Sort evidence by relevance score before building chain",
                    "Filter low-confidence hops using threshold",
                ],
            }
        },
        lineage=Lineage(parents=[seed.id], generation=2),
    )

    child2 = Program(
        code="def entrypoint():\n    return bfs(graph)",
        state=ProgramState.DONE,
        metrics={"fitness": 7.0},
        metadata={
            "mutation_output": {
                "archetype": "exploration",
                "insights_used": [],
                "changes": [
                    "Use BFS instead of DFS for multi-hop traversal",
                    "Cache intermediate retrieval results",
                ],
            }
        },
        lineage=Lineage(parents=[seed.id], generation=2),
    )

    child3 = Program(
        code="def entrypoint():\n    return combine(bfs, sort)",
        state=ProgramState.DONE,
        metrics={"fitness": 9.0},
        metadata={
            "mutation_output": {
                "archetype": "exploitation",
                "insights_used": ["BFS traversal", "sorting"],
                "changes": [
                    "Combine BFS traversal with relevance sorting",
                ],
            }
        },
        lineage=Lineage(parents=[child1.id, child2.id], generation=3),
    )

    return [seed, child1, child2, child3]


# ===========================================================================
# Tests
# ===========================================================================


class TestTwoRunMemoryLifecycle:
    """Full lifecycle: Run A extracts ideas → shared memory → Run B consumes them."""

    @pytest.fixture(autouse=True)
    def _reset(self):
        _reset_counter()
        yield

    @pytest.mark.asyncio
    async def test_run_a_ideas_flow_to_run_b_via_memory(self, tmp_path) -> None:
        """Complete two-run cycle:
        Run A: programs → IdeaTracker extracts → ideas saved to memory
        Run B: memory → selector → cards in mutation context → memory_used=True
        """
        # === RUN A: Extract ideas from evolved programs ===
        run_a_programs = _make_run_a_programs()

        # 1. IdeaTracker filters programs and converts to records
        with (
            patch(
                "gigaevo.memory.ideas_tracker.components.fabrics.llm_clients_fabric._create_llm_clients",
                return_value=(MagicMock(), MagicMock(), False),
            ),
            patch(
                "gigaevo.memory.ideas_tracker.ideas_tracker.summarize_task_description",
                return_value="Multi-hop fact verification",
            ),
        ):
            from gigaevo.memory.ideas_tracker.ideas_tracker import IdeaTracker

            tracker = IdeaTracker(
                task_description="Verify multi-hop claims using evidence chains",
                memory_write_enabled=False,
                memory_usage_tracking_enabled=False,
            )

        records = tracker._get_new_programs(run_a_programs)
        # Root (no parents) is filtered, 3 children remain
        assert len(records) == 3
        assert all(r.fitness > 0 for r in records)

        # Verify records carry mutation metadata
        best_record = max(records, key=lambda r: r.fitness)
        assert best_record.fitness == 9.0
        assert best_record.strategy == "exploitation"

        # 2. Memory usage tracking works with real programs
        usage = build_memory_usage_updates_from_programs(run_a_programs, "HoVer")
        # No memory cards were used in Run A, so usage is empty
        assert usage == {}

        # === BRIDGE: Save extracted ideas as memory cards ===
        memory = _make_memory(tmp_path)

        # Simulate ideas that IdeaTracker's analyzer would have extracted
        idea_cards = [
            {
                "id": "idea-sort",
                "description": "Sort evidence by relevance score before chain building",
                "keywords": ["sort", "relevance", "evidence", "chain"],
            },
            {
                "id": "idea-bfs",
                "description": "Use BFS instead of DFS for multi-hop traversal",
                "keywords": ["bfs", "traversal", "multi-hop", "graph"],
            },
            {
                "id": "idea-cache",
                "description": "Cache intermediate retrieval results to reduce latency",
                "keywords": ["cache", "retrieval", "latency", "performance"],
            },
        ]
        for card in idea_cards:
            memory.save_card(card)

        # Verify at least one card was saved successfully
        assert memory.get_card("idea-sort") is not None

        # === RUN B: Use memory during evolution ===
        selector = _make_selector(memory)

        # MemorySelectorAgent.select() finds relevant cards
        seed_prog = Program(
            code="def entrypoint():\n    return evidence",
            metadata={},
        )
        selection = await selector.select(
            input=[seed_prog],
            mutation_mode="rewrite",
            task_description="Multi-hop fact verification",
            metrics_description="fitness: accuracy on validation set",
            memory_text="",
            max_cards=3,
        )
        assert len(selection.cards) > 0, "Selector found no cards from populated memory"
        assert len(selection.card_ids) > 0

        # Wire selector into SelectorMemoryProvider
        provider = SelectorMemoryProvider(max_cards=3)
        provider._selector = selector

        # Run actual evolution with memory-aware DAG
        server = fakeredis.FakeServer()
        storage = _make_fakeredis_storage(server)
        await _add_seed(storage)

        engine, strategy = _build_engine(storage, max_generations=3)
        sm = ProgramStateManager(storage)
        runner = MemoryAwareFakeDagRunner(storage, sm, provider)

        runner.start()
        engine.start()
        try:
            await asyncio.wait_for(engine.task, timeout=30.0)
        except TimeoutError:
            pytest.fail("Engine did not finish within 30s")
        finally:
            await runner.stop()

        # === VERIFY: Run B children have memory metadata ===
        all_programs = await storage.get_all_by_status(ProgramState.DONE.value)
        children = [p for p in all_programs if p.lineage.parents]
        assert len(children) >= 2, f"Expected >= 2 children, got {len(children)}"

        for child in children:
            assert child.get_metadata("memory_used") is True, (
                f"Child {child.short_id} missing memory_used=True"
            )

        # All evaluated programs should have card IDs from Run A's memory
        for prog in all_programs:
            card_ids = prog.metadata.get(MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY)
            assert card_ids is not None, (
                f"Program {prog.short_id} missing card IDs from memory"
            )
            assert len(card_ids) > 0

    @pytest.mark.asyncio
    async def test_run_b_mutation_context_includes_memory_instructions(
        self, tmp_path
    ) -> None:
        """Verify that memory cards from Run A appear in MutationContextStage output."""
        # Setup memory with ideas from "Run A"
        memory = _make_memory(tmp_path)
        memory.save_card(
            {
                "id": "idea-relevance",
                "description": "Sort evidence by relevance score",
                "keywords": ["sort", "relevance", "evidence"],
            }
        )

        selector = _make_selector(memory)

        # MemoryContextStage produces card text
        provider = SelectorMemoryProvider(max_cards=3)
        provider._selector = selector

        memory_stage = MemoryContextStage(
            memory_provider=provider,
            task_description="Multi-hop fact verification",
            metrics_description="fitness: accuracy",
            timeout=60,
        )
        program = Program(code="def solve(): return search(evidence)")

        memory_output = await memory_stage.compute(program)
        assert isinstance(memory_output, StringContainer)
        assert len(memory_output.data) > 0

        # Card IDs written to program metadata
        assert MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY in program.metadata

        # MutationContextStage wraps it as "## Memory Instructions"
        ctx_stage = MutationContextStage(
            metrics_context=MetricsContext(
                specs={
                    "fitness": MetricSpec(
                        description="accuracy",
                        is_primary=True,
                        higher_is_better=True,
                        lower_bound=0.0,
                        upper_bound=1.0,
                    ),
                }
            ),
            timeout=60,
        )
        ctx_stage._raw_inputs = {
            "metrics": None,
            "insights": None,
            "lineage_ancestors": None,
            "lineage_descendants": None,
            "evolutionary_statistics": None,
            "formatted": None,
            "memory": memory_output,
        }
        ctx_stage._params_obj = None

        ctx_output = await ctx_stage.compute(program)
        context_str = ctx_output.data

        assert "Memory Instructions" in context_str
        assert MUTATION_CONTEXT_METADATA_KEY in program.metadata

    @pytest.mark.asyncio
    async def test_memory_usage_tracking_across_runs(self, tmp_path) -> None:
        """Run B programs that used memory cards have correct usage deltas."""
        # Simulate Run B output: child used memory cards and improved
        parent = Program(
            code="def f(): pass",
            state=ProgramState.DONE,
            metrics={"fitness": 3.0},
            metadata={},
            lineage=Lineage(parents=[], generation=1),
        )

        child = Program(
            code="def f(): return sorted(x)",
            state=ProgramState.DONE,
            metrics={"fitness": 8.0},
            metadata={"memory_selected_idea_ids": ["idea-sort", "idea-bfs"]},
            lineage=Lineage(parents=[parent.id], generation=2),
        )

        usage = build_memory_usage_updates_from_programs(
            [parent, child], "HoVer verification"
        )

        # Both cards get attributed the same delta (8.0 - 3.0 = 5.0)
        assert "idea-sort" in usage
        assert "idea-bfs" in usage

        sort_total = usage["idea-sort"]["used"]["total"]
        assert sort_total["total_used"] == 1
        assert sort_total["median_delta_fitness"] == 5.0

    @pytest.mark.asyncio
    async def test_post_run_hook_fires_after_run_b(self, tmp_path) -> None:
        """PostRunHook fires after Run B completes, receiving all programs."""
        # Recording hook
        captured: list[Program] = []

        class RecordingHook(PostRunHook):
            async def on_run_complete(self, stor) -> None:
                from gigaevo.programs.program import EXCLUDE_STAGE_RESULTS

                programs = await stor.get_all(exclude=EXCLUDE_STAGE_RESULTS)
                captured.extend(programs)

        # Use NullMemoryProvider for simplicity (hook verification is the goal)
        from gigaevo.memory.provider import NullMemoryProvider

        server = fakeredis.FakeServer()
        storage = _make_fakeredis_storage(server)
        await _add_seed(storage)

        engine, strategy = _build_engine(
            storage, max_generations=2, post_run_hook=RecordingHook()
        )
        sm = ProgramStateManager(storage)
        runner = MemoryAwareFakeDagRunner(storage, sm, NullMemoryProvider())

        runner.start()
        engine.start()
        try:
            await asyncio.wait_for(engine.task, timeout=30.0)
        except TimeoutError:
            pytest.fail("Engine did not finish within 30s")
        finally:
            await runner.stop()

        # Hook received all programs
        assert len(captured) >= 2
        # All programs should be in captured (hook ran and fetched from storage)
        assert all(isinstance(p, Program) for p in captured)


class TestRunAIdeaTrackerProgramNative:
    """Run A side: IdeaTracker works with Program objects directly."""

    def test_records_converter_maps_all_fields(self) -> None:
        """program_to_record correctly maps Program → ProgramRecord."""
        programs = _make_run_a_programs()
        records, ids = programs_to_records(
            programs, "Verify claims", "Multi-hop verification"
        )

        assert len(records) == len(programs)
        assert ids == {p.id for p in programs}

        # Best program record
        best = max(records, key=lambda r: r.fitness)
        assert best.fitness == 9.0
        assert best.generation == 3
        assert best.task_description == "Verify claims"
        assert best.task_description_summary == "Multi-hop verification"
        assert best.strategy == "exploitation"
        assert len(best.improvements) > 0
        assert "Combine BFS" in best.improvements[0]["description"]

    def test_idea_tracker_filters_correctly(self) -> None:
        """IdeaTracker._get_new_programs filters roots and zero-fitness."""
        with (
            patch(
                "gigaevo.memory.ideas_tracker.components.fabrics.llm_clients_fabric._create_llm_clients",
                return_value=(MagicMock(), MagicMock(), False),
            ),
            patch(
                "gigaevo.memory.ideas_tracker.ideas_tracker.summarize_task_description",
                return_value="Summary",
            ),
        ):
            from gigaevo.memory.ideas_tracker.ideas_tracker import IdeaTracker

            tracker = IdeaTracker(
                task_description="Test",
                memory_write_enabled=False,
                memory_usage_tracking_enabled=False,
            )

        programs = _make_run_a_programs()
        records = tracker._get_new_programs(programs)

        # Root (seed, no parents) is filtered
        assert len(records) == 3
        # All records have positive fitness
        assert all(r.fitness > 0 for r in records)
        # IDs tracked for deduplication
        assert len(tracker.programs_ids) == 3

    @pytest.mark.asyncio
    async def test_on_run_complete_calls_pipeline(self) -> None:
        """on_run_complete fetches programs from storage and runs pipeline."""
        with (
            patch(
                "gigaevo.memory.ideas_tracker.components.fabrics.llm_clients_fabric._create_llm_clients",
                return_value=(MagicMock(), MagicMock(), False),
            ),
            patch(
                "gigaevo.memory.ideas_tracker.ideas_tracker.summarize_task_description",
                return_value="Summary",
            ),
        ):
            from gigaevo.memory.ideas_tracker.ideas_tracker import IdeaTracker

            tracker = IdeaTracker(
                task_description="Test",
                memory_write_enabled=False,
                memory_usage_tracking_enabled=False,
            )

        tracker._run_on_programs = MagicMock()

        storage = AsyncMock()
        storage.get_all.return_value = _make_run_a_programs()

        await tracker.on_run_complete(storage)

        tracker._run_on_programs.assert_called_once()
        # Programs passed to pipeline
        passed = tracker._run_on_programs.call_args[0][0]
        assert len(passed) == 4  # seed + 3 children
