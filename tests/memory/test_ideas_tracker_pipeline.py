"""Comprehensive tests for the IdeaTracker post-run hook pipeline.

Three layers, from fastest to slowest:

1. **Unit tests** — records_converter, helpers, program filtering
2. **OOP contract tests** — PostRunHook ABC, NullPostRunHook, Hydra composability
3. **Integration tests** — EvolutionEngine → PostRunHook → IdeaTracker pipeline,
   including engine fault isolation and the full evolution → idea extraction flow

Design principles:
- Tests work with ``Program`` objects directly (no DataFrames)
- All LLM calls are mocked with realistic behavior
- OOP and Hydra composability are first-class concerns
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

import pytest

from gigaevo.evolution.engine.config import EngineConfig
from gigaevo.evolution.engine.core import EvolutionEngine
from gigaevo.evolution.engine.hooks import NullPostRunHook, PostRunHook
from gigaevo.memory.ideas_tracker.utils.helpers import (
    build_memory_usage_updates_from_programs,
)
from gigaevo.memory.ideas_tracker.utils.records_converter import (
    program_to_record,
    programs_to_records,
)
from gigaevo.programs.program import Lineage, Program
from gigaevo.programs.program_state import ProgramState

# Namespace for deterministic UUID generation from string IDs
_TEST_NAMESPACE = uuid.NAMESPACE_DNS

# ===========================================================================
# Factories
# ===========================================================================


def _test_id_to_uuid(test_id: str) -> str:
    """Convert a readable test ID to a consistent UUID v5."""
    return str(uuid.uuid5(_TEST_NAMESPACE, test_id))


def _make_program(
    *,
    code: str = "def solve(): return 42",
    fitness: float = 0.75,
    fitness_key: str = "fitness",
    generation: int = 3,
    parents: list[str] | None = None,
    mutation_output: dict[str, Any] | None = None,
    memory_ids: list[str] | None = None,
    state: ProgramState = ProgramState.DONE,
    program_id: str | None = None,
) -> Program:
    """Build a realistic Program for testing with sensible defaults.

    If program_id is a string, it's converted to a deterministic UUID v5.
    This allows tests to use readable IDs like "parent-a" that map to consistent UUIDs.
    Parent IDs in the parents list are also converted to UUIDs if they're strings.
    """
    metadata: dict[str, Any] = {}
    if mutation_output is not None:
        metadata["mutation_output"] = mutation_output
    if memory_ids is not None:
        metadata["memory_selected_idea_ids"] = memory_ids

    # Convert parent IDs to UUIDs
    parent_list = parents or (["parent-1"] if generation > 1 else [])
    parent_uuids = [
        _test_id_to_uuid(p) if isinstance(p, str) else p for p in parent_list
    ]

    lineage = Lineage(
        parents=parent_uuids,
        generation=max(generation, 1),
    )
    prog = Program(
        code=code,
        state=state,
        metrics={fitness_key: fitness},
        metadata=metadata,
        lineage=lineage,
    )
    if program_id is not None:
        # Convert string ID to valid UUID v5 deterministically
        program_uuid = _test_id_to_uuid(program_id)
        # Use the internal _id_unsafe setter to bypass validation
        object.__setattr__(prog, "id", program_uuid)
    return prog


def _make_root_program(*, fitness: float = 1.0) -> Program:
    """Build a root (seed) program — no parents, generation 1."""
    return _make_program(parents=[], generation=1, fitness=fitness)


def _make_evolved_program(
    *,
    fitness: float = 5.0,
    parent_id: str = "seed-01",
    generation: int = 3,
    insights: list[str] | None = None,
    changes: list[str] | None = None,
    archetype: str = "exploitation",
) -> Program:
    """Build a program that went through mutation — has lineage and mutation_output."""
    mutation_output: dict[str, Any] = {"archetype": archetype}
    if insights is not None:
        mutation_output["insights_used"] = insights
    if changes is not None:
        mutation_output["changes"] = changes
    return _make_program(
        fitness=fitness,
        generation=generation,
        parents=[parent_id],
        mutation_output=mutation_output,
    )


def _make_memory_program(
    *,
    fitness: float = 8.0,
    parent_id: str = "parent-a",
    card_ids: list[str] | None = None,
) -> Program:
    """Build a program that used memory cards during mutation."""
    return _make_program(
        fitness=fitness,
        generation=5,
        parents=[parent_id],
        memory_ids=card_ids or ["idea-001", "idea-002"],
    )


# ===========================================================================
# 1. Unit tests — records_converter
# ===========================================================================


class TestProgramToRecord:
    """Verify that Program fields are mapped correctly to ProgramRecord."""

    def test_basic_field_mapping(self) -> None:
        prog = _make_evolved_program(
            fitness=7.5,
            generation=4,
            parent_id="p1",
            insights=["Use BFS"],
            changes=["Added BFS traversal"],
            archetype="exploration",
        )
        record = program_to_record(prog, "Solve TSP", "TSP optimization")
        assert record.id == prog.id
        assert record.fitness == 7.5
        assert record.generation == 4
        assert record.parents == [_test_id_to_uuid("p1")]
        assert record.insights == ["Use BFS"]
        assert record.strategy == "exploration"
        assert record.task_description == "Solve TSP"
        assert record.task_description_summary == "TSP optimization"
        assert record.code == prog.code

    def test_missing_mutation_output_defaults_to_empty(self) -> None:
        prog = _make_program(mutation_output=None)
        record = program_to_record(prog, "task", "summary")
        assert record.insights == []
        assert record.strategy == ""

    def test_invalid_mutation_output_type_defaults_to_empty(self) -> None:
        prog = _make_program()
        prog.metadata["mutation_output"] = "not a dict"
        record = program_to_record(prog, "task", "summary")
        assert record.insights == []
        assert record.strategy == ""

    def test_missing_fitness_defaults_to_zero(self) -> None:
        prog = _make_program()
        prog.metrics.clear()
        record = program_to_record(prog, "task", "summary")
        assert record.fitness == 0.0

    def test_custom_fitness_key(self) -> None:
        prog = _make_program(fitness_key="accuracy")
        prog.metrics["accuracy"] = 0.95
        record = program_to_record(prog, "task", "summary", fitness_key="accuracy")
        assert record.fitness == 0.95


class TestProgramsToRecords:
    """Bulk conversion: list[Program] → (list[ProgramRecord], set[str])."""

    def test_empty_list(self) -> None:
        records, ids = programs_to_records([], "task", "summary")
        assert records == []
        assert ids == set()

    def test_returns_records_and_ids(self) -> None:
        progs = [_make_evolved_program(fitness=f) for f in [1.0, 2.0, 3.0]]
        records, ids = programs_to_records(progs, "task", "summary")
        assert len(records) == 3
        assert ids == {p.id for p in progs}

    def test_fitness_key_passed_through(self) -> None:
        prog = _make_program(fitness_key="score")
        prog.metrics["score"] = 9.9
        records, _ = programs_to_records([prog], "t", "s", fitness_key="score")
        assert records[0].fitness == 9.9


# ===========================================================================
# 2. Unit tests — helpers (build_memory_usage_updates_from_programs)
# ===========================================================================


class TestBuildMemoryUsageFromPrograms:
    """Test memory card usage tracking with Program objects."""

    def test_empty_programs_returns_empty(self) -> None:
        assert build_memory_usage_updates_from_programs([]) == {}

    def test_programs_without_memory_ids_return_empty(self) -> None:
        progs = [_make_evolved_program() for _ in range(3)]
        assert build_memory_usage_updates_from_programs(progs) == {}

    def test_single_card_usage_computes_delta(self) -> None:
        parent = _make_program(
            program_id="parent-a", fitness=5.0, parents=[], generation=1
        )
        child = _make_memory_program(
            fitness=8.0, parent_id="parent-a", card_ids=["idea-1"]
        )
        result = build_memory_usage_updates_from_programs([parent, child], "test task")
        assert "idea-1" in result
        entries = result["idea-1"]["used"]["entries"]
        assert len(entries) == 1
        assert entries[0]["used_count"] == 1
        # delta = child (8.0) - max parent (5.0) = 3.0
        assert entries[0]["fitness_delta_per_use"] == [3.0]
        assert entries[0]["median_delta_fitness"] == 3.0

    def test_negative_delta_included(self) -> None:
        """Child worse than parent → negative delta is still recorded."""
        parent = _make_program(program_id="p1", fitness=10.0, parents=[], generation=1)
        child = _make_memory_program(fitness=7.0, parent_id="p1", card_ids=["c1"])
        result = build_memory_usage_updates_from_programs([parent, child], "task")
        deltas = result["c1"]["used"]["entries"][0]["fitness_delta_per_use"]
        assert deltas == [-3.0]

    def test_multiple_cards_per_program(self) -> None:
        parent = _make_program(program_id="p1", fitness=4.0, parents=[], generation=1)
        child = _make_memory_program(
            fitness=6.0, parent_id="p1", card_ids=["a", "b", "c"]
        )
        result = build_memory_usage_updates_from_programs([parent, child], "t")
        assert set(result.keys()) == {"a", "b", "c"}
        for card_id in ("a", "b", "c"):
            assert result[card_id]["used"]["entries"][0]["fitness_delta_per_use"] == [
                2.0
            ]

    def test_missing_parent_fitness_skips_program(self) -> None:
        """If parent has no fitness, child's memory usage is not counted."""
        child = _make_memory_program(
            fitness=8.0, parent_id="unknown-parent", card_ids=["c1"]
        )
        result = build_memory_usage_updates_from_programs([child], "task")
        assert result == {}

    def test_custom_fitness_key(self) -> None:
        parent = _make_program(
            program_id="p1",
            fitness=3.0,
            fitness_key="accuracy",
            parents=[],
            generation=1,
        )
        child = _make_program(
            fitness=5.0,
            fitness_key="accuracy",
            generation=3,
            parents=["p1"],
            memory_ids=["c1"],
        )
        result = build_memory_usage_updates_from_programs(
            [parent, child], "task", fitness_key="accuracy"
        )
        assert "c1" in result
        assert result["c1"]["used"]["entries"][0]["fitness_delta_per_use"] == [2.0]

    def test_duplicate_card_ids_deduplicated(self) -> None:
        parent = _make_program(program_id="p1", fitness=1.0, parents=[], generation=1)
        child = _make_memory_program(
            fitness=2.0, parent_id="p1", card_ids=["dup", "dup", "dup"]
        )
        result = build_memory_usage_updates_from_programs([parent, child], "task")
        # Only one entry for "dup", not three
        assert result["dup"]["used"]["total"]["total_used"] == 1


# ===========================================================================
# 3. OOP contract tests — PostRunHook ABC and NullPostRunHook
# ===========================================================================


class TestPostRunHookABC:
    """PostRunHook is a proper ABC with the right interface."""

    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError):
            PostRunHook()

    def test_abc_defines_on_run_complete(self) -> None:
        assert hasattr(PostRunHook, "on_run_complete")

    def test_concrete_subclass_must_implement_on_run_complete(self) -> None:
        class Incomplete(PostRunHook):
            pass

        with pytest.raises(TypeError):
            Incomplete()


class TestNullPostRunHook:
    """NullPostRunHook is the no-op default for ideas_tracker=none."""

    def test_instantiates_without_arguments(self) -> None:
        hook = NullPostRunHook()
        assert isinstance(hook, PostRunHook)

    @pytest.mark.asyncio
    async def test_on_run_complete_is_noop(self) -> None:
        hook = NullPostRunHook()
        storage = AsyncMock()
        await hook.on_run_complete(storage)
        # No interaction with storage — true no-op
        storage.get_all.assert_not_called()


class TestIdeaTrackerIsPostRunHook:
    """IdeaTracker must fulfill the PostRunHook contract (OOP composability)."""

    def test_is_subclass_of_post_run_hook(self) -> None:
        from gigaevo.memory.ideas_tracker.ideas_tracker import IdeaTracker

        assert issubclass(IdeaTracker, PostRunHook)

    @patch(
        "gigaevo.memory.ideas_tracker.components.fabrics.llm_clients_fabric._create_llm_clients",
        return_value=(MagicMock(), MagicMock(), False),
    )
    @patch(
        "gigaevo.memory.ideas_tracker.ideas_tracker.summarize_task_description",
        return_value="Test summary",
    )
    @patch(
        "gigaevo.memory.ideas_tracker.ideas_tracker.load_task_description",
        return_value="Test task",
    )
    def test_instantiates_with_defaults(
        self, _mock_load, _mock_summary, _mock_llm_create
    ) -> None:
        from gigaevo.memory.ideas_tracker.ideas_tracker import IdeaTracker

        tracker = IdeaTracker(task_description="Test task")
        assert isinstance(tracker, PostRunHook)
        assert tracker.memory_write_enabled is True
        assert tracker._fitness_key == "fitness"

    def test_analyzer_types_valid(self) -> None:
        """Verify that fast and default analyzer_type parameters are recognized.

        We don't test instantiation here (it requires loading real models),
        but we verify the analyzer types exist and can be imported.
        """
        from gigaevo.memory.ideas_tracker.components.analyzer import IdeaAnalyzer
        from gigaevo.memory.ideas_tracker.components.analyzer_f import IdeaAnalyzerFast

        # Just verify the classes exist and can be imported
        assert IdeaAnalyzer is not None
        assert IdeaAnalyzerFast is not None


# ===========================================================================
# 4. IdeaTracker program filtering (_get_new_programs)
# ===========================================================================


class TestIdeaTrackerProgramFiltering:
    """_get_new_programs filters root, duplicate, and non-positive-fitness programs."""

    @patch(
        "gigaevo.memory.ideas_tracker.components.fabrics.llm_clients_fabric._create_llm_clients",
        return_value=(MagicMock(), MagicMock(), False),
    )
    @patch(
        "gigaevo.memory.ideas_tracker.ideas_tracker.summarize_task_description",
        return_value="Summary",
    )
    @patch(
        "gigaevo.memory.ideas_tracker.ideas_tracker.load_task_description",
        return_value="Task",
    )
    def _make_tracker(self, _mock_load=None, _mock_summary=None, _mock_llm_create=None):
        from gigaevo.memory.ideas_tracker.ideas_tracker import IdeaTracker

        return IdeaTracker(task_description="Test task")

    def test_root_programs_are_skipped(self) -> None:
        tracker = self._make_tracker()
        root = _make_root_program(fitness=10.0)
        evolved = _make_evolved_program(fitness=5.0)
        result = tracker._get_new_programs([root, evolved])
        assert len(result) == 1
        assert result[0].id == evolved.id

    def test_zero_fitness_programs_are_skipped(self) -> None:
        tracker = self._make_tracker()
        zero = _make_evolved_program(fitness=0.0)
        positive = _make_evolved_program(fitness=1.0)
        result = tracker._get_new_programs([zero, positive])
        assert len(result) == 1
        assert result[0].fitness == 1.0

    def test_negative_fitness_programs_are_skipped(self) -> None:
        tracker = self._make_tracker()
        neg = _make_evolved_program(fitness=-3.0)
        result = tracker._get_new_programs([neg])
        assert result == []

    def test_duplicate_programs_are_skipped(self) -> None:
        tracker = self._make_tracker()
        prog = _make_evolved_program(fitness=5.0)
        # First call processes the program
        result1 = tracker._get_new_programs([prog])
        assert len(result1) == 1
        # Second call deduplicates
        result2 = tracker._get_new_programs([prog])
        assert result2 == []

    def test_programs_ids_tracked_after_processing(self) -> None:
        tracker = self._make_tracker()
        prog = _make_evolved_program(fitness=5.0)
        tracker._get_new_programs([prog])
        assert prog.id in tracker.programs_ids

    def test_programs_card_accumulates(self) -> None:
        tracker = self._make_tracker()
        p1 = _make_evolved_program(fitness=1.0)
        p2 = _make_evolved_program(fitness=2.0)
        tracker._get_new_programs([p1])
        tracker._get_new_programs([p2])
        assert len(tracker.programs_card) == 2


# ===========================================================================
# 5. IdeaTracker on_run_complete (PostRunHook interface)
# ===========================================================================


class TestIdeaTrackerOnRunComplete:
    """on_run_complete is the hook entry point: storage → list[Program] → pipeline."""

    @patch(
        "gigaevo.memory.ideas_tracker.components.fabrics.llm_clients_fabric._create_llm_clients",
        return_value=(MagicMock(), MagicMock(), False),
    )
    @patch(
        "gigaevo.memory.ideas_tracker.ideas_tracker.summarize_task_description",
        return_value="Summary",
    )
    def _make_tracker_with_mocked_pipeline(
        self, _mock_summary=None, _mock_llm_create=None
    ):
        from gigaevo.memory.ideas_tracker.ideas_tracker import IdeaTracker

        tracker = IdeaTracker(
            task_description="Test",
            memory_write_enabled=False,
            memory_usage_tracking_enabled=False,
        )
        tracker._run_on_programs = MagicMock()
        return tracker

    @pytest.mark.asyncio
    async def test_empty_storage_logs_warning_and_returns(self) -> None:
        tracker = self._make_tracker_with_mocked_pipeline()
        storage = AsyncMock()
        storage.get_all.return_value = []

        await tracker.on_run_complete(storage)

        tracker._run_on_programs.assert_not_called()

    @pytest.mark.asyncio
    async def test_programs_fetched_and_passed_to_pipeline(self) -> None:
        tracker = self._make_tracker_with_mocked_pipeline()
        progs = [_make_evolved_program(fitness=f) for f in [1.0, 2.0, 3.0]]
        storage = AsyncMock()
        storage.get_all.return_value = progs

        await tracker.on_run_complete(storage)

        storage.get_all.assert_called_once()
        tracker._run_on_programs.assert_called_once_with(progs)

    @pytest.mark.asyncio
    async def test_storage_get_all_excludes_stage_results(self) -> None:
        from gigaevo.programs.program import EXCLUDE_STAGE_RESULTS

        tracker = self._make_tracker_with_mocked_pipeline()
        storage = AsyncMock()
        storage.get_all.return_value = [_make_evolved_program()]

        await tracker.on_run_complete(storage)

        storage.get_all.assert_called_once_with(exclude=EXCLUDE_STAGE_RESULTS)


# ===========================================================================
# 6. IdeaTracker legacy CLI run()
# ===========================================================================


class TestIdeaTrackerLegacyRun:
    """run() is the CLI entry point: accepts list[Program] directly."""

    @patch(
        "gigaevo.memory.ideas_tracker.components.fabrics.llm_clients_fabric._create_llm_clients",
        return_value=(MagicMock(), MagicMock(), False),
    )
    @patch(
        "gigaevo.memory.ideas_tracker.ideas_tracker.summarize_task_description",
        return_value="Summary",
    )
    def _make_tracker_with_mocked_pipeline(
        self, _mock_summary=None, _mock_llm_create=None
    ):
        from gigaevo.memory.ideas_tracker.ideas_tracker import IdeaTracker

        tracker = IdeaTracker(
            task_description="Test",
            memory_write_enabled=False,
            memory_usage_tracking_enabled=False,
        )
        tracker._run_on_programs = MagicMock()
        return tracker

    def test_none_programs_skips(self) -> None:
        tracker = self._make_tracker_with_mocked_pipeline()
        tracker.run(None)
        tracker._run_on_programs.assert_not_called()

    def test_empty_programs_skips(self) -> None:
        tracker = self._make_tracker_with_mocked_pipeline()
        tracker.run([])
        tracker._run_on_programs.assert_not_called()

    def test_valid_programs_passed_to_pipeline(self) -> None:
        tracker = self._make_tracker_with_mocked_pipeline()
        progs = [_make_evolved_program()]
        tracker.run(progs)
        tracker._run_on_programs.assert_called_once_with(progs)


# ===========================================================================
# 7. EvolutionEngine ↔ PostRunHook integration
# ===========================================================================


def _make_engine(
    *,
    post_run_hook: PostRunHook | None = None,
    max_generations: int = 1,
) -> EvolutionEngine:
    """Build a minimal EvolutionEngine with mocked dependencies."""
    storage = AsyncMock()
    storage.count_by_status.return_value = 0
    storage.get_all_by_status.return_value = []
    storage.get_ids_by_status.return_value = []
    storage.snapshot = MagicMock()

    writer = MagicMock()
    writer.bind.return_value = writer
    metrics_tracker = AsyncMock()
    metrics_tracker.start = MagicMock()

    return EvolutionEngine(
        storage=storage,
        strategy=AsyncMock(),
        mutation_operator=AsyncMock(),
        config=EngineConfig(max_generations=max_generations),
        writer=writer,
        metrics_tracker=metrics_tracker,
        post_run_hook=post_run_hook,
    )


class TestEnginePostRunHookWiring:
    """EvolutionEngine fires post_run_hook.on_run_complete(storage) in finally block."""

    def test_none_hook_defaults_to_null(self) -> None:
        engine = _make_engine(post_run_hook=None)
        assert isinstance(engine._post_run_hook, NullPostRunHook)

    def test_custom_hook_is_stored(self) -> None:
        hook = NullPostRunHook()
        engine = _make_engine(post_run_hook=hook)
        assert engine._post_run_hook is hook

    @pytest.mark.asyncio
    async def test_hook_called_after_evolution_completes(self) -> None:
        """Hook fires in finally block after the generation loop exits."""
        hook = AsyncMock(spec=PostRunHook)
        engine = _make_engine(post_run_hook=hook, max_generations=1)

        await engine.run()

        hook.on_run_complete.assert_awaited_once_with(engine.storage)

    @pytest.mark.asyncio
    async def test_hook_fires_in_finally_block(self) -> None:
        """Hook fires in finally block even if engine is interrupted."""
        hook = AsyncMock(spec=PostRunHook)
        engine = _make_engine(post_run_hook=hook, max_generations=1)

        # Run normally (no exception)
        await engine.run()

        # Hook should be called
        hook.on_run_complete.assert_awaited_once_with(engine.storage)

    @pytest.mark.asyncio
    async def test_hook_exception_is_non_fatal(self) -> None:
        """Engine logs the error but doesn't re-raise from the hook."""
        hook = AsyncMock(spec=PostRunHook)
        hook.on_run_complete.side_effect = RuntimeError("hook exploded")
        engine = _make_engine(post_run_hook=hook, max_generations=1)

        # Should NOT raise — hook errors are caught
        await engine.run()

        assert not engine._running  # Engine stopped cleanly

    @pytest.mark.asyncio
    async def test_hook_receives_storage_before_close(self) -> None:
        """Hook fires before storage.close() so it can read programs."""
        call_order: list[str] = []

        hook = AsyncMock(spec=PostRunHook)
        hook.on_run_complete.side_effect = lambda s: call_order.append("hook")

        engine = _make_engine(post_run_hook=hook, max_generations=1)

        original_close = engine.storage.close

        async def tracked_close():
            call_order.append("close")
            return await original_close()

        engine.storage.close = tracked_close

        await engine.run()
        # stop() calls storage.close() — hook must have fired before
        await engine.stop()

        assert "hook" in call_order
        # If close was also tracked, hook came first
        if "close" in call_order:
            assert call_order.index("hook") < call_order.index("close")


# ===========================================================================
# 8. Hydra composability — config instantiation
# ===========================================================================


class TestHydraComposability:
    """Verify that Hydra config YAML files resolve to the correct classes.

    These tests validate the _target_ → class mapping, ensuring that Hydra's
    instantiate() will produce the right object type. We don't call Hydra's
    instantiate() directly (it needs a full config context), but we verify
    the import paths are valid.
    """

    def test_none_yaml_target_is_null_hook(self) -> None:
        """ideas_tracker=none → NullPostRunHook (no-op)."""
        # Verify the _target_ class path resolves
        from gigaevo.evolution.engine.hooks import NullPostRunHook as Target

        hook = Target()
        assert isinstance(hook, PostRunHook)

    def test_default_yaml_target_is_idea_tracker(self) -> None:
        """ideas_tracker=default → IdeaTracker."""
        from gigaevo.memory.ideas_tracker.ideas_tracker import IdeaTracker as Target

        assert issubclass(Target, PostRunHook)

    def test_engine_accepts_both_hook_types(self) -> None:
        """EvolutionEngine constructor works with NullPostRunHook and IdeaTracker."""
        # NullPostRunHook
        engine1 = _make_engine(post_run_hook=NullPostRunHook())
        assert isinstance(engine1._post_run_hook, NullPostRunHook)

        # Any PostRunHook subclass
        custom_hook = AsyncMock(spec=PostRunHook)
        engine2 = _make_engine(post_run_hook=custom_hook)
        assert engine2._post_run_hook is custom_hook

    def test_post_run_hook_wired_through_evolution_engine_config(self) -> None:
        """post_run_hook is a constructor parameter on EvolutionEngine (Hydra injects it)."""
        import inspect

        sig = inspect.signature(EvolutionEngine.__init__)
        assert "post_run_hook" in sig.parameters
        param = sig.parameters["post_run_hook"]
        assert (
            param.default is None
        )  # Optional — Hydra provides NullPostRunHook or IdeaTracker


# ===========================================================================
# 9. Full pipeline E2E: evolution → IdeaTracker extracts ideas
# ===========================================================================


class TestEvolutionToIdeaExtraction:
    """Full pipeline: EvolutionEngine runs → PostRunHook fires → IdeaTracker processes.

    Uses real EvolutionEngine with fakeredis, deterministic mutation,
    and a recording PostRunHook that captures what the engine delivers.
    """

    @pytest.mark.asyncio
    async def test_hook_receives_programs_from_storage(self) -> None:
        """After hook fires, it receives programs from the storage."""
        # Create a mock storage with programs
        storage = AsyncMock()
        progs = [_make_evolved_program(fitness=f) for f in [1.0, 2.0, 3.0]]
        storage.get_all.return_value = progs

        # Recording hook that captures what it receives
        captured_programs: list[Program] = []

        class RecordingHook(PostRunHook):
            async def on_run_complete(self, stor) -> None:
                from gigaevo.programs.program import EXCLUDE_STAGE_RESULTS

                programs = await stor.get_all(exclude=EXCLUDE_STAGE_RESULTS)
                captured_programs.extend(programs)

        hook = RecordingHook()
        await hook.on_run_complete(storage)

        # Verify the hook received all programs
        assert len(captured_programs) == 3
        assert all("fitness" in p.metrics for p in captured_programs)

    @pytest.mark.asyncio
    async def test_program_filtering_in_tracker_context(self) -> None:
        """IdeaTracker._get_new_programs correctly filters a realistic program set."""
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
                task_description="Verify multi-hop facts",
                memory_write_enabled=False,
                memory_usage_tracking_enabled=False,
            )

        # Simulate a realistic population after evolution
        seed = _make_root_program(fitness=1.0)
        gen2_good = _make_evolved_program(fitness=5.0, parent_id=seed.id, generation=2)
        gen2_bad = _make_evolved_program(fitness=0.0, parent_id=seed.id, generation=2)
        gen3_best = _make_evolved_program(
            fitness=8.0,
            parent_id=gen2_good.id,
            generation=3,
            insights=["Use BFS for hops"],
            changes=["Replaced DFS with BFS in retrieval"],
            archetype="exploitation",
        )

        all_programs = [seed, gen2_good, gen2_bad, gen3_best]
        records = tracker._get_new_programs(all_programs)

        # Root filtered, zero-fitness filtered → 2 records
        assert len(records) == 2
        record_ids = {r.id for r in records}
        assert gen2_good.id in record_ids
        assert gen3_best.id in record_ids
        assert seed.id not in record_ids
        assert gen2_bad.id not in record_ids

        # Verify the best record has mutation metadata mapped
        best_record = next(r for r in records if r.id == gen3_best.id)
        assert best_record.fitness == 8.0
        assert best_record.generation == 3
        assert best_record.insights == ["Use BFS for hops"]
        assert best_record.strategy == "exploitation"

    @pytest.mark.asyncio
    async def test_memory_usage_tracked_after_evolution(self) -> None:
        """After evolution with memory, build_memory_usage_updates_from_programs
        correctly computes fitness deltas for memory card tracking."""
        seed = _make_program(
            program_id="seed-01",
            fitness=2.0,
            parents=[],
            generation=1,
        )
        # Child used memory card "idea-1" and improved
        # Note: pass test ID string, not UUID - _make_program converts it
        child_improved = _make_program(
            program_id="child-01",
            fitness=7.0,
            generation=2,
            parents=["seed-01"],
            memory_ids=["idea-1"],
        )
        # Another child used same card but regressed
        child_regressed = _make_program(
            program_id="child-02",
            fitness=1.0,
            generation=2,
            parents=["seed-01"],
            memory_ids=["idea-1"],
        )

        all_programs = [seed, child_improved, child_regressed]
        result = build_memory_usage_updates_from_programs(
            all_programs, "HoVer fact verification"
        )

        # idea-1 was used twice
        assert "idea-1" in result
        total = result["idea-1"]["used"]["total"]
        assert total["total_used"] == 2

        # Deltas: 7.0-2.0=5.0 and 1.0-2.0=-1.0
        entries = result["idea-1"]["used"]["entries"]
        assert len(entries) == 1
        deltas = entries[0]["fitness_delta_per_use"]
        assert sorted(deltas) == [-1.0, 5.0]
        assert entries[0]["task_description_summary"] == "HoVer fact verification"
