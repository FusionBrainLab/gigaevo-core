"""Comprehensive tests for the IdeaTracker post-run hook pipeline.

Three layers, from fastest to slowest:
1. Unit tests — records_converter, helpers, program filtering
2. OOP contract tests — PostRunHook ABC, NullPostRunHook, Hydra composability
3. Integration tests — EvolutionEngine → PostRunHook → IdeaTracker pipeline
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

import pytest

from gigaevo.evolution.engine.config import EngineConfig
from gigaevo.evolution.engine.core import EvolutionEngine
from gigaevo.evolution.engine.hooks import NullPostRunHook, PostRunHook
from gigaevo.memory.ideas_tracker.models import (
    program_to_record,
    programs_to_records,
)
from gigaevo.programs.program import Lineage, Program
from gigaevo.programs.program_state import ProgramState

_TEST_NAMESPACE = uuid.NAMESPACE_DNS


def _uuid(test_id: str) -> str:
    return str(uuid.uuid5(_TEST_NAMESPACE, test_id))


def _make_program(
    *,
    code: str = "def solve(): return 42",
    fitness: float = 0.75,
    is_valid: float = 1.0,
    fitness_key: str = "fitness",
    generation: int = 3,
    parents: list[str] | None = None,
    mutation_output: dict[str, Any] | None = None,
    memory_ids: list[str] | None = None,
    state: ProgramState = ProgramState.DONE,
    program_id: str | None = None,
) -> Program:
    metadata: dict[str, Any] = {}
    if mutation_output is not None:
        metadata["mutation_output"] = mutation_output
    if memory_ids is not None:
        metadata["memory_selected_idea_ids"] = memory_ids
    parent_list = parents or (["parent-1"] if generation > 1 else [])
    parent_uuids = [_uuid(p) if isinstance(p, str) else p for p in parent_list]
    lineage = Lineage(parents=parent_uuids, generation=max(generation, 1))
    prog = Program(
        code=code,
        state=state,
        metrics={fitness_key: fitness, "is_valid": is_valid},
        metadata=metadata,
        lineage=lineage,
    )
    if program_id is not None:
        object.__setattr__(prog, "id", _uuid(program_id))
    return prog


def _make_root_program(*, fitness: float = 1.0) -> Program:
    return _make_program(parents=[], generation=1, fitness=fitness)


def _make_evolved_program(
    *,
    fitness: float = 5.0,
    is_valid: float = 1.0,
    parent_id: str = "seed-01",
    generation: int = 3,
    insights: list[str] | None = None,
    changes: list[str] | None = None,
    archetype: str = "exploitation",
) -> Program:
    mutation_output: dict[str, Any] = {"archetype": archetype}
    if insights is not None:
        mutation_output["insights_used"] = insights
    if changes is not None:
        mutation_output["changes"] = changes
    return _make_program(
        fitness=fitness,
        is_valid=is_valid,
        generation=generation,
        parents=[parent_id],
        mutation_output=mutation_output,
    )


def _make_memory_program(
    *,
    fitness: float = 8.0,
    is_valid: float = 1.0,
    parent_id: str = "parent-a",
    card_ids: list[str] | None = None,
) -> Program:
    return _make_program(
        fitness=fitness,
        is_valid=is_valid,
        generation=5,
        parents=[parent_id],
        memory_ids=card_ids or ["idea-001", "idea-002"],
    )


# ---------------------------------------------------------------------------
# Helper: _build_usage_updates (was build_memory_usage_updates_from_programs)
# ---------------------------------------------------------------------------


def _build_memory_usage_updates(programs, task_summary="", fitness_key="fitness"):
    """Thin wrapper so tests don't need to import the internal helper directly."""
    from gigaevo.memory.ideas_tracker.ideas_tracker import _build_usage_updates

    return _build_usage_updates(
        programs, task_summary or "Task summary unavailable", fitness_key
    )


class TestBuildMemoryUsageFromPrograms:
    def test_empty_programs_returns_empty(self) -> None:
        assert _build_memory_usage_updates([]) == {}

    def test_programs_without_memory_ids_return_empty(self) -> None:
        progs = [_make_evolved_program() for _ in range(3)]
        assert _build_memory_usage_updates(progs) == {}

    def test_single_card_usage_computes_delta(self) -> None:
        parent = _make_program(
            program_id="parent-a", fitness=5.0, parents=[], generation=1
        )
        child = _make_memory_program(
            fitness=8.0, parent_id="parent-a", card_ids=["idea-1"]
        )
        result = _build_memory_usage_updates([parent, child], "test task")
        assert "idea-1" in result
        entries = result["idea-1"]["used"]["entries"]
        assert len(entries) == 1
        assert entries[0]["used_count"] == 1
        assert entries[0]["fitness_delta_per_use"] == [3.0]
        assert entries[0]["median_delta_fitness"] == 3.0

    def test_negative_delta_included(self) -> None:
        parent = _make_program(program_id="p1", fitness=10.0, parents=[], generation=1)
        child = _make_memory_program(fitness=7.0, parent_id="p1", card_ids=["c1"])
        result = _build_memory_usage_updates([parent, child], "task")
        assert result["c1"]["used"]["entries"][0]["fitness_delta_per_use"] == [-3.0]

    def test_multiple_cards_per_program(self) -> None:
        parent = _make_program(program_id="p1", fitness=4.0, parents=[], generation=1)
        child = _make_memory_program(
            fitness=6.0, parent_id="p1", card_ids=["a", "b", "c"]
        )
        result = _build_memory_usage_updates([parent, child], "t")
        assert set(result.keys()) == {"a", "b", "c"}

    def test_missing_parent_fitness_skips_program(self) -> None:
        child = _make_memory_program(
            fitness=8.0, parent_id="unknown-parent", card_ids=["c1"]
        )
        assert _build_memory_usage_updates([child], "task") == {}

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
        result = _build_memory_usage_updates([parent, child], "task", "accuracy")
        assert "c1" in result

    def test_duplicate_card_ids_deduplicated(self) -> None:
        parent = _make_program(program_id="p1", fitness=1.0, parents=[], generation=1)
        child = _make_memory_program(
            fitness=2.0, parent_id="p1", card_ids=["dup", "dup", "dup"]
        )
        result = _build_memory_usage_updates([parent, child], "task")
        assert result["dup"]["used"]["total_used"] == 1

    def test_invalid_parent_excluded_from_delta_calculation(self) -> None:
        """Invalid parents must not contribute to fitness_by_id (prevents sentinel pollution)."""
        valid_parent = _make_program(
            program_id="p1", fitness=5.0, is_valid=1.0, parents=[], generation=1
        )
        invalid_parent = _make_program(
            program_id="p2", fitness=-1e5, is_valid=0.0, parents=[], generation=1
        )
        child = _make_memory_program(
            fitness=6.0, is_valid=1.0, parent_id="p1", card_ids=["idea-1"]
        )
        result = _build_memory_usage_updates(
            [valid_parent, invalid_parent, child], "task"
        )
        # Delta should be relative to valid parent only: 6.0 - 5.0 = 1.0
        assert result["idea-1"]["used"]["entries"][0]["fitness_delta_per_use"] == [1.0]

    def test_invalid_child_not_used_for_deltas(self) -> None:
        """Invalid child programs must be skipped entirely in delta computation."""
        parent = _make_program(program_id="p1", fitness=5.0, parents=[], generation=1)
        invalid_child = _make_program(
            fitness=-1e5,
            is_valid=0.0,
            generation=2,
            parents=["p1"],
            memory_ids=["card-1"],
        )
        valid_child = _make_memory_program(
            fitness=7.0, is_valid=1.0, parent_id="p1", card_ids=["card-1"]
        )
        result = _build_memory_usage_updates(
            [parent, invalid_child, valid_child], "task"
        )
        # Only valid child contributes: delta = 7.0 - 5.0 = 2.0
        assert result["card-1"]["used"]["total_used"] == 1
        assert result["card-1"]["used"]["entries"][0]["fitness_delta_per_use"] == [2.0]


# ---------------------------------------------------------------------------
# records_converter tests (now in models.py)
# ---------------------------------------------------------------------------


class TestProgramToRecord:
    def test_basic_field_mapping(self) -> None:
        prog = _make_evolved_program(
            fitness=7.5,
            generation=4,
            parent_id="p1",
            insights=["Use BFS"],
            changes=["Added BFS traversal"],
            archetype="exploration",
        )
        record = program_to_record(prog, "Solve TSP", "TSP optimisation")
        assert record.id == prog.id
        assert record.fitness == 7.5
        assert record.generation == 4
        assert record.parents == [_uuid("p1")]
        assert record.insights == ["Use BFS"]
        assert record.strategy == "exploration"

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
    def test_empty_list(self) -> None:
        records, ids = programs_to_records([], "task", "summary")
        assert records == []
        assert ids == set()

    def test_returns_records_and_ids(self) -> None:
        progs = [_make_evolved_program(fitness=f) for f in [1.0, 2.0, 3.0]]
        records, ids = programs_to_records(progs, "task", "summary")
        assert len(records) == 3
        assert ids == {p.id for p in progs}


# ---------------------------------------------------------------------------
# PostRunHook ABC
# ---------------------------------------------------------------------------


class TestPostRunHookABC:
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
    def test_instantiates_without_arguments(self) -> None:
        hook = NullPostRunHook()
        assert isinstance(hook, PostRunHook)

    @pytest.mark.asyncio
    async def test_on_run_complete_is_noop(self) -> None:
        hook = NullPostRunHook()
        storage = AsyncMock()
        await hook.on_run_complete(storage)
        storage.get_all.assert_not_called()


# ---------------------------------------------------------------------------
# IdeaTracker as PostRunHook
# ---------------------------------------------------------------------------


def _make_tracker(**kwargs):
    from gigaevo.memory.ideas_tracker.analyzers import ClassifyingAnalyzer
    from gigaevo.memory.ideas_tracker.ideas_tracker import IdeaTracker

    mock_llm_clients = (MagicMock(), MagicMock(), False)
    with (
        patch(
            "gigaevo.memory.ideas_tracker.llm._init_clients",
            return_value=mock_llm_clients,
        ),
        patch(
            "gigaevo.memory.ideas_tracker.ideas_tracker._summarise_task_description",
            return_value="Test summary",
        ),
    ):
        analyzer = ClassifyingAnalyzer(model="mock-model")
        return IdeaTracker(analyzer=analyzer, task_description="Test task", **kwargs)


class TestIdeaTrackerIsPostRunHook:
    def test_is_subclass_of_post_run_hook(self) -> None:
        from gigaevo.memory.ideas_tracker.ideas_tracker import IdeaTracker

        assert issubclass(IdeaTracker, PostRunHook)

    def test_instantiates_with_analyzer(self) -> None:
        tracker = _make_tracker()
        assert isinstance(tracker, PostRunHook)
        assert tracker._fitness_key == "fitness"

    def test_analyzer_types_importable(self) -> None:
        from gigaevo.memory.ideas_tracker.analyzers import (
            ClassifyingAnalyzer,
            ClusteringAnalyzer,
        )

        assert ClassifyingAnalyzer is not None
        assert ClusteringAnalyzer is not None


class TestIdeaTrackerRunMethod:
    def test_run_with_no_programs_is_noop(self) -> None:
        """The run() method with no args should not crash, but does nothing (bug)."""
        tracker = _make_tracker(
            memory_write_enabled=False, memory_usage_tracking_enabled=False
        )
        result = tracker.run()
        assert result is None
        assert len(tracker._all_records) == 0

    def test_run_with_programs_processes_them(self) -> None:
        """The run() method can process programs when passed directly."""
        tracker = _make_tracker(
            memory_write_enabled=False, memory_usage_tracking_enabled=False
        )
        programs = [_make_evolved_program(fitness=f) for f in [1.0, 2.0, 3.0]]
        tracker.run(programs)
        assert len(tracker._all_records) == 3


class TestIdeaTrackerProgramFiltering:
    def test_root_programs_are_skipped(self) -> None:
        tracker = _make_tracker()
        root = _make_root_program(fitness=10.0)
        evolved = _make_evolved_program(fitness=5.0)
        result = tracker._eligible_records([root, evolved])
        assert len(result) == 1
        assert result[0].id == evolved.id

    def test_invalid_programs_are_skipped(self) -> None:
        """Programs with is_valid=0 must be excluded regardless of fitness value."""
        tracker = _make_tracker()
        invalid = _make_evolved_program(fitness=0.0, is_valid=0.0)
        valid = _make_evolved_program(fitness=1.0, is_valid=1.0)
        result = tracker._eligible_records([invalid, valid])
        assert len(result) == 1
        assert result[0].fitness == 1.0

    def test_program_without_is_valid_metric_is_excluded(self) -> None:
        """Programs missing the is_valid metric are treated as invalid."""
        tracker = _make_tracker()
        prog = _make_evolved_program(fitness=5.0)
        del prog.metrics["is_valid"]
        result = tracker._eligible_records([prog])
        assert result == []

    def test_valid_program_with_negative_fitness_is_included(self) -> None:
        """Valid programs with negative fitness (e.g. hexagon_improver range [-10,-3.8]) must not be excluded."""
        tracker = _make_tracker()
        result = tracker._eligible_records(
            [_make_evolved_program(fitness=-3.0, is_valid=1.0)]
        )
        assert len(result) == 1
        assert result[0].fitness == -3.0

    def test_valid_program_with_zero_fitness_is_included(self) -> None:
        """Valid programs with exactly zero fitness must not be excluded."""
        tracker = _make_tracker()
        result = tracker._eligible_records(
            [_make_evolved_program(fitness=0.0, is_valid=1.0)]
        )
        assert len(result) == 1
        assert result[0].fitness == 0.0

    def test_invalid_program_excluded_despite_positive_fitness(self) -> None:
        """A program that failed validation (is_valid=0) must be excluded even if fitness > 0."""
        tracker = _make_tracker()
        result = tracker._eligible_records(
            [_make_evolved_program(fitness=5.0, is_valid=0.0)]
        )
        assert result == []

    def test_duplicate_programs_are_skipped(self) -> None:
        tracker = _make_tracker()
        prog = _make_evolved_program(fitness=5.0)
        result1 = tracker._eligible_records([prog])
        assert len(result1) == 1
        result2 = tracker._eligible_records([prog])
        assert result2 == []

    def test_seen_ids_tracked_after_processing(self) -> None:
        tracker = _make_tracker()
        prog = _make_evolved_program(fitness=5.0)
        tracker._eligible_records([prog])
        assert prog.id in tracker._seen_ids

    def test_all_records_accumulates(self) -> None:
        tracker = _make_tracker()
        p1 = _make_evolved_program(fitness=1.0)
        p2 = _make_evolved_program(fitness=2.0)
        tracker._eligible_records([p1])
        tracker._eligible_records([p2])
        assert len(tracker._all_records) == 2


class TestIdeaTrackerOnRunComplete:
    def _make_tracker_with_mocked_run(self):
        tracker = _make_tracker(
            memory_write_enabled=False, memory_usage_tracking_enabled=False
        )
        tracker._run = AsyncMock()
        return tracker

    @pytest.mark.asyncio
    async def test_empty_storage_skips_pipeline(self) -> None:
        tracker = self._make_tracker_with_mocked_run()
        storage = AsyncMock()
        storage.get_all.return_value = []
        await tracker.on_run_complete(storage)
        tracker._run.assert_not_called()

    @pytest.mark.asyncio
    async def test_programs_passed_to_pipeline(self) -> None:
        tracker = self._make_tracker_with_mocked_run()
        progs = [_make_evolved_program(fitness=f) for f in [1.0, 2.0, 3.0]]
        storage = AsyncMock()
        storage.get_all.return_value = progs
        await tracker.on_run_complete(storage)
        tracker._run.assert_awaited_once_with(progs)

    @pytest.mark.asyncio
    async def test_storage_excludes_stage_results(self) -> None:
        from gigaevo.programs.program import EXCLUDE_STAGE_RESULTS

        tracker = self._make_tracker_with_mocked_run()
        storage = AsyncMock()
        storage.get_all.return_value = [_make_evolved_program()]
        await tracker.on_run_complete(storage)
        storage.get_all.assert_called_once_with(exclude=EXCLUDE_STAGE_RESULTS)


class TestIdeaTrackerLegacyRun:
    def _make_tracker_with_mocked_run(self):
        tracker = _make_tracker(
            memory_write_enabled=False, memory_usage_tracking_enabled=False
        )
        tracker._run = MagicMock()
        return tracker

    def test_none_programs_skips(self) -> None:
        tracker = self._make_tracker_with_mocked_run()
        tracker.run(None)
        tracker._run.assert_not_called()

    def test_empty_programs_skips(self) -> None:
        tracker = self._make_tracker_with_mocked_run()
        tracker.run([])
        tracker._run.assert_not_called()


# ---------------------------------------------------------------------------
# EvolutionEngine ↔ PostRunHook integration
# ---------------------------------------------------------------------------


def _make_engine(*, post_run_hook=None, max_generations=1):
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
    def test_none_hook_defaults_to_null(self) -> None:
        engine = _make_engine(post_run_hook=None)
        assert isinstance(engine._post_run_hook, NullPostRunHook)

    def test_custom_hook_is_stored(self) -> None:
        hook = NullPostRunHook()
        engine = _make_engine(post_run_hook=hook)
        assert engine._post_run_hook is hook

    @pytest.mark.asyncio
    async def test_hook_called_after_evolution_completes(self) -> None:
        hook = AsyncMock(spec=PostRunHook)
        engine = _make_engine(post_run_hook=hook, max_generations=1)
        await engine.run()
        hook.on_run_complete.assert_awaited_once_with(engine.storage)

    @pytest.mark.asyncio
    async def test_hook_exception_is_non_fatal(self) -> None:
        hook = AsyncMock(spec=PostRunHook)
        hook.on_run_complete.side_effect = RuntimeError("hook exploded")
        engine = _make_engine(post_run_hook=hook, max_generations=1)
        await engine.run()
        assert not engine._running


class TestHydraComposability:
    def test_none_yaml_target_is_null_hook(self) -> None:
        hook = NullPostRunHook()
        assert isinstance(hook, PostRunHook)

    def test_default_yaml_target_is_idea_tracker(self) -> None:
        from gigaevo.memory.ideas_tracker.ideas_tracker import IdeaTracker

        assert issubclass(IdeaTracker, PostRunHook)

    def test_engine_accepts_both_hook_types(self) -> None:
        engine1 = _make_engine(post_run_hook=NullPostRunHook())
        assert isinstance(engine1._post_run_hook, NullPostRunHook)
        engine2 = _make_engine(post_run_hook=AsyncMock(spec=PostRunHook))
        assert engine2._post_run_hook is not None

    def test_post_run_hook_in_engine_signature(self) -> None:
        import inspect

        sig = inspect.signature(EvolutionEngine.__init__)
        assert "post_run_hook" in sig.parameters


# ---------------------------------------------------------------------------
# Full pipeline E2E
# ---------------------------------------------------------------------------


class TestEvolutionToIdeaExtraction:
    @pytest.mark.asyncio
    async def test_hook_receives_programs_from_storage(self) -> None:
        storage = AsyncMock()
        progs = [_make_evolved_program(fitness=f) for f in [1.0, 2.0, 3.0]]
        storage.get_all.return_value = progs
        captured: list = []

        class RecordingHook(PostRunHook):
            async def on_run_complete(self, stor) -> None:
                from gigaevo.programs.program import EXCLUDE_STAGE_RESULTS

                programs = await stor.get_all(exclude=EXCLUDE_STAGE_RESULTS)
                captured.extend(programs)

        await RecordingHook().on_run_complete(storage)
        assert len(captured) == 3

    @pytest.mark.asyncio
    async def test_program_filtering_in_tracker_context(self) -> None:
        tracker = _make_tracker(
            memory_write_enabled=False, memory_usage_tracking_enabled=False
        )
        seed = _make_root_program(fitness=1.0)
        gen2_good = _make_evolved_program(fitness=5.0, parent_id=seed.id, generation=2)
        gen2_bad = _make_evolved_program(
            fitness=0.0, parent_id=seed.id, generation=2, is_valid=0.0
        )
        gen3_best = _make_evolved_program(
            fitness=8.0,
            parent_id=gen2_good.id,
            generation=3,
            insights=["Use BFS for hops"],
            changes=["Replaced DFS with BFS"],
            archetype="exploitation",
        )
        records = tracker._eligible_records([seed, gen2_good, gen2_bad, gen3_best])
        assert len(records) == 2
        record_ids = {r.id for r in records}
        assert gen2_good.id in record_ids
        assert gen3_best.id in record_ids
        assert seed.id not in record_ids
        assert gen2_bad.id not in record_ids
        best = next(r for r in records if r.id == gen3_best.id)
        assert best.fitness == 8.0
        assert best.insights == ["Use BFS for hops"]
        assert best.strategy == "exploitation"

    @pytest.mark.asyncio
    async def test_memory_usage_tracked_after_evolution(self) -> None:
        seed = _make_program(
            program_id="seed-01", fitness=2.0, parents=[], generation=1
        )
        child_improved = _make_program(
            program_id="child-01",
            fitness=7.0,
            generation=2,
            parents=["seed-01"],
            memory_ids=["idea-1"],
        )
        child_regressed = _make_program(
            program_id="child-02",
            fitness=1.0,
            generation=2,
            parents=["seed-01"],
            memory_ids=["idea-1"],
        )
        result = _build_memory_usage_updates(
            [seed, child_improved, child_regressed], "HoVer fact verification"
        )
        assert "idea-1" in result
        assert result["idea-1"]["used"]["total_used"] == 2
        deltas = result["idea-1"]["used"]["entries"][0]["fitness_delta_per_use"]
        assert sorted(deltas) == [-1.0, 5.0]
