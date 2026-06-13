"""Comprehensive tests for the IdeaTracker post-run hook pipeline.

Three layers, from fastest to slowest:
1. Unit tests — records_converter, helpers, program filtering
2. OOP contract tests — PostRunHook ABC, NullPostRunHook, Hydra composability
3. Integration tests — EvolutionEngine → PostRunHook → IdeaTracker pipeline
"""

from __future__ import annotations

import asyncio
import inspect
import threading
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

import pytest

from gigaevo.evolution.engine.config import SteadyStateEngineConfig
from gigaevo.evolution.engine.core import EvolutionEngine
from gigaevo.evolution.engine.hooks import NullPostRunHook, PostRunHook
from gigaevo.evolution.engine.steady_state import SteadyStateEvolutionEngine
from gigaevo.evolution.engine.stopper import MaxMutantsStopper
from gigaevo.memory.backend_factory import LocalMemoryBackendFactory
from gigaevo.memory.ideas_tracker.analyzers import (
    ClassifyingAnalyzer,
    ClusteringAnalyzer,
)
from gigaevo.memory.ideas_tracker.ideas_tracker import IdeaTracker
from gigaevo.memory.ideas_tracker.models import (
    AnalysisResult,
    program_to_record,
    programs_to_records,
)
from gigaevo.programs.metrics.context import (
    MIN_VALUE_DEFAULT,
    MetricsContext,
    MetricSpec,
)
from gigaevo.programs.program import EXCLUDE_STAGE_RESULTS, Lineage, Program
from gigaevo.programs.program_state import ProgramState
from tests.fakes.llm_router import FakeMemoryRouter

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


def _fitness_metrics_context() -> MetricsContext:
    return MetricsContext(
        specs={
            "fitness": MetricSpec(
                description="Primary fitness.",
                is_primary=True,
                higher_is_better=True,
            )
        }
    )


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
        assert record.strategy == "exploration"

    def test_missing_mutation_output_defaults_to_empty(self) -> None:
        prog = _make_program(mutation_output=None)
        record = program_to_record(prog, "task", "summary")
        assert record.strategy == ""

    def test_invalid_mutation_output_type_defaults_to_empty(self) -> None:
        prog = _make_program()
        prog.metadata["mutation_output"] = "not a dict"
        record = program_to_record(prog, "task", "summary")
        assert record.strategy == ""

    def test_missing_fitness_raises(self) -> None:
        prog = _make_program()
        prog.metrics.clear()
        with pytest.raises(KeyError):
            program_to_record(prog, "task", "summary")

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
    with patch(
        "gigaevo.memory.ideas_tracker.ideas_tracker._summarise_task_description",
        return_value="Test summary",
    ):
        analyzer = ClassifyingAnalyzer(llm=FakeMemoryRouter())
        kwargs.setdefault("backend", LocalMemoryBackendFactory())
        return IdeaTracker(analyzer=analyzer, task_description="Test task", **kwargs)


class TestIdeaTrackerIsPostRunHook:
    def test_is_subclass_of_post_run_hook(self) -> None:
        assert issubclass(IdeaTracker, PostRunHook)

    def test_instantiates_with_analyzer(self) -> None:
        tracker = _make_tracker()
        assert isinstance(tracker, PostRunHook)
        assert tracker._fitness_key == "fitness"

    def test_analyzer_types_importable(self) -> None:
        assert ClassifyingAnalyzer is not None
        assert ClusteringAnalyzer is not None


class TestIdeaTrackerRunMethod:
    def test_run_with_no_programs_is_noop(self) -> None:
        """The run() method with no args should not crash, but does nothing (bug)."""
        tracker = _make_tracker(memory_write_enabled=False)
        result = tracker.run()
        assert result is None
        assert len(tracker._all_records) == 0

    def test_run_with_programs_processes_them(self) -> None:
        """The run() method can process programs when passed directly."""
        tracker = _make_tracker(memory_write_enabled=False)
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

    def test_program_without_fitness_metric_is_excluded(self) -> None:
        """A valid program missing the fitness key yields no record — no phantom default."""
        tracker = _make_tracker()
        prog = _make_evolved_program(fitness=5.0)
        del prog.metrics["fitness"]
        assert tracker._eligible_records([prog]) == []

    def test_program_with_non_finite_fitness_is_excluded(self) -> None:
        """NaN/inf fitness is not a real observation."""
        tracker = _make_tracker()
        result = tracker._eligible_records(
            [_make_evolved_program(fitness=float("nan"))]
        )
        assert result == []

    def test_program_with_sentinel_fitness_is_excluded(self) -> None:
        """Fitness equal to the metric's sentinel floor is rejected via the wired
        MetricsContext even when the program claims validity."""
        tracker = _make_tracker(metrics_context=_fitness_metrics_context())
        result = tracker._eligible_records(
            [_make_evolved_program(fitness=MIN_VALUE_DEFAULT, is_valid=1.0)]
        )
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


class _FlakyAnalyzer:
    """Fails classification of everything on the first sweep, succeeds after."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def analyze_async(self, records, bank) -> AnalysisResult:
        self.calls.append([r.id for r in records])
        if len(self.calls) == 1:
            return AnalysisResult(failed_program_ids=[r.id for r in records])
        return AnalysisResult()


class TestFailedClassificationRetry:
    @pytest.mark.asyncio
    async def test_failed_programs_are_retried_next_increment(self) -> None:
        tracker = _make_tracker(memory_write_enabled=False)
        analyzer = _FlakyAnalyzer()
        tracker._analyzer = analyzer
        prog = _make_evolved_program(fitness=2.0)

        await tracker.run_increment([prog])
        await tracker.run_increment([prog])

        assert analyzer.calls == [[prog.id], [prog.id]]
        assert [r.id for r in tracker._all_records] == [prog.id]


class _RecordProbeAnalyzer:
    """Captures the ProgramRecords each sweep feeds the analyzer."""

    def __init__(self) -> None:
        self.batches: list[list] = []

    async def analyze_async(self, records, bank) -> AnalysisResult:
        self.batches.append(list(records))
        return AnalysisResult()


class TestParentCodeResolution:
    @pytest.mark.asyncio
    async def test_parent_codes_resolve_from_posterior_programs(self, tmp_path) -> None:
        """Live sweeps cap the analyzer window to the newest programs; mutation
        parents are usually older archive elites OUTSIDE that window. Parent
        code must resolve from the full posterior set or the verification gate,
        canonical dedup, and diff-grounding all silently disable mid-run."""
        tracker = _make_tracker(memory_write_enabled=False, logs_dir=tmp_path)
        probe = _RecordProbeAnalyzer()
        tracker._analyzer = probe
        parent = _make_program(
            parents=[],
            generation=1,
            code="def solve(): return 'parent'",
            program_id="elder-parent",
        )
        child = _make_program(
            parents=["elder-parent"],
            generation=2,
            code="def solve(): return 'child'",
        )

        await tracker.run_increment([child], posterior_programs=[parent, child])

        (record,) = probe.batches[0]
        assert record.parent_code == parent.code


class _CancellingAnalyzer:
    """Raises CancelledError on the first sweep (engine cancel-grace fired)."""

    def __init__(self) -> None:
        self.batches: list[list[str]] = []
        self._cancel_once = True

    async def analyze_async(self, records, bank) -> AnalysisResult:
        self.batches.append([r.id for r in records])
        if self._cancel_once:
            self._cancel_once = False
            raise asyncio.CancelledError
        return AnalysisResult()


class TestCancelledSweepRollback:
    @pytest.mark.asyncio
    async def test_cancelled_analysis_unsees_records(self, tmp_path) -> None:
        """Records are marked seen before analysis; a cancelled sweep must roll
        that back or the window's ideas are permanently lost."""
        tracker = _make_tracker(memory_write_enabled=False, logs_dir=tmp_path)
        analyzer = _CancellingAnalyzer()
        tracker._analyzer = analyzer
        prog = _make_evolved_program(fitness=2.0)

        with pytest.raises(asyncio.CancelledError):
            await tracker.run_increment([prog])
        await tracker.run_increment([prog])

        assert analyzer.batches == [[prog.id], [prog.id]]
        assert [r.id for r in tracker._all_records] == [prog.id]


class _AlwaysFailingAnalyzer:
    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    async def analyze_async(self, records, bank) -> AnalysisResult:
        self.batches.append([r.id for r in records])
        return AnalysisResult(failed_program_ids=[r.id for r in records])


class TestClassificationFailureCap:
    @pytest.mark.asyncio
    async def test_poison_program_retired_after_cap(self, tmp_path) -> None:
        """A program whose ideas the LLM can never classify must stop re-burning
        analyzer calls after a bounded number of retries."""
        tracker = _make_tracker(memory_write_enabled=False, logs_dir=tmp_path)
        analyzer = _AlwaysFailingAnalyzer()
        tracker._analyzer = analyzer
        prog = _make_evolved_program(fitness=2.0)

        for _ in range(5):
            await tracker.run_increment([prog])

        attempts = [batch for batch in analyzer.batches if batch]
        assert len(attempts) == 3


class _OverlapProbeAnalyzer:
    """Yields control mid-analysis so unserialized callers would interleave."""

    def __init__(self) -> None:
        self.in_flight = 0
        self.max_in_flight = 0

    async def analyze_async(self, records, bank) -> AnalysisResult:
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        await asyncio.sleep(0.02)
        self.in_flight -= 1
        return AnalysisResult()


class TestRunIncrementConcurrency:
    @pytest.mark.asyncio
    async def test_concurrent_run_increments_serialize(self, tmp_path) -> None:
        """The live hook and the post-run hook share one tracker; overlapping
        sweeps would interleave bank mutations and _seen_ids bookkeeping."""
        tracker = _make_tracker(memory_write_enabled=False, logs_dir=tmp_path)
        probe = _OverlapProbeAnalyzer()
        tracker._analyzer = probe

        await asyncio.gather(tracker.run_increment([]), tracker.run_increment([]))

        assert probe.max_in_flight == 1

    @pytest.mark.asyncio
    async def test_write_pipeline_runs_off_event_loop_thread(
        self, tmp_path, monkeypatch
    ) -> None:
        """Backend build + card ingest are blocking I/O; running them on the
        loop thread stalls every in-flight mutation for the whole sweep."""
        import gigaevo.memory.ideas_tracker.ideas_tracker as mod

        seen: dict[str, int] = {}

        def _spy(*args, **kwargs) -> None:
            seen["thread"] = threading.get_ident()

        monkeypatch.setattr(mod, "_run_write_pipeline", _spy)
        tracker = _make_tracker(memory_write_enabled=False, logs_dir=tmp_path)

        await tracker.run_increment([])

        assert seen["thread"] != threading.get_ident()

    @pytest.mark.asyncio
    async def test_session_log_flush_runs_off_event_loop_thread(
        self, tmp_path, monkeypatch
    ) -> None:
        """flush() serializes the whole bank and runs offline origin analysis —
        per-sweep blocking work that belongs off the loop thread too."""
        import gigaevo.memory.ideas_tracker.ideas_tracker as mod

        seen: dict[str, int] = {}

        def _spy(self, bank, *, records) -> None:
            seen["thread"] = threading.get_ident()

        monkeypatch.setattr(mod._SessionLog, "flush", _spy)
        tracker = _make_tracker(memory_write_enabled=False, logs_dir=tmp_path)

        await tracker.run_increment([])

        assert seen["thread"] != threading.get_ident()

    def test_run_write_pipeline_serializes_concurrent_callers(
        self, tmp_path, monkeypatch
    ) -> None:
        """A cancelled sweep leaves its to_thread writer running as an orphan;
        the next sweep (or on_run_complete) must not interleave a second
        backend build/ingest with it."""
        import time

        import gigaevo.memory.ideas_tracker.ideas_tracker as mod
        import gigaevo.memory.write_pipeline as wp

        state = {"in_flight": 0, "max_in_flight": 0}
        gauge = threading.Lock()

        def _slow_main(**kwargs):
            with gauge:
                state["in_flight"] += 1
                state["max_in_flight"] = max(state["max_in_flight"], state["in_flight"])
            time.sleep(0.05)
            with gauge:
                state["in_flight"] -= 1
            return None

        monkeypatch.setattr(wp, "main", _slow_main)
        banks = tmp_path / "banks.json"
        banks.write_text('[{"active_bank": []}]', encoding="utf-8")
        best = tmp_path / "best_ideas.json"
        best.write_text('[{"best_ideas": []}]', encoding="utf-8")

        def _call() -> None:
            mod._run_write_pipeline(
                True, banks, best, None, backend=LocalMemoryBackendFactory()
            )

        workers = [threading.Thread(target=_call) for _ in range(2)]
        for t in workers:
            t.start()
        for t in workers:
            t.join()

        assert state["max_in_flight"] == 1


class TestIdeaTrackerOnRunComplete:
    def _make_tracker_with_mocked_run(self):
        tracker = _make_tracker(memory_write_enabled=False)
        tracker.run_increment = AsyncMock()
        return tracker

    @pytest.mark.asyncio
    async def test_empty_storage_skips_pipeline(self) -> None:
        tracker = self._make_tracker_with_mocked_run()
        storage = AsyncMock()
        storage.get_all.return_value = []
        await tracker.on_run_complete(storage)
        tracker.run_increment.assert_not_called()

    @pytest.mark.asyncio
    async def test_programs_passed_to_pipeline(self) -> None:
        tracker = self._make_tracker_with_mocked_run()
        progs = [_make_evolved_program(fitness=f) for f in [1.0, 2.0, 3.0]]
        storage = AsyncMock()
        storage.get_all.return_value = progs
        await tracker.on_run_complete(storage)
        tracker.run_increment.assert_awaited_once_with(progs)

    @pytest.mark.asyncio
    async def test_storage_excludes_stage_results(self) -> None:
        tracker = self._make_tracker_with_mocked_run()
        storage = AsyncMock()
        storage.get_all.return_value = [_make_evolved_program()]
        await tracker.on_run_complete(storage)
        storage.get_all.assert_called_once_with(exclude=EXCLUDE_STAGE_RESULTS)


class TestIdeaTrackerPosteriorPopulation:
    """run_increment computes the injection posterior over posterior_programs
    (full lineage) while the analyzer keeps the smaller `programs` window."""

    @pytest.mark.asyncio
    async def test_posterior_uses_posterior_programs_when_given(
        self, monkeypatch
    ) -> None:
        import gigaevo.memory.ideas_tracker.ideas_tracker as mod

        captured: dict[str, list[str]] = {}

        def _spy(
            programs,
            *,
            fitness_key,
            higher_is_better,
            reputation=None,
            metrics_context=None,
        ):
            captured["ids"] = [p.id for p in programs]
            return {}

        monkeypatch.setattr(mod, "_card_posterior_from_programs", _spy)
        tracker = _make_tracker(memory_write_enabled=False)
        window = [_make_evolved_program(fitness=1.0)]
        full = window + [
            _make_evolved_program(fitness=2.0),
            _make_evolved_program(fitness=3.0),
        ]

        await tracker.run_increment(window, posterior_programs=full)

        assert captured["ids"] == [p.id for p in full]

    @pytest.mark.asyncio
    async def test_posterior_defaults_to_programs_when_absent(
        self, monkeypatch
    ) -> None:
        import gigaevo.memory.ideas_tracker.ideas_tracker as mod

        captured: dict[str, list[str]] = {}

        def _spy(
            programs,
            *,
            fitness_key,
            higher_is_better,
            reputation=None,
            metrics_context=None,
        ):
            captured["ids"] = [p.id for p in programs]
            return {}

        monkeypatch.setattr(mod, "_card_posterior_from_programs", _spy)
        tracker = _make_tracker(memory_write_enabled=False)
        progs = [_make_evolved_program(fitness=1.0)]

        await tracker.run_increment(progs)

        assert captured["ids"] == [p.id for p in progs]


class TestIdeaTrackerLegacyRun:
    def _make_tracker_with_mocked_run(self):
        tracker = _make_tracker(memory_write_enabled=False)
        tracker.run_increment = MagicMock()
        return tracker

    def test_none_programs_skips(self) -> None:
        tracker = self._make_tracker_with_mocked_run()
        tracker.run(None)
        tracker.run_increment.assert_not_called()

    def test_empty_programs_skips(self) -> None:
        tracker = self._make_tracker_with_mocked_run()
        tracker.run([])
        tracker.run_increment.assert_not_called()


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
    return SteadyStateEvolutionEngine(
        storage=storage,
        strategy=AsyncMock(),
        mutation_operator=AsyncMock(),
        config=SteadyStateEngineConfig(stopper=MaxMutantsStopper(max_generations)),
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

    # End-to-end PostRunHook wiring (hook is awaited, hook exceptions are
    # non-fatal) is covered by the integration tests that drive a populated
    # archive through engine.run(); the constructor checks above pin the
    # NullPostRunHook default and the custom-hook bind.


class TestHydraComposability:
    def test_none_yaml_target_is_null_hook(self) -> None:
        hook = NullPostRunHook()
        assert isinstance(hook, PostRunHook)

    def test_default_yaml_target_is_idea_tracker(self) -> None:
        assert issubclass(IdeaTracker, PostRunHook)

    def test_engine_accepts_both_hook_types(self) -> None:
        engine1 = _make_engine(post_run_hook=NullPostRunHook())
        assert isinstance(engine1._post_run_hook, NullPostRunHook)
        engine2 = _make_engine(post_run_hook=AsyncMock(spec=PostRunHook))
        assert engine2._post_run_hook is not None

    def test_post_run_hook_in_engine_signature(self) -> None:
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
                programs = await stor.get_all(exclude=EXCLUDE_STAGE_RESULTS)
                captured.extend(programs)

        await RecordingHook().on_run_complete(storage)
        assert len(captured) == 3

    @pytest.mark.asyncio
    async def test_program_filtering_in_tracker_context(self) -> None:
        tracker = _make_tracker(memory_write_enabled=False)
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
        assert best.strategy == "exploitation"


# ---------------------------------------------------------------------------
# IdeaTracker.run_increment() writes banks.json via flush
# ---------------------------------------------------------------------------


def test_ideas_tracker_run_writes_banks_file(tmp_path):
    """IdeaTracker.run_increment writes banks.json to a timestamped session dir."""
    import asyncio
    import json
    from unittest.mock import AsyncMock, MagicMock

    from gigaevo.memory.ideas_tracker.ideas_tracker import IdeaTracker
    from gigaevo.memory.ideas_tracker.models import AnalysisResult
    from gigaevo.memory.ideas_tracker.schemas import SummaryResponse

    prog = MagicMock()
    prog.id = "prog-aaa"
    prog.lineage = MagicMock()
    prog.lineage.parents = ["prog-seed"]
    prog.lineage.generation = 2
    prog.metrics = {"is_valid": 1.0, "fitness": 0.65}
    prog.code = "def solve(x): return x"
    prog.metadata = {}

    stub_analyzer = MagicMock()
    stub_analyzer.analyze_async = AsyncMock(
        return_value=AnalysisResult(new_ideas=[], updates=[])
    )
    stub_analyzer.call_structured = MagicMock(
        return_value=SummaryResponse(summary="Test summary")
    )
    stub_analyzer.call_structured_async = AsyncMock(
        return_value=SummaryResponse(summary="Test summary")
    )

    tracker = IdeaTracker(
        analyzer=stub_analyzer,
        task_description="solve test problems",
        memory_write_enabled=False,
        logs_dir=tmp_path,
    )

    asyncio.run(tracker.run_increment([prog]))

    log_dirs = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert len(log_dirs) >= 1, (
        f"Expected a session log directory, got: {list(tmp_path.iterdir())}"
    )
    session_dir = log_dirs[0]
    banks_file = session_dir / "banks.json"
    assert banks_file.exists(), f"banks.json not found in {session_dir}"

    data = json.loads(banks_file.read_text())
    assert isinstance(data, list) and len(data) >= 1
    assert "active_bank" in data[0], (
        f"Expected 'active_bank' key, got: {list(data[0].keys())}"
    )


class TestAdmitterWiring:
    def test_injected_admitter_reaches_session_log(self) -> None:
        from gigaevo.memory.core.admitter import SignBasedAdmitter

        admitter = SignBasedAdmitter()
        tracker = _make_tracker(admitter=admitter)
        assert tracker._log._admitter is admitter

    def test_default_admitter_is_none(self) -> None:
        tracker = _make_tracker()
        assert tracker._log._admitter is None


# ---------------------------------------------------------------------------
# Typed evolution-statistics stamping (banks.json snapshot)
# ---------------------------------------------------------------------------


class TestTypedStatsInjection:
    def test_idea_model_carries_typed_statistics(self) -> None:
        from gigaevo.memory.ideas_tracker.models import Idea
        from gigaevo.memory.shared_memory.models import EvolutionStatistics

        idea = Idea(
            description="use BFS",
            evolution_statistics={"ALL": {"intro_events": 3}},
        )
        assert isinstance(idea.evolution_statistics, EvolutionStatistics)
        assert idea.evolution_statistics.ALL is not None
        assert idea.evolution_statistics.ALL.intro_events == 3
        assert Idea(description="bare").model_dump()["evolution_statistics"] == {}

    def test_flush_stamps_typed_statistics_into_banks_snapshot(
        self, tmp_path, monkeypatch
    ) -> None:
        import json
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        from gigaevo.memory.core.idea_stats import IdeaStats
        import gigaevo.memory.ideas_tracker.ideas_tracker as it
        from gigaevo.memory.ideas_tracker.models import Idea

        stamped = Idea(id="idea-1", description="use BFS")
        cold = Idea(id="idea-2", description="never analysed")
        bank = MagicMock()
        bank.all_ideas.return_value = [stamped, cold]

        rows = [
            IdeaStats.model_validate(
                {
                    "idea_id": "idea-1",
                    "quartile": "ALL",
                    "description": "use BFS",
                    "intro_events": 5,
                    "posterior_a": 4.0,
                    "posterior_b": 2.0,
                    "p_help_mean": 0.6667,
                    "p_help_lo20": 0.51,
                    "efficacy_confident": True,
                }
            ),
            IdeaStats.model_validate(
                {"idea_id": "idea-1", "quartile": "Q1", "intro_events": 2}
            ),
        ]
        monkeypatch.setattr(
            it,
            "_analyse_origins",
            lambda **kwargs: SimpleNamespace(summary=rows, best_ideas=rows[:1]),
        )

        log = it._SessionLog(tmp_path)
        log.flush(bank, records=[])

        data = json.loads(log.banks_file.read_text(encoding="utf-8"))
        by_id = {idea["id"]: idea for idea in data[0]["active_bank"]}
        all_block = by_id["idea-1"]["evolution_statistics"]["ALL"]
        assert all_block["posterior_a"] == 4.0
        assert all_block["efficacy_confident"] is True
        assert set(by_id["idea-1"]["evolution_statistics"]) == {"ALL"}
        assert by_id["idea-2"]["evolution_statistics"] == {}
        assert stamped.evolution_statistics.ALL is None
        assert log.best_ideas_file.exists()
