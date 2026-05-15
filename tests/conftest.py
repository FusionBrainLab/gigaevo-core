"""Shared test fixtures for DAG, storage, and state manager tests."""

from __future__ import annotations

import asyncio
from typing import Any

import fakeredis.aioredis
import pytest

from gigaevo.database.redis import RedisProgramStorageConfig
from gigaevo.database.redis_program_storage import RedisProgramStorage
from gigaevo.database.state_manager import ProgramStateManager
from gigaevo.programs.core_types import (
    ProgramStageResult,
    StageIO,
    VoidInput,
    VoidOutput,
)
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState
from gigaevo.programs.stages.base import Stage
from gigaevo.programs.stages.cache_handler import NO_CACHE
from gigaevo.utils.trackers.base import LogWriter

# ---------------------------------------------------------------------------
# No-op LogWriter for tests
# ---------------------------------------------------------------------------


class NullWriter(LogWriter):
    def bind(self, path: list[str]) -> NullWriter:
        return self

    def scalar(self, metric: str, value: float, **kwargs: Any) -> None:
        pass

    def hist(self, metric: str, values: list[float], **kwargs: Any) -> None:
        pass

    def text(self, tag: str, text: str, **kwargs: Any) -> None:
        pass

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Mock stage I/O types
# ---------------------------------------------------------------------------


class MockOutput(StageIO):
    value: int = 42


class MockInput(StageIO):
    data: MockOutput


class OptionalInput(StageIO):
    data: MockOutput | None = None


# ---------------------------------------------------------------------------
# Mock stage classes
# ---------------------------------------------------------------------------


class FastStage(Stage):
    """Instant stage: VoidInput -> MockOutput."""

    InputsModel = VoidInput
    OutputModel = MockOutput

    async def compute(self, program: Program) -> MockOutput:
        return MockOutput(value=42)


class ChainedStage(Stage):
    """Reads input from an upstream stage: MockInput -> MockOutput."""

    InputsModel = MockInput
    OutputModel = MockOutput

    async def compute(self, program: Program) -> MockOutput:
        return MockOutput(value=self.params.data.value + 1)


class FailingStage(Stage):
    """Always raises RuntimeError (no inputs)."""

    InputsModel = VoidInput
    OutputModel = MockOutput

    async def compute(self, program: Program) -> MockOutput:
        raise RuntimeError("stage failed on purpose")


class FailingChainedStage(Stage):
    """Accepts input then fails."""

    InputsModel = MockInput
    OutputModel = MockOutput

    async def compute(self, program: Program) -> MockOutput:
        raise RuntimeError("chained stage failed on purpose")


class SlowStage(Stage):
    """Takes 0.5s to complete."""

    InputsModel = VoidInput
    OutputModel = MockOutput

    async def compute(self, program: Program) -> MockOutput:
        await asyncio.sleep(0.5)
        return MockOutput(value=99)


class TimeoutStage(Stage):
    """Sleeps forever (for timeout tests)."""

    InputsModel = VoidInput
    OutputModel = MockOutput

    async def compute(self, program: Program) -> MockOutput:
        await asyncio.sleep(3600)
        return MockOutput(value=0)  # pragma: no cover


class OptionalInputStage(Stage):
    """Accepts an optional input: OptionalInput -> MockOutput."""

    InputsModel = OptionalInput
    OutputModel = MockOutput

    async def compute(self, program: Program) -> MockOutput:
        if self.params.data is not None:
            return MockOutput(value=self.params.data.value + 10)
        return MockOutput(value=-1)


class VoidStage(Stage):
    """Returns None (VoidOutput)."""

    InputsModel = VoidInput
    OutputModel = VoidOutput

    async def compute(self, program: Program) -> None:
        return None


class SideEffectStage(Stage):
    """Writes to program.metrics during compute."""

    InputsModel = VoidInput
    OutputModel = MockOutput

    async def compute(self, program: Program) -> MockOutput:
        program.add_metrics({"side_effect_metric": 123.0})
        return MockOutput(value=77)


class NeverCachedStage(Stage):
    """Stage with NeverCached cache handler — always re-executes."""

    InputsModel = VoidInput
    OutputModel = MockOutput
    cache_handler = NO_CACHE

    async def compute(self, program: Program) -> MockOutput:
        return MockOutput(value=42)


# ---------------------------------------------------------------------------
# Loky executor cleanup
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def _shutdown_loky_executor_at_session_end():
    """Kill loky workers once at session teardown.

    Function-scoped teardown would pay the ``cpu_count``-worker spawn cost
    on every test; loky's reusable executor is safe to share across the
    session because :func:`get_reusable_executor` respawns on env/config
    change anyway.
    """
    yield
    try:
        from gigaevo.programs.stages.python_executors.wrapper import (
            shutdown_executor,
        )

        shutdown_executor()
    except Exception:
        pass


@pytest.fixture
def isolated_spill_dir(tmp_path, monkeypatch):
    """Point the wrapper's spill directory at a per-test temp directory.

    Lets tests count spill artefacts without interference from concurrent
    test runs or system-level tmp pollution.  Replaces the module-level
    ``_CONFIG`` so the new ``spill_dir`` takes effect on the next call.
    """
    from dataclasses import replace

    from gigaevo.programs.stages.python_executors import wrapper as _wrapper

    spill = tmp_path / "spill"
    spill.mkdir()
    monkeypatch.setattr(
        _wrapper, "_CONFIG", replace(_wrapper._CONFIG, spill_dir=spill)
    )
    return spill


@pytest.fixture
def fresh_executor():
    """Tear down the loky pool before and after the test.

    Use for tests that need a guaranteed-fresh worker state — e.g.
    inspecting env propagation, verifying lazy-spawn semantics, or
    asserting clean shutdown.  Costs ``cpu_count`` worker spawns.
    """
    from gigaevo.programs.stages.python_executors.wrapper import shutdown_executor

    shutdown_executor()
    yield
    shutdown_executor()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def fakeredis_storage():
    """RedisProgramStorage backed by fakeredis (async)."""
    server = fakeredis.FakeServer()
    config = RedisProgramStorageConfig(
        redis_url="redis://fake:6379/0",
        key_prefix="test",
    )
    storage = RedisProgramStorage(config)
    # Monkey-patch the connection to use fakeredis
    fake_redis = fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)
    storage._conn._redis = fake_redis
    storage._conn._closing = False
    yield storage
    await storage.close()


@pytest.fixture
async def state_manager(fakeredis_storage: RedisProgramStorage):
    """ProgramStateManager wrapping the fake storage."""
    return ProgramStateManager(fakeredis_storage)


@pytest.fixture
def null_writer():
    """No-op LogWriter for tests."""
    return NullWriter()


@pytest.fixture
def make_program():
    """Factory for creating test Program objects."""

    def _make(
        code: str = "def solve(): return 42",
        # DagRunner fetches programs that are already mid-flight (RUNNING).
        # Program's own default is QUEUED, but tests that exercise stage
        # execution want a program that is already past the scheduling gate.
        state: ProgramState = ProgramState.RUNNING,
        metrics: dict[str, float] | None = None,
        stage_results: dict[str, ProgramStageResult] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Program:
        # Set atomic_counter high so local program always wins in
        # the additive merge strategy (incoming.counter > existing.counter).
        # In production, the DagRunner fetches programs from Redis with
        # up-to-date counters; here we simulate the same effect.
        p = Program(code=code, state=state, atomic_counter=999_999_999)
        if metrics:
            p.add_metrics(metrics)
        if stage_results:
            p.stage_results = stage_results
        if metadata:
            p.metadata = metadata
        return p

    return _make
