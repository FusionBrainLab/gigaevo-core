"""Shared test fixtures for DAG, storage, and state manager tests."""

from __future__ import annotations

import asyncio
from typing import Any, Optional

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
    def bind(self, path: list[str]) -> "NullWriter":
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
    data: Optional[MockOutput] = None


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
# Exec-runner pool cleanup (prevents stale event-loop references)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
async def _clear_exec_runner_pool():
    """Clear the cached WorkerPool between tests to avoid stale event-loop refs.

    The default_exec_runner_pool() is an @lru_cache singleton whose asyncio
    Queue, Lock, and subprocess streams are bound to whatever event loop was
    running when they were created.  pytest-asyncio creates a fresh loop per
    test function, so the cached pool must be reset to prevent
    "Future attached to a different loop" errors.

    On teardown, shutdown() kills idle subprocess workers *before* the loop
    closes, preventing "Event loop is closed" warnings from transport __del__.
    """
    from gigaevo.programs.stages.python_executors.wrapper import (
        default_exec_runner_pool,
    )

    default_exec_runner_pool.cache_clear()
    yield
    # Kill idle subprocess workers while the event loop is still open.
    try:
        pool = default_exec_runner_pool()
        await pool.shutdown()
    except Exception:
        pass
    default_exec_runner_pool.cache_clear()


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
