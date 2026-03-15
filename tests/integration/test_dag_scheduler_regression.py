"""Regression tests for DagRunner scheduler bugs.

TOCTOU in DagRunner._maintain (fixed):
    _maintain classifies a task as timed-out based on elapsed time, but between
    that classification and the DISCARD write, _execute_dag may have already set
    the program to DONE.  Fix: fetch the program from Redis before discarding;
    skip discard if state is already DONE, record the timeout metric regardless.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from unittest.mock import MagicMock

import fakeredis.aioredis

from gigaevo.database.redis import RedisProgramStorageConfig
from gigaevo.database.redis_program_storage import RedisProgramStorage
from gigaevo.database.state_manager import ProgramStateManager
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState
from gigaevo.runner.dag_runner import DagRunner, DagRunnerConfig, TaskInfo

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_storage() -> tuple[RedisProgramStorage, object]:
    server = fakeredis.FakeServer()
    config = RedisProgramStorageConfig(
        redis_url="redis://fake:6379/0",
        key_prefix="test",
    )
    storage = RedisProgramStorage(config)
    fake_redis = fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)
    storage._conn._redis = fake_redis
    storage._conn._closing = False
    return storage, fake_redis


def _make_program() -> Program:
    return Program(
        code="def solve(): return 1",
        state=ProgramState.QUEUED,
        atomic_counter=999_999_999,
    )


def _make_runner(storage: RedisProgramStorage) -> DagRunner:
    dag_blueprint = MagicMock()
    writer = MagicMock()
    writer.bind = MagicMock(return_value=writer)
    return DagRunner(
        storage=storage,
        dag_blueprint=dag_blueprint,
        config=DagRunnerConfig(dag_timeout=1.0, poll_interval=0.1),
        writer=writer,
    )


# ---------------------------------------------------------------------------
# TOCTOU: _maintain must not discard a program that completed between the
# "timed-out" classification and the discard write.
# ---------------------------------------------------------------------------


async def test_maintain_skips_discard_when_program_already_done() -> None:
    """_maintain must not overwrite a DONE program with DISCARDED.

    Sequence:
      1. Program transitions QUEUED → RUNNING → DONE in storage (DAG finished).
      2. A sleeping asyncio task representing that program sits in _active with
         a started_at far in the past, so _maintain classifies it as timed-out.
      3. _maintain() is called.
      4. Because the program is already DONE in Redis, discard is skipped.
      5. Final state remains DONE; timeout counter is still incremented.
    """
    storage, _ = _make_storage()
    try:
        state_manager = ProgramStateManager(storage)
        runner = _make_runner(storage)

        program = _make_program()
        await storage.add(program)
        await state_manager.set_program_state(program, ProgramState.RUNNING)
        await state_manager.set_program_state(program, ProgramState.DONE)

        stored = await storage.get(program.id)
        assert stored is not None and stored.state == ProgramState.DONE

        async def _sleeper():
            await asyncio.sleep(3600)

        task = asyncio.create_task(_sleeper())
        old_start = time.monotonic() - 10_000.0  # way past dag_timeout=1.0
        runner._active[program.id] = TaskInfo(task, program.id, old_start)

        await runner._maintain()

        final = await storage.get(program.id)
        assert final is not None
        assert final.state == ProgramState.DONE, (
            f"_maintain discarded a successfully-completed program. "
            f"Final state: {final.state}. "
            "Fix: _maintain must check prog.state == DONE before discarding."
        )
        assert runner._metrics.dag_timeouts == 1

    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        await storage.close()


async def test_maintain_discards_running_program_on_timeout() -> None:
    """Normal timeout path: a still-RUNNING program must be DISCARDED.

    Ensures the DONE guard in _maintain does not break the normal discard path.
    """
    storage, _ = _make_storage()
    try:
        state_manager = ProgramStateManager(storage)
        runner = _make_runner(storage)

        program = _make_program()
        await storage.add(program)
        await state_manager.set_program_state(program, ProgramState.RUNNING)

        async def _sleeper():
            await asyncio.sleep(3600)

        task = asyncio.create_task(_sleeper())
        old_start = time.monotonic() - 10_000.0
        runner._active[program.id] = TaskInfo(task, program.id, old_start)

        await runner._maintain()

        final = await storage.get(program.id)
        assert final is not None
        assert final.state == ProgramState.DISCARDED, (
            f"A timed-out RUNNING program must be DISCARDED. Final state: {final.state}."
        )
        assert runner._metrics.dag_timeouts == 1

    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        await storage.close()
