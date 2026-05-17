"""Lifecycle / leak-prevention tests for ``run.run_experiment``.

Two failure modes covered:

* ``start()`` succeeds, then something between ``start()`` and the
  ``serve_until_signal`` return raises. The two background tasks
  (``evolution_engine.task`` and ``dag_runner.task``) must still receive
  ``stop()`` before the exception unwinds. The fix wraps the inner block in
  a ``try/finally`` that always awaits both ``stop`` coros.

* The recursive ``instantiate`` builds the engine, runner, writer, etc. into
  one container object. A later ``_target_`` raising mid-recursion leaves
  earlier-built objects unreachable from the ``finally`` block in the
  original code (it only checked ``writer is not None``). The fix walks
  every non-None local in the outer ``finally``; when ``instantiate``
  succeeds and a downstream step raises, every assigned component must be
  closed.
"""

from __future__ import annotations

import asyncio
import threading
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_start_then_exception_still_stops_runner_and_engine():
    """If ``serve_until_signal`` raises, both ``stop()`` calls fire.

    Mirrors the M3 shape: ``dag_runner.start()`` and
    ``evolution_engine.start()`` already created background asyncio tasks,
    then ``serve_until_signal`` raises (e.g. KeyboardInterrupt or an
    OSError installing the signal handler). The two tasks must not leak
    past ``run_experiment``'s return; both ``stop()`` calls must run.
    """
    import run as run_module

    fake_dag_runner = MagicMock()
    fake_dag_runner.start = MagicMock()
    fake_dag_runner.stop = AsyncMock()
    fake_dag_runner.task = None
    fake_dag_runner._dataplane = None
    fake_dag_runner._engine_root = None

    fake_engine = MagicMock()
    fake_engine.start = MagicMock()
    fake_engine.stop = AsyncMock()
    fake_engine.task = None
    fake_engine._dataplane = None
    fake_engine._engine_root = None
    fake_engine.mutation_operator = MagicMock(llm_wrapper=None, _prompt_fetcher=None)
    fake_engine.restore_state = AsyncMock()
    fake_engine.strategy = MagicMock(restore_state=AsyncMock())

    fake_storage = MagicMock()
    fake_storage.config = MagicMock(
        redis_url="redis://localhost:6379/0", key_prefix="test:"
    )
    fake_storage.acquire_instance_lock = AsyncMock()
    fake_storage.has_data = AsyncMock(return_value=False)
    fake_storage.close = AsyncMock()
    fake_storage.size = AsyncMock(return_value=0)
    fake_storage.recover_stranded_programs = AsyncMock(return_value=0)

    fake_loader = MagicMock()
    fake_loader.load = AsyncMock(return_value=[])

    fake_writer = MagicMock()
    fake_writer.close = MagicMock()

    config_with_instances = types.SimpleNamespace(
        redis_storage=fake_storage,
        program_loader=fake_loader,
        dag_runner=fake_dag_runner,
        evolution_engine=fake_engine,
        writer=fake_writer,
    )

    cfg = types.SimpleNamespace(
        problem=types.SimpleNamespace(name="test"),
        redis=MagicMock(
            db=0,
            host="localhost",
            port=6379,
            get=MagicMock(return_value=False),
        ),
        max_generations=None,
        get=MagicMock(return_value=None),
    )
    # ``cfg.get("pipeline_builder", {})`` is queried inside run_experiment.
    cfg.get = MagicMock(side_effect=lambda k, default=None: default)
    cfg.redis.get = MagicMock(return_value=False)

    fake_dataplane = MagicMock()
    fake_dataplane.shutdown = AsyncMock()

    boom = RuntimeError("serve_until_signal failed")

    async def fake_serve_until_signal(*, stop_coros=(), on_stop=()):
        # ``run_experiment`` constructs the stop coros eagerly; close them
        # so the unawaited-coroutine warning does not pollute the test run.
        for coro in stop_coros:
            coro.close()
        raise boom

    with (
        patch.object(run_module, "instantiate", return_value=config_with_instances),
        patch.object(
            run_module,
            "build_dataplane",
            AsyncMock(return_value=fake_dataplane),
        ),
        patch.object(run_module, "build_engine_root", MagicMock(return_value=object())),
        patch.object(run_module, "wire_storage", MagicMock()),
        patch.object(
            run_module,
            "build_actor_identity",
            MagicMock(return_value=object()),
        ),
        patch.object(run_module, "serve_until_signal", fake_serve_until_signal),
    ):
        with pytest.raises(RuntimeError, match="serve_until_signal failed"):
            await run_module.run_experiment(cfg)

    # Both stops must have fired before the exception escaped.
    fake_engine.stop.assert_awaited()
    fake_dag_runner.stop.assert_awaited()
    fake_storage.close.assert_awaited()
    fake_dataplane.shutdown.assert_awaited()
    fake_writer.close.assert_called()


@pytest.mark.asyncio
async def test_instantiate_failure_does_not_leak_pool_or_call_writer():
    """If ``instantiate`` raises, the outer ``finally`` still runs.

    No partial component is reachable from the local scope, so the
    finally block must tolerate every ``X is None`` check without
    crashing. The pool shutdown must still execute and the ambient
    pool token must be reset.
    """
    from gigaevo.programs.stages.python_executors.wrapper import (
        get_ambient_exec_runner_pool,
    )
    import run as run_module

    boom = RuntimeError("instantiate failed mid-tree")

    cfg = types.SimpleNamespace(
        problem=types.SimpleNamespace(name="test"),
        redis=MagicMock(),
        max_generations=None,
        get=MagicMock(side_effect=lambda k, default=None: default),
    )
    cfg.redis.get = MagicMock(return_value=False)

    before_ambient = get_ambient_exec_runner_pool()
    threads_before = {t.ident for t in threading.enumerate()}

    with patch.object(run_module, "instantiate", side_effect=boom):
        with pytest.raises(RuntimeError, match="instantiate failed mid-tree"):
            await run_module.run_experiment(cfg)

    # Ambient pool was bound for the call and reset in finally.
    assert get_ambient_exec_runner_pool() is before_ambient

    # No extra threads leaked. Allow a small window for daemon
    # threads spawned by unrelated test machinery to settle.
    await asyncio.sleep(0)
    threads_after = {t.ident for t in threading.enumerate()}
    leaked = threads_after - threads_before
    assert not leaked, f"Leaked {len(leaked)} thread(s) past run_experiment"


@pytest.mark.asyncio
async def test_engine_stop_is_idempotent_after_serve_already_stopped_it():
    """``stop()`` called twice on EvolutionEngine and DagRunner is a no-op.

    The fix relies on this: ``serve_until_signal`` awaits ``stop()`` once
    on the happy path; the outer ``finally`` awaits ``stop()`` again so
    failure paths still get cleanup. Both must coexist without raising.
    """
    # Engine.stop() with no task and no metrics collector
    engine = object.__new__(
        __import__(
            "gigaevo.evolution.engine.core", fromlist=["EvolutionEngine"]
        ).EvolutionEngine
    )
    engine._running = False
    engine._task = None
    engine._metrics_collector_task = None
    engine._metrics_tracker = None
    engine.storage = MagicMock(close=AsyncMock())

    await engine.stop()
    await engine.stop()  # second call must not raise
    assert engine._task is None

    # DagRunner.stop() with no task and no metrics collector
    runner = object.__new__(
        __import__("gigaevo.runner.dag_runner", fromlist=["DagRunner"]).DagRunner
    )
    runner._stopping = False
    runner._task = None
    runner._active = {}
    runner._done_queue = []
    runner._metrics_collector_task = None
    runner._storage = MagicMock(close=AsyncMock())
    runner._state_manager = MagicMock()

    async def _flush() -> None:
        return None

    runner._flush_done_queue = _flush

    await runner.stop()
    await runner.stop()  # second call must not raise
    assert runner._task is None
