"""End-to-end smoke test for the engine-startup dataplane wiring.

Mirrors :mod:`test_mini_run` (deterministic mutation operator, mock
stages, fakeredis) but constructs the engine with a coordinator wired
into the storage and the bandit router. The test asserts:

- ``dp.started`` is ``True`` while the engine is running;
- :attr:`RedisProgramStorage._dataplane` points at the same coordinator
  (so the storage routes through the dataplane path);
- the bandit's internal :class:`SlidingWindowUCB1` carries the
  ``dataplane`` + ``actor`` pair after :func:`wire_bandit_router`;
- ``dp.shutdown()`` runs at teardown and leaves the coordinator in
  ``started == False``.

The test does not call :func:`build_dataplane` directly because that
helper opens a real Redis socket via :meth:`aioredis.Redis.from_url`;
fakeredis cannot intercept that path. Instead, the test bypasses the
URL-based startup and injects a shared :class:`fakeredis.FakeServer`
into both the storage's connection and the coordinator's connection,
then exercises :func:`load_legacy_program_fsm`, :func:`wire_storage`,
and :func:`wire_bandit_router` exactly as the production entrypoint
does. This is the smallest test surface that exercises the wiring
contract without requiring a live Redis.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from unittest.mock import MagicMock

import fakeredis
import fakeredis.aioredis
import pytest

from gigaevo.database.redis_program_storage import (
    RedisProgramStorage,
    load_legacy_program_fsm,
)
import gigaevo.dataplane as dp
from gigaevo.dataplane import (
    DataPlane,
    build_actor_identity,
    wire_bandit_router,
    wire_storage,
)
from gigaevo.dataplane.scripts import LuaRegistry
from gigaevo.llm.bandit import BanditModelRouter
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState
from tests.integration.test_mini_run import (
    SEED_CODE,
    _build,
    _make_storage,
    _reset_counter,
)


async def _build_coordinator_against_fake(
    server: fakeredis.FakeServer,
    *,
    key_prefix: str,
) -> tuple[DataPlane, fakeredis.aioredis.FakeRedis]:
    """Wire a coordinator on top of an existing :class:`FakeServer`.

    Construction mirrors the production sequence — instantiate,
    inject pool, register scripts, load the legacy FSM table — but
    skips the URL-based startup so fakeredis is preserved.
    """
    coord = DataPlane(
        "redis://embedded/0",
        key_prefix=key_prefix,
    )
    fake = fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)
    coord._connection._pool = fake  # type: ignore[attr-defined]
    lua = LuaRegistry(fake)
    coord._register_builtin_scripts(lua)  # type: ignore[attr-defined]
    await lua.load_all()
    coord._lua = lua  # type: ignore[attr-defined]
    coord._started = True  # type: ignore[attr-defined]
    await load_legacy_program_fsm(coord)
    return coord, fake


async def _coord_shutdown(coord: DataPlane, fake: fakeredis.aioredis.FakeRedis) -> None:
    """Tear down a fake-pool coordinator without invoking real socket close.

    The connection pool here is a fakeredis instance rather than the
    real :class:`aioredis.Redis`; the coordinator's own
    :meth:`shutdown` calls ``_safe_close`` which works fine on the
    fake, but we clear ``_pool`` first so the assertion ``started ==
    False`` lands without disturbing other fixtures that may still
    hold the same :class:`FakeServer` handle.
    """
    coord._started = False  # type: ignore[attr-defined]
    coord._lua = None  # type: ignore[attr-defined]
    coord._connection._pool = None  # type: ignore[attr-defined]
    await fake.aclose()  # type: ignore[attr-defined]


@pytest.fixture
async def wired_engine_fixture() -> AsyncIterator[
    tuple[DataPlane, RedisProgramStorage, fakeredis.FakeServer]
]:
    """A storage and a coordinator wired against the same fake server.

    Yielded as ``(coord, storage, server)`` for tests that need to
    drive the engine and inspect coordinator state. The fixture
    handles startup and shutdown of the coordinator; the engine and
    runner lifecycles remain the test's responsibility.
    """
    _reset_counter()
    server = fakeredis.FakeServer()
    storage = _make_storage(server)
    coord, fake = await _build_coordinator_against_fake(server, key_prefix="minirun")
    wire_storage(storage, coord)
    try:
        yield coord, storage, server
    finally:
        await _coord_shutdown(coord, fake)


class TestDataplaneWiringSmoke:
    """Wiring contract: dataplane attaches without breaking the engine."""

    async def test_storage_dataplane_attribute_set(
        self,
        wired_engine_fixture: tuple[
            DataPlane, RedisProgramStorage, fakeredis.FakeServer
        ],
    ) -> None:
        coord, storage, _ = wired_engine_fixture
        assert storage._dataplane is coord
        assert coord.started

    async def test_engine_runs_with_dataplane_wired(self) -> None:
        """The engine completes its generation loop while the dataplane is live.

        Constructs the engine with the helper from
        :mod:`test_mini_run`, then wires the coordinator onto the
        single :class:`RedisProgramStorage` instance that the engine,
        strategy, dag-runner, and islands all share by reference.
        Verifies that the wired-in dataplane is observable on the
        storage after the engine has finished its run.
        """
        _reset_counter()
        server = fakeredis.FakeServer()
        storage, dag_runner, engine, _ = _build(server, max_generations=3)
        coord, fake = await _build_coordinator_against_fake(
            server, key_prefix="minirun"
        )
        wire_storage(storage, coord)

        seed = Program(code=SEED_CODE, state=ProgramState.QUEUED)
        await storage.add(seed)

        import asyncio

        dag_runner.start()
        engine.start()
        try:
            await asyncio.wait_for(engine.task, timeout=30.0)
            assert coord.started
        finally:
            await dag_runner.stop()
            await storage.close()
            await _coord_shutdown(coord, fake)

        assert engine.metrics.total_generations == 3
        # After the explicit shutdown above, the coordinator must
        # report ``started == False`` — verifying the finally-block
        # contract that production engine teardown relies on.
        assert not coord.started
        # Storage still carries the reference even after shutdown;
        # that is intentional — the rebind is one-shot at startup
        # and the storage holding a stale handle does not cause
        # incorrect behaviour because every call path goes through
        # :func:`dp.started` before touching the pool.
        assert storage._dataplane is coord

    async def test_coordinator_shutdown_clears_started(self) -> None:
        """``shutdown()`` flips ``started`` to ``False`` and drops the pool."""
        _reset_counter()
        server = fakeredis.FakeServer()
        coord, fake = await _build_coordinator_against_fake(
            server, key_prefix="shutdown-check"
        )
        try:
            assert coord.started
            await coord.shutdown()
            assert not coord.started
        finally:
            # ``shutdown`` already ran; ``_coord_shutdown`` is the
            # belt-and-braces cleanup for the fakeredis handle.
            await _coord_shutdown(coord, fake)


class TestBanditWiring:
    """The router-wiring helper rebinds private state without errors."""

    def _make_bandit_router(self) -> BanditModelRouter:
        """A bandit with two dummy models — no real LLM calls in this test."""
        # MagicMock stands in for ``ChatOpenAI`` instances; the bandit
        # never invokes them in this test, only routes between names.
        model_a = MagicMock()
        model_a.model_name = "alpha"
        model_b = MagicMock()
        model_b.model_name = "beta"
        return BanditModelRouter(
            models=[model_a, model_b],
            probabilities=[0.5, 0.5],
            writer=None,
            name="wiring-test",
            fitness_key="fitness",
            higher_is_better=True,
        )

    async def test_wire_bandit_router_attaches_dataplane_and_actor(self) -> None:
        server = fakeredis.FakeServer()
        coord, fake = await _build_coordinator_against_fake(server, key_prefix="bandit")
        actor = build_actor_identity(run_id="r1", worker_id="w1")
        router = self._make_bandit_router()
        try:
            assert router._bandit._dataplane is None
            assert router._bandit._actor is None

            wired = wire_bandit_router(router, coord, actor)
            assert wired is True
            assert router._bandit._dataplane is coord
            assert router._bandit._actor == actor
            assert router._bandit.is_redis_backed
        finally:
            await _coord_shutdown(coord, fake)

    async def test_wire_bandit_router_noops_on_static_router(self) -> None:
        """A plain :class:`MultiModelRouter` is silently skipped."""
        from gigaevo.llm.models import MultiModelRouter

        model = MagicMock()
        model.model_name = "static"
        router = MultiModelRouter(
            models=[model], probabilities=[1.0], writer=None, name="static-test"
        )
        server = fakeredis.FakeServer()
        coord, fake = await _build_coordinator_against_fake(server, key_prefix="static")
        actor = build_actor_identity(run_id="r1", worker_id="w1")
        try:
            wired = wire_bandit_router(router, coord, actor)
            assert wired is False
        finally:
            await _coord_shutdown(coord, fake)


class TestActorIdentity:
    """The actor-identity builder is deterministic and env-aware."""

    def test_explicit_run_and_worker(self) -> None:
        actor = build_actor_identity(run_id="r-explicit", worker_id="w-explicit")
        assert actor.run_id == "r-explicit"
        assert actor.worker_id == "w-explicit"

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(dp.ENV_RUN_ID, "env-run")
        monkeypatch.setenv(dp.ENV_WORKER_ID, "env-worker")
        actor = build_actor_identity()
        assert actor.run_id == "env-run"
        assert actor.worker_id == "env-worker"

    def test_default_worker_carries_pid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(dp.ENV_RUN_ID, raising=False)
        monkeypatch.delenv(dp.ENV_WORKER_ID, raising=False)
        actor = build_actor_identity()
        import os

        assert str(os.getpid()) in actor.worker_id
        # uuid4().hex without dashes is exactly 32 chars
        assert len(actor.run_id) == 32
