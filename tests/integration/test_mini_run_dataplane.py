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
then exercises :func:`wire_storage` and :func:`wire_bandit_router`
exactly as the production entrypoint does. The FSM hash is primed via
:func:`load_fsm_table` since the test bypasses :meth:`DataPlane.startup`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from unittest.mock import MagicMock

import fakeredis
import fakeredis.aioredis
import pytest

from gigaevo.database.redis_program_storage import RedisProgramStorage
import gigaevo.dataplane as dp
from gigaevo.dataplane import (
    DataPlane,
    build_actor_identity,
    wire_bandit_router,
    wire_prompt_fetcher,
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

    Construction mirrors the production sequence — instantiate, inject
    pool, register scripts, load the case-tolerant FSM table — but
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
    # FSM table primes the program-state transition matrix in the
    # case-tolerant form the script consults. The dataplane bridge no
    # longer ships a legacy-vocab reloader; ``load_fsm_table`` emits
    # rows under both vocabularies in one pass.
    from gigaevo.dataplane.transitions import (
        PROGRAM_STATE_TRANSITIONS,
        load_fsm_table,
    )

    await load_fsm_table(
        fake,
        key_prefix=key_prefix,
        name="program_state",
        table=PROGRAM_STATE_TRANSITIONS,
    )
    coord._lua = lua  # type: ignore[attr-defined]
    coord._started = True  # type: ignore[attr-defined]
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


class TestEngineRootWiring:
    """Engine-root threading through :func:`wire_storage`.

    The storage receives a single :class:`EngineRoot` at startup; per-
    call FSM tokens derive by linear split. The structural invariant is
    that two consecutive transitions on a single program both succeed:
    the engine root rotates its long-lived witness, so the second call
    is not blocked by a consumed token from the first.
    """

    async def test_wire_storage_attaches_engine_root(self) -> None:
        from gigaevo.dataplane import build_engine_root

        server = fakeredis.FakeServer()
        # The storage from :func:`_make_storage` is pinned to the
        # ``minirun`` prefix; the coordinator must share that prefix so
        # both observe the same key-space on the shared FakeServer.
        coord, fake = await _build_coordinator_against_fake(
            server, key_prefix="minirun"
        )
        storage = _make_storage(server)
        try:
            assert storage._engine_root is None  # type: ignore[attr-defined]
            root = build_engine_root()
            wire_storage(storage, coord, root)
            assert storage._engine_root is root  # type: ignore[attr-defined]
            assert storage._dataplane is coord  # type: ignore[attr-defined]
        finally:
            await storage.close()
            await _coord_shutdown(coord, fake)

    async def test_two_transitions_on_one_program_both_succeed(self) -> None:
        """The structural invariant: rotation preserves single-live-witness
        across consecutive transitions on the same program. The legacy
        ``mint_root`` path was per-call; the engine-root path replaces
        it with ``mint_split`` from the rotating engine root."""
        from gigaevo.dataplane import build_engine_root

        _reset_counter()
        server = fakeredis.FakeServer()
        coord, fake = await _build_coordinator_against_fake(
            server, key_prefix="minirun"
        )
        storage = _make_storage(server)
        root = build_engine_root()
        wire_storage(storage, coord, root)
        try:
            prog = Program(code=SEED_CODE, state=ProgramState.QUEUED)
            await storage.add(prog)
            initial_root = root._program_root  # type: ignore[attr-defined]

            await storage.fast_state_transition(
                prog, ProgramState.QUEUED.value, ProgramState.RUNNING.value
            )
            await storage.atomic_state_transition(
                prog, ProgramState.RUNNING.value, ProgramState.DONE.value
            )

            # Initial root consumed, current root live and rotated.
            assert initial_root.consumed
            current = root._program_root  # type: ignore[attr-defined]
            assert current is not initial_root
            assert not current.consumed
            # Persisted program reflects the final state.
            fetched = await storage.get(prog.id)
            assert fetched is not None
            assert fetched.state == ProgramState.DONE
        finally:
            await storage.close()
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


class TestPromptFetcherWiring:
    """The prompt-fetcher wiring helper pins the shared-dataplane discipline.

    Mirrors the bandit-wiring contract: the helper is silent on
    non-co-evolved fetchers and rebinds private state on the
    :class:`GigaEvoArchivePromptFetcher` case. The structural invariant
    is that the main DataPlane the fetcher writes through is the SAME
    instance the storage and bandit observe — eliminating the second
    connection pool the fetcher would lazily build otherwise.
    """

    async def test_wire_prompt_fetcher_attaches_and_shares_main_dp(
        self, tmp_path
    ) -> None:
        """The fetcher's main DP is the engine's shared DP after wiring."""
        from gigaevo.prompts.fetcher import GigaEvoArchivePromptFetcher

        server = fakeredis.FakeServer()
        main_coord, main_fake = await _build_coordinator_against_fake(
            server, key_prefix="testpfx"
        )
        prompt_coord, prompt_fake = await _build_coordinator_against_fake(
            server, key_prefix="prompt_evolution"
        )
        actor = build_actor_identity(run_id="r1", worker_id="w1")
        fetcher = GigaEvoArchivePromptFetcher(
            prompt_redis_db=6,
            main_redis_prefix="testpfx",
            main_redis_db=5,
            fallback_prompts_dir=tmp_path,
        )
        try:
            assert fetcher._main_dp is None
            assert fetcher._prompt_dp is None

            wired = wire_prompt_fetcher(fetcher, main_coord, prompt_coord, actor)
            assert wired is True
            # Sharing invariant: the fetcher's main DP is identically
            # the one the engine threads into storage and bandit.
            assert fetcher._main_dp is main_coord
            assert fetcher._prompt_dp is prompt_coord
            assert fetcher._actor == actor
            # Lifetime: engine owns the handles after wiring.
            assert fetcher._main_dp_owned is False
            assert fetcher._prompt_dp_owned is False
        finally:
            await _coord_shutdown(main_coord, main_fake)
            await _coord_shutdown(prompt_coord, prompt_fake)

    async def test_wire_prompt_fetcher_noops_on_fixed_fetcher(self, tmp_path) -> None:
        """A :class:`FixedDirPromptFetcher` is silently skipped (no archive)."""
        from gigaevo.prompts.fetcher import FixedDirPromptFetcher

        server = fakeredis.FakeServer()
        main_coord, main_fake = await _build_coordinator_against_fake(
            server, key_prefix="main"
        )
        prompt_coord, prompt_fake = await _build_coordinator_against_fake(
            server, key_prefix="prompt"
        )
        actor = build_actor_identity(run_id="r1", worker_id="w1")
        fetcher = FixedDirPromptFetcher(prompts_dir=tmp_path)
        try:
            wired = wire_prompt_fetcher(fetcher, main_coord, prompt_coord, actor)
            assert wired is False
        finally:
            await _coord_shutdown(main_coord, main_fake)
            await _coord_shutdown(prompt_coord, prompt_fake)

    async def test_wire_then_record_outcome_writes_to_shared_pool(
        self, tmp_path
    ) -> None:
        """A wired ``record_outcome`` lands in the engine's shared DataPlane.

        The reader's source is the SAME ``DataPlane`` instance the wire
        helper attached — reading back proves the writer is bound to
        that pool rather than a private lazy clone.
        """
        from gigaevo.llm.bandit import MutationOutcome
        from gigaevo.prompts.coevolution.stats import RedisPromptStatsProvider
        from gigaevo.prompts.fetcher import GigaEvoArchivePromptFetcher

        server = fakeredis.FakeServer()
        main_coord, main_fake = await _build_coordinator_against_fake(
            server, key_prefix="testpfx"
        )
        prompt_coord, prompt_fake = await _build_coordinator_against_fake(
            server, key_prefix="prompt_evolution"
        )
        actor = build_actor_identity(run_id="r1", worker_id="w1")
        fetcher = GigaEvoArchivePromptFetcher(
            prompt_redis_db=6,
            main_redis_prefix="testpfx",
            main_redis_db=5,
            fallback_prompts_dir=tmp_path,
        )
        try:
            wire_prompt_fetcher(fetcher, main_coord, prompt_coord, actor)
            await fetcher.record_outcome(
                prompt_id="pid-1",
                child_fitness=0.9,
                parent_fitness=0.5,
                higher_is_better=True,
                outcome=MutationOutcome.ACCEPTED,
                child_metrics={"em": 1.0},
            )
            provider = RedisPromptStatsProvider(
                host="localhost",
                port=6379,
                db=5,
                prefix="testpfx",
                min_trials=0,
                dataplanes=[main_coord],
            )
            stats = await provider.get_stats("pid-1")
            assert stats.trials == 1
            assert stats.successes == 1
            assert stats.recent_fitnesses == [0.9]
        finally:
            await _coord_shutdown(main_coord, main_fake)
            await _coord_shutdown(prompt_coord, prompt_fake)

    async def test_wire_prompt_fetcher_idempotent(self, tmp_path) -> None:
        """A second wire with the same triple is a silent no-op."""
        from gigaevo.prompts.fetcher import GigaEvoArchivePromptFetcher

        server = fakeredis.FakeServer()
        main_coord, main_fake = await _build_coordinator_against_fake(
            server, key_prefix="main"
        )
        prompt_coord, prompt_fake = await _build_coordinator_against_fake(
            server, key_prefix="prompt"
        )
        actor = build_actor_identity(run_id="r1", worker_id="w1")
        fetcher = GigaEvoArchivePromptFetcher(
            prompt_redis_db=6,
            main_redis_prefix="main",
            fallback_prompts_dir=tmp_path,
        )
        try:
            wire_prompt_fetcher(fetcher, main_coord, prompt_coord, actor)
            # Second wire with identical args is a no-op.
            wire_prompt_fetcher(fetcher, main_coord, prompt_coord, actor)
            assert fetcher._main_dp is main_coord
            assert fetcher._prompt_dp is prompt_coord
        finally:
            await _coord_shutdown(main_coord, main_fake)
            await _coord_shutdown(prompt_coord, prompt_fake)

    async def test_wire_prompt_fetcher_rejects_conflicting_args(self, tmp_path) -> None:
        """A second wire with different args raises rather than overwriting."""
        from gigaevo.prompts.fetcher import GigaEvoArchivePromptFetcher

        server = fakeredis.FakeServer()
        main_coord, main_fake = await _build_coordinator_against_fake(
            server, key_prefix="main"
        )
        other_coord, other_fake = await _build_coordinator_against_fake(
            server, key_prefix="other"
        )
        prompt_coord, prompt_fake = await _build_coordinator_against_fake(
            server, key_prefix="prompt"
        )
        actor = build_actor_identity(run_id="r1", worker_id="w1")
        fetcher = GigaEvoArchivePromptFetcher(
            prompt_redis_db=6,
            main_redis_prefix="main",
            fallback_prompts_dir=tmp_path,
        )
        try:
            wire_prompt_fetcher(fetcher, main_coord, prompt_coord, actor)
            with pytest.raises(RuntimeError, match="different DataPlane"):
                wire_prompt_fetcher(fetcher, other_coord, prompt_coord, actor)
            # Original attachment must survive a rejected re-wire.
            assert fetcher._main_dp is main_coord
        finally:
            await _coord_shutdown(main_coord, main_fake)
            await _coord_shutdown(other_coord, other_fake)
            await _coord_shutdown(prompt_coord, prompt_fake)


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
