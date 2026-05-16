"""Round-trip tests for prompt outcome stats over the DataPlane substrate.

Exercises the write/read contract between
:meth:`GigaEvoArchivePromptFetcher.record_outcome` and
:meth:`RedisPromptStatsProvider.get_stats`. Each outcome flows through
the typed :class:`DataPlane` primitives (G-counter for trials /
successes / metrics, bounded list for fitness, set for the metric-name
directory); the reader aggregates across one or more source DataPlanes.

Uses ``fakeredis.aioredis`` to back the DataPlane's connection pool so
both sides observe the same in-memory database without any direct
``redis``-py import.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import fakeredis
import fakeredis.aioredis
import pytest

from gigaevo.dataplane import ActorIdentity, DataPlane, RunId, WorkerId
from gigaevo.llm.bandit import MutationOutcome
from gigaevo.prompts.coevolution.stats import RedisPromptStatsProvider
from gigaevo.prompts.fetcher import (
    _FITNESS_WINDOW,
    GigaEvoArchivePromptFetcher,
)


async def _wired_dataplane(
    server: fakeredis.FakeServer, *, key_prefix: str
) -> DataPlane:
    """Build a started DataPlane whose connection pool is a fakeredis client.

    Mirrors the fixture pattern used by the dataplane unit tests so a
    pool dialled at a real Redis URL is never required for these tests.
    """
    dp = DataPlane("redis://embedded/0", key_prefix=key_prefix)
    fake = fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)
    dp._connection._pool = fake  # type: ignore[attr-defined]
    from gigaevo.dataplane.scripts import LuaRegistry

    lua = LuaRegistry(fake)
    dp._register_builtin_scripts(lua)  # type: ignore[attr-defined]
    await lua.load_all()
    dp._lua = lua  # type: ignore[attr-defined]
    dp._started = True  # type: ignore[attr-defined]
    return dp


@pytest.fixture
def shared_server() -> fakeredis.FakeServer:
    """Single in-memory server backing both writer and reader DataPlanes."""
    return fakeredis.FakeServer()


@pytest.fixture
async def main_dp(
    shared_server: fakeredis.FakeServer,
) -> AsyncIterator[DataPlane]:
    dp = await _wired_dataplane(shared_server, key_prefix="testpfx")
    try:
        yield dp
    finally:
        await dp._connection._pool.aclose()  # type: ignore[union-attr,attr-defined]
        dp._started = False  # type: ignore[attr-defined]
        dp._lua = None  # type: ignore[attr-defined]
        dp._connection._pool = None  # type: ignore[attr-defined]


@pytest.fixture
def actor() -> ActorIdentity:
    return ActorIdentity(run_id=RunId("test-run"), worker_id=WorkerId("w-1"))


@pytest.fixture
def fetcher(
    tmp_path, main_dp: DataPlane, actor: ActorIdentity
) -> GigaEvoArchivePromptFetcher:
    """Fetcher pre-wired with the shared main DataPlane (writes)."""
    return GigaEvoArchivePromptFetcher(
        prompt_redis_db=6,
        main_redis_prefix="testpfx",
        main_redis_db=5,
        fallback_prompts_dir=tmp_path,
        main_dataplane=main_dp,
        actor=actor,
    )


@pytest.fixture
def provider(main_dp: DataPlane) -> RedisPromptStatsProvider:
    """Reader pointed at the same main DataPlane the fetcher writes to."""
    return RedisPromptStatsProvider(
        host="localhost",
        port=6379,
        db=5,
        prefix="testpfx",
        min_trials=0,
        dataplanes=[main_dp],
    )


class TestAtomicWriteRoundTrip:
    """Write a few outcomes, read aggregates back via the provider."""

    @pytest.mark.asyncio
    async def test_single_outcome_round_trip(
        self,
        fetcher: GigaEvoArchivePromptFetcher,
        provider: RedisPromptStatsProvider,
    ) -> None:
        await fetcher.record_outcome(
            prompt_id="abc",
            child_fitness=0.8,
            parent_fitness=0.5,
            higher_is_better=True,
            outcome=MutationOutcome.ACCEPTED,
            child_metrics={"em": 1.0, "f1": 0.75},
        )
        stats = await provider.get_stats("abc")
        assert stats.trials == 1
        assert stats.successes == 1
        assert stats.recent_fitnesses == [0.8]
        assert stats.mean_metrics == {"em": 1.0, "f1": 0.75}

    @pytest.mark.asyncio
    async def test_failure_outcome_increments_trials_only(
        self,
        fetcher: GigaEvoArchivePromptFetcher,
        provider: RedisPromptStatsProvider,
    ) -> None:
        await fetcher.record_outcome(
            prompt_id="abc",
            child_fitness=0.3,
            parent_fitness=0.5,
            higher_is_better=True,
            outcome=MutationOutcome.REJECTED_STRATEGY,
            child_metrics=None,
        )
        stats = await provider.get_stats("abc")
        assert stats.trials == 1
        assert stats.successes == 0
        assert stats.recent_fitnesses == [0.3]
        assert stats.mean_metrics is None

    @pytest.mark.asyncio
    async def test_rejected_acceptor_is_skipped(
        self,
        fetcher: GigaEvoArchivePromptFetcher,
        provider: RedisPromptStatsProvider,
    ) -> None:
        await fetcher.record_outcome(
            prompt_id="abc",
            child_fitness=0.9,
            parent_fitness=0.4,
            higher_is_better=True,
            outcome=MutationOutcome.REJECTED_ACCEPTOR,
        )
        stats = await provider.get_stats("abc")
        assert stats.trials == 0
        assert stats.successes == 0
        assert stats.recent_fitnesses is None

    @pytest.mark.asyncio
    async def test_metrics_sums_accumulate(
        self,
        fetcher: GigaEvoArchivePromptFetcher,
        provider: RedisPromptStatsProvider,
    ) -> None:
        for f in (0.4, 0.6, 0.8):
            await fetcher.record_outcome(
                prompt_id="abc",
                child_fitness=f,
                parent_fitness=0.5,
                higher_is_better=True,
                outcome=MutationOutcome.ACCEPTED,
                child_metrics={"em": 1.0},
            )
        stats = await provider.get_stats("abc")
        assert stats.trials == 3
        # 0.4 < 0.5 fails the improvement test, the other two succeed.
        assert stats.successes == 2
        # metrics_count == 3, sum(em) == 3.0 → mean 1.0
        assert stats.mean_metrics == {"em": 1.0}
        # Fitness list is newest-first.
        assert stats.recent_fitnesses == [0.8, 0.6, 0.4]

    @pytest.mark.asyncio
    async def test_fitness_window_caps_list(
        self,
        fetcher: GigaEvoArchivePromptFetcher,
        provider: RedisPromptStatsProvider,
    ) -> None:
        # Push more than the window — the oldest entries must drop.
        n = _FITNESS_WINDOW + 5
        for i in range(n):
            await fetcher.record_outcome(
                prompt_id="abc",
                child_fitness=float(i),
                parent_fitness=-1.0,  # every entry is an improvement
                higher_is_better=True,
                outcome=MutationOutcome.ACCEPTED,
            )
        stats = await provider.get_stats("abc")
        assert stats.trials == n
        assert stats.successes == n
        assert stats.recent_fitnesses is not None
        assert len(stats.recent_fitnesses) == _FITNESS_WINDOW
        # Newest first → most recent value is n-1.
        assert stats.recent_fitnesses[0] == float(n - 1)
        # Oldest retained value is n - _FITNESS_WINDOW (entries 0 .. n - W - 1
        # were trimmed).
        assert stats.recent_fitnesses[-1] == float(n - _FITNESS_WINDOW)

    @pytest.mark.asyncio
    async def test_min_trials_floor_zeros_success_rate(
        self,
        fetcher: GigaEvoArchivePromptFetcher,
        main_dp: DataPlane,
    ) -> None:
        p = RedisPromptStatsProvider(
            host="localhost",
            port=6379,
            db=5,
            prefix="testpfx",
            min_trials=5,
            dataplanes=[main_dp],
        )
        await fetcher.record_outcome(
            prompt_id="abc",
            child_fitness=0.9,
            parent_fitness=0.5,
            higher_is_better=True,
            outcome=MutationOutcome.ACCEPTED,
        )
        stats = await p.get_stats("abc")
        assert stats.trials == 1
        assert stats.successes == 1
        assert stats.success_rate == 0.0  # below floor

    @pytest.mark.asyncio
    async def test_unknown_prompt_id_returns_zeros(
        self,
        provider: RedisPromptStatsProvider,
    ) -> None:
        stats = await provider.get_stats("never-written")
        assert stats.trials == 0
        assert stats.successes == 0
        assert stats.recent_fitnesses is None
        assert stats.mean_metrics is None


class TestCrossInstanceSharing:
    """Two fetcher instances writing to the same prefix converge via CRDT.

    The trials / successes counters partition per actor so disjoint
    writers commute under the G-counter merge invariant. Both writers
    address the same logical prompt id and the reader observes the
    cross-actor sum, which is the headline guarantee the redesign
    promises (§5.6 of the dataplane design doc).
    """

    @pytest.mark.asyncio
    async def test_two_fetchers_share_via_dataplane(
        self,
        tmp_path,
        main_dp: DataPlane,
    ) -> None:
        actor_a = ActorIdentity(run_id=RunId("run-a"), worker_id=WorkerId("w-a"))
        actor_b = ActorIdentity(run_id=RunId("run-b"), worker_id=WorkerId("w-b"))
        fetcher_a = GigaEvoArchivePromptFetcher(
            prompt_redis_db=6,
            main_redis_prefix="testpfx",
            main_redis_db=5,
            fallback_prompts_dir=tmp_path,
            main_dataplane=main_dp,
            actor=actor_a,
        )
        fetcher_b = GigaEvoArchivePromptFetcher(
            prompt_redis_db=6,
            main_redis_prefix="testpfx",
            main_redis_db=5,
            fallback_prompts_dir=tmp_path,
            main_dataplane=main_dp,
            actor=actor_b,
        )
        # Three improvements from A, two from B; all five are trials.
        for _ in range(3):
            await fetcher_a.record_outcome(
                prompt_id="shared",
                child_fitness=0.8,
                parent_fitness=0.5,
                higher_is_better=True,
                outcome=MutationOutcome.ACCEPTED,
            )
        for _ in range(2):
            await fetcher_b.record_outcome(
                prompt_id="shared",
                child_fitness=0.7,
                parent_fitness=0.4,
                higher_is_better=True,
                outcome=MutationOutcome.ACCEPTED,
            )
        provider = RedisPromptStatsProvider(
            host="localhost",
            port=6379,
            db=5,
            prefix="testpfx",
            min_trials=0,
            dataplanes=[main_dp],
        )
        stats = await provider.get_stats("shared")
        assert stats.trials == 5
        assert stats.successes == 5


class TestContextVarPackIsolation:
    """``_CURRENT_PACK`` is per-task — concurrent fetches do not stomp."""

    @pytest.mark.asyncio
    async def test_two_tasks_get_independent_packs(self) -> None:
        import asyncio

        from gigaevo.prompts.fetcher import _CURRENT_PACK, _PromptPack

        async def task(name: str, value: str) -> str | None:
            _CURRENT_PACK.set(_PromptPack(system=value, user=None, prompt_id=name))
            await asyncio.sleep(0)  # yield to the other task
            pack = _CURRENT_PACK.get()
            return pack.prompt_id if pack else None

        a, b = await asyncio.gather(task("alpha", "AAA"), task("beta", "BBB"))
        assert a == "alpha"
        assert b == "beta"
