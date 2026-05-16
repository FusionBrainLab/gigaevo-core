"""Round-trip tests for prompt outcome stats over real Redis primitives.

Exercises the write/read contract between
:meth:`GigaEvoArchivePromptFetcher.record_outcome` and
:meth:`RedisPromptStatsProvider.get_stats`. Each outcome lands in a
hash + list pair (HINCRBY / HINCRBYFLOAT / LPUSH + LTRIM), and the
reader aggregates trials, successes, per-metric sums, and the
fitness window across one or more source DBs.

Uses ``fakeredis`` for the sync write client and
``fakeredis.aioredis`` for the async read client so the same in-memory
DB sees both sides.
"""

from __future__ import annotations

import fakeredis
import fakeredis.aioredis
import pytest

from gigaevo.llm.bandit import MutationOutcome
from gigaevo.prompts.coevolution.stats import RedisPromptStatsProvider
from gigaevo.prompts.fetcher import (
    _FITNESS_WINDOW,
    GigaEvoArchivePromptFetcher,
)


@pytest.fixture
def shared_server() -> fakeredis.FakeServer:
    """Single in-memory server backing both sync and async clients."""
    return fakeredis.FakeServer()


@pytest.fixture
def fetcher(
    tmp_path, shared_server: fakeredis.FakeServer
) -> GigaEvoArchivePromptFetcher:
    """Fetcher whose sync write client is swapped to a fakeredis instance."""
    f = GigaEvoArchivePromptFetcher(
        prompt_redis_db=6,
        main_redis_prefix="testpfx",
        main_redis_db=5,
        fallback_prompts_dir=tmp_path,
    )
    f._redis_main_sync = fakeredis.FakeStrictRedis(
        server=shared_server, db=5, decode_responses=True
    )
    return f


@pytest.fixture
def provider(shared_server: fakeredis.FakeServer) -> RedisPromptStatsProvider:
    """Reader pointed at the same fakeredis db=5 as the fetcher."""
    p = RedisPromptStatsProvider(
        host="localhost", port=6379, db=5, prefix="testpfx", min_trials=0
    )
    p._redis_clients[5] = fakeredis.aioredis.FakeRedis(
        server=shared_server, db=5, decode_responses=True
    )
    return p


class TestAtomicWriteRoundTrip:
    """Write a few outcomes, read aggregates back via the provider."""

    @pytest.mark.asyncio
    async def test_single_outcome_round_trip(
        self,
        fetcher: GigaEvoArchivePromptFetcher,
        provider: RedisPromptStatsProvider,
    ) -> None:
        fetcher.record_outcome(
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
        fetcher.record_outcome(
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
        fetcher.record_outcome(
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
            fetcher.record_outcome(
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
            fetcher.record_outcome(
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
        shared_server: fakeredis.FakeServer,
    ) -> None:
        p = RedisPromptStatsProvider(
            host="localhost", port=6379, db=5, prefix="testpfx", min_trials=5
        )
        p._redis_clients[5] = fakeredis.aioredis.FakeRedis(
            server=shared_server, db=5, decode_responses=True
        )
        fetcher.record_outcome(
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


class TestContextVarPackIsolation:
    """``_CURRENT_PACK`` is per-task — concurrent fetches do not stomp."""

    @pytest.mark.asyncio
    async def test_two_tasks_get_independent_packs(
        self, tmp_path, shared_server: fakeredis.FakeServer
    ) -> None:
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
