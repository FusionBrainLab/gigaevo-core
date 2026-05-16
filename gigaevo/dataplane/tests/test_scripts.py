"""Tests for :class:`gigaevo.dataplane.scripts.LuaRegistry`.

Coverage:
    - register / load_all happy path;
    - register-after-load-all is repaired by the missing-SHA path;
    - re-register with a different source invalidates the SHA;
    - evalsha on an unregistered name raises ``ScriptNotRegisteredError``;
    - NOSCRIPT triggers exactly one reload-and-retry;
    - a second consecutive NOSCRIPT raises ``ScriptLostError``;
    - concurrent evalsha racing on NOSCRIPT does not double-load;
    - properties ``registered`` and ``loaded_count`` reflect state.

Tests build a tiny mock Redis client because we want to control when
NOSCRIPT fires; ``fakeredis`` always replies cleanly to EVALSHA, which
makes the NOSCRIPT recovery path untestable against it.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from redis.exceptions import NoScriptError

from gigaevo.dataplane.errors import ScriptLostError, ScriptNotRegisteredError
from gigaevo.dataplane.ids import ScriptName
from gigaevo.dataplane.scripts import LuaRegistry


class _MockRedis:
    """Minimal Redis stand-in that records script loads and EVALSHA calls.

    Tracks the "loaded" SHA set so EVALSHA(sha) raises ``NoScriptError``
    iff ``sha`` is not currently loaded — i.e. it models the actual
    Redis contract instead of a fixed counter.

    ``forget_on_next_load`` simulates the flush-script-cache scenario:
    while True, every successful ``script_load`` immediately invalidates
    every prior SHA so the next EVALSHA on a stale handle hits NOSCRIPT.
    The default is False (loads accumulate).
    """

    __slots__ = (
        "loaded_shas",
        "loaded_sources",
        "evalsha_calls",
        "next_sha",
        "flush_before_load",
    )

    def __init__(self) -> None:
        self.loaded_shas: set[str] = set()
        self.loaded_sources: list[str] = []
        self.evalsha_calls: list[tuple[str, tuple[Any, ...]]] = []
        self.next_sha = 0
        self.flush_before_load = False

    async def script_load(self, source: str) -> str:
        if self.flush_before_load:
            self.loaded_shas.clear()
        self.loaded_sources.append(source)
        self.next_sha += 1
        sha = f"sha-{self.next_sha:04d}"
        self.loaded_shas.add(sha)
        return sha

    async def evalsha(self, sha: str, *args: Any) -> Any:
        self.evalsha_calls.append((sha, args))
        if sha not in self.loaded_shas:
            raise NoScriptError("simulated NOSCRIPT")
        return "OK"

    def flush_scripts(self) -> None:
        """Simulate ``SCRIPT FLUSH`` — every prior SHA becomes invalid."""
        self.loaded_shas.clear()


def _name(s: str) -> ScriptName:
    return ScriptName(s)


class TestRegistration:
    def test_register_and_query(self) -> None:
        reg = LuaRegistry(_MockRedis())  # type: ignore[arg-type]
        reg.register(_name("a"), "return 1")
        assert reg.is_registered(_name("a"))
        assert not reg.is_registered(_name("missing"))
        assert reg.registered == (_name("a"),)
        assert reg.loaded_count == 0

    def test_reregister_same_source_quiet(self) -> None:
        reg = LuaRegistry(_MockRedis())  # type: ignore[arg-type]
        reg.register(_name("a"), "return 1")
        reg.register(_name("a"), "return 1")  # no warning, no change
        assert reg.registered == (_name("a"),)

    def test_reregister_different_source_drops_sha(self) -> None:
        """Re-registering with a new source must invalidate the cached SHA.

        Otherwise the next ``evalsha`` would EVALSHA the old SHA and
        either hit NOSCRIPT (best case) or — worse — run the wrong
        script if a different registrant happened to load the same
        source under another name.
        """
        mock = _MockRedis()
        reg = LuaRegistry(mock)  # type: ignore[arg-type]
        reg.register(_name("a"), "return 1")
        # Force the SHA cache to be populated.
        reg._sha[_name("a")] = "sha-stale"  # noqa: SLF001
        reg.register(_name("a"), "return 2")
        assert _name("a") not in reg._sha  # noqa: SLF001


class TestLoadAll:
    async def test_load_all_populates_shas(self) -> None:
        mock = _MockRedis()
        reg = LuaRegistry(mock)  # type: ignore[arg-type]
        reg.register(_name("a"), "return 1")
        reg.register(_name("b"), "return 2")
        await reg.load_all()
        assert reg.loaded_count == 2
        assert len(mock.loaded_sources) == 2

    async def test_load_all_no_scripts_is_noop(self) -> None:
        reg = LuaRegistry(_MockRedis())  # type: ignore[arg-type]
        await reg.load_all()
        assert reg.loaded_count == 0


class TestEvalsha:
    async def test_evalsha_unregistered_raises(self) -> None:
        reg = LuaRegistry(_MockRedis())  # type: ignore[arg-type]
        with pytest.raises(ScriptNotRegisteredError):
            await reg.evalsha(_name("missing"), keys=[], args=[])

    async def test_evalsha_happy_path(self) -> None:
        mock = _MockRedis()
        reg = LuaRegistry(mock)  # type: ignore[arg-type]
        reg.register(_name("a"), "return 1")
        await reg.load_all()
        result = await reg.evalsha(_name("a"), keys=["k"], args=["v"])
        assert result == "OK"
        assert len(mock.evalsha_calls) == 1

    async def test_register_after_load_all_works_via_missing_sha(
        self,
    ) -> None:
        """Registering a new script after load_all must still be invokable.

        The first :meth:`evalsha` for the late-registered script sees a
        ``None`` SHA, takes the reload path, and proceeds normally.
        """
        mock = _MockRedis()
        reg = LuaRegistry(mock)  # type: ignore[arg-type]
        reg.register(_name("a"), "return 1")
        await reg.load_all()
        assert reg.loaded_count == 1
        # Late registration.
        reg.register(_name("late"), "return 2")
        assert reg.loaded_count == 1
        result = await reg.evalsha(_name("late"), keys=[], args=[])
        assert result == "OK"
        # The late script is now SHA-cached.
        assert reg.loaded_count == 2

    async def test_noscript_triggers_one_reload(self) -> None:
        mock = _MockRedis()
        # SCRIPT FLUSH (re-load invalidates prior SHAs) on the next load
        # so the post-load EVALSHA still sees the latest SHA.
        mock.flush_before_load = True
        reg = LuaRegistry(mock)  # type: ignore[arg-type]
        reg.register(_name("a"), "return 1")
        await reg.load_all()
        # Drop the loaded SHA so the original EVALSHA hits NOSCRIPT.
        mock.flush_scripts()
        initial_loads = len(mock.loaded_sources)
        result = await reg.evalsha(_name("a"), keys=[], args=[])
        assert result == "OK"
        # Exactly one extra SCRIPT LOAD happened on the NOSCRIPT branch.
        assert len(mock.loaded_sources) == initial_loads + 1
        # Two EVALSHA calls total: original + retry.
        assert len(mock.evalsha_calls) == 2

    async def test_two_consecutive_noscript_raises_script_lost(self) -> None:
        """If SCRIPT FLUSH fires *between* the reload and the retry,
        the retry hits NOSCRIPT a second time and ``ScriptLostError`` is
        raised.
        """

        class _ChaosRedis(_MockRedis):
            async def script_load(self, source: str) -> str:
                sha = await super().script_load(source)
                # Immediately invalidate the just-loaded SHA so the
                # retry EVALSHA inside the registry hits NOSCRIPT too.
                self.loaded_shas.discard(sha)
                return sha

        mock = _ChaosRedis()
        reg = LuaRegistry(mock)  # type: ignore[arg-type]
        reg.register(_name("a"), "return 1")
        # First load_all also has the post-load invalidation, so the
        # SHA cache holds a sha that Redis doesn't know about.
        await reg.load_all()
        with pytest.raises(ScriptLostError):
            await reg.evalsha(_name("a"), keys=[], args=[])


class TestConcurrentNoScript:
    async def test_concurrent_noscript_coalesces_reload(self) -> None:
        """Twenty coroutines racing on NOSCRIPT must reload once.

        The double-checked-locking discipline inside ``_reload``
        guarantees the first caller through the lock issues exactly one
        ``SCRIPT LOAD`` and every subsequent waiter reuses the cached
        SHA. Failure mode without the lock: 20 SCRIPT LOAD round-trips,
        a thundering herd against Redis under churn.
        """
        mock = _MockRedis()
        reg = LuaRegistry(mock)  # type: ignore[arg-type]
        reg.register(_name("a"), "return 1")
        await reg.load_all()
        initial_loads = len(mock.loaded_sources)
        # Drop the loaded SHA so every concurrent EVALSHA hits NOSCRIPT
        # — but ONLY once: after the first reload, the new SHA is
        # loaded and stays loaded, so retries succeed.
        mock.flush_scripts()

        async def call() -> Any:
            return await reg.evalsha(_name("a"), keys=[], args=[])

        results = await asyncio.gather(*[call() for _ in range(20)])
        assert all(r == "OK" for r in results)
        # Without the reload lock this would be ~20; with it, it's 1.
        extra_loads = len(mock.loaded_sources) - initial_loads
        assert extra_loads == 1, (
            f"Expected exactly one reload across concurrent NOSCRIPT recoveries, "
            f"saw {extra_loads}."
        )

    async def test_concurrent_load_all_safe(self) -> None:
        """Multiple ``load_all`` racers stay internally consistent."""
        mock = _MockRedis()
        reg = LuaRegistry(mock)  # type: ignore[arg-type]
        reg.register(_name("a"), "return 1")
        reg.register(_name("b"), "return 2")
        await asyncio.gather(*[reg.load_all() for _ in range(5)])
        # Both scripts always end up in the SHA cache.
        assert reg.loaded_count == 2

    async def test_partial_load_all_repaired_by_missing_sha_path(self) -> None:
        """A load_all that fails midway leaves the registry in a state
        the missing-SHA branch of evalsha repairs.

        The second ``script_load`` raises; the first script lands in
        the SHA cache, the second does not. A subsequent ``evalsha`` for
        the un-loaded script must repair itself with a single SCRIPT
        LOAD on the missing-SHA fallback.
        """
        call_count = {"n": 0}

        class _FlakyRedis(_MockRedis):
            async def script_load(self, source: str) -> str:
                call_count["n"] += 1
                if call_count["n"] == 2:
                    raise RuntimeError("simulated partial-load failure")
                return await _MockRedis.script_load(self, source)

        mock = _FlakyRedis()
        reg = LuaRegistry(mock)  # type: ignore[arg-type]
        reg.register(_name("a"), "return 1")
        reg.register(_name("b"), "return 2")
        with pytest.raises(RuntimeError, match="partial-load failure"):
            await reg.load_all()
        # Only one script is in the SHA cache.
        assert reg.loaded_count == 1
        # ``b`` repairs on first use via the missing-SHA branch.
        loads_before = len(mock.loaded_sources)
        result = await reg.evalsha(_name("b"), keys=[], args=[])
        assert result == "OK"
        assert len(mock.loaded_sources) == loads_before + 1
        assert reg.loaded_count == 2
