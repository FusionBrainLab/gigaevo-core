"""Progress-based synchronization hook for adversarial co-evolution.

ProgressBasedSyncHook blocks a population's engine until the drift between
this population and the opponent(s) falls within a configured cap. Unlike
MainRunSyncHook (which waits for generation boundaries), this hook operates
on continuous program-count progress — designed for SteadyStateEvolutionEngine
where "generation" is an epoch boundary.

The wait condition is asymmetric by design (drift-cap semantics):

    while own_progress - min(opponent_progress) > drift_cap:
        sleep(poll_interval)

Only the *ahead* population ever blocks; the *behind* population's wait
condition reduces to a negative number and is always satisfied, so it
proceeds freely. Deadlock is impossible by construction: there is no state
in which both sides can be waiting for each other simultaneously.

This replaces the prior symmetric "opponent advanced by min_delta since my
last sync" semantics, which were observed to deadlock in k5-budget-v3's
K3_1 pair (KF-07 in experiments/PATTERNS.md) when both populations landed
in the pre-step hook at the same time.

Supports asymmetric update ratios via ``sync_every_n_epochs``: set to K for
K:1 updates (this population runs K epochs per sync). Inspired by GAN
training routines (critic iterations > generator).
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
import warnings

from loguru import logger
from redis import asyncio as aioredis


class ProgressBasedSyncHook:
    """Pre-step hook enforcing a drift cap between this population and opponent(s).

    Polls both this run's and each opponent run's ``engine:programs_processed``
    counter in Redis and waits only while
    ``own_progress - min(opponent_progress) > drift_cap``.

    Supports K:1 asymmetric update ratios via ``sync_every_n_epochs``: when set
    to K > 1, the hook is a no-op for K-1 out of every K calls.

    Args:
        host: Redis host.
        port: Redis port.
        sources: List of {"db": int, "prefix": str} — opponent run(s).
            Must be non-empty.
        own_db: Redis DB of this population's own run. Required.
        own_prefix: Redis key prefix of this population's own run. Required.
        drift_cap: Maximum number of programs this population is allowed to
            lead the opponent by before blocking. Must be a non-negative int.
            Required — pass explicitly per experiment (no silent default).
        min_delta: Deprecated alias for ``drift_cap``. If ``drift_cap`` is
            ``None`` and ``min_delta`` is provided, ``min_delta`` is used and
            a DeprecationWarning is emitted.
        sync_every_n_epochs: Only sync every N epochs (default: 1). Set to K
            for K:1 asymmetric updates.
        timeout: Last-resort safety net — maximum seconds to wait before
            proceeding anyway (default: 7200). Under drift-cap semantics this
            should almost never fire; kept in case of Redis outage or similar.
        poll_interval: Seconds between polls (default: 5.0).
    """

    def __init__(
        self,
        host: str,
        port: int,
        sources: list[dict[str, int | str]],
        own_db: int,
        own_prefix: str,
        drift_cap: int | None = None,
        min_delta: int | None = None,
        sync_every_n_epochs: int = 1,
        timeout: float = 7200.0,
        poll_interval: float = 5.0,
        **kwargs: Any,  # absorb extra keys from Hydra config inheritance
    ):
        if not sources:
            raise ValueError("ProgressBasedSyncHook requires at least one source")

        if drift_cap is None and min_delta is None:
            raise ValueError(
                "ProgressBasedSyncHook requires `drift_cap` (or deprecated "
                "alias `min_delta`) to be set explicitly. Example: drift_cap=8 "
                "means this population may lead the opponent by at most 8 "
                "programs before blocking."
            )
        if drift_cap is not None and min_delta is not None:
            logger.warning(
                "[ProgressBasedSyncHook] Both `drift_cap` ({}) and `min_delta` "
                "({}) were passed; using `drift_cap` and ignoring `min_delta`.",
                drift_cap,
                min_delta,
            )
        if drift_cap is None:
            warnings.warn(
                "`min_delta` is a deprecated alias; pass `drift_cap` instead. "
                "The semantic has changed from 'opponent must advance by N "
                "since my last sync' (symmetric, deadlock-prone) to 'I must "
                "not be more than N ahead of opponent' (asymmetric, "
                "deadlock-free). The old name is kept as a compatibility "
                "shim for existing configs.",
                DeprecationWarning,
                stacklevel=2,
            )
            drift_cap = min_delta  # type: ignore[assignment]
        assert drift_cap is not None  # for mypy
        if drift_cap < 0:
            raise ValueError(f"drift_cap must be >= 0, got {drift_cap}")

        self._host = host
        self._port = port
        self._sources = [(int(s["db"]), str(s["prefix"])) for s in sources]
        self._own_db = int(own_db)
        self._own_prefix = str(own_prefix)
        self._drift_cap = drift_cap
        self._sync_every_n_epochs = max(1, sync_every_n_epochs)
        self._timeout = timeout
        self._poll_interval = poll_interval
        self._epoch_call_count: int = 0
        self._first_call: bool = True
        self._redis_clients: dict[int, aioredis.Redis] = {}  # type: ignore[type-arg]

        sources_desc = ", ".join(f"db={db} prefix={pfx!r}" for db, pfx in self._sources)
        logger.info(
            "[ProgressBasedSyncHook] Init | own=db={} prefix={!r} "
            "opponents=[{}] drift_cap={} sync_every={}epochs timeout={}s poll={}s",
            self._own_db,
            self._own_prefix,
            sources_desc,
            self._drift_cap,
            self._sync_every_n_epochs,
            self._timeout,
            self._poll_interval,
        )

    def _get_redis(self, db: int) -> aioredis.Redis:  # type: ignore[type-arg]
        if db not in self._redis_clients:
            self._redis_clients[db] = aioredis.Redis(
                host=self._host,
                port=self._port,
                db=db,
                decode_responses=True,
            )
        return self._redis_clients[db]

    async def _read_progress(self, db: int, prefix: str) -> int:
        """Read ``engine:programs_processed`` for a single run. Missing -> 0."""
        try:
            r = self._get_redis(db)
            raw = await r.hget(f"{prefix}:run_state", "engine:programs_processed")
            return int(raw) if raw else 0
        except Exception as exc:
            logger.warning(
                "[ProgressBasedSyncHook] Error reading progress from db={} prefix={!r}: {}",
                db,
                prefix,
                exc,
            )
            return 0

    async def _get_min_opponent_progress(self) -> int:
        """Minimum ``programs_processed`` across all tracked opponent runs."""
        values = [await self._read_progress(db, prefix) for db, prefix in self._sources]
        return min(values) if values else 0

    async def _get_own_progress(self) -> int:
        """This population's own ``programs_processed`` at the latest epoch boundary."""
        return await self._read_progress(self._own_db, self._own_prefix)

    async def __call__(self) -> None:
        """Block until drift between this population and opponent(s) is within cap.

        Respects ``sync_every_n_epochs``: skips K-1 out of every K calls.

        On the first call, logs a baseline reading and returns immediately
        (drift check is skipped for the first hook call to mirror the prior
        "baseline, don't block" contract).
        """
        # Asymmetric ratio: skip this epoch if not a sync epoch
        self._epoch_call_count += 1
        if self._epoch_call_count < self._sync_every_n_epochs:
            return
        self._epoch_call_count = 0

        # First call: log baseline and skip drift check (contract preserved from
        # prior semantics so engine startup is unaffected).
        if self._first_call:
            self._first_call = False
            own_p = await self._get_own_progress()
            opp_p = await self._get_min_opponent_progress()
            logger.info(
                "[ProgressBasedSyncHook] Baseline | own={} opp_min={} (drift={}, cap={})",
                own_p,
                opp_p,
                own_p - opp_p,
                self._drift_cap,
            )
            return

        start = time.monotonic()
        last_progress_log = start

        while True:
            own_progress = await self._get_own_progress()
            opp_progress = await self._get_min_opponent_progress()
            drift = own_progress - opp_progress

            # Drift-cap condition: block only while we are MORE than cap ahead.
            # When drift <= cap (including the behind-side case drift < 0), proceed.
            if drift <= self._drift_cap:
                elapsed = time.monotonic() - start
                if elapsed > self._poll_interval:
                    logger.info(
                        "[ProgressBasedSyncHook] Drift within cap | own={} opp_min={} "
                        "drift={} cap={} (waited {:.1f}s)",
                        own_progress,
                        opp_progress,
                        drift,
                        self._drift_cap,
                        elapsed,
                    )
                return

            elapsed = time.monotonic() - start
            if elapsed > self._timeout:
                logger.warning(
                    "[ProgressBasedSyncHook] Timeout after {:.0f}s with drift={} "
                    "> cap={} (own={}, opp_min={}). Proceeding anyway — this "
                    "should be rare under drift-cap semantics and likely "
                    "indicates Redis or opponent-run failure, not mutual wait.",
                    elapsed,
                    drift,
                    self._drift_cap,
                    own_progress,
                    opp_progress,
                )
                return

            now = time.monotonic()
            if (now - last_progress_log) >= 60.0:
                logger.info(
                    "[ProgressBasedSyncHook] Leading opponent by {} (cap={}) "
                    "— waited {:.0f}s | own={} opp_min={}",
                    drift,
                    self._drift_cap,
                    elapsed,
                    own_progress,
                    opp_progress,
                )
                last_progress_log = now

            await asyncio.sleep(self._poll_interval)
