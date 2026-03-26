"""Adaptive concurrency control for LLM throughput — AIMD congestion control.

Dynamically adjusts the number of concurrent in-flight mutations based on
observed LLM response latency.  Inspired by TCP congestion control:

* **Slow start**: begins at ``floor`` concurrent slots, calibrates a latency
  baseline from the first few requests.
* **Additive increase**: after each window of ``target`` completions, if the
  median latency is below threshold, add one slot.
* **Multiplicative decrease**: if median latency exceeds threshold, halve slots.

The result is automatic adaptation to shared GPU load — when other evolution
runs or external users are competing for the same vLLM servers, each run
independently backs off to the per-server sweet spot.

Usage::

    sema = AdaptiveSemaphore(ceiling=16, floor=2)
    await sema.acquire()
    t0 = time.monotonic()
    result = await llm_call(...)
    sema.report_latency(time.monotonic() - t0)
    sema.release()
"""

from __future__ import annotations

import asyncio
import statistics

from loguru import logger

# ---------------------------------------------------------------------------
# AIMD parameters (tuned from Qwen3-235B contention data)
# ---------------------------------------------------------------------------
# Measured on production server (INTERNAL_IP:8777):
#   N=1: 5.6s mean     N=2: 4.3s (batching)
#   N=4: 5.2s          N=8: 10.6s (saturated)
# The knee is at ~4 concurrent/server.  With latency target = 1.5x baseline,
# the controller detects saturation before throughput degrades.

_CALIBRATION_COUNT = 3  # requests before setting baseline
_TARGET_MULTIPLIER = 1.5  # latency target = baseline * this
_DECREASE_FACTOR = 0.5  # multiplicative decrease on congestion
_MIN_WINDOW = 2  # minimum samples before adjusting


class AdaptiveSemaphore:
    """AIMD-based adaptive concurrency limiter.

    Simple async semaphore with latency-based AIMD control. Updates to
    ``_target`` require reacquisition to take effect (no fast-path update
    for active waiters, but that's OK — AIMD changes slowly anyway).
    """

    def __init__(self, ceiling: int, floor: int = 2):
        if floor < 1:
            raise ValueError(f"floor must be >= 1, got {floor}")
        if ceiling < floor:
            raise ValueError(f"ceiling ({ceiling}) must be >= floor ({floor})")

        self._ceiling = ceiling
        self._floor = floor
        self._target = floor
        self._sem = asyncio.Semaphore(floor)

        # Calibration state
        self._latency_threshold: float | None = None
        self._calibration_samples: list[float] = []

        # AIMD window
        self._window: list[float] = []

        logger.info("[AdaptiveSema] Init: floor={}, ceiling={}", floor, ceiling)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def capacity(self) -> int:
        """Current target concurrency."""
        return self._target

    @property
    def ceiling(self) -> int:
        return self._ceiling

    @property
    def calibrated(self) -> bool:
        return self._latency_threshold is not None

    # ------------------------------------------------------------------
    # Acquire / Release
    # ------------------------------------------------------------------

    async def acquire(self) -> None:
        """Acquire one slot. Blocks if at capacity."""
        await self._sem.acquire()

    def release(self) -> None:
        """Release one slot."""
        self._sem.release()

    # ------------------------------------------------------------------
    # Latency reporting + AIMD
    # ------------------------------------------------------------------

    def report_latency(self, latency: float) -> None:
        """Report an observed LLM call latency for AIMD adjustment.

        Call after each successful LLM mutation completes.  Failed or
        cancelled calls should NOT be reported (they would corrupt the
        baseline or trigger false congestion signals).
        """
        # Phase 1: calibration
        if self._latency_threshold is None:
            self._calibration_samples.append(latency)
            if len(self._calibration_samples) >= _CALIBRATION_COUNT:
                baseline = statistics.median(self._calibration_samples)
                self._latency_threshold = baseline * _TARGET_MULTIPLIER
                logger.info(
                    "[AdaptiveSema] Calibrated: baseline={:.1f}s, "
                    "threshold={:.1f}s ({}x)",
                    baseline,
                    self._latency_threshold,
                    _TARGET_MULTIPLIER,
                )
            return

        # Phase 2: collect window
        self._window.append(latency)
        window_size = max(self._target, _MIN_WINDOW)
        if len(self._window) < window_size:
            return

        # Window complete — evaluate and adjust
        median_lat = statistics.median(self._window)
        self._window.clear()

        old_target = self._target
        if median_lat <= self._latency_threshold:
            # Additive increase
            new_target = min(self._ceiling, self._target + 1)
        else:
            # Multiplicative decrease
            new_target = max(self._floor, int(self._target * _DECREASE_FACTOR))

        if new_target != old_target:
            self._target = new_target
            # Update semaphore capacity for next acquire
            if new_target > old_target:
                # Add slots
                for _ in range(new_target - old_target):
                    self._sem.release()
            logger.info(
                "[AdaptiveSema] {} {} -> {} (median={:.1f}s, threshold={:.1f}s)",
                "INCREASE" if new_target > old_target else "DECREASE",
                old_target,
                new_target,
                median_lat,
                self._latency_threshold,
            )
