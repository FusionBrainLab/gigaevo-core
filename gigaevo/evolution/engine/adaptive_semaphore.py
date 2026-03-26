"""Adaptive concurrency control for LLM throughput.

Starts at full capacity (``ceiling``) and backs off when GPU contention is
detected via rising LLM latency.  Recovers gradually when conditions improve.

Strategy: **full-start, back-off** (like TCP BBR, not Reno).

* **Calibration**: first ``_CALIBRATION_COUNT`` requests establish a latency
  baseline.  Threshold = baseline * ``_TARGET_MULTIPLIER``.
* **Multiplicative decrease**: if window median latency > threshold, halve.
* **Additive increase**: if window median <= threshold, add one slot.

Measured on Qwen3-235B (INTERNAL_IP:8777):
  N=1: 5.6s   N=2: 4.3s   N=4: 5.2s   N=8: 10.6s (saturated)
Sweet spot = 4 concurrent/server.  1.5x threshold catches the knee.
"""

from __future__ import annotations

import asyncio
import statistics

from loguru import logger

_CALIBRATION_COUNT = 3
_TARGET_MULTIPLIER = 1.5
_DECREASE_FACTOR = 0.5
_MIN_WINDOW = 2


class AdaptiveSemaphore:
    """Full-start, back-off adaptive concurrency limiter.

    Tracks ``_active`` count to correctly handle capacity decreases:
    on release, a permit is only returned to the pool if ``_active``
    is below ``_target``.  This naturally drains excess capacity after
    a multiplicative decrease without corrupting the semaphore state.
    """

    def __init__(self, ceiling: int, floor: int = 2):
        if floor < 1:
            raise ValueError(f"floor must be >= 1, got {floor}")
        if ceiling < floor:
            raise ValueError(f"ceiling ({ceiling}) must be >= floor ({floor})")

        self._ceiling = ceiling
        self._floor = floor
        self._target = ceiling  # start at full capacity
        self._active = 0  # currently held permits
        self._sem = asyncio.Semaphore(ceiling)

        # Calibration
        self._latency_threshold: float | None = None
        self._calibration_samples: list[float] = []

        # AIMD window
        self._window: list[float] = []

        logger.info("[AdaptiveSema] Init: ceiling={}, floor={}", ceiling, floor)

    @property
    def capacity(self) -> int:
        return self._target

    @property
    def ceiling(self) -> int:
        return self._ceiling

    @property
    def active(self) -> int:
        return self._active

    @property
    def calibrated(self) -> bool:
        return self._latency_threshold is not None

    # ------------------------------------------------------------------
    # Acquire / Release
    # ------------------------------------------------------------------

    async def acquire(self) -> None:
        """Acquire one concurrency slot.  Blocks when at capacity."""
        await self._sem.acquire()
        self._active += 1

    def release(self) -> None:
        """Release one concurrency slot.

        Only returns the permit to the underlying semaphore if current
        active count is below the target.  When the target has been
        decreased, excess permits are absorbed here (not returned),
        so the effective capacity converges to the new target.
        """
        self._active -= 1
        if self._active < self._target:
            self._sem.release()
        # else: absorb the permit — effective capacity shrinks by 1

    # ------------------------------------------------------------------
    # Latency reporting + AIMD
    # ------------------------------------------------------------------

    def report_latency(self, latency: float) -> None:
        """Report LLM call latency for AIMD adjustment.

        Call after each successful mutation.  Do NOT report failures.
        """
        # Phase 1: calibration
        if self._latency_threshold is None:
            self._calibration_samples.append(latency)
            if len(self._calibration_samples) >= _CALIBRATION_COUNT:
                baseline = statistics.median(self._calibration_samples)
                # Small epsilon floor prevents oscillation when baseline ≈ 0
                # (pure mocks with no sleep).  Any real LLM call (even with
                # TIMESCALE compression) produces latencies >> 0.001s.
                self._latency_threshold = max(0.001, baseline * _TARGET_MULTIPLIER)
                logger.info(
                    "[AdaptiveSema] Calibrated: baseline={:.1f}s, "
                    "threshold={:.1f}s ({}x, min=1.0s)",
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

        # Window complete — evaluate
        median_lat = statistics.median(self._window)
        self._window.clear()

        old_target = self._target
        if median_lat <= self._latency_threshold:
            new_target = min(self._ceiling, self._target + 1)
        else:
            new_target = max(self._floor, int(self._target * _DECREASE_FACTOR))

        if new_target == old_target:
            return

        self._target = new_target

        if new_target > old_target:
            # Increase: release extra permits so blocked acquirers wake up.
            # Only release what's actually needed (don't over-allocate).
            headroom = new_target - self._active
            available = self._sem._value  # noqa: SLF001
            to_release = max(0, headroom - available)
            for _ in range(to_release):
                self._sem.release()

        # Decrease: no action needed — release() will absorb excess permits
        # naturally as active tasks complete.

        logger.info(
            "[AdaptiveSema] {} {} -> {} (median={:.1f}s, threshold={:.1f}s, active={})",
            "INCREASE" if new_target > old_target else "DECREASE",
            old_target,
            new_target,
            median_lat,
            self._latency_threshold,
            self._active,
        )
