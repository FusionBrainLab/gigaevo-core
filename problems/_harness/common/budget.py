"""Cumulative wall-clock budget shared by every representation.

Budget equality is the load-bearing fairness property of the whole comparison:
if one representation gets more wall time than another, no downstream statistic
means anything. So the budget is anchored to a monotonic clock at construction
and counts *everything* that happens afterwards — proposal time, validation
time, controller overhead — rather than being hand-charged per call, which can
silently under-count whatever the caller forgets to charge.
"""

from __future__ import annotations

from collections.abc import Callable
import time


class WallClockBudget:
    def __init__(
        self, total_s: float, clock: Callable[[], float] = time.monotonic
    ) -> None:
        if total_s < 0:
            raise ValueError(f"budget must be non-negative, got {total_s}")
        self._total_s = total_s
        self._clock = clock
        self._started_at = clock()

    @property
    def total_s(self) -> float:
        return self._total_s

    @property
    def elapsed_s(self) -> float:
        return self._clock() - self._started_at

    @property
    def remaining_s(self) -> float:
        return max(0.0, self._total_s - self.elapsed_s)

    def exhausted(self) -> bool:
        return self.remaining_s <= 0.0
