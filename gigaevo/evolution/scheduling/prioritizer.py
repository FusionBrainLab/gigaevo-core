"""Program prioritizers for DAG evaluation scheduling.

A ``ProgramPrioritizer`` controls the order in which queued programs
are launched for DAG evaluation.  The DagRunner calls ``prioritize()``
on its batch of candidate programs before creating async tasks.

The default (FIFO) preserves insertion order.  LPT and SJF use an
:class:`~gigaevo.evolution.scheduling.predictor.EvalTimePredictor`
to reorder programs by predicted evaluation time.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from gigaevo.evolution.scheduling.predictor import EvalTimePredictor

if TYPE_CHECKING:
    from gigaevo.programs.program import Program


class ProgramPrioritizer(ABC):
    """Control program launch order in the DagRunner.

    ``prioritize()`` receives a list of candidate programs (already
    fetched from Redis) and returns them in the desired launch order.
    The DagRunner creates tasks in this order, and the semaphore
    controls actual concurrency.
    """

    @abstractmethod
    def prioritize(self, programs: list[Program]) -> list[Program]:
        """Return *programs* in desired launch order.

        First element is launched first.  The input list is NOT modified.
        """

    @property
    def predictor(self) -> EvalTimePredictor | None:
        """Return the underlying predictor, if any (for online learning)."""
        return None


class FIFOPrioritizer(ProgramPrioritizer):
    """Preserve insertion order (default behavior)."""

    def prioritize(self, programs: list[Program]) -> list[Program]:
        return list(programs)


class LPTPrioritizer(ProgramPrioritizer):
    """Longest Processing Time first.

    Start predicted-longest programs first so they finish closer to
    predicted-shorter ones, minimizing tail idle time.  Falls back to
    FIFO when the predictor is cold.

    Optimal for minimizing makespan on parallel identical machines
    (Graham, 1969).
    """

    def __init__(self, eval_predictor: EvalTimePredictor) -> None:
        self._predictor = eval_predictor

    def prioritize(self, programs: list[Program]) -> list[Program]:
        if not programs:
            return []
        if not self._predictor.is_warm():
            return list(programs)
        return sorted(
            programs,
            key=lambda p: self._predictor.predict(p),
            reverse=True,
        )

    @property
    def predictor(self) -> EvalTimePredictor:
        return self._predictor


class SJFPrioritizer(ProgramPrioritizer):
    """Shortest Job First — for comparison benchmarking.

    Opposite of LPT.  Expected to perform worse for makespan but
    better for average latency.
    """

    def __init__(self, eval_predictor: EvalTimePredictor) -> None:
        self._predictor = eval_predictor

    def prioritize(self, programs: list[Program]) -> list[Program]:
        if not programs:
            return []
        if not self._predictor.is_warm():
            return list(programs)
        return sorted(
            programs,
            key=lambda p: self._predictor.predict(p),
        )

    @property
    def predictor(self) -> EvalTimePredictor:
        return self._predictor


class CachedFirstPrioritizer(ProgramPrioritizer):
    """Re-evaluations before fresh mutants.

    A program with non-empty ``stage_results`` has already been DAG-evaluated
    once, so on re-eval most of its stages will hit ``cached_skip`` and finish
    in milliseconds. Surfacing those to the front of the launch queue directly
    unblocks producer tasks that are pinned on
    :meth:`ParentRefresher._await_done` (each pinned task holds an in-flight
    slot, so when ``N`` mutants × ``M``-second refresh queues collide,
    throughput collapses even though per-DAG exec is near-zero).

    Within each tier (cached, fresh) the input order is preserved — Redis
    SMEMBERS hash order, which the runner uses upstream, has no meaningful
    semantics, so we don't attempt to re-sort.

    No predictor — the cache signal lives on the program itself, no online
    learning required. The class is therefore safe to swap in as default and
    drop later if a higher-fidelity predictor-backed strategy supersedes it.
    """

    def prioritize(self, programs: list[Program]) -> list[Program]:
        if not programs:
            return []
        cached: list[Program] = []
        fresh: list[Program] = []
        for p in programs:
            if p.stage_results:
                cached.append(p)
            else:
                fresh.append(p)
        return cached + fresh
