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
