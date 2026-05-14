"""Pluggable stopping criteria for evolution engines.

Hydra config group: ``config/stopper/``.
Engine calls ``stopper.should_stop(ctx)`` once per dispatched mutant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class StopContext:
    total_mutants: int = 0
    elapsed_seconds: float = 0.0
    best_fitness: float | None = None
    programs_processed: int = 0


@dataclass(frozen=True)
class StopDecision:
    stop: bool
    reason: str


class EvolutionStopper:
    def should_stop(self, ctx: StopContext) -> StopDecision:
        return StopDecision(stop=False, reason="")


class MaxMutantsStopper(EvolutionStopper):
    def __init__(self, max_mutants: int) -> None:
        self.max_mutants = max_mutants

    def should_stop(self, ctx: StopContext) -> StopDecision:
        if ctx.total_mutants >= self.max_mutants:
            return StopDecision(
                stop=True,
                reason=f"Reached max_mutants={self.max_mutants}",
            )
        return StopDecision(stop=False, reason="")


class WallClockStopper(EvolutionStopper):
    def __init__(self, budget_seconds: float) -> None:
        self.budget_seconds = budget_seconds

    def should_stop(self, ctx: StopContext) -> StopDecision:
        if ctx.elapsed_seconds >= self.budget_seconds:
            return StopDecision(
                stop=True,
                reason=f"Wall clock budget exceeded: {ctx.elapsed_seconds:.0f}s >= {self.budget_seconds:.0f}s",
            )
        return StopDecision(stop=False, reason="")


class FitnessPlateauStopper(EvolutionStopper):
    # NOTE: best_fitness is wired into StopContext by
    # EvolutionEngine._build_stop_context() via MetricsTracker.get_best_fitness().
    # If the engine cannot determine a best-fitness value (no MetricsTracker or
    # no metric data yet), ctx.best_fitness is None and this stopper short-circuits.
    def __init__(self, window: int, min_delta: float = 0.001) -> None:
        self.window = window
        self.min_delta = min_delta
        self._best_seen: float | None = None
        self._stagnant_count: int = 0

    def should_stop(self, ctx: StopContext) -> StopDecision:
        if ctx.best_fitness is None:
            return StopDecision(stop=False, reason="")

        if (
            self._best_seen is None
            or (ctx.best_fitness - self._best_seen) >= self.min_delta
        ):
            self._best_seen = ctx.best_fitness
            self._stagnant_count = 0
        else:
            self._stagnant_count += 1

        if self._stagnant_count >= self.window:
            return StopDecision(
                stop=True,
                reason=f"Fitness plateau: no improvement >= {self.min_delta} for {self.window} mutants",
            )
        return StopDecision(stop=False, reason="")


class CompositeStopper(EvolutionStopper):
    def __init__(
        self,
        mode: Literal["any", "all"] = "any",
        children: list[EvolutionStopper] | None = None,
    ) -> None:
        self.mode = mode
        self.children: list[EvolutionStopper] = children or []

    def should_stop(self, ctx: StopContext) -> StopDecision:
        if not self.children:
            return StopDecision(stop=False, reason="")

        decisions = [c.should_stop(ctx) for c in self.children]
        triggered = [d for d in decisions if d.stop]

        if self.mode == "any" and triggered:
            reasons = "; ".join(d.reason for d in triggered)
            return StopDecision(stop=True, reason=reasons)

        if self.mode == "all" and len(triggered) == len(self.children):
            reasons = "; ".join(d.reason for d in triggered)
            return StopDecision(stop=True, reason=reasons)

        return StopDecision(stop=False, reason="")
