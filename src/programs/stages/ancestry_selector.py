from __future__ import annotations

import random
from typing import Literal

from src.database.program_storage import ProgramStorage
from src.programs.metrics.context import MetricsContext
from src.programs.program import Program

AncestrySelectionStrategy = Literal["random", "best_fitness"]


class AncestrySelector:
    """
    Selects IDs for lineage analysis based on a strategy and max_selected.
    Strategies:
      - "random": random sample up to N
      - "best_fitness": rank by primary metric (direction-aware), take top N
    """

    def __init__(
        self,
        metrics_context: MetricsContext,
        strategy: AncestrySelectionStrategy = "random",
        max_selected: int = 1,
    ) -> None:
        self.metrics_context = metrics_context
        self.strategy = strategy
        self.max_selected = max(1, int(max_selected))

    async def select(self, programs: list[Program]) -> list[Program]:
        if self.strategy == "best_fitness":
            fitness_key = self.metrics_context.get_primary_key()
            higher_better = bool(
                self.metrics_context.get_primary_spec().higher_is_better
            )
            scored: list[tuple[float, str]] = []
            for program in programs:
                if fitness_key not in program.metrics:
                    raise ValueError(
                        f"Missing fitness key '{fitness_key}' in program {program.id}"
                    )
                val = program.metrics[fitness_key]
                scored.append((val, program))

            scored.sort(key=lambda t: t[0], reverse=higher_better)
            return [program for _, program in scored[: self.max_selected]]

        elif self.strategy == "random":
            return random.sample(programs, min(self.max_selected, len(programs)))
        else:
            raise ValueError(f"Unknown program selection strategy: {self.strategy}")
