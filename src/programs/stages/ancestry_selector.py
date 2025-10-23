from __future__ import annotations

import random
from typing import List, Literal

from src.database.program_storage import ProgramStorage
from src.programs.metrics.context import MetricsContext

ParentSelectionStrategy = Literal["first", "random", "best_fitness"]


class AncestrySelector:
    """
    Selects parent IDs for lineage analysis based on a strategy and max_parents.
    Strategies:
      - "first": keep original order, take first N
      - "random": random sample up to N
      - "best_fitness": rank by primary metric (direction-aware), take top N
        * Missing/non-numeric metrics are coalesced via
          MetricsContext.get_sentinels()[metric_key].
    """

    def __init__(
        self,
        storage: ProgramStorage,
        metrics_context: MetricsContext,
        strategy: ParentSelectionStrategy = "first",
        max_parents: int = 1,
    ) -> None:
        self.storage = storage
        self.metrics_context = metrics_context
        self.strategy = strategy
        self.max_parents = max(1, int(max_parents))

    async def select(self, program) -> List[str]:
        if not program.lineage.parents:
            return []

        parents = list(program.lineage.parents)
        n = min(self.max_parents, len(parents))
        if n == 0:
            return []

        if self.strategy == "first":
            return parents[:n]

        if self.strategy == "random":
            return random.sample(parents, n) if len(parents) > n else parents

        if self.strategy == "best_fitness":
            fitness_key = self.metrics_context.get_primary_key()
            higher_better = bool(self.metrics_context.get_primary_spec().higher_is_better)
            worst_map = self.metrics_context.get_sentinels()
            worst = float(worst_map[fitness_key])

            scored: list[tuple[float, str]] = []
            for pid in parents:
                p = await self.storage.get(pid)
                val = p.metrics[fitness_key]
                scored.append((val, pid))

            scored.sort(key=lambda t: t[0], reverse=higher_better)
            return [pid for _, pid in scored[:n]]

        raise ValueError(f"Unknown parent selection strategy: {self.strategy}")