from abc import ABC, abstractmethod
import random

from gigaevo.evolution.strategies.utils import dominates, extract_fitness_values
from gigaevo.programs.program import Program


class MigrantSelector(ABC):
    """Abstract base class for selecting programs to migrate."""

    @abstractmethod
    def __call__(self, programs: list[Program], count: int) -> list[Program]: ...


class RandomMigrantSelector(MigrantSelector):
    """Selects random programs."""

    def __call__(self, programs: list[Program], count: int) -> list[Program]:
        if len(programs) <= count:
            return programs
        else:
            return random.sample(programs, count)


class TopFitnessMigrantSelector(MigrantSelector):
    """Selects top programs by scalar fitness."""

    def __init__(self, fitness_key: str, fitness_key_higher_is_better: bool = True):
        self.fitness_key = fitness_key
        self.fitness_key_higher_is_better = fitness_key_higher_is_better

    def __call__(self, programs: list[Program], count: int) -> list[Program]:
        if not programs:
            return []

        fitness_values = [
            extract_fitness_values(
                program,
                [self.fitness_key],
                {self.fitness_key: self.fitness_key_higher_is_better},
            )
            for program in programs
        ]
        scored_programs = [
            (prog, fitness_values[i])
            for i, prog in enumerate(programs)
            if fitness_values[i] is not None
        ]

        if not scored_programs:
            return random.sample(programs, min(count, len(programs)))

        sorted_programs = sorted(scored_programs, key=lambda x: x[1], reverse=True)
        return [p for p, _ in sorted_programs[:count]]


class ParetoFrontMigrantSelector(MigrantSelector):
    """Selects from the Pareto front (non-dominated set)."""

    def __init__(
        self,
        fitness_keys: list[str],
        fitness_key_higher_is_better: dict[str, bool] | None = None,
    ):
        self.fitness_keys = fitness_keys
        self.fitness_key_higher_is_better = fitness_key_higher_is_better or {
            key: True for key in fitness_keys
        }

    def __call__(self, programs: list[Program], count: int) -> list[Program]:
        if not programs:
            return []

        pareto_front = self._compute_pareto_front(
            programs, self.fitness_keys, self.fitness_key_higher_is_better
        )

        if len(pareto_front) >= count:
            return random.sample(pareto_front, count)
        else:
            remaining = list(set(programs) - set(pareto_front))
            filler = random.sample(
                remaining, min(count - len(pareto_front), len(remaining))
            )
            return pareto_front + filler

    def _compute_pareto_front(
        self,
        programs: list[Program],
        fitness_keys: list[str],
        fitness_key_higher_is_better: dict[str, bool],
    ) -> list[Program]:
        pareto_front = []

        for p in programs:
            if all(
                not dominates(
                    extract_fitness_values(
                        other, fitness_keys, fitness_key_higher_is_better
                    ),
                    extract_fitness_values(
                        p, fitness_keys, fitness_key_higher_is_better
                    ),
                )
                for other in programs
                if other != p
            ):
                pareto_front.append(p)

        return pareto_front
