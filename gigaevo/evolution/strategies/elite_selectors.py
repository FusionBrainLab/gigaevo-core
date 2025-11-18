from abc import ABC, abstractmethod
import random
from typing import Callable, List, Optional, Protocol

from loguru import logger

from gigaevo.evolution.strategies.utils import dominates, extract_fitness_values
from gigaevo.programs.program import Program


class EliteSelectorProtocol(Protocol):
    def __call__(self, programs: list[Program], total: int) -> list[Program]:
        pass


class EliteSelector(ABC):
    @abstractmethod
    def __call__(self, programs: list[Program], total: int) -> list[Program]:
        pass


class RandomEliteSelector(EliteSelector):
    def __call__(self, programs: list[Program], total: int) -> list[Program]:
        logger.debug(
            "RandomEliteSelector: selecting {} from {} programs",
            total,
            len(programs),
        )

        if len(programs) <= total:
            logger.debug(
                "RandomEliteSelector: returning all {} programs (≤ requested {})",
                len(programs),
                total,
            )
            return programs

        selected = random.sample(programs, total)
        logger.debug(
            "RandomEliteSelector: selected {} programs randomly",
            len(selected),
        )
        return selected


class FitnessProportionalEliteSelector(EliteSelector):
    def __init__(self, fitness_key: str, fitness_key_higher_is_better: bool = True):
        self.fitness_key = fitness_key
        self.higher_is_better = fitness_key_higher_is_better

    def __call__(self, programs: list[Program], total: int) -> list[Program]:
        logger.debug(
            "FitnessProportionalEliteSelector: selecting {} from {} programs (key='{}', higher_is_better={})",
            total,
            len(programs),
            self.fitness_key,
            self.higher_is_better,
        )

        if len(programs) <= total:
            logger.debug(
                "FitnessProportionalEliteSelector: returning all {} programs (≤ requested {})",
                len(programs),
                total,
            )
            return programs

        fitnesses = []
        for p in programs:
            if self.fitness_key not in p.metrics:
                raise ValueError(
                    f"Missing fitness key '{self.fitness_key}' in program {p.id}"
                )
            val = p.metrics[self.fitness_key]
            fitnesses.append(val if self.higher_is_better else -val)

        min_fitness = min(fitnesses)
        max_fitness = max(fitnesses)
        logger.debug(
            "FitnessProportionalEliteSelector: fitness range [{:.3f}, {:.3f}]",
            min_fitness,
            max_fitness,
        )

        if min_fitness < 0:
            fitnesses = [
                f - min_fitness + 1e-6 for f in fitnesses
            ]  # shift to positive space
            logger.debug(
                "FitnessProportionalEliteSelector: shifted fitnesses to positive space"
            )

        # FIXED: Proper sampling without replacement using numpy-style approach
        selected = []
        remaining_programs = list(programs)
        remaining_fitnesses = list(fitnesses)

        for _ in range(min(total, len(programs))):
            if not remaining_programs:
                break

            # Select one program based on fitness weights
            chosen = random.choices(
                remaining_programs, weights=remaining_fitnesses, k=1
            )[0]
            selected.append(chosen)

            # Remove selected program and its fitness from remaining pools
            idx = remaining_programs.index(chosen)
            remaining_programs.pop(idx)
            remaining_fitnesses.pop(idx)

        logger.debug(
            "FitnessProportionalEliteSelector: selected {} programs",
            len(selected),
        )
        return selected


class ScalarTournamentEliteSelector(EliteSelector):
    def __init__(
        self,
        fitness_key: str,
        fitness_key_higher_is_better: bool = True,
        tournament_size: int = 3,
    ):
        self.fitness_key = fitness_key
        self.higher_is_better = fitness_key_higher_is_better
        self.tournament_size = tournament_size

    def _rank(self, program: Program) -> float:
        values = extract_fitness_values(
            program,
            [self.fitness_key],
            {self.fitness_key: self.higher_is_better},
        )
        return values[0]

    def __call__(self, programs: list[Program], total: int) -> list[Program]:
        if len(programs) <= total:
            logger.warning(
                f"Only {len(programs)} programs available, requested {total}. Returning all."
            )
            return programs

        # FIXED: Proper sampling without replacement
        selected = []
        remaining_programs = list(programs)

        while len(selected) < total and remaining_programs:
            candidates = random.sample(
                remaining_programs,
                min(self.tournament_size, len(remaining_programs)),
            )
            ranked = [(p, -self._rank(p)) for p in candidates]
            ranked.sort(key=lambda x: x[1])
            winner = ranked[0][0]
            selected.append(winner)

            # Remove winner from remaining programs
            remaining_programs.remove(winner)

        return selected


class ParetoTournamentEliteSelector(EliteSelector):
    def __init__(
        self,
        fitness_keys: List[str],
        fitness_key_higher_is_better: Optional[dict[str, bool]] = None,
        tie_breaker: Optional[Callable[[Program], float]] = None,
        tournament_size: int = 3,
    ):
        if not fitness_keys or len(fitness_keys) < 2:
            raise ValueError("ParetoTournament requires at least two fitness keys.")

        self.fitness_keys = fitness_keys
        self.higher_is_better = fitness_key_higher_is_better or {
            k: True for k in fitness_keys
        }
        self.tie_breaker = tie_breaker or (lambda p: p.created_at.timestamp())
        self.tournament_size = tournament_size

    def _pareto_rank(self, target: Program, population: List[Program]) -> int:
        vec = extract_fitness_values(target, self.fitness_keys, self.higher_is_better)
        return sum(
            1
            for other in population
            if other is not target
            and dominates(
                extract_fitness_values(other, self.fitness_keys, self.higher_is_better),
                vec,
            )
        )

    def __call__(self, programs: List[Program], total: int) -> List[Program]:
        if len(programs) <= total:
            logger.warning(
                f"Only {len(programs)} programs available, requested {total}. Returning all."
            )
            return programs

        # FIXED: Proper sampling without replacement
        selected = []
        remaining_programs = list(programs)

        while len(selected) < total and remaining_programs:
            candidates = random.sample(
                remaining_programs,
                min(self.tournament_size, len(remaining_programs)),
            )
            ranked = [
                (p, self._pareto_rank(p, candidates), self.tie_breaker(p))
                for p in candidates
            ]
            ranked.sort(
                key=lambda x: (x[1], x[2])
            )  # by dominated count, then tie-breaker
            winner = ranked[0][0]
            selected.append(winner)

            # Remove winner from remaining programs
            remaining_programs.remove(winner)

        return selected
