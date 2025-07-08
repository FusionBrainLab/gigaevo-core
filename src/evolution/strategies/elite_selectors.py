from abc import ABC, abstractmethod
from typing import Protocol, List, Callable, Optional
import random
from loguru import logger
from src.programs.program import Program
from src.evolution.strategies.utils import extract_fitness_values, dominates

class EliteSelectorProtocol(Protocol):
    def __call__(self, programs: List[Program], total: int) -> List[Program]:
        pass

class EliteSelector(ABC):
    @abstractmethod
    def __call__(self, programs: List[Program], total: int) -> List[Program]:
        pass

class RandomEliteSelector(EliteSelector):
    def __call__(self, programs: List[Program], total: int) -> List[Program]:
        if len(programs) <= total:
            logger.warning(f"Only {len(programs)} elites available, requested {total}. Returning all available elites without duplicates.")
            return programs
        return random.sample(programs, total)


class FitnessProportionalEliteSelector(EliteSelector):
    def __init__(self, fitness_key: str, fitness_key_higher_is_better: bool = True):
        self.fitness_key = fitness_key
        self.higher_is_better = fitness_key_higher_is_better

    def __call__(self, programs: List[Program], total: int) -> List[Program]:
        if len(programs) <= total:
            logger.warning(f"Only {len(programs)} elites available, requested {total}. Returning all available elites without duplicates.")
            return programs

        fitnesses = []
        for p in programs:
            if self.fitness_key not in p.metrics:
                raise ValueError(f"Missing fitness key '{self.fitness_key}' in program {p.id}")
            val = p.metrics[self.fitness_key]
            fitnesses.append(val if self.higher_is_better else -val)

        min_fitness = min(fitnesses)
        if min_fitness < 0:
            fitnesses = [f - min_fitness + 1e-6 for f in fitnesses]  # shift to positive space

        selected = set()
        while len(selected) < total:
            chosen = random.choices(programs, weights=fitnesses, k=1)[0]
            selected.add(chosen)

        return list(selected)
    

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
        values = extract_fitness_values(program, [self.fitness_key], {self.fitness_key: self.higher_is_better})
        return values[0]

    def __call__(self, programs: List[Program], total: int) -> List[Program]:
        if len(programs) <= total:
            logger.warning(f"Only {len(programs)} programs available, requested {total}. Returning all.")
            return programs

        selected = set()
        attempts = 0
        while len(selected) < total and attempts < total * 10:
            candidates = random.sample(programs, min(self.tournament_size, len(programs)))
            ranked = [(p, -self._rank(p)) for p in candidates]
            ranked.sort(key=lambda x: x[1])
            winner = ranked[0][0]
            selected.add(winner)
            attempts += 1

        if len(selected) < total:
            logger.warning(f"Tournament selected only {len(selected)} of {total} requested elites.")
        return list(selected)
    
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
        self.higher_is_better = fitness_key_higher_is_better or {k: True for k in fitness_keys}
        self.tie_breaker = tie_breaker or (lambda p: p.created_at.timestamp())
        self.tournament_size = tournament_size

    def _pareto_rank(self, target: Program, population: List[Program]) -> int:
        vec = extract_fitness_values(target, self.fitness_keys, self.higher_is_better)
        return sum(
            1 for other in population
            if other is not target and dominates(
                extract_fitness_values(other, self.fitness_keys, self.higher_is_better),
                vec
            )
        )

    def __call__(self, programs: List[Program], total: int) -> List[Program]:
        if len(programs) <= total:
            logger.warning(f"Only {len(programs)} programs available, requested {total}. Returning all.")
            return programs

        selected = set()
        attempts = 0
        while len(selected) < total and attempts < total * 10:
            candidates = random.sample(programs, min(self.tournament_size, len(programs)))
            ranked = [
                (p, self._pareto_rank(p, candidates), self.tie_breaker(p)) for p in candidates
            ]
            ranked.sort(key=lambda x: (x[1], x[2]))  # by dominated count, then tie-breaker
            winner = ranked[0][0]
            selected.add(winner)
            attempts += 1

        if len(selected) < total:
            logger.warning(f"Tournament selected only {len(selected)} of {total} requested elites.")
        return list(selected)