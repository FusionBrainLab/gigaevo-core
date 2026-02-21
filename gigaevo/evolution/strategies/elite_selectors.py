from abc import ABC, abstractmethod
import math
import random
import statistics
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
    """Fitness-proportional sampling with optional Boltzmann temperature control.

    When ``temperature`` is ``None`` (default), weights are the raw fitness
    values (shifted to be non-negative). This is the classic roulette-wheel
    behaviour where selection probability is directly proportional to fitness.

    When ``temperature`` is set, a Boltzmann (softmax) transform is applied:
        w_i = exp(f_i / temperature)

    * **High temperature** → flattens fitness differences, approaching uniform
      sampling. Useful for exploration: minor fitness shifts that might indicate
      radical new ideas still have a fair chance of being selected.
    * **Low temperature** → amplifies fitness differences, approaching greedy
      selection. Useful for exploitation.
    * ``temperature`` is a single float easily tunable by an optimization agent.
    """

    def __init__(
        self,
        fitness_key: str,
        fitness_key_higher_is_better: bool = True,
        temperature: float | None = None,
    ):
        self.fitness_key = fitness_key
        self.higher_is_better = fitness_key_higher_is_better
        self.temperature = temperature

    def _compute_weights(self, fitnesses: list[float]) -> list[float]:
        """Convert raw fitnesses into sampling weights."""
        if self.temperature is not None:
            # Boltzmann / softmax weighting
            # Subtract max for numerical stability (doesn't change the distribution)
            max_f = max(fitnesses)
            weights = []
            for f in fitnesses:
                exp_arg = (f - max_f) / self.temperature
                clamped = max(-500.0, min(500.0, exp_arg))
                weights.append(math.exp(clamped))
            return weights

        # Default: linear proportional (shift to non-negative)
        min_f = min(fitnesses)
        if min_f < 0:
            logger.debug(
                "FitnessProportionalEliteSelector: shifted fitnesses to positive space"
            )
            return [f - min_f + 1e-6 for f in fitnesses]
        return list(fitnesses)

    def __call__(self, programs: list[Program], total: int) -> list[Program]:
        logger.debug(
            "FitnessProportionalEliteSelector: selecting {} from {} programs "
            "(key='{}', higher_is_better={}, temperature={})",
            total,
            len(programs),
            self.fitness_key,
            self.higher_is_better,
            self.temperature,
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

        weights = self._compute_weights(fitnesses)

        # Sampling without replacement
        selected = []
        remaining_programs = list(programs)
        remaining_weights = list(weights)

        for _ in range(min(total, len(programs))):
            if not remaining_programs:
                break

            total_weight = sum(remaining_weights)
            if total_weight == 0:
                logger.warning(
                    "FitnessProportionalEliteSelector: all remaining weights are zero; "
                    "falling back to uniform sampling for the rest "
                    "(remaining={}, already_selected={}, requested_total={})",
                    len(remaining_programs),
                    len(selected),
                    total,
                )
                selected.extend(
                    random.sample(remaining_programs, total - len(selected))
                )
                break

            # Select one program based on fitness weights
            chosen = random.choices(remaining_programs, weights=remaining_weights, k=1)[
                0
            ]
            selected.append(chosen)

            # Remove selected program and its fitness from remaining pools
            idx = remaining_programs.index(chosen)
            remaining_programs.pop(idx)
            remaining_weights.pop(idx)

        logger.debug(
            "FitnessProportionalEliteSelector: selected {} programs",
            len(selected),
        )
        return selected


class WeightedEliteSelector(EliteSelector):
    """ShinkaEvolve-inspired weighted sampling combining sigmoid-scaled fitness
    with a children-count novelty penalty.

    Weight for program i:
        s_i = sigmoid(lambda_ * (F(P_i) - median(F)))
        h_i = 1 / (1 + child_count_i)
        w_i = max(s_i * h_i, epsilon)
    """

    def __init__(
        self,
        fitness_key: str,
        fitness_key_higher_is_better: bool = True,
        lambda_: float = 10.0,
        epsilon: float = 1e-8,
    ):
        self.fitness_key = fitness_key
        self.higher_is_better = fitness_key_higher_is_better
        self.lambda_ = lambda_
        self.epsilon = epsilon

    def _sigmoid(self, x: float) -> float:
        clamped = max(-500.0, min(500.0, x))
        return 1.0 / (1.0 + math.exp(-clamped))

    def __call__(self, programs: list[Program], total: int) -> list[Program]:
        logger.debug(
            "WeightedEliteSelector: selecting {} from {} programs (key='{}', higher_is_better={}, lambda={}, epsilon={})",
            total,
            len(programs),
            self.fitness_key,
            self.higher_is_better,
            self.lambda_,
            self.epsilon,
        )

        if len(programs) <= total:
            logger.debug(
                "WeightedEliteSelector: returning all {} programs (≤ requested {})",
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

        median_f = statistics.median(fitnesses)

        weights = []
        for p, f in zip(programs, fitnesses):
            s_i = self._sigmoid(self.lambda_ * (f - median_f))
            h_i = 1.0 / (1.0 + p.lineage.child_count)
            w_i = max(s_i * h_i, self.epsilon)
            weights.append(w_i)

        # Sample without replacement
        selected: list[Program] = []
        remaining_programs = list(programs)
        remaining_weights = list(weights)

        for _ in range(min(total, len(programs))):
            if not remaining_programs:
                break

            total_weight = sum(remaining_weights)
            if total_weight == 0:
                logger.warning(
                    "WeightedEliteSelector: all remaining weights are zero; "
                    "falling back to uniform sampling "
                    "(remaining={}, already_selected={}, requested_total={})",
                    len(remaining_programs),
                    len(selected),
                    total,
                )
                selected.extend(
                    random.sample(remaining_programs, total - len(selected))
                )
                break

            chosen = random.choices(remaining_programs, weights=remaining_weights, k=1)[
                0
            ]
            selected.append(chosen)

            idx = remaining_programs.index(chosen)
            remaining_programs.pop(idx)
            remaining_weights.pop(idx)

        logger.debug(
            "WeightedEliteSelector: selected {} programs",
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
