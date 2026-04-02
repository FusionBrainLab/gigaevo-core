from abc import ABC, abstractmethod
from collections.abc import Callable
import random
from typing import Protocol

from loguru import logger
import numpy as np
from scipy.special import expit, softmax

from gigaevo.evolution.strategies.utils import (
    dominates,
    extract_fitness_values,
    weighted_sample_without_replacement,
)
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
    """Softmax (Boltzmann) fitness-proportional sampling.

    Fitnesses are always normalized to [0, 1] before applying softmax,
    making the selector fully scale- and shift-invariant regardless of
    the problem's fitness range.

    When ``temperature`` is ``None`` (default), it is auto-computed as
    ``max(std(normalized_fitnesses), 0.01)``.  This means a 1-sigma
    advantage in normalized fitness yields roughly an ``e ≈ 2.7×``
    higher unnormalized weight — moderate exploration that adapts to
    the current fitness landscape.

    When ``temperature`` is set explicitly, it operates in normalized
    [0, 1] space: high temperature (e.g. 10.0) → near-uniform,
    low temperature (e.g. 0.001) → near-greedy.
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
        """Convert raw fitnesses into softmax sampling weights.

        Fitnesses are normalized to [0, 1] so that the temperature is
        problem-independent.  Temperature is then either the user-supplied
        value or auto-computed from the spread of normalized fitnesses.
        """
        arr = np.asarray(fitnesses, dtype=np.float64)

        # --- Normalize to [0, 1] -----------------------------------------
        fitness_range = float(np.ptp(arr))
        if fitness_range < 1e-10:
            # Fully converged — no fitness signal, select uniformly.
            n = len(arr)
            return [1.0 / n] * n
        arr = (arr - arr.min()) / fitness_range

        # --- Determine temperature ----------------------------------------
        temp = self.temperature
        if temp is None:
            # Auto-temperature: use the sample std of the normalized
            # fitnesses, floored at 0.01.  Because fitnesses live in
            # [0, 1], std is always in (0, ~0.5], so the floor only
            # matters when nearly all programs have identical fitness.
            # A floor of 0.01 gives mild differentiation (best/worst
            # ratio ≈ e^(1/0.01) in the extreme 2-program case).
            std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
            temp = max(std, 0.01)

        return softmax(arr / temp).tolist()

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

        if not all(np.isfinite(f) for f in fitnesses):
            logger.warning(
                "FitnessProportionalEliteSelector: non-finite fitnesses detected; "
                "falling back to uniform sampling"
            )
            return random.sample(programs, min(total, len(programs)))

        min_fitness = min(fitnesses)
        max_fitness = max(fitnesses)
        logger.debug(
            "FitnessProportionalEliteSelector: fitness range [{:.3f}, {:.3f}]",
            min_fitness,
            max_fitness,
        )

        weights = self._compute_weights(fitnesses)

        selected = weighted_sample_without_replacement(programs, weights, total)
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

        arr = np.asarray(fitnesses, dtype=np.float64)
        median_f = float(np.median(arr))
        child_counts = np.array(
            [p.lineage.child_count for p in programs], dtype=np.float64
        )

        s = expit(self.lambda_ * (arr - median_f))
        h = 1.0 / (1.0 + child_counts)
        weights = np.maximum(s * h, self.epsilon).tolist()

        selected = weighted_sample_without_replacement(programs, weights, total)
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
        selected: list[Program] = []
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
        fitness_keys: list[str],
        fitness_key_higher_is_better: dict[str, bool] | None = None,
        tie_breaker: Callable[[Program], float] | None = None,
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

    def _pareto_rank(self, target: Program, population: list[Program]) -> int:
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

    def __call__(self, programs: list[Program], total: int) -> list[Program]:
        if len(programs) <= total:
            logger.warning(
                f"Only {len(programs)} programs available, requested {total}. Returning all."
            )
            return programs

        # FIXED: Proper sampling without replacement
        selected: list[Program] = []
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
