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
            "[RandomEliteSelector] selecting {} from {} programs",
            total,
            len(programs),
        )

        if len(programs) <= total:
            logger.debug(
                "[RandomEliteSelector] returning all {} programs (≤ requested {})",
                len(programs),
                total,
            )
            return programs

        selected = random.sample(programs, total)
        logger.debug(
            "[RandomEliteSelector] selected {} programs randomly",
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
            "[FitnessProportionalEliteSelector] selecting {} from {} programs "
            "(key='{}', higher_is_better={}, temperature={})",
            total,
            len(programs),
            self.fitness_key,
            self.higher_is_better,
            self.temperature,
        )

        if len(programs) <= total:
            logger.debug(
                "[FitnessProportionalEliteSelector] returning all {} programs (≤ requested {})",
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
                "[FitnessProportionalEliteSelector] non-finite fitnesses detected; "
                "falling back to uniform sampling"
            )
            return random.sample(programs, min(total, len(programs)))

        min_fitness = min(fitnesses)
        max_fitness = max(fitnesses)
        logger.debug(
            "[FitnessProportionalEliteSelector] fitness range [{:.3f}, {:.3f}]",
            min_fitness,
            max_fitness,
        )

        weights = self._compute_weights(fitnesses)

        selected = weighted_sample_without_replacement(programs, weights, total)
        logger.debug(
            "[FitnessProportionalEliteSelector] selected {} programs",
            len(selected),
        )
        return selected


class FitnessProportionalTournamentEliteSelector(FitnessProportionalEliteSelector):
    """Two-pass selector: fitness-proportional pool, then uniform tournament.

    Stage 1 — softmax fitness-proportional sample of a candidate pool of size
    ``pool_multiplier * total`` (clamped to the population size). Reuses the
    weighting / temperature logic from :class:`FitnessProportionalEliteSelector`.

    Stage 2 — uniform random sample of ``total`` programs from the pool, without
    replacement. The uniform second pass widens exploration relative to plain
    FP without dropping fitness pressure entirely: fitness controls which
    programs reach the pool, then chance decides which ones win.
    """

    def __init__(
        self,
        fitness_key: str,
        fitness_key_higher_is_better: bool = True,
        temperature: float | None = None,
        pool_multiplier: int = 5,
    ):
        super().__init__(
            fitness_key=fitness_key,
            fitness_key_higher_is_better=fitness_key_higher_is_better,
            temperature=temperature,
        )
        self.pool_multiplier = max(1, int(pool_multiplier))

    def __call__(self, programs: list[Program], total: int) -> list[Program]:
        logger.debug(
            "[FitnessProportionalTournamentEliteSelector] selecting {} from {} "
            "programs (key='{}', higher_is_better={}, temperature={}, "
            "pool_multiplier={})",
            total,
            len(programs),
            self.fitness_key,
            self.higher_is_better,
            self.temperature,
            self.pool_multiplier,
        )

        if len(programs) <= total:
            return programs

        pool_size = min(total * self.pool_multiplier, len(programs))

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
                "[FitnessProportionalTournamentEliteSelector] non-finite "
                "fitnesses detected; falling back to uniform sampling"
            )
            return random.sample(programs, total)

        weights = self._compute_weights(fitnesses)
        pool = weighted_sample_without_replacement(programs, weights, pool_size)

        if len(pool) <= total:
            return pool
        return random.sample(pool, total)


class FitnessProportionalTournamentBoundedGapEliteSelector(
    FitnessProportionalTournamentEliteSelector
):
    """FPT for parent #1; FPT for parent #2.. restricted to elites within a
    bounded fitness gap of parent #1 — a same-stratum coupler.

    Derived from the 2026-05-26 pair-coupling audit on two
    ``num_parents=2`` runs (HoVer + tabular_regression): pairs whose
    ``|p1.fitness - p2.fitness|`` lands in the top quartile of pairwise
    gaps advance ~3-5x worse than inner-quartile pairs in both runs.
    LCA-based and iter/gen-gap couplers do not survive the data
    (non-monotonic / sign-flip).

    Behaviour
    ---------
    For ``total >= 2`` the call decomposes into:

    1. parent #1 via inherited FPT — preserves the dominant
       ``pair_max_fit`` signal (r~+0.43 on tabular child fitness).
    2. ``natural_gap = 2 * MAD(elite_fitnesses)`` — O(n), deterministic
       estimator of the median pairwise gap. ``MAD =
       median(|f - median(f)|)``.
    3. ``delta = gap_factor * natural_gap``.
    4. ``candidates = {e in elites : e != parent_1 and
       |e.fit - parent_1.fit| <= delta}``.
    5. If fewer than ``total - 1`` candidates survive, fall back to
       ``elites \\ {parent_1}`` so a run with a high-fitness outlier
       does not deadlock.
    6. parent #2.. via inherited FPT on ``candidates``.

    For ``total <= 1`` the call delegates to the inherited FPT
    unchanged — there is nothing to couple.

    Parameters
    ----------
    gap_factor:
        Multiplier on the natural gap. ``1.5`` (default) drops
        approximately the top quartile of fitness-gap pairs in the
        audit's two runs. ``+inf`` is equivalent to plain FPT. Must be
        non-negative.
    """

    def __init__(
        self,
        fitness_key: str,
        fitness_key_higher_is_better: bool = True,
        temperature: float | None = None,
        pool_multiplier: int = 5,
        gap_factor: float = 1.5,
    ):
        super().__init__(
            fitness_key=fitness_key,
            fitness_key_higher_is_better=fitness_key_higher_is_better,
            temperature=temperature,
            pool_multiplier=pool_multiplier,
        )
        if not np.isfinite(gap_factor) and gap_factor != float("inf"):
            raise ValueError(f"gap_factor must be finite or +inf, got {gap_factor}")
        if gap_factor < 0:
            raise ValueError(f"gap_factor must be non-negative, got {gap_factor}")
        self.gap_factor = float(gap_factor)

    @staticmethod
    def _natural_gap(fitnesses: list[float]) -> float:
        arr = np.asarray(fitnesses, dtype=np.float64)
        if len(arr) < 2:
            return 0.0
        return float(2.0 * np.median(np.abs(arr - np.median(arr))))

    def __call__(self, programs: list[Program], total: int) -> list[Program]:
        logger.debug(
            "[FitnessProportionalTournamentBoundedGapEliteSelector] selecting {} "
            "from {} programs (gap_factor={})",
            total,
            len(programs),
            self.gap_factor,
        )

        if len(programs) <= total:
            return programs
        if total <= 1:
            return super().__call__(programs, total)

        first = super().__call__(programs, 1)
        if not first:
            return super().__call__(programs, total)
        p1 = first[0]

        fitnesses = []
        for p in programs:
            if self.fitness_key not in p.metrics:
                raise ValueError(
                    f"Missing fitness key '{self.fitness_key}' in program {p.id}"
                )
            fitnesses.append(p.metrics[self.fitness_key])

        if not all(np.isfinite(f) for f in fitnesses):
            logger.warning(
                "[FitnessProportionalTournamentBoundedGapEliteSelector] non-finite "
                "fitnesses; coupling disabled, uniform-sampling the remainder"
            )
            remaining = [p for p in programs if p is not p1]
            rest = random.sample(remaining, min(total - 1, len(remaining)))
            return [p1, *rest]

        natural_gap = self._natural_gap(fitnesses)
        delta = self.gap_factor * natural_gap
        p1_fit = p1.metrics[self.fitness_key]

        candidates = [
            p
            for p in programs
            if p is not p1 and abs(p.metrics[self.fitness_key] - p1_fit) <= delta
        ]
        if len(candidates) < total - 1:
            logger.debug(
                "[FitnessProportionalTournamentBoundedGapEliteSelector] only {} "
                "candidates within delta={:.4g} of parent#1 (need {}); "
                "falling back to full elite pool minus parent#1",
                len(candidates),
                delta,
                total - 1,
            )
            candidates = [p for p in programs if p is not p1]

        rest = super().__call__(candidates, total - 1)
        return [p1, *rest]


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
            "[WeightedEliteSelector] selecting {} from {} programs (key='{}', higher_is_better={}, lambda={}, epsilon={})",
            total,
            len(programs),
            self.fitness_key,
            self.higher_is_better,
            self.lambda_,
            self.epsilon,
        )

        if len(programs) <= total:
            logger.debug(
                "[WeightedEliteSelector] returning all {} programs (≤ requested {})",
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
            "[WeightedEliteSelector] selected {} programs",
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
                f"[ScalarTournamentEliteSelector] Only {len(programs)} programs available, requested {total}. Returning all."
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
                f"[ParetoTournamentEliteSelector] Only {len(programs)} programs available, requested {total}. Returning all."
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
