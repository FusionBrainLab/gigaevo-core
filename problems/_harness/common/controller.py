"""The one controller every representation runs under.

This is the experiment. DirectEvolve, UnifiedEvolve, ImprovEvolve and Generic-BH
differ only in the adapter handed to `run`; the search itself — how many
initialization proposals, the sigma schedule, when a candidate is accepted, how
seeds are drawn — is identical. A method-specific branch anywhere below would
turn the comparison into a comparison of controllers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from problems._harness.common.adapters import DirectEvaluator, attempt
from problems._harness.common.contracts import (
    Budget,
    Config,
    ProposalResult,
    SearchAdapter,
    SupervisedCall,
)
from problems._harness.common.events import EventLogger


@runtime_checkable
class Benchmark(Protocol):
    """Everything task-specific the controller is allowed to know."""

    def validate(self, config: Config) -> float | None:
        """Objective value if the config is feasible, None if it is not."""

    def better(self, candidate: float, incumbent: float) -> bool:
        """Strictly better by the benchmark's frozen acceptance tolerance."""


@dataclass(frozen=True)
class ControllerConfig:
    """Stage B sweeps the whole sigma ladder repeatedly, not once.

    The ladder restarts coarse at the top of every round — lifted from the pipeline
    the published numbers came from (problems/hexagon_improver/validate.py:263) — so
    a candidate that has settled into a basin keeps being kicked out of it. A single
    monotone decay would visit each sigma once and make Stage B a different algorithm.

    There is deliberately no round count. Stage B runs until the budget is spent, so
    what bounds every arm is the wall clock and nothing else. A fixed round count
    bounded the modular arms by call count instead, which let them finish and idle
    while DirectEvolve — bounded only by the clock — spent the rest: measured on the
    first HEX-11 campaign, 97.3% of valid candidates exhausted the ladder with time
    still on the clock. That is not a matched comparison, and it also hides the
    trade the interface exists to expose: under a clock, a cheaper transition buys
    more rounds and an expensive one has to earn its keep, so the search can price
    its own moves.
    """

    stage_a_proposals: int
    sigma_schedule: tuple[float, ...]


@dataclass(frozen=True)
class RunOutcome:
    config: Config | None
    objective: float | None
    proposals: int
    accepted: int
    # Same definition as DirectOutcome.valid_rate — the fraction of calls the
    # benchmark's own verifier accepted. The arms are only comparable on this if it
    # counts the same thing in both, so it comes from the same Incumbent.
    valid_rate: float
    best_valid: float | None = None


@dataclass(frozen=True)
class DirectOutcome:
    """What DirectEvolve did with the budget it was given.

    `calls` and `valid_rate` are the numbers that answer the reviewer's objection.
    A DirectEvolve run reporting calls=1 is a crippled baseline; the whole point of
    the cumulative-budget loop is that this field comes out large.
    """

    config: Config | None
    objective: float | None
    calls: int
    accepted: int
    best_call_index: int | None
    valid_rate: float
    best_valid: float | None = None


class Incumbent:
    """The single place a candidate is allowed to become the incumbent.

    Every representation — modular, unified, direct — promotes through this object.
    A second copy of the acceptance rule anywhere would be exactly the confound the
    fixed controller exists to remove: whoever got the laxer rule would win on the
    rule, not on the method.
    """

    def __init__(
        self,
        benchmark: Benchmark,
        logger: EventLogger | None = None,
        warm_start: Config | None = None,
    ) -> None:
        self._benchmark = benchmark
        self._logger = logger
        self.config: Config | None = None
        self.objective: float | None = None
        self.accepted = 0
        self.considered = 0
        self.valid = 0
        self.best_index: int | None = None

        # The best objective any proposal was VALIDATED at, whether or not it was
        # promoted — which on a warm-started benchmark is the only sub-baseline signal
        # there is. `objective` cannot carry it: the incumbent starts AT the warm start
        # and promotion is strictly-better, so an arm that never beats it ends at
        # exactly the warm start's value, and an arm that missed by 1e-9 is then
        # indistinguishable from one that never came close. This field separates them.
        self.best_valid: float | None = None

        # A warm-started benchmark hands the arm an incumbent instead of asking it to
        # build one, so the search begins where the shipped grader begins
        # (validate.py: `current = best-known config` BEFORE stage A). Two consequences
        # the cold path does not have: a failed initialize costs that one call rather
        # than the whole configuration, and Stage B still runs. Hex passes None and is
        # unaffected.
        #
        # It is seeded, not `consider`ed: the arm did not propose this config, so
        # counting it would credit every arm with one free valid call and make
        # valid_rate incomparable to the cold benchmark's.
        if warm_start is not None:
            objective = benchmark.validate(warm_start)
            if objective is None:
                raise ValueError(
                    "the warm start is infeasible under the benchmark's own verifier — "
                    "every arm would be graded against a baseline that does not exist"
                )
            self.config = warm_start
            self.objective = objective

    def consider(self, result: ProposalResult, index: int, **fields) -> bool:
        """A proposal may only ever replace the incumbent by being validated and
        strictly better. Every other outcome — invalid, timeout, exception, worse —
        leaves it exactly as it was.

        `objective is None` is the valid-call test: a call is valid iff it returned
        and the benchmark's own verifier accepted what it returned. Nothing else
        counts as valid, so a program cannot inflate its valid-call rate by
        returning something that merely looks like a config.
        """
        objective = self._benchmark.validate(result.config) if result.ok else None
        promote = objective is not None and (
            self.objective is None or self._benchmark.better(objective, self.objective)
        )
        if promote:
            self.config = result.config
            self.objective = objective
            self.best_index = index
            self.accepted += 1
        if objective is not None and (
            self.best_valid is None
            or self._benchmark.better(objective, self.best_valid)
        ):
            self.best_valid = objective
        self.considered += 1
        self.valid += objective is not None
        if self._logger is not None:
            self._logger.emit_proposal(
                result,
                index=index,
                objective=objective,
                accepted=promote,
                incumbent_objective=self.objective,
                **fields,
            )
        return promote

    @property
    def valid_rate(self) -> float:
        """Of the calls that were made, the fraction that produced a feasible config."""
        return self.valid / self.considered if self.considered else 0.0


class SeedSchedule:
    """Deterministic and representation-blind: the nth proposal of a run gets the
    same seed whichever adapter is driving it."""

    def __init__(self, base_seed: int) -> None:
        self._base_seed = base_seed
        self._issued = 0

    def next(self) -> int:
        seed = self._base_seed * 1_000_003 + self._issued
        self._issued += 1
        return seed


def run(
    adapter: SearchAdapter,
    benchmark: Benchmark,
    config: ControllerConfig,
    budget: Budget,
    seeds: SeedSchedule,
    supervise: SupervisedCall = attempt,
    logger: EventLogger | None = None,
    warm_start: Config | None = None,
) -> RunOutcome:
    incumbent = Incumbent(benchmark, logger, warm_start)
    proposals = 0

    for _ in range(config.stage_a_proposals):
        if budget.exhausted():
            break
        seed = seeds.next()
        incumbent.consider(
            supervise(lambda s=seed: adapter.initialize(s), "initialize"),
            proposals,
            stage="initialize",
            sigma=0.0,
        )
        proposals += 1

    round_index = 0
    while not budget.exhausted() and incumbent.config is not None:
        for sigma in config.sigma_schedule:
            if budget.exhausted():
                break
            seed = seeds.next()
            incumbent.consider(
                supervise(
                    lambda s=seed, g=sigma, c=incumbent.config: adapter.transition(
                        c, g, s
                    ),
                    "transition",
                ),
                proposals,
                stage="transition",
                round=round_index,
                sigma=sigma,
            )
            proposals += 1
        round_index += 1

    return RunOutcome(
        config=incumbent.config,
        objective=incumbent.objective,
        proposals=proposals,
        accepted=incumbent.accepted,
        valid_rate=incumbent.valid_rate,
        best_valid=incumbent.best_valid,
    )


def run_direct(
    evaluator: DirectEvaluator,
    benchmark: Benchmark,
    budget: Budget,
    seeds: SeedSchedule,
    supervise: SupervisedCall = attempt,
    logger: EventLogger | None = None,
    warm_start: Config | None = None,
) -> DirectOutcome:
    """DirectEvolve under the same budget and the same acceptance rule.

    The point of this function is what it does NOT do. It gives the direct program
    the whole budget — as many `solve` calls as fit, not one artificial attempt —
    and then holds every returned candidate to the identical `Incumbent.consider`
    the evolved representations go through. Cripple the baseline here and the
    headline comparison is worthless; give it a laxer accept rule and the headline
    comparison is worthless in the other direction.

    `warm_start` is the same object the other two arms are seeded with, and on a
    warm-started benchmark the direct program is ALSO handed it as its instance — so
    it does not merely start level, it starts identical.
    """
    incumbent = Incumbent(benchmark, logger, warm_start)

    # Consume the evaluator's stream lazily: each proposal is graded in-band, before
    # the generator's next budget check, so validation time is charged to the same
    # clock the evolved arms pay on and only the incumbent is retained (O(1) state).
    calls = 0
    for result in evaluator.run(budget, seeds, supervise):
        incumbent.consider(result, calls, stage="direct", call=calls)
        calls += 1

    return DirectOutcome(
        config=incumbent.config,
        objective=incumbent.objective,
        calls=calls,
        accepted=incumbent.accepted,
        best_call_index=incumbent.best_index,
        valid_rate=incumbent.valid_rate,
        best_valid=incumbent.best_valid,
    )
