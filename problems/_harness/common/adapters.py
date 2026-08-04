"""Adapters reducing each evolved representation to the SearchAdapter contract.

The controller sees only initialize/transition. Whether a representation splits
that work across three methods (ImprovEvolve) or one (UnifiedEvolve) is hidden
here, so no representation can gain an advantage from the calling convention.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
import copy
from dataclasses import dataclass
import time
from typing import Any

from problems._harness.common.contracts import (
    Budget,
    Config,
    DirectProgram,
    ModularProgram,
    ProposalResult,
    ProposalStatus,
    SeedSchedule,
    SupervisedCall,
    UnifiedProgram,
)


def _isolate(config: Config) -> Config:
    """Sever every reference between candidate code and the controller.

    Copying the input stops a program mutating the incumbent in place; copying
    the output stops a program mutating it later through a retained reference.
    """
    return copy.deepcopy(config)


def attempt(call: Callable[[], Config], label: str) -> ProposalResult:
    """Run one atomic proposal, converting any failure into a status object."""
    start = time.perf_counter()
    try:
        config = call()
    except Exception as err:
        return ProposalResult(
            status=ProposalStatus.EXCEPTION,
            label=label,
            elapsed_s=time.perf_counter() - start,
            error_type=type(err).__name__,
            error_message=str(err),
        )
    elapsed = time.perf_counter() - start
    if config is None:
        return ProposalResult(
            status=ProposalStatus.INVALID_RETURN, label=label, elapsed_s=elapsed
        )
    return ProposalResult(
        status=ProposalStatus.SUCCESS, label=label, config=config, elapsed_s=elapsed
    )


# On a warm-started benchmark every arm is handed the same incumbent to begin from, and
# `initialize` is where each interface receives it: the unified operator gets it as its
# input_config instead of None, the modular one improves it instead of its own
# generate_config, and the direct one takes it as its `instance` (DirectEvaluator needs
# no warm-start field of its own — passing the config AS the instance is exactly
# `solve(warm_start, seed)`).
#
# `None` is the cold path and is what hex passes, so hex behaves exactly as before.
# Warm-starting is deliberately NOT an improve-arm privilege: it is shared information,
# and an arm that received a better starting point than another would win on the start
# and not on the interface.


@dataclass(frozen=True)
class UnifiedAdapter:
    """UnifiedEvolve: one incumbent-conditioned operator handles both roles."""

    program: UnifiedProgram
    warm_start: Config | None = None

    def initialize(self, seed: int) -> Config:
        start = None if self.warm_start is None else _isolate(self.warm_start)
        return _isolate(self.program.propose(start, 0.0, seed))

    def transition(self, input_config: Config, intensity: float, seed: int) -> Config:
        return _isolate(self.program.propose(_isolate(input_config), intensity, seed))


@dataclass(frozen=True)
class ModularAdapter:
    """ImprovEvolve: explicit generate / perturb / improve decomposition."""

    program: ModularProgram
    warm_start: Config | None = None

    def initialize(self, seed: int) -> Config:
        # generate_config is not called when warm-started — which is what the shipped
        # prompt already tells the model ("the grader warm-starts from the best-known
        # config, not from generate_config"), so this is the published contract, not a
        # weakening of the modular interface.
        generated = (
            self.program.generate_config(seed)
            if self.warm_start is None
            else _isolate(self.warm_start)
        )
        return _isolate(self.program.improve(generated, seed))

    def transition(self, input_config: Config, intensity: float, seed: int) -> Config:
        perturbed = self.program.perturb(_isolate(input_config), intensity, seed)
        return _isolate(self.program.improve(perturbed, seed))


@dataclass(frozen=True)
class DirectEvaluator:
    """DirectEvolve: repeated seeded solve() calls against a cumulative budget.

    A program that consumes the whole budget gets one attempt; a fast program
    gets several. `run` yields each proposal lazily so the caller can grade it
    in-band, against the same clock: the grading of a fast program's backlog is
    not deferred to after the budget expires, which would hand the baseline free
    wall time the evolved arms — which grade every proposal inside the loop — pay
    for. Neither a slow nor a fast program is granted extra wall time.
    """

    program: DirectProgram
    instance: Any
    minimum_launch_budget_s: float

    def run(
        self,
        budget: Budget,
        seeds: SeedSchedule,
        supervise: SupervisedCall = attempt,
    ) -> Iterator[ProposalResult]:
        while budget.remaining_s > self.minimum_launch_budget_s:
            seed = seeds.next()
            yield supervise(
                # The instance is isolated on the way IN as well as the way out.
                # On a warm-started benchmark it is the Cohn array, and a program
                # that normalized or scaled it in place would corrupt the warm start
                # for every later call in the same config — silently, since the
                # damage lands on the thing the score is measured against. Hex's
                # instance is an int, so this costs it nothing.
                lambda s=seed: _isolate(self.program.solve(_isolate(self.instance), s)),
                "solve",
            )
