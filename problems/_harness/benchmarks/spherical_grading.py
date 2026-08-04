"""Grade one spherical candidate program. Every arm of the matched study lands here.

The hex analogue of this file grades ONE instance. This one grades a SET — the 14
development configurations from the manifest — because a single instance cannot carry
the generalisation claim the resubmission needs, and that is the whole reason the
benchmark moved. The set is read from the manifest, so the three arms cannot end up
evolving on different configurations.

Everything that could decide the outcome — the controller, the intensity ladder, the
per-config budget, the sandbox, the seed schedule, the verifier, the accept rule, and
the warm start itself — is this file, once. The three problem packages under
problems/spherical_<arm> differ in exactly one line: which adapter wraps the evolved
class.

THE SCORE IS SIGNED, AND IT IS SCORED ON THE PROPOSAL, NOT ON THE INCUMBENT.

Beating Cohn is the hard part of this benchmark; most of a run is spent below the
baseline. The published score cannot see down there, for two compounding reasons, and
both had to go:

  The floor. The shipped grader scores max(0, relative reduction of mu). Non-improvement
  is worth nothing, so every arm that has not yet won reads exactly 0.0 and the search
  climbs a flat surface.

  The incumbent. Removing the floor alone changes NOTHING, which is the part that is
  easy to get wrong: the incumbent is pre-seeded with Cohn and promotion is strictly
  better, so `mu_best` can never come out above the baseline and the floor never fires
  in the first place. The score has to be taken from the best mu a proposal was
  VALIDATED at — promoted or not — for the sign to ever be negative.

So `fitness` is 100 * mean over configs of the SIGNED relative reduction of the best
validated proposal's mu. Above the baseline it is identical to the published score. Below
it, it keeps resolving: an arm closing on Cohn and an arm nowhere near it are now
different numbers, and evolution has a gradient to climb the whole way up.

A config where the arm validated NOTHING scores as mu = 1.0, the worst code that is
still valid (two coincident points). Not as 0.0 — that would make silence worth more
than a bad attempt, and a search told that will learn to be silent. It is bounded for
the same reason: mu <= 1 and mu_cohn >= 0.15 everywhere in the snapshot, so the worst
attainable score on the development panel is -118.1 and INVALID_FITNESS sits below it.

`gain_floored` is the published number, kept so the campaign's figures stay comparable.
`mean_mu` is the ACCEPTED coherence — the physical quantity the paper reports against
Cohn's table — and it is pinned at the baseline unless the arm actually won.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np

from problems._harness.benchmarks.spherical import SphericalBenchmark
from problems._harness.common.adapters import DirectEvaluator
from problems._harness.common.budget import WallClockBudget
from problems._harness.common.controller import SeedSchedule, run, run_direct
from problems._harness.common.sandbox import (
    CandidateWorker,
    RemoteProgram,
    SandboxSupervisor,
    require_cpu_only_grading_process,
)
from problems._harness.protocol.catalogue import floored, gain, warm_start
from problems._harness.protocol.settings import (
    config_seconds,
    controller_config,
    direct_minimum_launch_seconds,
    evolution_split,
    grading_seed,
    sandbox_limits,
    startup_seconds,
)

SPHERICAL_CONFIGS = evolution_split("spherical")

# The worst code that is still VALID: two coincident unit vectors, so mu = 1. What a
# config scores when the arm validated nothing on it — silence is graded as the worst
# attempt, never as no attempt, or the cheapest way to a good score is to propose nothing.
WORST_MU = 1.0

# GigaEvo's sentinel for a program that produced nothing valid ANYWHERE. It must equal the
# sentinel_value each arm's metrics.yaml declares, or the engine and the grader disagree
# about what "invalid" is, and it must sit strictly BELOW the worst attainable score —
# -118.1 on the development panel, since WORST_MU on every config is as bad as grading
# gets. The shipped grader's -1.0 was safe only while the score was floored at zero; under
# a signed score it would rank a program that produces nothing ABOVE one that produces
# something bad, which is the incentive this metric exists to remove.
INVALID_FITNESS = -200.0


@dataclass(frozen=True)
class ConfigOutcome:
    dimension: int
    count: int
    mu_cohn: float
    mu_best: float
    mu_proposed: float | None
    gain: float
    gain_floored: float
    proposals: int
    accepted: int
    valid_rate: float


@contextmanager
def _caged(program_class: Any, dimension: int, count: int, seconds: float):
    """The candidate, alive in a spawned worker for as long as this config has budget.

    One worker per CONFIGURATION rather than per candidate, because the evolved class
    is constructed with (n, d) — a single worker cannot serve 14 different geometries.
    This process must never instantiate the program itself: GigaEvo already unpickled
    the class here, so a JAX candidate has imported JAX into THIS interpreter, and
    building the program on this side would put the run back where it started.
    """
    limits = sandbox_limits(seconds)
    require_cpu_only_grading_process(limits)
    worker = CandidateWorker(
        program_class=program_class,
        kwargs={"n": count, "d": dimension, "seed": grading_seed()},
        limits=limits,
    )
    try:
        # The clock starts when the candidate is ready to be called, not when we begin
        # getting it ready: see CandidateWorker.start. Ordering, not decoration --
        # building the budget first is what charged the candidate for the harness's own
        # startup. Inside the try, because start() is now the first thing here that can
        # hold a process or a temp dir, and both have to be released on any exit.
        worker.start(startup_seconds("spherical"))
        budget = WallClockBudget(total_s=seconds)
        yield RemoteProgram(worker), budget, SandboxSupervisor(worker, budget)
    finally:
        worker.close()


def _grade_config(
    adapter_of: Any | None,
    program_class: Any,
    dimension: int,
    count: int,
    seconds: float,
) -> ConfigOutcome:
    """One (d, N): warm-start from frozen Cohn, search under the shared controller.

    `adapter_of` None is the direct arm. It is the only branch in this file, and it
    selects a CALLING CONVENTION, not a search: both paths get the same warm start, the
    same budget, the same seed schedule and the same `Incumbent.consider`.
    """
    cohn = warm_start(dimension, count)
    benchmark = SphericalBenchmark(dimension=dimension, count=count)

    with _caged(program_class, dimension, count, seconds) as (
        program,
        budget,
        supervise,
    ):
        if adapter_of is None:
            outcome = run_direct(
                DirectEvaluator(
                    program=program,
                    # The warm start IS the instance: `solve(cohn, seed)`. The direct
                    # arm starts from the identical configuration the other two do.
                    instance=cohn.points,
                    minimum_launch_budget_s=direct_minimum_launch_seconds("spherical"),
                ),
                benchmark,
                budget,
                SeedSchedule(grading_seed()),
                supervise,
                warm_start=cohn.points,
            )
            proposals = outcome.calls
        else:
            outcome = run(
                adapter_of(program, cohn.points),
                benchmark,
                controller_config("spherical"),
                budget,
                SeedSchedule(grading_seed()),
                supervise,
                warm_start=cohn.points,
            )
            proposals = outcome.proposals

    # The incumbent is pre-seeded with Cohn and the accept rule is monotone, so
    # `objective` can never be worse than the baseline and never None. It is what the
    # PUBLISHED score reads, and the reason that score cannot see below the baseline.
    mu_best = cohn.mu_cohn if outcome.objective is None else outcome.objective

    # What the SIGNED score reads: the best mu a proposal was validated at, which is the
    # only quantity in this run that is free to be worse than Cohn. Validating nothing
    # scores as the worst valid code rather than as the baseline.
    scored = WORST_MU if outcome.best_valid is None else outcome.best_valid

    return ConfigOutcome(
        dimension=dimension,
        count=count,
        mu_cohn=cohn.mu_cohn,
        mu_best=mu_best,
        # None when the arm never produced ONE feasible config for this geometry. It is
        # not filled in with WORST_MU: `gain` grades that case, but `mean_mu_proposed`
        # reports coherence actually reached, and must not claim a mu nothing reached.
        mu_proposed=outcome.best_valid,
        gain=gain(scored, cohn.mu_cohn),
        gain_floored=floored(gain(mu_best, cohn.mu_cohn)),
        proposals=proposals,
        accepted=outcome.accepted,
        valid_rate=outcome.valid_rate,
    )


def _counts(outcomes: list[ConfigOutcome]) -> dict:
    """What the candidate DID, as distinct from how well it did it.

    Reported whether or not it ever produced a valid code, because these are the only
    numbers that separate a program whose every call raised (proposals = 0) from one that
    was called three hundred times and returned garbage every time. Both score the
    sentinel; they are not the same failure and they do not have the same fix. The seed
    gate writes its rejection message from exactly these keys, so omitting one here does
    not merely hide it — it prints as a measured zero.
    """
    return {
        "proposals": int(np.sum([o.proposals for o in outcomes])),
        "accepted": int(np.sum([o.accepted for o in outcomes])),
        "valid_rate": float(np.mean([o.valid_rate for o in outcomes])),
        "configs_graded": len(outcomes),
        "configs_improved": sum(1 for o in outcomes if o.gain > 0.0),
        # Configs the arm validated nothing on. Each costs the score its config's full
        # WORST_MU penalty — up to 227 points on (16, 296) — so a fitness deep in the
        # negatives is read here first.
        "configs_dead": sum(1 for o in outcomes if o.mu_proposed is None),
    }


def _metrics(outcomes: list[ConfigOutcome]) -> dict:
    if not outcomes:
        return {"fitness": INVALID_FITNESS, "is_valid": 0}

    # A candidate is invalid only if it never produced ONE feasible config anywhere.
    # Below that bar it is a bad program, not a broken one, and it is SCORED — poorly —
    # rather than sentinelled: the signed gain already grades a dead config as the worst
    # valid code, and that penalty is what the sentinel used to have to stand in for.
    #
    # The mu metrics are absent from this branch and the counters are not: there is no
    # coherence to report, but there is plenty to report about the attempt.
    if not any(outcome.valid_rate > 0.0 for outcome in outcomes):
        return {"fitness": INVALID_FITNESS, "is_valid": 0, **_counts(outcomes)}

    # Averaged over the configs that produced one, not over all 14 — a config where the
    # arm was never feasible has no mu to contribute, and substituting one there would
    # report the arm as having reached a coherence it never reached. `fitness` does grade
    # that case (at WORST_MU); this metric only ever reports mu that was really seen.
    proposed = [o.mu_proposed for o in outcomes if o.mu_proposed is not None]

    return {
        "fitness": 100.0 * float(np.mean([o.gain for o in outcomes])),
        "is_valid": 1,
        # The published score, floored and read off the incumbent. Kept only so the
        # campaign's figures stay comparable; nothing selects on it.
        "gain_floored": 100.0 * float(np.mean([o.gain_floored for o in outcomes])),
        # The mu that was ACCEPTED: pinned at the baseline unless the arm beat it.
        "mean_mu": float(np.mean([o.mu_best for o in outcomes])),
        "mean_mu_cohn": float(np.mean([o.mu_cohn for o in outcomes])),
        # The best mu that was VALIDATED, promoted or not — what `fitness` is scored on.
        "mean_mu_proposed": float(np.mean(proposed)),
        **_counts(outcomes),
    }


def grade(
    adapter_of: Any | None,
    program_class: Any,
    configs: tuple[tuple[int, int], ...] = SPHERICAL_CONFIGS,
    seconds: float | None = None,
) -> dict:
    """Grade a candidate across the development configurations, concurrently.

    Concurrent because the budget is PER CONFIG: 14 configs at 45s serially would cost
    630s per candidate and price spherical back up to hex. Each config gets its own
    worker pinned to its own core, so the searches do not compete — `cpu_limit: 1` still
    holds exactly where it decides anything.

    `seconds` exists so a smoke run can grade a seed in seconds rather than the
    protocol's minutes. The runs leave it None and get the frozen budget.
    """
    seconds = config_seconds("spherical") if seconds is None else seconds

    with ThreadPoolExecutor(max_workers=len(configs)) as pool:
        outcomes = list(
            pool.map(
                lambda config: _grade_config(
                    adapter_of, program_class, config[0], config[1], seconds
                ),
                configs,
            )
        )

    return _metrics(outcomes)
