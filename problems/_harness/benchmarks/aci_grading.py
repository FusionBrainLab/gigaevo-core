"""Grade one ACI candidate program. Every arm of the matched study lands here.

The instance is the manifest's, not this file's: `problems/aci_*` and the prompts are
both named off `evolution_instances`, so an instance change moves the packages, the
prompts and the grader together or it does not happen at all.

The three problem packages under `problems/aci_*` differ in exactly one line: which
adapter wraps the evolved class. Everything that could decide the outcome -- the
controller, the sigma ladder, the wall-clock budget, the sandbox, the seed schedule,
the verifier, the accept rule -- is this file, once. A validator that carried its own
copy of any of them would turn the interface study into a comparison of validators.

COLD, like hex (not spherical): the arm builds f from scratch, `warm_start=None`. The
fitness the search optimizes is the incumbent's C directly -- ACI maximizes, so it is
the raw objective, the mirror of hex's `-L`. `proposals`, `accepted` and `valid_rate`
are reported as diagnostics only, to answer the reviewer's objection to the DirectEvolve
baseline: a basic arm reporting proposals=1 is crippled, and the number has to be
visible to say it isn't.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

from problems._harness.benchmarks.aci import ACIBenchmark
from problems._harness.common.adapters import DirectEvaluator
from problems._harness.common.budget import WallClockBudget
from problems._harness.common.controller import SeedSchedule, run, run_direct
from problems._harness.common.sandbox import (
    CandidateWorker,
    RemoteProgram,
    SandboxSupervisor,
    require_cpu_only_grading_process,
)
from problems._harness.protocol.settings import (
    candidate_seconds,
    controller_config,
    direct_minimum_launch_seconds,
    evolution_instance,
    grading_seed,
    sandbox_limits,
    startup_seconds,
)

ACI_N = evolution_instance("aci")

# GigaEvo's sentinel for a program that produced nothing valid. Matches
# problems/alphaevolve/second_autocorr_ineq/metrics.yaml, so the two are comparable in
# the archive. Any valid C lies in (0, 1], so -1000 is unambiguously worse than valid.
INVALID_FITNESS = -1000.0


def _metrics(objective: float | None, proposals: int, accepted: int, valid_rate: float):
    return {
        # ACI MAXIMIZES C, so the fitness IS the objective (hex minimizes L and stores
        # -L; the sign is the only difference and it is the whole difference).
        "fitness": objective if objective is not None else INVALID_FITNESS,
        "is_valid": 1 if objective is not None else 0,
        "proposals": proposals,
        "accepted": accepted,
        "valid_rate": valid_rate,
    }


@contextmanager
def _caged(program_class: Any, items: int, seconds: float):
    """The candidate, alive in a spawned worker for as long as it has budget.

    This process must never instantiate the program itself. GigaEvo already unpickled
    the class here, so a JAX candidate has imported JAX into THIS interpreter -- the
    worker is what keeps that from mattering, and building the program on this side
    would put the run back where it started.
    """
    limits = sandbox_limits(seconds)
    require_cpu_only_grading_process(limits)
    worker = CandidateWorker(
        program_class=program_class,
        kwargs={"n": items, "seed": grading_seed()},
        limits=limits,
    )
    try:
        # The clock starts when the candidate is ready to be called: see
        # CandidateWorker.start. Inside the try, so a worker that fails to spawn still
        # releases its process and its temp dir.
        worker.start(startup_seconds("aci"))
        budget = WallClockBudget(total_s=seconds)
        yield RemoteProgram(worker), budget, SandboxSupervisor(worker, budget)
    finally:
        worker.close()


def grade_controller(
    adapter_of: Any,
    program_class: Any,
    items: int = ACI_N,
    seconds: float | None = None,
) -> dict:
    """The improve and mixed arms: one incumbent, Stage A then Stage B.

    `seconds` exists so a smoke run can grade a seed in seconds rather than the
    protocol's minutes. The runs leave it None and get the frozen budget.
    """
    seconds = candidate_seconds("aci") if seconds is None else seconds
    with _caged(program_class, items, seconds) as (program, budget, supervise):
        outcome = run(
            adapter_of(program),
            ACIBenchmark(items=items),
            controller_config("aci"),
            budget,
            SeedSchedule(grading_seed()),
            supervise,
        )

    return _metrics(
        outcome.objective, outcome.proposals, outcome.accepted, outcome.valid_rate
    )


def grade_direct(
    program_class: Any, items: int = ACI_N, seconds: float | None = None
) -> dict:
    """The basic arm: repeated seeded solve() calls against the same wall clock.

    It is given the whole budget rather than one attempt, and its candidates go through
    the identical acceptance rule -- see problems._harness.common.controller.run_direct.
    """
    seconds = candidate_seconds("aci") if seconds is None else seconds
    with _caged(program_class, items, seconds) as (program, budget, supervise):
        outcome = run_direct(
            DirectEvaluator(
                program=program,
                instance=items,
                minimum_launch_budget_s=direct_minimum_launch_seconds("aci"),
            ),
            ACIBenchmark(items=items),
            budget,
            SeedSchedule(grading_seed()),
            supervise,
        )

    return _metrics(
        outcome.objective, outcome.calls, outcome.accepted, outcome.valid_rate
    )
