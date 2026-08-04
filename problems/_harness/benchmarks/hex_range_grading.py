"""Range-generalization grading for Exp G: one candidate, five instance sizes.

Each train size is graded independently by the frozen Exp A machinery
(hex_grading — same sandbox, verifier, acceptance rule and grading seed),
with an EQUAL wall-clock slice per size for every arm. The aggregate is

    fitness = -( mean over valid n of L_n / sqrt(n) + PENALTY * #invalid n )

with the all-invalid sentinel. L/sqrt(n) is the scale that needs no record
constants: the container's area bound gives L >= sqrt(n), so the ratio is
O(1) at every size and each n contributes comparably to the mean. The +1
per invalid size makes full-range validity dominate single-size brilliance.

is_valid is 1 when AT LEAST ONE size graded valid: a program that packs
four sizes and crashes on the fifth carries signal the archive must keep.
Per-size fitness_{n} = -L_n diagnostics are prompt-visible in both arms
identically, so neither arm receives feedback the other does not.
"""

from __future__ import annotations

import math
from typing import Any

from problems._harness.benchmarks.hex_grading import (
    INVALID_FITNESS,
    grade_controller,
    grade_direct,
)

N_TRAIN = (11, 12, 15, 17, 23)
N_HOLDOUT = (13, 14, 16)
PER_N_SECONDS = 30.0
INVALID_PENALTY = 1.0


def _aggregate(per_n: dict[int, dict[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    ratios: list[float] = []
    proposals = accepted = 0
    feasible = 0.0
    for n, res in sorted(per_n.items()):
        metrics[f"fitness_{n}"] = res["fitness"]
        proposals += int(res.get("proposals", 0))
        accepted += int(res.get("accepted", 0))
        feasible += float(res.get("valid_rate", 0.0)) * int(res.get("proposals", 0))
        if res["is_valid"]:
            ratios.append(-res["fitness"] / math.sqrt(n))
    n_invalid = len(per_n) - len(ratios)
    if ratios:
        fitness = -(sum(ratios) / len(ratios) + INVALID_PENALTY * n_invalid)
    else:
        fitness = INVALID_FITNESS
    metrics.update(
        fitness=fitness,
        is_valid=1 if ratios else 0,
        n_valid=len(ratios),
        proposals=proposals,
        accepted=accepted,
        valid_rate=feasible / proposals if proposals else 0.0,
    )
    return metrics


def grade_range_controller(
    adapter_of: Any,
    program_class: Any,
    instances: tuple[int, ...] = N_TRAIN,
    seconds: float | None = None,
) -> dict[str, Any]:
    seconds = PER_N_SECONDS if seconds is None else seconds
    return _aggregate(
        {
            n: grade_controller(adapter_of, program_class, items=n, seconds=seconds)
            for n in instances
        }
    )


def grade_range_direct(
    program_class: Any,
    instances: tuple[int, ...] = N_TRAIN,
    seconds: float | None = None,
) -> dict[str, Any]:
    seconds = PER_N_SECONDS if seconds is None else seconds
    return _aggregate(
        {n: grade_direct(program_class, items=n, seconds=seconds) for n in instances}
    )
