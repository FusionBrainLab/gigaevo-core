"""The protocol constants, read from `protocol.yaml` rather than re-declared in code.

Every arm's validator builds its controller and its budget from here. If the
controller settings were literals in three validate.py files, one of them could
drift and the three-way comparison would quietly become a comparison of
controllers — which is the one confound this whole apparatus exists to remove.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import yaml

from problems._harness import HARNESS
from problems._harness.common.controller import ControllerConfig
from problems._harness.common.sandbox import SandboxLimits

PROTOCOL = HARNESS / "protocol.yaml"

_UNITS = {"KiB": 2**10, "MiB": 2**20, "GiB": 2**30}


@cache
def protocol() -> dict:
    return yaml.safe_load(PROTOCOL.read_text())


@cache
def controller_config(benchmark: str) -> ControllerConfig:
    """Keyed by benchmark, shared by every ARM of it.

    The ladder's units are the benchmark's: hex sigma is a displacement, spherical
    intensity is a fraction in [0, 1]. What the study requires is that the three arms
    of one benchmark run the identical controller, and they do — there is one entry
    per benchmark here, never one per arm.
    """
    block = protocol()["controller"][benchmark]
    return ControllerConfig(
        stage_a_proposals=block["stage_a_seeds"],
        sigma_schedule=tuple(float(sigma) for sigma in block["sigma_schedule"]),
    )


def grading_seed() -> int:
    return int(protocol()["controller"]["grading_seed"])


def catalogue_snapshot() -> Path:
    return HARNESS / protocol()["benchmarks"]["spherical"]["catalogue_snapshot"]


@cache
def evolution_split(benchmark: str) -> tuple[tuple[int, int], ...]:
    """The (dimension, count) configurations every arm evolves on.

    Spherical's analogue of `evolution_instance`, and plural where that one is
    singular: the whole reason for moving off hex is that one instance cannot carry a
    generalisation claim. Same guarantee though — it is read from the protocol file, so
    the three arms cannot end up evolving on different sets.
    """
    split = protocol()["benchmarks"][benchmark]["development_split"]
    return tuple((int(dimension), int(count)) for dimension, count in split)


def evolution_instance(benchmark: str) -> int:
    """The one instance every arm evolves on.

    Singular on purpose. The problem packages, the prompts, the seed pool and the
    grader all key off this, so if the protocol ever declared two the three arms could
    silently end up evolving on different ones — which is not a slower experiment, it
    is a different experiment.
    """
    instances = protocol()["benchmarks"][benchmark]["evolution_instances"]
    if len(instances) != 1:
        raise ValueError(
            f"{benchmark} declares {len(instances)} evolution instances, expected 1: "
            f"{instances}. Every arm evolves on the same single instance."
        )
    return int(instances[0])


def candidate_seconds(benchmark: str) -> float:
    return float(protocol()["budgets"][benchmark]["candidate_evaluation_seconds"])


def config_seconds(benchmark: str) -> float:
    """Spherical's budget unit: seconds per (d, N) configuration, not per candidate.

    Distinct from `candidate_seconds` on purpose. The 14 configs are graded
    concurrently, so a candidate's wall clock is one config's budget — reusing the
    per-candidate key here would silently mean a 14x different thing.
    """
    return float(protocol()["budgets"][benchmark]["config_evaluation_seconds"])


def startup_seconds(benchmark: str) -> float:
    """How long the harness gives itself to get a candidate ready to be CALLED.

    A separate key from the search budget because it is a separate thing, paid by the
    harness: spawning the worker and importing what the candidate imports is not search,
    and on a busy box it dwarfed it. Billed to the search it made a candidate's score
    depend on how many rivals happened to be starting alongside it.
    """
    return float(protocol()["budgets"][benchmark]["candidate_startup_seconds"])


def direct_minimum_launch_seconds(benchmark: str) -> float:
    return float(protocol()["budgets"][benchmark]["direct_minimum_launch_seconds"])


def _bytes(allowance: str) -> int:
    number, unit = allowance[:-3], allowance[-3:]
    return int(float(number) * _UNITS[unit])


def sandbox_limits(wall_timeout_s: float) -> SandboxLimits:
    """The same cage for every arm. An unequal sandbox is a confound: a candidate
    given more cores, or a GPU, turns equal wall time into unequal compute."""
    block = protocol()["security"]
    return SandboxLimits(
        wall_timeout_s=wall_timeout_s,
        resident_bytes=_bytes(block["candidate_resident_limit"]),
        cpu_cores=block["cpu_limit"],
        gpu_visible=block["gpu_visible"],
    )
