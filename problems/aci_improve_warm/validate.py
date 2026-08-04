"""Warm-started ImprovEvolve arm of the ACI study: the 10th, best-effort run.

Grading is byte-identical to problems/aci_improve (same grade_controller /
ModularAdapter path); the ONLY difference between this package and aci_improve is
initial_programs/ -- here the archive is seeded with strong pre-existing ImprovEvolve
programs instead of the cold ones() floor. This package is NOT part of the matched
9-run ablation; it exists so the strong run can warm-start without perturbing the
cold aci_improve package the ablation shares.
"""

from problems._harness.benchmarks.aci_grading import grade_controller
from problems._harness.common.adapters import ModularAdapter


def validate(program_class):
    return grade_controller(ModularAdapter, program_class)
