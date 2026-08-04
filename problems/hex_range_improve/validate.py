"""ImprovEvolve arm of the matched range-generalization study (Exp G):
generate_config / perturb / improve, graded independently at every train size.

The grading lives in problems._harness.benchmarks.hex_range_grading, shared with the
basic arm. The only thing this file chooses is the adapter.
"""

from problems._harness.benchmarks.hex_range_grading import grade_range_controller
from problems._harness.common.adapters import ModularAdapter


def validate(program_class):
    return grade_range_controller(ModularAdapter, program_class)
