"""UnifiedEvolve arm of the matched HEX-23 study: one propose() operator.

The grading lives in problems._harness.benchmarks.hex_grading, shared with the improve and
basic arms. The only thing this file chooses is the adapter.
"""

from problems._harness.benchmarks.hex_grading import grade_controller
from problems._harness.common.adapters import UnifiedAdapter


def validate(program_class):
    return grade_controller(UnifiedAdapter, program_class)
