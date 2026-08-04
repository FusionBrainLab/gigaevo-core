"""UnifiedEvolve arm of the matched spherical study: one propose() operator.

The grading lives in problems._harness.benchmarks.spherical_grading, shared with the improve and
basic arms. The only thing this file chooses is the adapter.
"""

from problems._harness.benchmarks.spherical_grading import grade
from problems._harness.common.adapters import UnifiedAdapter


def validate(program_class):
    return grade(UnifiedAdapter, program_class)
