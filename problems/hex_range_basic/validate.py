"""DirectEvolve arm of the matched range-generalization study (Exp G):
repeated seeded solve() calls, graded independently at every train size.

The grading lives in problems._harness.benchmarks.hex_range_grading, shared with the
improve arm — same per-size wall clock, sandbox, seed schedule and acceptance
rule. This arm has no incumbent, so each size runs the direct evaluator.
"""

from problems._harness.benchmarks.hex_range_grading import grade_range_direct


def validate(program_class):
    return grade_range_direct(program_class)
