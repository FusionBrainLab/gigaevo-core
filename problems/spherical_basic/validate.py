"""DirectEvolve arm of the matched spherical study: repeated seeded solve() calls.

The grading lives in problems._harness.benchmarks.spherical_grading, shared with the improve and
mixed arms. `None` selects the direct path — run_direct rather than the two-stage
controller — under the same warm start, the same wall clock, the same sandbox, the same
seed schedule and the same acceptance rule.

Unlike the hex basic arm, this one is NOT cold. The spherical grader floors its score at
the Cohn baseline, so a from-scratch solver would score 0.0 on every configuration in
every run and the arm would be unmeasurable rather than merely weak. It is handed the
same warm start as the other two arms and given the whole budget on top of it; what
still differs is only who runs the outer search loop.
"""

from problems._harness.benchmarks.spherical_grading import grade


def validate(program_class):
    return grade(None, program_class)
