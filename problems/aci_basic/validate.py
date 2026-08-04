"""DirectEvolve arm of the matched ACI study: repeated seeded solve() calls.

The grading lives in problems._harness.benchmarks.aci_grading, shared with the improve and
mixed arms. This arm has no incumbent, so it runs run_direct rather than the
two-stage controller -- under the same wall clock, the same sandbox, the same seed
schedule and the same acceptance rule.
"""

from problems._harness.benchmarks.aci_grading import grade_direct


def validate(program_class):
    return grade_direct(program_class)
