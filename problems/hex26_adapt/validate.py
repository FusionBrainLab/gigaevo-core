"""Expert-hint adaptation study: HEX n=26 from p0.

Same grading as `problems/hex23_*`, retargeted to n=26 — one of two instances
where a gap to the published +E value survives equal clock. The 600 s candidate
budget is the study constant set from the n=26 saturation probe; every one of
the four conditions runs at this identical budget, so the override is a study
constant, not a knob.
"""

from problems._harness.benchmarks.hex_grading import grade_controller
from problems._harness.common.adapters import ModularAdapter


def validate(program_class):
    return grade_controller(ModularAdapter, program_class, items=26, seconds=600)
