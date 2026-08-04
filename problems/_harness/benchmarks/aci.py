"""Independent ACI optimizer benchmark: maximize C = ||f*f||_2^2 / (||f*f||_1 ||f*f||_inf).

The controller's view of the second autocorrelation inequality as a search target.
Config is a length-N nonnegative array f on a 1-D grid; the objective is C(f), and a
run maximizes it from the trivial constant seed (C=2/3) toward the AlphaEvolve record
(C=0.96102). This is the ACI analogue of `benchmarks/hex.py::HexBenchmark`.

The C convention and its validity checks are LIFTED from `verifier.py` (co-located)
(calibrated 2026-07-18 against both published anchors to ~1e-15) rather than re-derived:
a second convention here could silently disagree with the record the arms chase, and a
verifier that shared the optimizer's own metric could be gamed by it. `compute_c`
returns None where `verifier.verify` raises, because the controller's contract is
objective-or-None and an infeasible proposal is a normal event in a search, not an error.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from problems._harness.benchmarks.verifier import (
    MIN_RESOLUTION,
    NONNEG_TOL,
    autoconvolution,
    metrics_from_conv,
)

DEFAULT_ITEMS = 50000

# The published search accepts on strict improvement with no epsilon; hex and spherical
# hardcode theirs the same way. This is deliberately NOT
# benchmarks.aci.acceptance_abs_tolerance (1.56e-4) from the manifest -- that is expD's
# mechanism-control gate, a different experiment. Zero here, applied to every arm
# identically, so a noise-sized acceptance cannot favour one interface.
ACCEPTANCE_ABS_TOLERANCE = 0.0


def compute_c(config, items: int) -> float | None:
    """C if `config` is a valid length-`items` nonnegative nontrivial array, else None.

    Mirrors the checks in `verifier.verify` but returns None instead of raising. The
    resolution floor is kept even though `items` is fixed at evolution time so that a
    test-time sweep to a tiny n cannot silently produce a meaningless C.
    """
    try:
        f = np.asarray(config, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if f.ndim != 1 or f.shape[0] != items:
        return None
    if items < MIN_RESOLUTION:
        return None
    if not np.all(np.isfinite(f)):
        return None
    if float(f.min()) < NONNEG_TOL:
        return None
    if float(f.max()) <= 0.0:
        return None
    f = np.maximum(f, 0.0)
    c = metrics_from_conv(autoconvolution(f))["c"]
    if not np.isfinite(c):
        return None
    return float(c)


@dataclass(frozen=True)
class ACIBenchmark:
    """The controller's view: C if feasible, None if not.

    `items` is the grid resolution N and a parameter for the same reason hex's is:
    the champion is resolution-swept at test time, and a benchmark hardcoded to 50000
    would silently reject every swept config as the wrong shape.
    """

    items: int = DEFAULT_ITEMS

    def validate(self, config) -> float | None:
        return compute_c(config, self.items)

    def better(self, candidate: float, incumbent: float) -> bool:
        return candidate > incumbent + ACCEPTANCE_ABS_TOLERANCE
