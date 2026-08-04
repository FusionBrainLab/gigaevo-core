"""The frozen Cohn warm start: the one configuration all three arms begin from.

This is the most load-bearing input in Experiment A. Every arm starts here, and the
floored fitness is measured against this configuration's mu, so if the coordinates and
the mu ever disagree the whole comparison is scored against a baseline that does not
exist.

They can disagree, because they live in different places. `mu_cohn` is frozen in
`protocol/cohn_snapshot.json`, which is in git. The COORDINATES are
not: they sit in problems/spherical_codes_improver_general/cohn_cache/, which is
generated data, gitignored, and re-downloaded on a miss from a catalogue the manifest
already records as having moved since the freeze (catalogue_snapshot_is_dated_not_current).
A cold cache on run day would silently hand every arm a different warm start than the
mu they are graded against, and nothing downstream would notice.

So the snapshot's per-config `packing_sha256` is enforced as a gate rather than kept as
a record: the cached bytes are hashed and must match, and the mu recomputed from them
must match the frozen mu. A miss is a hard failure, never a re-download — a run that
quietly refetched would produce numbers against an undeclared baseline.

The mu cross-check is also what licenses this module parsing the packings itself
instead of importing the grader's loader: if this parser disagreed with the one that
built the snapshot, the recomputed mu would not match to 1e-12 and the gate would fire.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from problems._harness import REPO
from problems._harness.benchmarks.spherical import mu_of

CACHE_DIR = REPO / "problems/spherical_codes_improver_general/cohn_cache"

_FLOAT = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")

# The recomputed mu must land this close to the frozen one. Both are float64 reductions
# over the same coordinates, so the only spread is summation order: ~1e-16, not 1e-12.
MU_ABS_TOLERANCE = 1e-12


class CatalogueError(RuntimeError):
    """The cached packings do not match the frozen snapshot."""


@dataclass(frozen=True)
class WarmStart:
    dimension: int
    count: int
    points: np.ndarray
    mu_cohn: float


@cache
def snapshot() -> dict:
    from problems._harness.protocol.settings import catalogue_snapshot

    return json.loads(catalogue_snapshot().read_text())


@cache
def _frozen() -> dict[tuple[int, int], dict]:
    return {
        (int(entry["dimension"]), int(entry["count"])): entry
        for entry in snapshot()["configs"]
    }


def _read(path: Path, dimension: int, count: int) -> np.ndarray:
    values = _FLOAT.findall(path.read_text())
    if len(values) != dimension * count:
        raise CatalogueError(
            f"packing (d={dimension}, N={count}): parsed {len(values)} floats, "
            f"expected {dimension * count} — {path} is not the packing it claims to be"
        )
    points = np.asarray([float(v) for v in values], dtype=np.float64).reshape(
        count, dimension
    )
    if not np.all(np.isfinite(points)):
        raise CatalogueError(
            f"packing (d={dimension}, N={count}) has non-finite entries"
        )
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return points / np.where(norms > 0, norms, 1.0)


@cache
def _load(dimension: int, count: int) -> WarmStart:
    entry = _frozen().get((dimension, count))
    if entry is None:
        raise CatalogueError(
            f"(d={dimension}, N={count}) is not in the frozen snapshot. The arms may "
            "only be graded on configurations whose baseline was frozen."
        )

    path = CACHE_DIR / f"packing_{dimension}_{count}.txt"
    if not path.exists():
        raise CatalogueError(
            f"{path} is missing. It is generated data and is NOT re-downloaded here: "
            "the live catalogue has moved since the snapshot was frozen, so a refetch "
            "would warm-start the arms from coordinates the frozen mu does not "
            "describe. Restore the cache from the run archive."
        )

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != entry["packing_sha256"]:
        raise CatalogueError(
            f"packing (d={dimension}, N={count}) does not match the frozen snapshot:\n"
            f"  cached  {digest}\n  frozen  {entry['packing_sha256']}\n"
            "The cache has been refetched from the live catalogue. Every arm would "
            "start from a configuration the frozen mu_cohn does not describe."
        )

    points = _read(path, dimension, count)
    mu = mu_of(points)
    if abs(mu - float(entry["mu_cohn"])) > MU_ABS_TOLERANCE:
        raise CatalogueError(
            f"packing (d={dimension}, N={count}): recomputed mu {mu!r} disagrees with "
            f"the frozen mu {entry['mu_cohn']!r}. The bytes hash correctly, so this is "
            "a reader disagreeing with the one that built the snapshot."
        )
    return WarmStart(dimension, count, points, mu)


def warm_start(dimension: int, count: int) -> WarmStart:
    """The frozen Cohn configuration for one (d, N), verified against the snapshot.

    Every arm calls this and gets the identical array, so the warm start is shared
    information and cannot be what separates them.
    """
    frozen = _load(dimension, count)
    return WarmStart(dimension, count, frozen.points.copy(), frozen.mu_cohn)


def gain(mu: float, mu_cohn: float) -> float:
    """The per-config score: SIGNED relative reduction of mu below the Cohn baseline.

    Not floored. The shipped grader
    (problems/spherical_codes_improver_general/validate.py) floors this at zero, and on
    a warm-started arm that floor is not a floor but a ceiling on the information: the
    incumbent starts AT Cohn and the accept rule is strictly-better, so nothing can ever
    score below the baseline and every arm that fails to improve reads exactly 0.0. An
    arm that missed by 1e-9 and an arm that was never close report the same number, and
    the search has no gradient to climb — on a benchmark where beating Cohn is the hard
    part, that is most of the run.

    Signed, this is the CONTINUOUS EXTENSION of the published score: identical wherever
    the published one is positive, and below the baseline it keeps resolving. It is
    bounded because mu is: the worst valid code has two coincident points, mu = 1, and
    mu_cohn is 0.15 or more everywhere in the snapshot, so the score cannot run away.
    `floored` recovers the published number for continuity.
    """
    return (mu_cohn - mu) / abs(mu_cohn)


def floored(value: float) -> float:
    """The published semantics: non-improvement is worth nothing, never a negative."""
    return max(0.0, value)
