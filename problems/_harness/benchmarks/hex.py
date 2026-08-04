"""Independent HEX-11 verifier: pack 11 unit hexagons, minimize the enclosing one.

Written from the geometry, not from the evolved program's penalty. It shares no
code with `problems/hexagon_improver` on purpose — a verifier that imports the
optimizer's own overlap test cannot catch the optimizer exploiting that test.

Overlap is decided by the separating-axis theorem rather than by edge crossings,
because SAT reports a *penetration depth* and is correct for containment and
coincidence, which an edge-intersection test can miss by construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_ITEMS = 11
UNIT_SIDE = 1.0

# Largest interpenetration still called "touching". Lifted, not re-derived: the
# paper's own overlap test allows 1e-8 of slack at a shared endpoint
# (problems/hexagon_improver/helper.py:38). Optimal packings put hexagons in
# exact contact, so some slack is required either way.
VERIFIER_ABS_TOLERANCE = 1e-8

# The paper's search accepts on strict improvement with no epsilon
# (problems/hexagon_improver/validate.py:230: `if fitness > best_global_fitness`).
# Kept at zero to match it. Every representation is held to the same rule, so a
# noise-sized acceptance cannot favour one of them.
ACCEPTANCE_ABS_TOLERANCE = 0.0

_VERTEX_ANGLES = np.linspace(0.0, 2 * np.pi, 6, endpoint=False)
# Edge normals of a flat-topped regular hexagon: the outer hexagon's three
# distinct constraint directions.
_OUTER_NORMALS = np.stack(
    [
        np.cos(_VERTEX_ANGLES + np.pi / 6),
        np.sin(_VERTEX_ANGLES + np.pi / 6),
    ],
    axis=-1,
)


@dataclass(frozen=True)
class HexReport:
    feasible: bool
    outer_side: float
    max_penetration: float
    worst_pair: tuple[int, int] | None
    reason: str | None = None


def vertices_of(center: np.ndarray, angle: float) -> np.ndarray:
    offsets = _VERTEX_ANGLES + angle
    return np.asarray(center, dtype=float) + UNIT_SIDE * np.stack(
        [np.cos(offsets), np.sin(offsets)], axis=-1
    )


def penetration_depth(poly_a: np.ndarray, poly_b: np.ndarray) -> float:
    """Positive is overlap depth; zero or negative is separation (SAT).

    For convex polygons the minimum overlap over all edge normals is exactly the
    penetration depth, so containment and coincidence give a large positive value
    rather than the false negative an edge-crossing test would return.
    """
    depth = np.inf
    for poly in (poly_a, poly_b):
        edges = np.roll(poly, -1, axis=0) - poly
        axes = np.stack([-edges[:, 1], edges[:, 0]], axis=-1)
        axes /= np.linalg.norm(axes, axis=1, keepdims=True)
        projected_a, projected_b = poly_a @ axes.T, poly_b @ axes.T
        overlaps = np.minimum(projected_a.max(0), projected_b.max(0)) - np.maximum(
            projected_a.min(0), projected_b.min(0)
        )
        depth = min(depth, float(overlaps.min()))
    return depth


def enclosing_side(all_vertices: np.ndarray) -> float:
    """Side of the smallest origin-centred flat-topped regular hexagon containing
    every vertex. Its apothem is side*sqrt(3)/2, so the binding constraint is the
    largest absolute projection onto the three edge normals."""
    extents = np.abs(all_vertices @ _OUTER_NORMALS.T).max(axis=0)
    return float(2.0 * extents.max() / np.sqrt(3.0))


def verify(config, items: int = DEFAULT_ITEMS) -> HexReport:
    # A config is the (centers, angles) tuple every hex prompt declares as the return
    # type, and that problems/hexagon_improver/validate.py:32 already unpacks. Reading
    # it any other way here would fail every program that obeyed its own prompt.
    try:
        raw_centers, raw_angles = config
        centers = np.asarray(raw_centers, dtype=float)
        angles = np.asarray(raw_angles, dtype=float)
    except (TypeError, KeyError, ValueError) as err:
        return HexReport(False, np.inf, np.inf, None, f"unreadable config: {err}")

    if centers.shape != (items, 2) or angles.shape != (items,):
        return HexReport(
            False,
            np.inf,
            np.inf,
            None,
            f"expected {items} centers and angles, "
            f"got {centers.shape} and {angles.shape}",
        )
    if not (np.all(np.isfinite(centers)) and np.all(np.isfinite(angles))):
        return HexReport(False, np.inf, np.inf, None, "non-finite coordinates")

    offsets = angles[:, None] + _VERTEX_ANGLES[None, :]
    polygons = centers[:, None, :] + UNIT_SIDE * np.stack(
        [np.cos(offsets), np.sin(offsets)], axis=-1
    )

    # All-pairs SAT, identical arithmetic to penetration_depth but batched: the
    # per-pair Python loop cost ~12 ms per config, which run_direct's post-budget
    # scoring pass multiplies by every call a fast proposer makes (~300k in one
    # 600 s budget — hours of wall clock per candidate).
    edges = np.roll(polygons, -1, axis=1) - polygons
    axes = np.stack([-edges[..., 1], edges[..., 0]], axis=-1)
    axes /= np.linalg.norm(axes, axis=2, keepdims=True)
    # projected[m, k, v, a] = vertex v of hexagon m on axis a of hexagon k
    projected = np.einsum("mvd,kad->mkva", polygons, axes)
    hi, lo = projected.max(axis=2), projected.min(axis=2)
    # overlap[i, j, a] on i's own axes; depth(i, j) = min over both axis sets
    overlap = np.minimum(
        np.einsum("iia->ia", hi)[:, None, :], np.swapaxes(hi, 0, 1)
    ) - np.maximum(np.einsum("iia->ia", lo)[:, None, :], np.swapaxes(lo, 0, 1))
    own_axis_depth = overlap.min(axis=2)
    pair_depth = np.minimum(own_axis_depth, own_axis_depth.T)

    upper = np.triu_indices(items, k=1)
    if upper[0].size:
        flat = np.argmax(pair_depth[upper])
        worst_depth = float(pair_depth[upper][flat])
        worst_pair = (int(upper[0][flat]), int(upper[1][flat]))
    else:
        worst_depth, worst_pair = -np.inf, None

    side = enclosing_side(polygons.reshape(-1, 2))
    overlapping = worst_depth > VERIFIER_ABS_TOLERANCE
    return HexReport(
        feasible=not overlapping,
        outer_side=side,
        max_penetration=worst_depth,
        worst_pair=worst_pair,
        reason=f"hexagons {worst_pair} interpenetrate by {worst_depth:.3e}"
        if overlapping
        else None,
    )


@dataclass(frozen=True)
class HexBenchmark:
    """The controller's view: objective if feasible, None if not.

    `items` is a parameter because Experiment A's transfer study evaluates the
    same evolved program at n = 12..17, and a verifier hardcoded to 11 would
    silently reject every transfer config as the wrong shape.
    """

    items: int = DEFAULT_ITEMS

    def validate(self, config) -> float | None:
        report = verify(config, self.items)
        return report.outer_side if report.feasible else None

    def better(self, candidate: float, incumbent: float) -> bool:
        return candidate < incumbent - ACCEPTANCE_ABS_TOLERANCE
