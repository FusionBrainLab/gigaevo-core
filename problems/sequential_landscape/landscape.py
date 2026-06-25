"""Sequential-landscape generator.

Realizes an optimization landscape whose disconnectivity (barrier) tree is
prescribed by hand, so that deep minima are reachable only by descending the
chain of local minima in order. Topology is set by a 1-D floor (the barrier
tree); geometry (dimension, no-Euclidean-shortcut) is set by a winding
serpentine embedding. The two are decoupled.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import PchipInterpolator

SKIP_FACTOR = 5.0


class InvalidTreeError(ValueError):
    """Raised when a barrier tree violates the heap (monotone-saddle) property."""


@dataclass(frozen=True)
class Leaf:
    depth: float


@dataclass(frozen=True)
class Node:
    height: float
    children: tuple = field(default_factory=tuple)


@dataclass(frozen=True)
class GeometrySpec:
    dim: int = 2
    arm_gap: float = 3.0
    well_spacing: float = 1.0
    wall_stiffness: float | None = None
    row_length: int | None = None
    seed: int = 0


def _rep(node) -> float:
    return node.depth if isinstance(node, Leaf) else node.height


def _validate_heap(node) -> None:
    if isinstance(node, Leaf):
        return
    for child in node.children:
        if _rep(child) > node.height:
            raise InvalidTreeError(
                f"saddle height {node.height} below descendant level {_rep(child)}"
            )
        _validate_heap(child)


def compile_tree(node) -> tuple[list[float], list[float]]:
    """In-order traversal -> (minima depths, barriers between consecutive minima).

    The barrier between two minima equals the height of their lowest common
    ancestor (the Cartesian-tree bijection); an n-ary node emits its height
    once between each pair of adjacent child blocks.
    """
    if isinstance(node, Leaf):
        return [node.depth], []
    depths: list[float] = []
    barriers: list[float] = []
    for i, child in enumerate(node.children):
        cd, cb = compile_tree(child)
        if i > 0:
            barriers.append(node.height)
        barriers.extend(cb)
        depths.extend(cd)
    return depths, barriers


def _serpentine(num: int, well_spacing: float, arm_gap: float, row_length: int):
    pts = []
    for i in range(num):
        row, col = divmod(i, row_length)
        x = col if row % 2 == 0 else (row_length - 1 - col)
        pts.append((x * well_spacing, row * arm_gap))
    return np.array(pts, dtype=float)


class Landscape:
    def __init__(self, depths, barriers, geom: GeometrySpec):
        if geom.dim < 2:
            raise InvalidTreeError("dim must be >= 2 for a serpentine embedding")
        self.depths = list(depths)
        self.barriers = list(barriers)
        self.dim = geom.dim
        self.arm_gap = geom.arm_gap
        self.num_minima = len(depths)

        k = self.num_minima
        row_length = geom.row_length or max(2, round(k**0.5))
        self._vertices2d = _serpentine(k, geom.well_spacing, geom.arm_gap, row_length)

        seg = np.diff(self._vertices2d, axis=0)
        seg_len = np.linalg.norm(seg, axis=1)
        self.min_arclengths = np.concatenate([[0.0], np.cumsum(seg_len)])
        self._seg = seg
        self._seg_len2 = np.maximum(seg_len**2, 1e-18)

        xs, ys = [], []
        for i in range(k):
            xs.append(self.min_arclengths[i])
            ys.append(depths[i])
            if i < k - 1:
                xs.append(0.5 * (self.min_arclengths[i] + self.min_arclengths[i + 1]))
                ys.append(barriers[i])
        self._floor = PchipInterpolator(np.array(xs), np.array(ys))
        self._s_max = float(self.min_arclengths[-1])

        max_span = (max(barriers) - min(depths)) if barriers else 0.0
        if geom.wall_stiffness is not None:
            self.wall_stiffness = geom.wall_stiffness
        else:
            self.wall_stiffness = SKIP_FACTOR * max_span / (geom.arm_gap / 2.0) ** 2
        self.wall_height = self.wall_stiffness * (geom.arm_gap / 2.0) ** 2

        self.min_points = []
        for p in self._vertices2d:
            full = np.zeros(self.dim)
            full[:2] = p
            self.min_points.append(full)

        gi = int(np.argmin(depths))
        self.global_min_value = float(depths[gi])
        self.global_min_x = self.min_points[gi]

        self._barrier_arclengths = np.array(
            [
                0.5 * (self.min_arclengths[i] + self.min_arclengths[i + 1])
                for i in range(k - 1)
            ]
        )

        margin = geom.arm_gap
        xmin, ymin = self._vertices2d.min(axis=0)
        xmax, ymax = self._vertices2d.max(axis=0)
        self.bounds = [
            (float(xmin - margin), float(xmax + margin)),
            (float(ymin - margin), float(ymax + margin)),
        ] + [(-margin, margin)] * (self.dim - 2)

    def _project(self, x2d):
        a = self._vertices2d[:-1]
        t = np.clip(np.sum((x2d - a) * self._seg, axis=1) / self._seg_len2, 0.0, 1.0)
        foot = a + t[:, None] * self._seg
        d2 = np.sum((x2d - foot) ** 2, axis=1)
        j = int(np.argmin(d2))
        s = self.min_arclengths[j] + t[j] * np.sqrt(self._seg_len2[j])
        return float(s), float(d2[j])

    def floor(self, s: float) -> float:
        return float(self._floor(np.clip(s, 0.0, self._s_max)))

    def __call__(self, x) -> float:
        x = np.asarray(x, dtype=float)
        s, perp2 = self._project(x[:2])
        extra2 = float(np.sum(x[2:] ** 2)) if self.dim > 2 else 0.0
        return self.floor(s) + self.wall_stiffness * (perp2 + extra2)

    def basin_of(self, x) -> int:
        x = np.asarray(x, dtype=float)
        s, _ = self._project(x[:2])
        return int(np.searchsorted(self._barrier_arclengths, s))

    def merge_height(self, i: int, j: int) -> float:
        lo, hi = sorted((i, j))
        return float(max(self.barriers[lo:hi]))


def build_sequence(depths, barriers, geom: GeometrySpec | None = None) -> Landscape:
    geom = geom or GeometrySpec()
    if len(barriers) != len(depths) - 1:
        raise InvalidTreeError(
            f"need exactly {len(depths) - 1} barriers for {len(depths)} minima"
        )
    for i, b in enumerate(barriers):
        if b <= depths[i] or b <= depths[i + 1]:
            raise InvalidTreeError(
                f"barrier {b} not strictly above adjacent minima "
                f"({depths[i]}, {depths[i + 1]})"
            )
    return Landscape(list(depths), list(barriers), geom)


def build(tree, geom: GeometrySpec | None = None) -> Landscape:
    _validate_heap(tree)
    depths, barriers = compile_tree(tree)
    return build_sequence(depths, barriers, geom)
