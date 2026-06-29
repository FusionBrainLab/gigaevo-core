"""Insane maze landscape.

Embeds a barrier *tree* (internal nodes = saddles, leaves = minima) as a winding,
branching canyon in 2D plus confined nuisance dimensions, with footprints and tube
radii that shrink geometrically with tree depth. The global minimum is an unsamplable
needle reachable only by sequential discovery down the correct branch, with deep decoy
branches that strand depth-first followers and force backtracking.

Credit is graph-path-based (progress down the unique root->global path), because with
deceptive deep decoys a value-based score is gameable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import PchipInterpolator

from problems.sequential_landscape.landscape import (
    InvalidTreeError,
    Leaf,
    _rep,
    _validate_heap,
)

NO_SKIP_FACTOR = 5.0


@dataclass(frozen=True)
class MazeSpec:
    dim: int = 4
    shrink_rate: float = 0.6
    lane_base: float = 12.0
    row_gap: float = 10.0
    switchbacks: int = 4
    turn_jitter: float = 0.7
    base_stiffness: float = 4.0
    seed: int = 0


@dataclass
class _N:
    id: int
    value: float
    is_leaf: bool
    children: list = field(default_factory=list)
    parent: int = -1
    depth: int = 0
    path: tuple = ()
    x: float = 0.0
    y: float = 0.0


def _flatten(tree) -> list[_N]:
    nodes: list[_N] = []

    def walk(t, parent, depth, prefix):
        nid = len(nodes)
        n = _N(nid, _rep(t), isinstance(t, Leaf), [], parent, depth, prefix + (nid,))
        nodes.append(n)
        if not n.is_leaf:
            for c in t.children:
                n.children.append(walk(c, nid, depth + 1, n.path))
        return nid

    walk(tree, -1, 0, ())
    return nodes


class MazeLandscape:
    def __init__(self, nodes: list[_N], spec: MazeSpec):
        self.dim = spec.dim
        self.spec = spec
        self._nodes = nodes

        leaves = [n for n in nodes if n.is_leaf]
        depths = [n.value for n in leaves]
        gi = int(np.argmin(depths))
        if depths.count(min(depths)) > 1:
            raise InvalidTreeError("global minimum must be a unique strict minimum")
        self._global = leaves[gi]
        self._global_depth = len(self._global.path) - 1

        self._assign_positions(spec)
        self._build_edges(spec)

        self.num_minima = len(leaves)
        self.leaf_depths = [n.value for n in leaves]
        self.min_points = [self._lift(np.array([n.x, n.y])) for n in leaves]
        self.barriers = [n.value for n in nodes if not n.is_leaf]
        self.global_min_value = float(self._global.value)
        self.global_min_x = self._lift(np.array([self._global.x, self._global.y]))

        progs = [self.progress(p) for p in self.min_points]
        self.decoy_index = int(np.argmin(progs))

        pts = np.vstack([e["pts"] for e in self._edges])
        margin = spec.lane_base
        lo2, hi2 = pts.min(axis=0), pts.max(axis=0)
        self.bounds = [
            (float(lo2[0] - margin), float(hi2[0] + margin)),
            (float(lo2[1] - margin), float(hi2[1] + margin)),
        ] + [(-margin, margin)] * (self.dim - 2)

    def _slot(self, depth: int) -> float:
        # constant lane spacing keeps non-adjacent branches separated; basin
        # footprints shrink with depth through the tube radius (stiffness), not
        # through crowding the lanes together.
        return self.spec.lane_base

    def _stiffness(self, depth: int) -> float:
        return self.spec.base_stiffness / self.spec.shrink_rate ** (2 * depth)

    def _lift(self, p2d) -> np.ndarray:
        full = np.zeros(self.dim)
        full[:2] = p2d
        return full

    def _assign_positions(self, spec: MazeSpec):
        cursor = [0.0]

        def place(n: _N):
            n.y = n.depth * spec.row_gap
            if n.is_leaf:
                n.x = cursor[0]
                cursor[0] += self._slot(n.depth)
            else:
                for c in n.children:
                    place(self._nodes[c])
                n.x = float(np.mean([self._nodes[c].x for c in n.children]))

        place(self._nodes[0])

    def _build_edges(self, spec: MazeSpec):
        self._edges = []
        seg_a, seg_b, seg_edge, seg_s0, seg_len = [], [], [], [], []
        for n in self._nodes:
            if n.parent < 0:
                continue
            p = self._nodes[n.parent]
            poly = self._wind(
                np.array([p.x, p.y]), np.array([n.x, n.y]), n.depth, n.id, spec
            )
            seg = np.diff(poly, axis=0)
            slen = np.linalg.norm(seg, axis=1)
            arctot = float(slen.sum())
            s_cum = np.concatenate([[0.0], np.cumsum(slen)])
            pchip = PchipInterpolator(
                np.array([0.0, arctot]), np.array([p.value, n.value])
            )
            ei = len(self._edges)
            for j in range(len(seg)):
                seg_a.append(poly[j])
                seg_b.append(poly[j + 1])
                seg_edge.append(ei)
                seg_s0.append(s_cum[j])
                seg_len.append(max(slen[j], 1e-12))
            self._edges.append(
                {
                    "pts": poly,
                    "pchip": pchip,
                    "depth": n.depth,
                    "child": n.id,
                    "endpoints": {n.parent, n.id},
                }
            )
        self._A = np.array(seg_a)
        self._B = np.array(seg_b)
        self._edge = np.array(seg_edge)
        self._s0 = np.array(seg_s0)
        self._L = np.array(seg_len)

    def _wind(self, p, c, depth, child_id, spec: MazeSpec):
        d = c - p
        L = float(np.linalg.norm(d))
        if L < 1e-9 or spec.switchbacks <= 0:
            return np.array([p, c])
        u = d / L
        perp = np.array([-u[1], u[0]])
        amp = min(spec.turn_jitter * self._slot(depth), self._slot(depth) / 4.0)
        rng = np.random.default_rng(spec.seed * 1000 + child_id)
        pts = [p]
        k = spec.switchbacks
        for i in range(1, k + 1):
            frac = i / (k + 1)
            off = perp * amp * float(rng.uniform(-1.0, 1.0))
            pts.append(p + frac * d + off)
        pts.append(c)
        return np.array(pts)

    def _project(self, p2d):
        ab = self._B - self._A
        t = np.clip(
            np.sum((p2d - self._A) * ab, axis=1) / np.sum(ab * ab, axis=1), 0.0, 1.0
        )
        foot = self._A + t[:, None] * ab
        d2 = np.sum((p2d - foot) ** 2, axis=1)
        i = int(np.argmin(d2))
        ei = int(self._edge[i])
        s = self._s0[i] + t[i] * self._L[i]
        return ei, float(s), float(d2[i])

    def __call__(self, x) -> float:
        x = np.asarray(x, dtype=float)
        ei, s, perp2 = self._project(x[:2])
        e = self._edges[ei]
        extra2 = float(np.sum(x[2:] ** 2)) if self.dim > 2 else 0.0
        return float(e["pchip"](s)) + self._stiffness(e["depth"]) * (perp2 + extra2)

    def progress(self, x) -> float:
        x = np.asarray(x, dtype=float)
        ei, _, _ = self._project(x[:2])
        child = self._nodes[self._edges[ei]["child"]]
        common = 0
        for a, b in zip(child.path, self._global.path):
            if a != b:
                break
            common += 1
        lca_depth = common - 1
        return max(0.0, lca_depth) / self._global_depth

    def tube_radius_by_depth(self) -> list[float]:
        depths = sorted({e["depth"] for e in self._edges})
        return [
            self.spec.shrink_rate**d / self.spec.base_stiffness**0.5 for d in depths
        ]

    def _lca_height(self, a: _N, b: _N) -> float:
        common = self._nodes[0]
        for x, y in zip(a.path, b.path):
            if x != y:
                break
            common = self._nodes[x]
        return common.value

    def worst_shortcut_margin(self) -> float:
        """How much higher the straight-bridge peak between two non-adjacent
        branches sits above the legitimate canyon saddle (their LCA) you'd
        otherwise climb. >= 0 everywhere means no shortcut is ever cheaper than
        traversing the maze, i.e. no skipping."""
        worst = np.inf
        for i, ei in enumerate(self._edges):
            ci = self._nodes[ei["child"]]
            for ej in self._edges[i + 1 :]:
                if ei["endpoints"] & ej["endpoints"]:
                    continue
                diff = ei["pts"][:, None, :] - ej["pts"][None, :, :]
                d2 = (diff**2).sum(axis=2)
                a, b = np.unravel_index(int(d2.argmin()), d2.shape)
                bridge = np.linspace(ei["pts"][a], ej["pts"][b], 11)
                peak = max(self(self._lift(p)) for p in bridge)
                lca = self._lca_height(ci, self._nodes[ej["child"]])
                worst = min(worst, peak - lca)
        return worst

    def true_path_points(self) -> list[np.ndarray]:
        return [
            self._lift(np.array([self._nodes[i].x, self._nodes[i].y]))
            for i in self._global.path
        ]

    def sample_true_path(self, n: int) -> list[np.ndarray]:
        path_ids = set(self._global.path[1:])
        polys = [e["pts"] for e in self._edges if e["child"] in path_ids]
        order = {e["child"]: e["pts"] for e in self._edges if e["child"] in path_ids}
        ordered = [order[i] for i in self._global.path[1:]]
        line = np.vstack(ordered) if ordered else polys[0]
        idx = np.linspace(0, len(line) - 1, n).astype(int)
        return [self._lift(line[i]) for i in idx]


def build_maze(tree, spec: MazeSpec | None = None) -> MazeLandscape:
    spec = spec or MazeSpec()
    if spec.dim < 2:
        raise InvalidTreeError("dim must be >= 2 for a maze embedding")
    _validate_heap(tree)
    nodes = _flatten(tree)
    return MazeLandscape(nodes, spec)
