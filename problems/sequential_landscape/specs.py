"""The graded difficulty ladder of hand-authored landscapes (easy -> hard).

Each Instance is frozen and reproducible. The breakdown study runs optimizers
down this ladder and reports where success collapses.
"""

from __future__ import annotations

from dataclasses import dataclass

from problems.sequential_landscape.landscape import (
    GeometrySpec,
    Landscape,
    Leaf,
    Node,
    _rep,
    build,
    build_sequence,
)
from problems.sequential_landscape.maze import MazeLandscape, MazeSpec, build_maze


def _chain(k: int, drop: float = 1.0, bump: float = 0.5, start: float = 0.0):
    depths = [start - i * drop for i in range(k)]
    barriers = [depths[i] + bump for i in range(k - 1)]
    return depths, barriers


def _balanced(depths, gap: float = 1.0):
    nodes = [Leaf(d) for d in depths]
    while len(nodes) > 1:
        nxt = []
        for i in range(0, len(nodes) - 1, 2):
            a, b = nodes[i], nodes[i + 1]
            nxt.append(Node(max(_rep(a), _rep(b)) + gap, (a, b)))
        if len(nodes) % 2 == 1:
            nxt.append(nodes[-1])
        nodes = nxt
    return nodes[0]


@dataclass(frozen=True)
class Instance:
    name: str
    budget: int
    geom: GeometrySpec
    _builder: object

    def landscape(self) -> Landscape:
        return self._builder()


def _seq(depths, barriers, geom):
    return lambda: build_sequence(depths, barriers, geom)


def _tree(node, geom):
    return lambda: build(node, geom)


def _double_funnel():
    left = _balanced([-1.0, -2.0, -1.5, -2.5], gap=0.5)
    right = _balanced([-3.0, -4.0, -3.5, -5.0], gap=0.5)
    root = Node(max(_rep(left), _rep(right)) + 4.0, (left, right))
    return root


def get_ladder() -> list[Instance]:
    cat3 = _chain(3)
    cat7 = _chain(7)
    cat12 = _chain(12)
    cat20 = _chain(20)
    balanced8 = _balanced([-1, -2, -1.5, -3, -2.5, -5, -2, -4], gap=1.0)
    return [
        Instance(
            "caterpillar_3", 300, GeometrySpec(dim=2), _seq(*cat3, GeometrySpec(dim=2))
        ),
        Instance(
            "caterpillar_7", 700, GeometrySpec(dim=2), _seq(*cat7, GeometrySpec(dim=2))
        ),
        Instance(
            "caterpillar_12",
            1500,
            GeometrySpec(dim=5),
            _seq(*cat12, GeometrySpec(dim=5)),
        ),
        Instance(
            "balanced_8",
            1200,
            GeometrySpec(dim=3),
            _tree(balanced8, GeometrySpec(dim=3)),
        ),
        Instance(
            "double_funnel_10",
            1500,
            GeometrySpec(dim=4),
            _tree(_double_funnel(), GeometrySpec(dim=4)),
        ),
        Instance(
            "deep_caterpillar_20",
            3000,
            GeometrySpec(dim=8),
            _seq(*cat20, GeometrySpec(dim=8)),
        ),
    ]


def _spine_maze(length, branch=2, gap=0.5, global_val=-10.0, decoy_val=-8.0):
    """Spine root..global of `length` saddles; each spine node also spawns
    `branch-1` deceptively-deep decoy dead-ends that strand greedy followers."""
    node = Leaf(global_val)
    for i in range(length):
        decoys = tuple(Leaf(decoy_val - 0.1 * j) for j in range(branch - 1))
        node = Node((i + 1) * gap, (*decoys, node))
    return node


@dataclass(frozen=True)
class MazeInstance:
    name: str
    budget: int
    spec: MazeSpec
    _tree: object

    def landscape(self) -> MazeLandscape:
        return build_maze(self._tree, self.spec)


def get_maze_ladder() -> list[MazeInstance]:
    return [
        MazeInstance(
            "maze_easy",
            2500,
            MazeSpec(
                dim=7,
                shrink_rate=0.55,
                switchbacks=4,
                turn_jitter=0.9,
                lane_base=16.0,
                base_stiffness=8.0,
            ),
            _spine_maze(8, branch=2),
        ),
        MazeInstance(
            "maze_medium",
            3500,
            MazeSpec(
                dim=8,
                shrink_rate=0.55,
                switchbacks=5,
                turn_jitter=0.9,
                lane_base=16.0,
                base_stiffness=8.0,
            ),
            _spine_maze(9, branch=3),
        ),
        MazeInstance(
            "maze_hard",
            5000,
            MazeSpec(
                dim=10,
                shrink_rate=0.5,
                switchbacks=5,
                turn_jitter=0.9,
                lane_base=16.0,
                base_stiffness=8.0,
            ),
            _spine_maze(10, branch=3),
        ),
        MazeInstance(
            "maze_insane",
            7000,
            MazeSpec(
                dim=12,
                shrink_rate=0.45,
                switchbacks=6,
                turn_jitter=1.0,
                lane_base=16.0,
                base_stiffness=8.0,
            ),
            _spine_maze(12, branch=4),
        ),
    ]
