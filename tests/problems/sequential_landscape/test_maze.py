import numpy as np
import pytest

from problems.sequential_landscape.landscape import InvalidTreeError, Leaf, Node
from problems.sequential_landscape.maze import MazeSpec, build_maze


def spine_with_decoy():
    # spine root -> n1 -> global(-5); at each spine node a deep decoy dead-end.
    # heap: saddle height >= every descendant value.
    deep_decoy = Leaf(-4.0)  # deceptively deep, but a dead end
    inner = Node(0.0, (Leaf(-3.0), Leaf(-5.0)))  # global -5 is here, behind saddle 0
    return Node(2.0, (deep_decoy, inner))


class TestTopology:
    def test_minima_count_and_global(self):
        ls = build_maze(spine_with_decoy(), MazeSpec(dim=3))
        assert ls.num_minima == 3
        assert ls.global_min_value == pytest.approx(-5.0)
        assert ls(ls.global_min_x) == pytest.approx(-5.0, abs=1e-6)

    def test_leaf_points_realize_their_depths(self):
        ls = build_maze(spine_with_decoy(), MazeSpec(dim=3))
        for pt, d in zip(ls.min_points, ls.leaf_depths):
            assert ls(pt) == pytest.approx(d, abs=1e-6)

    def test_heap_violation_rejected(self):
        bad = Node(-1.0, (Leaf(-0.5), Leaf(-2.0)))  # saddle below a child
        with pytest.raises(InvalidTreeError):
            build_maze(bad, MazeSpec())

    def test_global_must_be_unique_minimum(self):
        tied = Node(2.0, (Leaf(-5.0), Leaf(-5.0)))
        with pytest.raises(InvalidTreeError):
            build_maze(tied, MazeSpec())


class TestProgressCredit:
    def test_global_point_scores_full(self):
        ls = build_maze(spine_with_decoy(), MazeSpec(dim=3))
        assert ls.progress(ls.global_min_x) == pytest.approx(1.0)

    def test_decoy_point_scores_partial(self):
        ls = build_maze(spine_with_decoy(), MazeSpec(dim=3))
        # the deep decoy diverges at the root -> credit should be near zero
        decoy_pt = ls.min_points[ls.decoy_index]
        assert ls.progress(decoy_pt) < ls.progress(ls.global_min_x)
        assert ls.progress(decoy_pt) == pytest.approx(0.0, abs=1e-9)


class TestShrinkingGeometry:
    def test_deeper_edges_are_narrower(self):
        ls = build_maze(spine_with_decoy(), MazeSpec(dim=3, shrink_rate=0.5))
        # tube radius (1/sqrt(stiffness)) strictly decreases with edge depth
        radii = ls.tube_radius_by_depth()
        assert all(radii[d + 1] < radii[d] for d in range(len(radii) - 1))

    def test_no_straight_shortcut_beats_the_canyon(self):
        ls = build_maze(spine_with_decoy(), MazeSpec(dim=3))
        # every straight bridge between non-adjacent branches peaks at least as
        # high as the legitimate saddle it would bypass -> no skipping
        assert ls.worst_shortcut_margin() >= 0.0


class TestFloorShape:
    def test_no_spurious_minima_along_true_path(self):
        ls = build_maze(spine_with_decoy(), MazeSpec(dim=3))
        pts = ls.sample_true_path(2000)
        vals = np.array([ls(p) for p in pts])
        # the only interior local minima along the descent are the basins themselves
        is_min = (vals[1:-1] < vals[:-2]) & (vals[1:-1] < vals[2:])
        assert int(is_min.sum()) <= ls.num_minima

    def test_confined_extra_dimensions(self):
        ls = build_maze(spine_with_decoy(), MazeSpec(dim=4))
        base = ls.global_min_x.copy()
        off = base.copy()
        off[3] += 1.0
        assert ls(off) > ls(base)
