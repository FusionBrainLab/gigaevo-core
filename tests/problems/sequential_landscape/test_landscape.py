import numpy as np
import pytest

from problems.sequential_landscape.landscape import (
    GeometrySpec,
    InvalidTreeError,
    Leaf,
    Node,
    build,
    compile_tree,
)


def caterpillar():
    # depths [-1, -2, -3], barriers [2, 1]; deepest leaf is last (in-order)
    return Node(2.0, (Leaf(-1.0), Node(1.0, (Leaf(-2.0), Leaf(-3.0)))))


def double_funnel():
    # global minimum (-5) hidden in the SECOND funnel, behind a tall central wall
    left = Node(0.0, (Leaf(-1.0), Leaf(-2.0)))
    right = Node(0.0, (Leaf(-3.0), Leaf(-5.0)))
    return Node(4.0, (left, right))


class TestCompileTree:
    def test_inorder_depths_and_barriers(self):
        depths, barriers = compile_tree(caterpillar())
        assert depths == [-1.0, -2.0, -3.0]
        assert barriers == [2.0, 1.0]

    def test_barriers_count_is_one_less_than_minima(self):
        depths, barriers = compile_tree(double_funnel())
        assert len(barriers) == len(depths) - 1

    def test_single_leaf(self):
        depths, barriers = compile_tree(Leaf(-1.0))
        assert depths == [-1.0]
        assert barriers == []

    def test_nary_node_emits_repeated_height_between_blocks(self):
        # three children merging at the same height -> two equal barriers
        depths, barriers = compile_tree(Node(3.0, (Leaf(-1.0), Leaf(-2.0), Leaf(-3.0))))
        assert depths == [-1.0, -2.0, -3.0]
        assert barriers == [3.0, 3.0]


class TestHeapValidation:
    def test_child_node_taller_than_parent_rejected(self):
        bad = Node(1.0, (Leaf(0.0), Node(2.0, (Leaf(-1.0), Leaf(-1.5)))))
        with pytest.raises(InvalidTreeError):
            build(bad)

    def test_barrier_not_above_adjacent_minimum_rejected(self):
        # barrier height -1.0 is below the -0.5 minimum next to it -> not a real saddle
        bad = Node(-1.0, (Leaf(-0.5), Leaf(-2.0)))
        with pytest.raises(InvalidTreeError):
            build(bad)


class TestRealizedFloor:
    def test_minima_match_prescribed_depths(self):
        ls = build(caterpillar())
        assert ls.num_minima == 3
        for pt, depth in zip(ls.min_points, ls.depths):
            assert ls(pt) == pytest.approx(depth, abs=1e-6)

    def test_barrier_peaks_match_prescribed_heights(self):
        ls = build(caterpillar())
        # floor evaluated at the arclength midpoint between consecutive minima
        for i, b in enumerate(ls.barriers):
            s_mid = 0.5 * (ls.min_arclengths[i] + ls.min_arclengths[i + 1])
            assert ls.floor(s_mid) == pytest.approx(b, abs=1e-6)

    def test_floor_has_no_spurious_extrema(self):
        ls = build(caterpillar())
        ss = np.linspace(0.0, ls.min_arclengths[-1], 4000)
        g = np.array([ls.floor(s) for s in ss])
        # count interior local maxima of the sampled floor == number of barriers
        is_peak = (g[1:-1] > g[:-2]) & (g[1:-1] > g[2:])
        assert int(is_peak.sum()) == len(ls.barriers)

    def test_global_min_is_true_argmin(self):
        ls = build(caterpillar())
        assert ls.global_min_value == pytest.approx(-3.0, abs=1e-6)
        assert ls(ls.global_min_x) == pytest.approx(-3.0, abs=1e-6)
        # deepest leaf, sampled densely, is never beaten
        rng = np.random.default_rng(0)
        for _ in range(2000):
            x = np.array([rng.uniform(lo, hi) for lo, hi in ls.bounds])
            assert ls(x) >= ls.global_min_value - 1e-6


class TestGeometry:
    def test_basin_of_recovers_minimum_index(self):
        ls = build(caterpillar())
        for i, pt in enumerate(ls.min_points):
            assert ls.basin_of(pt) == i

    def test_no_skip_wall_dwarfs_along_canyon_barriers(self):
        ls = build(caterpillar(), GeometrySpec(arm_gap=3.0))
        max_span = max(ls.barriers) - ls.global_min_value
        assert ls.wall_height >= 4.0 * max_span

    def test_dimension_and_bounds(self):
        ls = build(caterpillar(), GeometrySpec(dim=5))
        assert ls.dim == 5
        assert len(ls.bounds) == 5
        assert ls(ls.global_min_x) == pytest.approx(-3.0, abs=1e-6)

    def test_extra_dimensions_are_confined(self):
        ls = build(caterpillar(), GeometrySpec(dim=4))
        base = ls.global_min_x.copy()
        off = base.copy()
        off[2] += 1.0  # step out along a confined dimension
        assert ls(off) > ls(base)


class TestMergeHeights:
    def test_merge_height_is_max_barrier_between_leaves(self):
        ls = build(caterpillar())
        assert ls.merge_height(0, 1) == pytest.approx(2.0)
        assert ls.merge_height(1, 2) == pytest.approx(1.0)
        assert ls.merge_height(0, 2) == pytest.approx(2.0)

    def test_double_funnel_global_behind_tall_wall(self):
        ls = build(double_funnel())
        assert ls.global_min_value == pytest.approx(-5.0)
        # the two shallow-side minima merge with the deep side only at the tall wall
        assert ls.merge_height(0, 3) == pytest.approx(4.0)
        assert ls.merge_height(2, 3) == pytest.approx(0.0)
