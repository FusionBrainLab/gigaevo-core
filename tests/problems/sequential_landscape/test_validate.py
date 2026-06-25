import numpy as np

from problems.sequential_landscape.landscape import GeometrySpec, build_sequence
from problems.sequential_landscape.specs import get_ladder
from problems.sequential_landscape.validate import _instance_score, validate


def _ls():
    return build_sequence([0.0, -1.0, -2.0], [0.5, -0.5], GeometrySpec(dim=2))


class TestInstanceScore:
    def test_global_scores_one(self):
        ls = _ls()
        assert _instance_score(ls, ls.global_min_x) == 1.0

    def test_shallowest_minimum_scores_zero(self):
        ls = _ls()
        shallow_pt = ls.min_points[int(np.argmax(ls.depths))]
        assert _instance_score(ls, shallow_pt) == 0.0

    def test_off_canyon_point_scores_zero(self):
        ls = _ls()
        far = ls.global_min_x.copy()
        far[1] += 100.0  # climb the wall
        assert _instance_score(ls, far) == 0.0

    def test_intermediate_basin_scores_between(self):
        ls = _ls()
        mid = ls.min_points[1]
        assert 0.0 < _instance_score(ls, mid) < 1.0


class _RandomOptimizer:
    def optimize(self, f, bounds, budget):
        import random

        rng = random.Random(0)
        best, best_v = None, float("inf")
        for _ in range(min(budget, 200)):
            x = [rng.uniform(lo, hi) for lo, hi in bounds]
            v = f(x)
            if v < best_v:
                best_v, best = v, x
        return best


class TestValidateContract:
    def test_non_optimizer_is_invalid(self):
        out = validate(123)
        assert out["is_valid"] == 0.0
        assert out["fitness"] == 0.0

    def test_instance_is_accepted(self):
        out = validate(_RandomOptimizer())
        assert out["is_valid"] == 1.0
        assert 0.0 <= out["fitness"] <= 1.0

    def test_class_is_instantiated_and_accepted(self):
        out = validate(_RandomOptimizer)  # entrypoint may return the class itself
        assert out["is_valid"] == 1.0
        assert 0.0 <= out["fitness"] <= 1.0

    def test_perfect_optimizer_scores_high(self):
        ladder = {inst.name: inst.landscape() for inst in get_ladder()}

        class Oracle:
            def optimize(self, f, bounds, budget):
                # cheat using the known global for whichever landscape matches dim+bounds
                for ls in ladder.values():
                    if ls.bounds == bounds:
                        return list(ls.global_min_x)
                return [0.0] * len(bounds)

        out = validate(Oracle())
        assert out["fitness"] > 0.99
