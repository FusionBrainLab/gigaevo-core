"""Tests for adversarial optimizer validation (Task 1).

Tests both Pop A (optimizer) and Pop B (landscape) validators
in static-fallback mode (no Redis opponent).
"""

import importlib.util
import math
import os


def _load_seed(relative_path):
    """Load a seed program from the problems directory."""
    seed_path = os.path.join(os.path.dirname(__file__), relative_path)
    seed_path = os.path.abspath(seed_path)
    spec = importlib.util.spec_from_file_location("seed", seed_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestPopAOptimizerValidate:
    def test_valid_optimizer_gets_positive_fitness(self):
        import random

        from problems.adversarial.optimizer.pop_a.validate import validate

        def good_optimizer(f, bounds, budget):
            best_x, best_val = None, float("inf")
            for _ in range(budget):
                x = [random.uniform(lo, hi) for lo, hi in bounds]
                val = f(x)
                if val < best_val:
                    best_val, best_x = val, x[:]
            return best_x

        result = validate(good_optimizer)
        assert result["is_valid"] == 1
        assert result["fitness"] > 0.0

    def test_non_callable_is_invalid(self):
        from problems.adversarial.optimizer.pop_a.validate import validate

        assert validate("not a function") == {"is_valid": 0, "fitness": 0.0}

    def test_crashing_optimizer_scores_zero(self):
        from problems.adversarial.optimizer.pop_a.validate import validate

        result = validate(lambda f, b, n: (_ for _ in ()).throw(RuntimeError("crash")))
        assert result["fitness"] == 0.0

    def test_seed_program_is_valid(self):
        mod = _load_seed(
            "../../problems/adversarial/optimizer/pop_a/initial_programs/seed.py"
        )
        optimizer = mod.entrypoint()
        assert callable(optimizer)

        from problems.adversarial.optimizer.pop_a.validate import validate

        result = validate(optimizer)
        assert result["is_valid"] == 1
        assert result["fitness"] > 0.0


class TestPopBLandscapeValidate:
    def test_deceptive_landscape_gets_positive_fitness(self):
        from problems.adversarial.optimizer.pop_b.validate import validate

        dim = 5
        optimum = [3.7, -2.1, 4.5, -1.3, 2.8]
        bounds = [(-5.12, 5.12)] * dim

        def landscape(x):
            rastrigin = 10 * dim + sum(
                xi**2 - 10 * math.cos(2 * math.pi * xi) for xi in x
            )
            dist = sum((xi - oi) ** 2 for xi, oi in zip(x, optimum))
            return rastrigin - 50.0 * math.exp(-2.0 * dist)

        result = validate((landscape, bounds, optimum, 500))
        assert result["is_valid"] == 1
        assert result["fitness"] > 0.0

    def test_non_tuple_is_invalid(self):
        from problems.adversarial.optimizer.pop_b.validate import validate

        assert validate("not a tuple")["is_valid"] == 0

    def test_trivial_landscape_low_fitness(self):
        from problems.adversarial.optimizer.pop_b.validate import validate

        optimum, bounds = [0.0] * 5, [(-5, 5)] * 5
        result = validate((lambda x: sum(xi**2 for xi in x), bounds, optimum, 500))
        assert result["is_valid"] == 1
        assert result["fitness"] < 0.5  # easy landscape → low deceptiveness

    def test_seed_landscape_is_valid(self):
        mod = _load_seed(
            "../../problems/adversarial/optimizer/pop_b/initial_programs/seed.py"
        )
        data = mod.entrypoint()
        assert isinstance(data, tuple) and len(data) == 4

        from problems.adversarial.optimizer.pop_b.validate import validate

        result = validate(data)
        assert result["is_valid"] == 1
        assert result["fitness"] > 0.0
