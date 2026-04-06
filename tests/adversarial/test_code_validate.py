"""Tests for adversarial code synthesis validation (Task 2).

Tests both Pop A (expression parser) and Pop B (test generator) validators
in static-fallback mode (no Redis opponent).
"""

import ast
import importlib.util
import os


def _load_seed(relative_path):
    seed_path = os.path.abspath(os.path.join(os.path.dirname(__file__), relative_path))
    spec = importlib.util.spec_from_file_location("seed", seed_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestPopAExpressionParserValidate:
    def test_good_parser_high_fitness(self):
        from problems.adversarial.code.pop_a.validate import validate

        def good_parser(expr):
            return float(eval(compile(ast.parse(expr, mode="eval"), "", "eval")))

        result = validate(good_parser)
        assert result["is_valid"] == 1
        assert result["fitness"] > 0.8

    def test_non_callable_is_invalid(self):
        from problems.adversarial.code.pop_a.validate import validate

        assert validate(42) == {"is_valid": 0, "fitness": 0.0}

    def test_always_wrong_parser(self):
        from problems.adversarial.code.pop_a.validate import validate

        result = validate(lambda expr: 999999.0)
        assert result["fitness"] == 0.0

    def test_seed_parser_is_valid(self):
        mod = _load_seed(
            "../../problems/adversarial/code/pop_a/initial_programs/seed.py"
        )
        parser = mod.entrypoint()
        assert callable(parser)

        from problems.adversarial.code.pop_a.validate import validate

        result = validate(parser)
        assert result["is_valid"] == 1
        assert result["fitness"] > 0.0


class TestPopBTestGeneratorValidate:
    def test_valid_generator_gets_fitness(self):
        from problems.adversarial.code.pop_b.validate import validate

        def gen():
            return ["2 + 3 * 4", "-1", "(1 + 2) * 3", "1 - 2 - 3", "--1"]

        result = validate(gen)
        assert result["is_valid"] == 1
        assert result["fitness"] > 0.0

    def test_non_callable_is_invalid(self):
        from problems.adversarial.code.pop_b.validate import validate

        assert validate("not a function")["is_valid"] == 0

    def test_empty_generator_is_invalid(self):
        from problems.adversarial.code.pop_b.validate import validate

        assert validate(lambda: [])["is_valid"] == 0

    def test_invalid_expressions_filtered(self):
        from problems.adversarial.code.pop_b.validate import validate

        result = validate(lambda: ["2 + 3", "not math", "+ + +"])
        assert result["is_valid"] == 1

    def test_seed_generator_is_valid(self):
        mod = _load_seed(
            "../../problems/adversarial/code/pop_b/initial_programs/seed.py"
        )
        generator = mod.entrypoint()
        assert callable(generator)

        from problems.adversarial.code.pop_b.validate import validate

        result = validate(generator)
        assert result["is_valid"] == 1
        assert result["fitness"] > 0.0
