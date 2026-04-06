"""Validate expression parsers against adversarial test inputs (Pop A).

Fitness: fraction of tests passed (correctness on adversarial inputs).
Reference oracle: Python's ast module for safe evaluation.
Falls back to static test suite when opponent archive is empty.
"""

import ast
import operator
import signal

from problems.adversarial.shared import (
    exec_entrypoint,
    get_opponent_config,
    sample_opponents,
)

# --- Reference oracle (ast-based, safe) ---

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _reference_eval(expr: str) -> float:
    """Evaluate expression using Python's AST — safe reference oracle."""
    tree = ast.parse(expr, mode="eval")

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp):
            left, right = _eval(node.left), _eval(node.right)
            if isinstance(node.op, ast.Div) and right == 0:
                raise ValueError("Division by zero")
            return _OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp):
            return _OPS[type(node.op)](_eval(node.operand))
        raise ValueError(f"Unsupported: {type(node).__name__}")

    return _eval(tree)


# --- Static fallback tests ---

STATIC_TESTS = [
    "2 + 3",
    "2 + 3 * 4",
    "(2 + 3) * 4",
    "10 / 3",
    "-3 + 4",
    "-(2 + 3)",
    "1 + 2 + 3 + 4",
    "2 * 3 + 4 * 5",
    "((1 + 2) * (3 + 4))",
    "100 / 4 / 5",
    "1 - 2 - 3",
    "-1 * -2",
    "0.5 + 0.5",
    "(((3)))",
    "1 + -2",
]


def _test_one(solver, expr, timeout=1.0):
    """Test solver on one expression. Returns True if correct."""
    try:
        expected = _reference_eval(expr)
    except (ValueError, SyntaxError):
        # Oracle rejects → solver should also reject
        try:
            _call_with_timeout(solver, (expr,), timeout)
            return False
        except Exception:
            return True

    try:
        result = _call_with_timeout(solver, (expr,), timeout)
        if not isinstance(result, (int, float)):
            return False
        return abs(result - expected) < 1e-6 * max(1.0, abs(expected))
    except Exception:
        return False


def _call_with_timeout(fn, args, timeout):
    """Call fn(*args) with SIGALRM timeout."""

    def _handler(signum, frame):
        raise TimeoutError("timeout")

    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        return fn(*args)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def _load_adversarial_tests():
    """Load test expressions from Pop B's archive."""
    cfg = get_opponent_config()
    if not cfg["prefix"]:
        return []

    opponents = sample_opponents(
        host=cfg["host"],
        port=cfg["port"],
        db=cfg["db"],
        prefix=cfg["prefix"],
        n=5,
    )
    tests = []
    for _pid, _fit, code in opponents:
        try:
            generator = exec_entrypoint(code, timeout=2.0)
            if callable(generator):
                cases = generator()
                if isinstance(cases, (list, tuple)):
                    tests.extend(tc for tc in cases if isinstance(tc, str))
        except Exception:
            continue
    return tests


def validate(data):
    if not callable(data):
        return {"is_valid": 0, "fitness": 0.0}

    tests = _load_adversarial_tests() or STATIC_TESTS
    correct = sum(_test_one(data, expr) for expr in tests)
    fitness = correct / len(tests) if tests else 0.0
    return {"is_valid": 1 if correct > 0 else 0, "fitness": fitness}
