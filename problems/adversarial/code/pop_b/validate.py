"""Validate adversarial test generators against opponent parsers (Pop B).

Fitness: mean failure rate of opponent parsers on generated tests.
Reference oracle: Python's ast module determines ground truth.
Falls back to a deliberately buggy parser when opponent archive is empty.
"""

import ast
import operator
import signal

from problems.adversarial.shared import (
    exec_entrypoint,
    get_opponent_config,
    sample_opponents,
)

# --- Reference oracle (same as Pop A) ---

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _reference_eval(expr: str) -> float:
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


# --- Static fallback: deliberately buggy parser ---


def _buggy_parser(expr: str) -> float:
    """Left-to-right evaluation ignoring operator precedence."""
    expr = expr.strip()
    tokens, i = [], 0
    while i < len(expr):
        if expr[i].isspace():
            i += 1
        elif expr[i] in "+-*/()" and not (
            expr[i] == "-"
            and (i == 0 or expr[i - 1] in "+-*/(")
            and i + 1 < len(expr)
            and (expr[i + 1].isdigit() or expr[i + 1] == ".")
        ):
            tokens.append(expr[i])
            i += 1
        elif (
            expr[i].isdigit()
            or expr[i] == "."
            or (expr[i] == "-" and (i == 0 or expr[i - 1] in "+-*/("))
        ):
            j = i
            if expr[j] == "-":
                j += 1
            while j < len(expr) and (expr[j].isdigit() or expr[j] == "."):
                j += 1
            tokens.append(float(expr[i:j]))
            i = j
        else:
            raise ValueError(f"Bad char: {expr[i]}")

    if not tokens or not isinstance(tokens[0], float):
        raise ValueError("Empty expression")
    result, i = tokens[0], 1
    while i + 1 < len(tokens):
        op, right = tokens[i], tokens[i + 1]
        if not isinstance(right, float):
            raise ValueError("Expected number")
        i += 2
        if op == "+":
            result += right
        elif op == "-":
            result -= right
        elif op == "*":
            result *= right
        elif op == "/":
            result /= right
    return result


STATIC_PARSERS = [_buggy_parser]


def _is_valid_expression(expr: str) -> bool:
    if not isinstance(expr, str) or not expr.strip():
        return False
    try:
        _reference_eval(expr)
        return True
    except Exception:
        return False


def _call_with_timeout(fn, args, timeout=1.0):
    def _handler(signum, frame):
        raise TimeoutError("timeout")

    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        return fn(*args)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def _parser_fails_on(parser_fn, expr):
    """Returns True if parser gets the expression WRONG (adversarial success)."""
    try:
        expected = _reference_eval(expr)
    except Exception:
        return False  # Skip invalid expressions

    try:
        result = _call_with_timeout(parser_fn, (expr,))
        if not isinstance(result, (int, float)):
            return True
        return abs(result - expected) >= 1e-6 * max(1.0, abs(expected))
    except Exception:
        return True  # Any failure = adversarial success


def _load_opponent_parsers():
    """Load parsers from Pop A's archive."""
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
    parsers = []
    for _pid, _fit, code in opponents:
        try:
            result = exec_entrypoint(code, timeout=2.0)
            if callable(result):
                parsers.append(result)
        except Exception:
            continue
    return parsers


def validate(data):
    if not callable(data):
        return {"is_valid": 0, "fitness": 0.0}

    try:
        test_cases = _call_with_timeout(data, (), timeout=2.0)
    except Exception:
        return {"is_valid": 0, "fitness": 0.0}

    if not isinstance(test_cases, (list, tuple)) or not test_cases:
        return {"is_valid": 0, "fitness": 0.0}

    valid_tests = [tc for tc in test_cases if _is_valid_expression(tc)]
    if not valid_tests:
        return {"is_valid": 0, "fitness": 0.0}

    parsers = _load_opponent_parsers() or STATIC_PARSERS

    total_failures = sum(
        _parser_fails_on(p, expr) for p in parsers for expr in valid_tests
    )
    total_tests = len(parsers) * len(valid_tests)
    fitness = total_failures / total_tests if total_tests > 0 else 0.0
    return {"is_valid": 1, "fitness": fitness}
