def entrypoint():
    """Expression evaluator — deliberately buggy seed.

    Handles basic arithmetic and parentheses but has known bugs:
    - Unary minus not handled correctly in all positions
    - Double negation fails
    - Division by zero not caught
    """

    def evaluate(expression: str) -> float:
        tokens = _tokenize(expression)
        pos = [0]
        result = _parse_expr(tokens, pos)
        if pos[0] != len(tokens):
            raise ValueError("Unexpected token")
        return result

    def _tokenize(expr):
        tokens = []
        i = 0
        while i < len(expr):
            if expr[i].isspace():
                i += 1
            elif expr[i] in "+-*/()":
                tokens.append(expr[i])
                i += 1
            elif expr[i].isdigit() or expr[i] == ".":
                j = i
                while j < len(expr) and (expr[j].isdigit() or expr[j] == "."):
                    j += 1
                tokens.append(float(expr[i:j]))
                i = j
            else:
                raise ValueError(f"Invalid character: {expr[i]}")
        return tokens

    def _parse_expr(tokens, pos):
        left = _parse_term(tokens, pos)
        while pos[0] < len(tokens) and tokens[pos[0]] in ("+", "-"):
            op = tokens[pos[0]]
            pos[0] += 1
            right = _parse_term(tokens, pos)
            if op == "+":
                left += right
            else:
                left -= right
        return left

    def _parse_term(tokens, pos):
        left = _parse_factor(tokens, pos)
        while pos[0] < len(tokens) and tokens[pos[0]] in ("*", "/"):
            op = tokens[pos[0]]
            pos[0] += 1
            right = _parse_factor(tokens, pos)
            if op == "*":
                left *= right
            else:
                left /= right  # Bug: no division by zero check
        return left

    def _parse_factor(tokens, pos):
        if pos[0] >= len(tokens):
            raise ValueError("Unexpected end of expression")
        token = tokens[pos[0]]
        if isinstance(token, float):
            pos[0] += 1
            return token
        elif token == "(":
            pos[0] += 1
            result = _parse_expr(tokens, pos)
            if pos[0] >= len(tokens) or tokens[pos[0]] != ")":
                raise ValueError("Missing closing parenthesis")
            pos[0] += 1
            return result
        elif token == "-":
            pos[0] += 1
            return -_parse_factor(tokens, pos)
        # Bug: unary + not handled
        else:
            raise ValueError(f"Unexpected token: {token}")

    return evaluate
