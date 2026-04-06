def entrypoint():
    """Adversarial test generator seed — targets common parser bugs."""

    def generate():
        return [
            # Unary minus edge cases
            "-1",
            "--1",
            "-(3)",
            "-(-3)",
            "1 + -2",
            "1 * -2",
            # Precedence
            "2 + 3 * 4",
            "2 * 3 + 4",
            "1 + 2 * 3 + 4",
            # Parentheses nesting
            "(((1)))",
            "((1 + 2) * (3 + 4))",
            "(1 + (2 * (3 + 4)))",
            # Chained same-precedence (left associativity)
            "1 - 2 - 3",
            "8 / 4 / 2",
            "1 - 2 + 3 - 4",
            # Whitespace variations
            "  2  +  3  ",
            "2+3",
            # Floats
            "0.1 + 0.2",
            "1.5 * 2.0",
            # Mixed unary/binary
            "-1 * -1",
        ]

    return generate
