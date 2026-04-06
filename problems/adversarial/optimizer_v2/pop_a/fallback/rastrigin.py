def entrypoint():
    """Shifted Rastrigin — classic deceptive landscape."""
    import math

    dim = 5
    optimum = [2.5, -1.5, 3.0, -2.0, 1.0]
    bounds = [(-5.12, 5.12)] * dim
    budget = 500

    def objective(x):
        return 10 * dim + sum(
            (xi - oi) ** 2 - 10 * math.cos(2 * math.pi * (xi - oi))
            for xi, oi in zip(x, optimum)
        )

    return objective, bounds, optimum, budget
