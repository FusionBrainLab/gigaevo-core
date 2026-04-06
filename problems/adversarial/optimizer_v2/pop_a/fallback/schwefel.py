def entrypoint():
    """Shifted Schwefel — global optimum far from the search center."""
    import math

    dim = 5
    optimum = [420.9687, 420.9687, 420.9687, 420.9687, 420.9687]
    bounds = [(0.0, 500.0)] * dim
    budget = 500

    def objective(x):
        return 418.9829 * dim - sum(xi * math.sin(math.sqrt(abs(xi))) for xi in x)

    return objective, bounds, optimum, budget
