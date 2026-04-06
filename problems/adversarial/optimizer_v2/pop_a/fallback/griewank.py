def entrypoint():
    """Shifted Griewank — many local minima with wide basins."""
    import math

    dim = 5
    optimum = [3.0, -2.0, 1.5, -4.0, 2.5]
    bounds = [(-10.0, 10.0)] * dim
    budget = 500

    def objective(x):
        shifted = [xi - oi for xi, oi in zip(x, optimum)]
        s = sum(si**2 for si in shifted) / 4000.0
        p = 1.0
        for i, si in enumerate(shifted):
            p *= math.cos(si / math.sqrt(i + 1))
        return s - p + 1.0

    return objective, bounds, optimum, budget
