def entrypoint():
    """Deceptive landscape seed: shifted Rastrigin with a hidden global minimum.

    The dominant basin of attraction is around the origin (decoy).
    The true global minimum is hidden at an offset location with a very deep
    narrow basin that overcomes the Rastrigin penalty.
    """
    import math

    dim = 5
    # True global minimum — hidden away from the origin
    optimum = [3.7, -2.1, 4.5, -1.3, 2.8]
    bounds = [(-5.12, 5.12)] * dim
    budget = 500

    def objective(x):
        # Base: standard Rastrigin (global min at origin)
        rastrigin = 10 * dim + sum(xi**2 - 10 * math.cos(2 * math.pi * xi) for xi in x)

        # Decoy: deep wide basin at origin to attract optimizers
        dist_to_origin = sum(xi**2 for xi in x)
        decoy_basin = -5.0 * math.exp(-0.1 * dist_to_origin)

        # Hidden global minimum: very deep narrow basin at optimum
        # Must be deep enough to overcome Rastrigin penalty at the offset
        dist_to_optimum = sum((xi - oi) ** 2 for xi, oi in zip(x, optimum))
        hidden_basin = -200.0 * math.exp(-2.0 * dist_to_optimum)

        return rastrigin + decoy_basin + hidden_basin

    return objective, bounds, optimum, budget
