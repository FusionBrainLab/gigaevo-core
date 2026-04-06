def entrypoint():
    """Simple random search optimizer — deliberately naive seed.

    Samples random points uniformly, then does greedy local search around
    the best found. Easily trapped by local minima on deceptive landscapes.
    """
    import random

    def optimizer(f, bounds, budget):
        dim = len(bounds)
        best_x = None
        best_val = float("inf")
        evals_used = 0

        # Phase 1: Random sampling (60% of budget)
        sample_budget = int(budget * 0.6)
        for _ in range(sample_budget):
            x = [random.uniform(lo, hi) for lo, hi in bounds]
            val = f(x)
            evals_used += 1
            if val < best_val:
                best_val = val
                best_x = x[:]

        # Phase 2: Local search around best point (remaining budget)
        step_size = 0.1
        while evals_used < budget:
            # Perturb each dimension slightly
            candidate = []
            for i, (lo, hi) in enumerate(bounds):
                delta = random.gauss(0, step_size * (hi - lo))
                candidate.append(max(lo, min(hi, best_x[i] + delta)))
            val = f(candidate)
            evals_used += 1
            if val < best_val:
                best_val = val
                best_x = candidate[:]
            else:
                step_size *= 0.95  # Shrink step on failure

        return best_x

    return optimizer
