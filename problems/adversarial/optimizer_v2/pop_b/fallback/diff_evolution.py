def entrypoint():
    """Differential evolution — population-based, harder to deceive."""
    import random

    def optimizer(f, bounds, budget):
        dim = len(bounds)
        pop_size = min(20, max(4, budget // 10))
        F = 0.8  # mutation factor
        CR = 0.9  # crossover rate

        # Initialize population
        pop = []
        fit = []
        for _ in range(pop_size):
            x = [random.uniform(lo, hi) for lo, hi in bounds]
            pop.append(x)
            fit.append(f(x))
        evals = pop_size

        while evals < budget:
            for i in range(pop_size):
                if evals >= budget:
                    break
                # Select 3 distinct others
                candidates = [j for j in range(pop_size) if j != i]
                a, b, c = random.sample(candidates, 3)

                # Mutate + crossover
                j_rand = random.randint(0, dim - 1)
                trial = []
                for j in range(dim):
                    if random.random() < CR or j == j_rand:
                        v = pop[a][j] + F * (pop[b][j] - pop[c][j])
                        v = max(bounds[j][0], min(bounds[j][1], v))
                        trial.append(v)
                    else:
                        trial.append(pop[i][j])

                f_trial = f(trial)
                evals += 1
                if f_trial <= fit[i]:
                    pop[i] = trial
                    fit[i] = f_trial

        best_idx = min(range(pop_size), key=lambda i: fit[i])
        return pop[best_idx]

    return optimizer
