"""Cross-play evaluator: adversarial landscape vs opponent optimizers.

Called by CallValidatorFunction as evaluate(opponent_results, program_output).
All results are pre-computed by the pipeline -- no exec() here.

opponent_results: list of callables optimizer(f, bounds, budget) -> x_best
    from FetchOpponentResultsStage (each opponent's entrypoint() result).
program_output: tuple (objective, bounds, optimum, budget)
    from CallProgramFunction (this program's entrypoint() result).
"""


def evaluate(opponent_results, program_output):
    import math
    import random

    if not isinstance(program_output, tuple) or len(program_output) != 4:
        return {"fitness": 0.0, "is_valid": 0.0, "n_opponents": 0.0}

    objective, bounds, optimum, budget = program_output
    dim = len(optimum)

    # Validate landscape: optimum must actually be the minimum
    try:
        f_opt = objective(optimum)
    except Exception:
        return {"fitness": 0.0, "is_valid": 0.0, "n_opponents": 0.0}

    # Spot-check: random points should not be better than claimed optimum
    rng = random.Random(42)
    for _ in range(200):
        x_rand = [rng.uniform(lo, hi) for lo, hi in bounds]
        try:
            if objective(x_rand) < f_opt - 1e-6:
                return {"fitness": 0.0, "is_valid": 0.0, "n_opponents": 0.0}
        except Exception:
            return {"fitness": 0.0, "is_valid": 0.0, "n_opponents": 0.0}

    # Cross-play: measure how far optimizers land from the true optimum
    max_dist = math.sqrt(dim) * 10  # diagonal of search space
    scores = []
    for optimize in opponent_results:
        try:
            if not callable(optimize):
                scores.append(
                    1.0
                )  # non-callable = infinite distance = max deceptiveness
                continue
            x_best = optimize(objective, bounds, int(budget))
            if x_best is None or len(x_best) != dim:
                scores.append(1.0)
                continue
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(x_best, optimum)))
            # Deceptiveness: higher distance = better for the landscape
            scores.append(min(dist / max_dist, 1.0))
        except Exception:
            scores.append(1.0)  # crash = max deceptiveness

    fitness = sum(scores) / len(scores) if scores else 0.0
    return {
        "fitness": fitness,
        "is_valid": 1.0,
        "n_opponents": float(len(scores)),
    }
