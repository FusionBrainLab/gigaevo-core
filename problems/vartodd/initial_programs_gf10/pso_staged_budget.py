from typing import Iterable
import numpy as np
import random
import pyswarms as ps
from helper import BaseEvaluator, Matrix, ExplorationScore, FinalizationScore

np.random.seed(42)
random.seed(40)

def _w_tanh(z: float, scale: float = 3.0, sharp: float = 1.5) -> float:
    return float(scale * np.tanh(z / sharp))

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))

class Evaluator(BaseEvaluator):
    seeds = [random.randint(1, 10000) for _ in range(1)]

    def policy_mapping(self):
        ranks = [440, 0]

        wred = self.map_par(_w_tanh, 0)
        wdim = self.map_par(_w_tanh, 0)
        wbucket = self.map_par(_w_tanh, 0)
        fwred = self.map_par(_w_tanh, 0)
        fwdim = self.map_par(_w_tanh, 0)
        fwbucket = self.map_par(_w_tanh, 0)
        fwtohpe = self.map_par(_w_tanh, 0)

        budget = self.map_par(sigmoid, 0)

        w_pool = [ExplorationScore(weights=[wred, wdim, wbucket, 0.0, 0.0], pow=1) for _ in ranks]
        w_final = [FinalizationScore(
            weights=[fwred, fwdim, fwbucket, 0.0, 0.0, fwtohpe],
            centers=[1, 0, 0, 0, 0, 0],
            pow=2
        ) for _ in ranks]

        self.set_pool_scores(ranks, w_pool)
        self.set_final_scores(ranks, w_final)

        beam = int(1 + 3 * budget)
        todd_w = int(1 + 3 * budget)
        max_pool = int(15 + 25 * budget)
        num_samples = int(6 + 10 * (1.0 - budget))
        min_z = 10 + 180 * budget
        max_tohpe = int(2 + 3 * budget)
        gen_part = 0.2 + 0.6 * budget

        self.set_min_pool_size(2)
        self.set_min_z_to_research(min_z)
        self.set_temperature(0.55)
        self.set_num_samples(ranks, [num_samples, max(4, num_samples - 2)])
        self.set_max_pool_size(max_pool)
        self.set_max_tohpe(max_tohpe)
        self.set_try_only_tohpe(ranks, [1, 1])
        self.set_max_reduction(10)
        self.set_min_reduction(1)
        self.set_max_from_single_ns(2)
        self.set_tohpe_num_best(8)
        self.set_gen_part(gen_part)
        self.set_todd_width(todd_w)
        self.set_beamsearch_width(beam)

    def __call__(self, params: Iterable):
        early = self.run(params, self.seeds[:1])
        if self.best_rank < 100000 and early[0] > self.best_rank + 25:
            return float(early[0] + 10.0)

        rest = self.run(params, self.seeds[1:]) if len(self.seeds) > 1 else []
        tcounts = early + rest
        q = float(np.quantile(tcounts, 0.7))
        spread = float(np.std(tcounts)) if len(tcounts) > 1 else 0.0
        return q + 0.02 * spread


def run_pso(fun: Evaluator, num_eval: int = 12, particles: int = 6) -> np.ndarray:
    x = fun.extract_active()
    n_params = len(x)

    def objective_function(positions):
        return np.array([fun(pos) for pos in positions])

    bounds = (np.array([-2.0] * n_params), np.array([2.0] * n_params))
    options = {"c1": 0.7, "c2": 0.6, "w": 0.8}

    optimizer = ps.single.GlobalBestPSO(
        n_particles=particles,
        dimensions=n_params,
        options=options,
        bounds=bounds,
    )

    best_cost, best_position = optimizer.optimize(
        objective_function,
        iters=num_eval,
        verbose=False,
    )
    return best_position


def entrypoint():
    fun = Evaluator(path_name="init", max_depth=240)

    xopt = run_pso(fun, num_eval=18, particles=6)
    fun(xopt)

    best_rank = fun.best_rank
    for thr in [best_rank + 30, best_rank + 10]:
        x_active = fun.set_up_new_init(0, rank_thr=int(thr), xopt=xopt)
        if x_active is None:
            break
        xopt = run_pso(fun, num_eval=10, particles=5)
        fun(xopt)

    return fun.get_best()
