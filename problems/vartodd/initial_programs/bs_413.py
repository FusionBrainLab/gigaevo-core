from typing import Iterable
import numpy as np
from helper import BaseEvaluator, Matrix, ExplorationScore, FinalizationScore
import random
import pyswarms as ps

np.random.seed(42)
random.seed(40)

def _w_tanh(z: float, scale: float = 4.0, sharp: float = 1.5) -> float:
    return float(scale * np.tanh(z / sharp))

def softmin(xs, beta=6.0):
    xs = np.asarray(xs, dtype=float)
    m = xs.min()
    return float(m - (1.0/beta) * np.log(np.exp(-beta*(xs - m)).sum()))

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

class Evaluator(BaseEvaluator):
    seeds = [random.randint(1, 10000) for _ in range(1)]

    def policy_mapping(self):
        eweigts = [self.map_par(_w_tanh, 0) for i in range(5)]
        budget = self.map_par(sigmoid, 0)

        w_pool = [ExplorationScore(
            weights=eweigts,
            pow=1
        )]
        
        w_final = [FinalizationScore(
            weights=[*eweigts, self.map_par(_w_tanh, 0)],
            pow=2
        )]
        
        self.set_pool_scores(w_pool)
        self.set_final_scores(w_final)
        self.set_temperature(0.4)
        self.set_min_z_to_research(80 + 120 * budget)
        
        max_pool = int(8 + 12 * budget)
        self.set_max_pool_size(max_pool)
        self.set_min_pool_size(1)

        self.set_max_tohpe(4)
        self.set_try_only_tohpe(1)
        
        self.set_max_reduction(8)
        self.set_min_reduction(1)
        self.set_tohpe_num_best(6)
        self.set_max_from_single_ns(3)
        self.set_num_samples(int(6 + 6 * (1.0 - budget)))

        self.set_gen_part(0.3 + 0.4 * budget)
        self.set_todd_width(int(1 + 2 * budget))
        self.set_beamsearch_width(int(1 + 2 * budget))

    def __call__(self, params: Iterable): 
        tcounts = self.run(params, self.seeds)
        bestish = softmin(tcounts, beta=6.0)
        spread = float(np.std(tcounts)) if len(tcounts) > 1 else 0.0
        return bestish + 0.01 * spread

def run_pso(fun: Evaluator, num_eval: int=5, options={ 'c1': 1.0, 'c2': 1.0, 'w': 0.9 }) -> Evaluator:
    x = fun.extract_active()
    n_params = len(x)
    def objective_function(positions):
        return np.array([fun(pos) for pos in positions])
    
    bounds = (np.array([-2.0] * n_params), np.array([2.0] * n_params))

    optimizer = ps.single.GlobalBestPSO(
        n_particles=4, 
        dimensions=n_params,
        options=options,
        bounds=bounds
    )

    best_cost, best_position = optimizer.optimize(
        objective_function, 
        iters=num_eval,
        verbose=False
    )
    return best_position
        

def entrypoint():
    fun = Evaluator(path_name="init", init_rank_thr=0)
    xopt = run_pso(fun, 15)
    
    return fun.get_best()
