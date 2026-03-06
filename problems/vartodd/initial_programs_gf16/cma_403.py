from typing import Iterable
import cma
import nlopt
from pyswarm import pso
import numpy as np
from helper import get_matrix, BaseEvaluator, Matrix, ExplorationScore, FinalizationScore, get_matrix
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
    seeds = [random.randint(1, 10000) for _ in range(2)]

    def policy_mapping(self):
        init_rank = self.init_rank
        ranks = [0]
        eweigts = [self.map_par(_w_tanh, 0) for i in range(5)]
        fweigts = [self.map_par(_w_tanh, 0) for i in range(6)]

        w_pool = [ExplorationScore(
            weights=eweigts,
            pow=1
        )]
        
        w_final = [FinalizationScore(
            weights=fweigts,
            pow=2
        )]
        
        self.set_pool_scores(w_pool)
        self.set_final_scores(w_final)
        self.set_min_pool_size(2)
        self.set_min_z_to_research(10 + 500*self.map_par(sigmoid,0)) 
        self.set_temperature(0.7)
        
        self.set_num_samples(30)
        self.set_max_pool_size((460, 0), (20, 100))
        self.set_max_tohpe(6)
        
        self.set_try_only_tohpe(1)
        
        self.set_max_reduction(10)
        self.set_min_reduction(1)
        self.set_max_from_single_ns(2)
        self.set_tohpe_num_best(1)
        self.set_gen_part(0.8)
        self.set_todd_width(2)  
        self.set_beamsearch_width(3)  

    def __call__(self, params: Iterable): 
        tcounts = self.run(params, self.seeds)
        bestish = softmin(tcounts, beta=6.0)
        spread = float(np.std(tcounts)) if len(tcounts) > 1 else 0.0
        return bestish + 0.01 * spread

def run_cma(fun, num_eval: int = 5, initial_sigma: float = 0.5) -> np.ndarray:
    x0 = fun.extract_active()
    n_params = len(x0)
    bounds = [-2.0, 2.0]  # Lower and upper bounds for all dimensions
    def objective_function(x):
        if x.ndim == 1:
            return float(fun(x))
        else:
            return np.array([fun(xi) for xi in x])
    
    options = {
        'maxfevals': num_eval,
        'popsize' : 6,
        'bounds': bounds,
        'verbose': 0,
        'tolfun': 1e-4,
        'tolx': 1e-4,
    }
    
    xopt, es = cma.fmin2(
        objective_function,
        x0,
        initial_sigma,  
        options=options,
        restarts=0,
        bipop=False
    )
    
    return np.array(xopt)

def entrypoint(mat):
    fun = Evaluator(mat=mat, max_depth=200)
    
    xopt = run_cma(fun, 30)
    # fun(xopt)
    x_active = fun.set_up_new_init(0, rank_thr=470, xopt=xopt)
    if x_active is not None and len(x_active) > 0:
        xopt = run_cma(fun, 50)
        fun(xopt)
    
    return fun.get_best()