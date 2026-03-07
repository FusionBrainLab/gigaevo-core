from typing import List
import numpy as np
from node import Matrix, Node, FinalizationScore, ExplorationScore, Tensor3D
from mcts_dao import Dao, RankSchedule, Path
from todd import Todd
from typing import Iterable, Sequence, Any, Tuple
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field


def _worker_run_one_from_template(seed, path: Path, todd: Todd, bs_width: RankSchedule = RankSchedule.constant(1), todd_width: RankSchedule = RankSchedule.constant(1)):
    todd = deepcopy(todd)
    # self.todd = Todd(self.dao, max_depth)
    node, counters = todd.run(path, bs_width, todd_width, True, seed, )
    return seed, node, counters


def find_rank(path, rank):
    for i, mat in enumerate(path):
        if mat.rows < rank:
            return path[max(i-1, 0)]
    return path[-1]
        
def get_matrix(name:str=None) -> Matrix:
    if name is None:
        # return Matrix.from_numpy(np.load("problems/vartodd/npy/gf2^16_1612310.npy") )
        return Matrix.from_numpy(np.load("problems/vartodd/npy/gf2^10_1030.npy") )
        # return Matrix.from_numpy(np.load("problems/vartodd/npy/gf2^9_940.npy") )
    return Matrix.from_numpy(np.load(f"npy/{name}.npy") )


#--------------------------- RANK ----------------------------
def _is_rank_list(x: Any) -> bool:
    if not isinstance(x, Iterable): 
        return False
    x = list(x) 
    return len(x) > 0 and isinstance(x[0], (list, tuple)) and len(x[0]) == 2

def _to_rank_schedule(x: Any) -> "RankSchedule":
    """Accept RankSchedule | [(rank, value), ...] | scalar and return RankSchedule."""
    if isinstance(x, RankSchedule):
        return x
    if isinstance(x, zip):
        x = [obj for obj in x]
    if _is_rank_list(x):
        return RankSchedule.from_any(list(x))
    return RankSchedule.constant(x)

def _to_erank_schedule(x: Any) -> "RankSchedule":
    """Accept RankSchedule | [(rank, value), ...] | scalar and return RankSchedule."""
    if isinstance(x, RankSchedule):
        return x
    if isinstance(x, zip):
        x = [obj for obj in x]
    if _is_rank_list(x):
        return RankSchedule.from_any([(rank, _to_exploration_score(el)) for (rank, el) in x])
    return RankSchedule.constant(_to_exploration_score(x))

def _to_frank_schedule(x: Any) -> "RankSchedule":
    """Accept RankSchedule | [(rank, value), ...] | scalar and return RankSchedule."""
    if isinstance(x, RankSchedule):
        return x
    if isinstance(x, zip):
        x = [obj for obj in x]
    if _is_rank_list(x):
        return RankSchedule.from_any([(rank, _to_finalization_score(el)) for (rank, el) in x])
    return RankSchedule.constant(_to_finalization_score(x))

def _to_exploration_score(x: Any) -> "ExplorationScore":
    """Accept ExplorationScore | (wred, wdim, wpossible_red) and return ExplorationScore."""
    if isinstance(x, ExplorationScore):
        return x

    x = list(x)
    xs = np.asarray(x)
    x = np.asarray(xs)/np.sqrt(np.sum(xs*xs)) if np.any(xs) else xs
    return ExplorationScore(*x)

def _to_finalization_score(x: Any) -> "FinalizationScore":
    """Accept FinalizationScore | (wred, wdim, wpossible_red, wtohpe_dim) and return FinalizationScore."""
    if isinstance(x, FinalizationScore):
        return x
    x = list(x)
    xs = np.asarray(x)
    x = np.asarray(xs)/np.sqrt(np.sum(xs*xs)) if np.any(xs) else xs
    return FinalizationScore(*x)

def float_rank_shedule_to_str(dss: List[RankSchedule], ranks: List[int]):
    output_r = []
    output_v = []
    for i, ds in enumerate(dss):
        down = ranks[i]
        up = ranks[i-1] if i > 0 else 1000000            
        for r, v in ds.points:
            if r < up and r >= down:
                output_r.append(r)
                output_v.append(float(v))
            elif r < down:
                output_r.append(down)
                output_v.append(float(v))
                break
    return [tuple(output_r), tuple(output_v)]

def int_rank_shedule_to_str(dss: List[RankSchedule], ranks: List[int]):
    output_r = []
    output_v = []
    for i, ds in enumerate(dss):
        down = ranks[i]
        up = ranks[i-1] if i > 0 else 1000000            
        for r, v in ds.points:
            if r < up and r >= down:
                output_r.append(r)
                output_v.append(int(v))
            elif r < down:
                output_r.append(down)
                output_v.append(int(v))
                break
    return [tuple(output_r), tuple(output_v)]

def score_rank_shedule_to_str(dss: List[RankSchedule], ranks: List[int]):
    output_r = []
    output_v = []
    for i, ds in enumerate(dss):
        down = ranks[i]
        up = ranks[i-1] if i > 0 else 1000000            
        for r, v in ds.points:
            if r < up and r >= down:
                output_r.append(r)
                weights = [float(f"{v[i]:.3f}") for i in range(len(v))]
                centers = [float(f"{v[i+len(v)]:.3f}") for i in range(len(v))]
                pow = float(f"{v.pow():.3f}")
                output = "("
                if any(weights):
                    output = output + f"{weights=},"
                if any(centers):
                    output = output + f"{centers=},"  
                output = output + f"{pow=})"  
                output_v.append(output)
                # output_v.append(f"{weights=}, {centers=}, {pow=}")
            elif r < down:
                output_r.append(down)
                weights = [float(f"{v[i]:.3f}") for i in range(len(v))]
                centers = [float(f"{v[i+len(v)]:.3f}") for i in range(len(v))]
                pow = float(f"{v.pow():.3f}")
                output = "("
                if any(weights):
                    output = output + f"{weights=},"
                if any(centers):
                    output = output + f"{centers=},"  
                output = output + f"{pow=})"  
                output_v.append(output)
                break
    return [tuple(output_r), tuple(output_v)]
    
def dao_rank_to_str(daos: List[Dao], ranks: List[int]):
    dict = {}
    dict["num_samples"] = int_rank_shedule_to_str([dao.mode.num_samples for dao in daos], ranks)
    dict["top_pool"] = int_rank_shedule_to_str([dao.mode.top_pool for dao in daos], ranks)
    # dict["try_only_tohpe"] = int_rank_shedule_to_str([dao.mode.try_only_tohpe for dao in daos], ranks)
    # dict["max_tohpe"] = int_rank_shedule_to_str([dao.mode.max_tohpe for dao in daos], ranks)
    # dict["num_tohpe_sample"] = int_rank_shedule_to_str([dao.mode.num_tohpe_sample for dao in daos], ranks)
    # dict["max_from_single_ns"] = int_rank_shedule_to_str([dao.mode.max_from_single_ns for dao in daos], ranks)
    # dict["min_reduction"] = int_rank_shedule_to_str([dao.mode.min_reduction for dao in daos], ranks)
    # dict["max_reduction"] = int_rank_shedule_to_str([dao.mode.max_reduction for dao in daos], ranks)
    dict["pool_scores"] = score_rank_shedule_to_str([dao.mode.pool_scores for dao in daos], ranks)
    dict["final_scores"] = score_rank_shedule_to_str([dao.mode.final_scores for dao in daos], ranks)
    dict["max_z_to_research"] = float_rank_shedule_to_str([dao.mode.max_z_to_research for dao in daos], ranks)
    dict["gen_part"] = float_rank_shedule_to_str([dao.mode.gen_part for dao in daos], ranks)
    # dict["temperature"] = float_rank_shedule_to_str([dao.mode.temperature for dao in daos], ranks)
    # dict["non_improving_prob"] = float_rank_shedule_to_str(dao.mode.non_improving_prob)
    return dict

def print_uniform_by_rank(best_ranks, best_evals, max_lines=10):
    """
    Print entries with uniformly spaced rank values, always including the last entry.
    """
    s = ""
    n = len(best_ranks)
    if n <= max_lines:
        for i, (rank, eval_step) in enumerate(zip(best_ranks, best_evals)):
            if i == n - 1:
                s += "Final "
            s += f"Rank={rank} at eval={eval_step}\n"
        return s
    
    # Get unique ranks and their first occurrence
    rank_to_step = {}
    for i, rank in enumerate(best_ranks):
        if rank not in rank_to_step:  # Keep first occurrence
            rank_to_step[rank] = i
    
    # Sort ranks in ascending order (better ranks first if lower is better)
    unique_ranks = sorted(rank_to_step.keys())
    
    if len(unique_ranks) <= max_lines:
        # If we have few unique ranks, print them all plus last
        selected_steps = set()
        for rank in unique_ranks:
            selected_steps.add(rank_to_step[rank])
        selected_steps.add(n - 1)  # Always include last step
        selected_steps = sorted(selected_steps)
    else:
        selected_steps = []
        
        first_rank = unique_ranks[0]
        selected_steps.append(rank_to_step[first_rank])
        step_size = (len(unique_ranks) - 1) / (max_lines - 2)  # -2 for first and last
        
        for i in range(1, max_lines - 1):
            rank_idx = int(i * step_size)
            if rank_idx < len(unique_ranks):
                rank = unique_ranks[rank_idx]
                selected_steps.append(rank_to_step[rank])
        
        last_step = n - 1
        if last_step not in selected_steps:
            selected_steps.append(last_step)
        
        selected_steps = sorted(set(selected_steps))
    
    for i, step in enumerate(selected_steps):
        rank = best_ranks[step]
        eval_step = best_evals[step]
        if i == len(selected_steps) - 1:
            s+= "Final "
        s += f"Rank={rank} at eval={eval_step}\n"
    return s

class BaseEvaluator:
    todd: Todd
    _best_rank: int=100000
    best_matrix: np.ndarray
    best_pathes: List[Path]
    best_report: str
    best_pcfg: str
    best_eval: int = 0
    total_eval: int = 0
    best_seen: int = 0
    shedule: str = "rank"
    bs_width: RankSchedule = RankSchedule.constant(1)
    todd_width: RankSchedule = RankSchedule.constant(1)
    current_path: Path
    best_ranks: List[int] 
    best_evals: List[int] 
    def __init__(self, mat: Matrix, max_depth: int, fin_rank: int = 161, shedule: str = "rank", fill_tcounts=False):
        self.with_report = False
        self.current_path = Path()
        self.init_rank = mat.rows
        self.current_path.final_node = Node(mat)
        self.fin_rank = fin_rank
        self.shedule = shedule
        self.dao: Dao = Dao()
        self.dao.threads = 4
        self.todd = Todd(self.dao, max_depth)
        self.max_depth = max_depth
        self.tcount = []
        self.best_pathes = []
        self.best_ranks = []
        self.best_evals = []
        self.active_params = []
        self.best_params = []
        self.x0 = [0 for i in range(200)]
        self.reinit()    

    def set_up_new_init(self, path_num:int, rank_thr:int, xopt=None):

        if path_num >= len(self.best_pathes):
            return None
        new_path = self.best_pathes[path_num].branch_path_at(rank_thr=rank_thr)
        if new_path is None:
            return None
        self.current_path = new_path
        self.dao = deepcopy(self.current_path.daos[-1])
        self.todd = Todd(self.dao, self.max_depth)
        if xopt is not None:
            self.insert(xopt)
        else:
            self.x0 = new_path.x0s[-1]
        self.reinit()
        return self.extract_active()
    
    @property
    def init(self):
        return self.current_path.final_node.state

    @property
    def path_num(self):
        return len(self.best_pathes)
        
    @property
    def best_rank(self):
        return self._best_rank

    def map_par(self, mapping: callable, thr: int, **kwargs):
        if self.init.rows > thr:
            self.active_params.append(self.idx)
        self.idx += 1
        return mapping(self.x0[self.idx - 1], **kwargs)
        
    def insert(self, x):
        for i, a in enumerate(self.active_params):
            self.x0[a] = x[i]
        return self.x0
    
    def reinit(self):
        self.active_params = []
        self.idx = 0
        self.policy_mapping()
        return self.extract_active()
        
    def extract_active(self):
        x = []
        for i, a in enumerate(self.active_params):
            x.append(self.x0[a])
        return x
        
    def __call__(self, params: Iterable):
        pass

    def policy_mapping(self):
        pass

    def run(self, params, seeds, max_workers=2):
        if len(params) != len(self.active_params):
            raise RuntimeError(f"Num of params {len(params)} is not equal to the num of active params {len(self.active_params)}")
        self.insert(params)
        self.reinit()
        # self.policy_setup(params)
        if max_workers == 1:
            results = [(*_worker_run_one_from_template(seed, self.current_path, self.todd, self.bs_width, self.todd_width), seed) for seed in seeds]
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                futures = [
                    ex.submit(_worker_run_one_from_template, seed, self.current_path, self.todd, self.bs_width, self.todd_width)
                    for seed in seeds
                ]
                results = [(*f.result(), s) for f, s in zip(futures, seeds)]

        # process deterministically in seed order
        seed_to_idx = {s:i for i,s in enumerate(seeds)}
        results.sort(key=lambda x: seed_to_idx[x[0]])

        mats_ranks = []
        for _, node, counters, seed in results:
            rank = node.state.rows
            mats_ranks.append(rank)
            self.total_eval += counters[0]
            self.tcount.append(rank)
            if rank < self._best_rank:
                self.best_seen = 0
                self.best_ranks.append(rank)
                self.best_evals.append(self.total_eval)
                self.best_eval = self.total_eval
                self._best_rank = rank
                self.best_pathes = [self.current_path.branch_path(node, self.dao, self.x0)]
            if rank == self._best_rank:
                self.best_pathes.append(self.current_path.branch_path(node, self.dao, self.x0))
                self.best_seen += counters[1]
                self.best_seed = seed
        return mats_ranks

    def get_best(self):
        return (
            self.best_pathes[0].final_node.state.to_numpy(), 
            self.best_pathes[0].format_path_stats_tiny(),
            "\nbest_policy:\n" +
            str(dao_rank_to_str(self.best_pathes[0].daos, self.best_pathes[0].ranks_thr + [0])) + "\nsearch_stat:\n" +
            f"rank 0.9q={np.quantile(self.tcount, 0.9)} \n" +
            f"rank 0.1q={np.quantile(self.tcount, 0.1)} \n" +
            print_uniform_by_rank(self.best_ranks, self.best_evals, 8) +
            f"total_evals: {self.total_eval}" +
            f"\nbest_seen_times: {self.best_seen}"
            )
    def set_final_scores(self, x: Any, vals=None):
        if vals is not None:
            x = list(zip(x, vals))
        self.dao.mode.final_scores = _to_frank_schedule(x)
        
    def set_pool_scores(self, x: Any, vals=None):
        if vals is not None:
            x = list(zip(x, vals))
        self.dao.mode.pool_scores = _to_erank_schedule(x)

    def set_num_samples(self, x: Any, vals=None):
        if vals is not None:
            x = list(zip(x, vals))
        self.dao.mode.num_samples = _to_rank_schedule(x)

    def set_gen_part(self, x: Any, vals=None):
        if vals is not None:
            x = list(zip(x, vals))
        self.dao.mode.gen_part = _to_rank_schedule(x)

    def set_beamsearch_width(self, x: Any, vals=None):
        if vals is not None:
            x = list(zip(x, vals))
        self.bs_width = _to_rank_schedule(x)
        
    def set_todd_width(self, x: Any, vals=None):
        if vals is not None:
            x = list(zip(x, vals))
        self.todd_width = _to_rank_schedule(x)

    def set_min_z_to_research(self, x: Any, vals=None):
        if vals is not None:
            x = list(zip(x, vals))
        self.dao.mode.max_z_to_research = _to_rank_schedule(x)
        self.dao.threads = 3

    def set_min_pool_size(self, x: Any, vals=None):
        if vals is not None:
            x = list(zip(x, vals))
        self.dao.mode.min_pool_size = _to_rank_schedule(x)

    def set_temperature(self, x: Any, vals=None):
        if vals is not None:
            x = list(zip(x, vals))
        self.dao.mode.temperature = _to_rank_schedule(x)

    def set_non_improving_prob(self, x: Any, vals=None):
        if vals is not None:
            x = list(zip(x, vals))
        self.dao.mode.non_improving_prob = _to_rank_schedule(x)

    def set_max_pool_size(self, x: Any, vals=None):
        if vals is not None:
            x = list(zip(x, vals))
        self.dao.mode.top_pool = _to_rank_schedule(x)

    def set_max_tohpe(self, x: Any, vals=None):
        if vals is not None:
            x = list(zip(x, vals))
        self.dao.mode.max_tohpe = _to_rank_schedule(x)

    def set_try_only_tohpe(self, x: Any, vals=None):
        if vals is not None:
            x = list(zip(x, vals))
        self.dao.mode.try_only_tohpe = _to_rank_schedule(x)

    def set_max_from_single_ns(self, x: Any, vals=None):
        if vals is not None:
            x = list(zip(x, vals))
        self.dao.mode.max_from_single_ns = _to_rank_schedule(x)

    def set_tohpe_num_best(self, x: Any, vals=None):
        if vals is not None:
            x = list(zip(x, vals))
        self.dao.mode.num_tohpe_sample = _to_rank_schedule(x)

    def set_min_reduction(self, x: Any, vals=None):
        if vals is not None:
            x = list(zip(x, vals))
        self.dao.mode.min_reduction = _to_rank_schedule(x)

    def set_max_reduction(self, x: Any, vals=None):
        if vals is not None:
            x = list(zip(x, vals))
        self.dao.mode.max_reduction  = _to_rank_schedule(x)