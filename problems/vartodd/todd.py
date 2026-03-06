
from __future__ import annotations

import math
import random
from typing import Any, Callable, Dict, Optional, Tuple, List

import numpy as np

from mcts_dao import Dao, Path
from node import Node, ActionInfo, Matrix, Result, Stats, policy_iteration

import heapq
class Todd:
    def __init__(self, dao: Dao, depth):
        self.dao: Dao = dao
        self.depth = depth

    def run(self, path: Path, width, todd_width, with_report=False, seed=1):
        root = path.final_node
        # root = displ
        node = root
        best_node = root
        counter = 0
        best_counter = 0
        nodes = [root]
        for i in range(self.depth):
            new_nodes = []
            counter = max(counter, len(nodes))
            for node in nodes:
                # width.at(node.state.rows)
                pcfg = self.dao.policy_config_at(depth=node.state.rows, mode="default", num_candidates=todd_width.at(node.state.rows))
                out: Result = policy_iteration(cur_mat=node.state, policy_cfg=pcfg, seed=seed, add_seed=0)
                chosen = out.chosen
                states = out.states
                if not chosen or not states:
                    break
                if len(states) == len(chosen) + 1:
                    states = states[1:]

                for cand, state in zip(chosen, states):
                    info = ActionInfo.from_candidate(cand, global_info=out.stats, source="rollout")
                    child = node.add_child(
                        state=state,
                        incoming=info,
                        prior=1.0
                    )
                    if child.state.rows < best_node.state.rows:
                        best_counter = 0
                        best_node = child
                    if child.state.rows == best_node.state.rows:
                        best_counter += 1
                    new_nodes.append(child)
            nodes = heapq.nlargest(width.at(best_node.state.rows), new_nodes, lambda x : x.incoming.cand.final_score)
        if with_report:
            best_counter = min(counter, best_counter)
            return best_node, (counter, best_counter)
        else:
            return node.state.to_numpy()