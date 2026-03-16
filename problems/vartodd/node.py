from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional
from pyvartodd.pyvartodd import Matrix, CandidateExport, Stats, Result,  PolicyConfig, ExplorationScore, FinalizationScore, policy_iteration, Tensor3D, Function, ScoringFunction


@dataclass(slots=True)
class ActionInfo:
    cand: CandidateExport
    global_info: Stats
    source: str = ""

    @staticmethod
    def from_candidate(
        cand: CandidateExport,
        *,
        global_info: Optional[Stats] = None,
        source: str = "",
    ) -> "ActionInfo":
        return ActionInfo(
            cand=cand,
            global_info=global_info,
            source=source,
        )

    @property
    def reduction(self):
        return self.cand.reduction

@dataclass(slots=True)
class Node:
    state: Matrix
    parent: Optional["Node"] = None
    incoming: Optional[ActionInfo] = None
    depth: int = 0

    @property
    def value_mean(self) -> float:
        return 0.0 if self.visits == 0 else self.value_sum / self.visits

    def add_child(
        self,
        *,
        state: Matrix,
        incoming: ActionInfo,
        prior: float,
        frozen_until: int = 0,
        active: bool = True,
        init_rank: int = 1000000
    ) -> "Node":
        child = Node(
            state=state,
            parent=self,
            incoming=incoming,
            depth=self.depth + 1,
        )
        return child

    def path_from_root(self) -> List["Node"]:
        out: List["Node"] = []
        cur: Optional["Node"] = self
        while cur is not None:
            out.append(cur)
            cur = cur.parent
        out.reverse()
        return out