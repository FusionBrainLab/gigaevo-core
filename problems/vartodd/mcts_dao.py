
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generic, List, Mapping, Optional, Sequence, Tuple, TypeVar, Union

from node import Node, PolicyConfig, ExplorationScore, FinalizationScore
from copy import deepcopy

def _get(obj, *names, default=None):
    """First existing attribute in names, else default."""
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return default

def _q_ge(num_better: int, total: int) -> float:
    if not total:
        return 0.0
    return 1.0 - (num_better / total)

@dataclass(slots=True)
class Path:
    final_node: Node = None
    ranks_thr: List[int] = field(default_factory=list)
    daos: List[Dao] = field(default_factory=list)
    active_params: List[List[float]] = field(default_factory=list)
    x0s: List[List[float]] = field(default_factory=list)
    def branch_path(self, node: Node, dao: Dao, x0: List[float]):
        new_path = Path()
        new_path.final_node = node
        new_path.daos = self.daos + [deepcopy(dao)]
        new_path.x0s = self.x0s + [deepcopy(x0)]
        new_path.ranks_thr = deepcopy(self.ranks_thr)
        return new_path
        
    def branch_path_at(self, rank_thr: int):
        print(f"{self.final_node.state.rows=}")
        node = self.final_node
        if node.state.rows > rank_thr:
            return None
        while node is not None and node.state.rows < rank_thr:
            node = node.parent
        if node is None:
            return None
        new_path = Path()
        new_path.final_node = node
        new_path.ranks_thr = self.ranks_thr + [node.state.rows]
        new_path.daos = deepcopy(self.daos)
        new_path.x0s = deepcopy(self.x0s)
        return new_path

    def format_path_stats_tiny(self) -> str:
        _out = []
        # for node in out
        tohpe_zero = 0
        path = []
        node = self.final_node
        while node is not None:
            path.append(node)
            node = node.parent
        path = list(reversed(path))
        for index, node in enumerate(path[1:]):
            if node.incoming is None:
                return "path stats unavailable (missing incoming info)"

            cand = node.incoming.cand
            s = node.incoming.global_info
            maxbucket = node.incoming.global_info.max_bucket
            bs = node.incoming.cand.bucket_size
            n = int(_get(s, "nonzero", default=0) or 0)

            r = int(_get(cand, "reduction", default=0) or 0)
            rmax = int(_get(s, "max_reduction", "best_reduction", default=r) or r)
            mean_dim = s.mean_basis
            max_dim = s.max_basis
            dim = cand.basis_dim
            rd = r - rmax
            rbetter = int(_get(cand, "num_better_red", "num_better_reduction", default=0) or 0)
            rq = _q_ge(rbetter, n)

            is_beyond = n == s.accepted_non_improving
            if not is_beyond:
                n = n - s.accepted_non_improving 
                    
            if tohpe_zero == 0 and s.accepted_tohpe == 0:
                tohpe_zero = index
            _out.append(
                f"r{r}/d{rd:+d}/q{rq:.2f}%;bd{dim}/d{dim - max_dim}/m{mean_dim:.3g};bs{bs}/d{bs-maxbucket:+d}"
                f"tha{s.accepted_tohpe}/tda{s.accepted}/b{int(is_beyond)}"
            )

        out = []
        for index, node in enumerate(path[:-1]):
            rank = node.state.rows
            out.append(f"{rank}:{_out[index]}")
        out.append(f"{path[-1].state.rows}:final")
        l = len(out)
        if len(out) > 15:
            if tohpe_zero <= 7:
                first_part = out[0:7]
            else:
                first_part = out[0:5]
            if len(out) - tohpe_zero <= 7:
                second_part = out[-7:]
            else:
                second_part = out[-5:]
            if tohpe_zero > 7 and len(out) - tohpe_zero > 7:
                return "\n".join(first_part + ["..."] + out[tohpe_zero - 1 : tohpe_zero + 3] + ["..."] + second_part) 
            return "\n".join(first_part + ["..."] + second_part) 
        return "\n".join(out)
T = TypeVar("T")


def _as_int(x: Any) -> int:
    if isinstance(x, bool):
        return int(x)
    return int(x)


def _as_float(x: Any) -> float:
    if isinstance(x, bool):
        return float(int(x))
    return float(x)


def _as_bool(x: Any) -> bool:
    if isinstance(x, str):
        lx = x.strip().lower()
        if lx in {"1", "true", "yes", "y", "on"}:
            return True
        if lx in {"0", "false", "no", "n", "off"}:
            return False
    return bool(x)


@dataclass(slots=True)
class DepthSchedule(Generic[T]):
    points: List[Tuple[int, T]] = field(default_factory=list)

    @staticmethod
    def constant(value: T) -> "DepthSchedule[T]":
        return DepthSchedule(points=[(0, value)])

    @staticmethod
    def from_any(value: Union["DepthSchedule[T]", Sequence[Tuple[int, T]], T]) -> "DepthSchedule[T]":
        if isinstance(value, DepthSchedule):
            return value
        if isinstance(value, (list, tuple)):
            if value and isinstance(value[0], tuple) and len(value[0]) == 2:
                pts = [(int(d), v) for d, v in value]  # type: ignore[misc]
                pts.sort(key=lambda x: x[0])
                return DepthSchedule(points=pts)
        return DepthSchedule.constant(value)  # type: ignore[arg-type]

    def at(self, depth: int) -> T:
        if not self.points:
            raise ValueError("empty schedule")
        d = int(depth)
        cur = self.points[0][1]
        for dd, vv in self.points:
            if dd <= d:
                cur = vv
            else:
                break
        return cur

@dataclass(slots=True)
class RankSchedule(Generic[T]):
    points: List[Tuple[int, T]] = field(default_factory=list)

    @staticmethod
    def constant(value: T) -> "RankSchedule[T]":
        return RankSchedule(points=[(0, value)])

    @staticmethod
    def from_any(value: Union["RankSchedule[T]", Sequence[Tuple[int, T]], T]) -> "RankSchedule[T]":
        if isinstance(value, RankSchedule):
            return value
        if isinstance(value, (list, tuple)):
            if value and isinstance(value[0], tuple) and len(value[0]) == 2:
                pts = [(int(d), v) for d, v in value]  # type: ignore[misc]
                pts.sort(key=lambda x: x[0], reverse=True)
                return RankSchedule(points=pts)
        return RankSchedule.constant(value)  # type: ignore[arg-type]

    def at(self, rank: int) -> T:
        if not self.points:
            raise ValueError("empty schedule")
        r = int(rank)
        cur = self.points[0][1]
        for rr, vv in self.points:
            if r <= rr:
                cur = vv
            else:
                cur = vv
                break
        return cur


@dataclass(slots=True)
class UctDao:
    name: str = "puct"
    c: DepthSchedule[float] = field(default_factory=lambda: DepthSchedule.constant(2.5))
    fn: Optional[Callable[..., float]] = None

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "UctDao":
        return UctDao(
            name=str(d.get("name", "puct")),
            c=DepthSchedule.from_any(d.get("c", 2.5)),
            fn=d.get("fn", None),
        )

    def c_at(self, depth: int) -> float:
        return _as_float(self.c.at(depth))


@dataclass(slots=True)
class TreeDao:
    rollout_add: bool = True
    rollout_active: bool = False
    rollout_frozen_until: int = 0

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "TreeDao":
        return TreeDao(
            rollout_add=_as_bool(d.get("rollout_add", True)),
            rollout_active=_as_bool(d.get("rollout_active", False)),
            rollout_frozen_until=_as_int(d.get("rollout_frozen_until", 0)),
        )


@dataclass(slots=True)
class NonImprovingDao:
    non_improving_prob: DepthSchedule[float] = field(default_factory=lambda: DepthSchedule.constant(0.0))

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "NonImprovingDao":
        return NonImprovingDao(
            non_improving_prob=DepthSchedule.from_any(d.get("non_improving_prob", 0.0)),
        )

    def prob_at(self, depth: int) -> float:
        return _as_float(self.non_improving_prob.at(depth))


@dataclass(slots=True)
class ModeDao:
    num_samples: DepthSchedule[int] = field(default_factory=lambda: DepthSchedule.constant(64))
    top_pool: DepthSchedule[int] = field(default_factory=lambda: DepthSchedule.constant(1))
    selection: DepthSchedule[str] = field(default_factory=lambda: DepthSchedule.constant("softmax"))
    temperature: DepthSchedule[float] = field(default_factory=lambda: DepthSchedule.constant(0.0))
    max_tohpe: DepthSchedule[int] = field(default_factory=lambda: DepthSchedule.constant(1))
    gen_part: DepthSchedule[float] = field(default_factory=lambda: DepthSchedule.constant(1.0))
    max_z_to_research: DepthSchedule[float] = field(default_factory=lambda: DepthSchedule.constant(5000))
    pool_scores: DepthSchedule[ExplorationScore] = field(default_factory=lambda: DepthSchedule.constant(ExplorationScore([0.5, 0.5, 0.0, 0.0, 0])))
    final_scores: DepthSchedule[FinalizationScore] = field(default_factory=lambda: DepthSchedule.constant(FinalizationScore([0.5, 0.5, 0.0, 0.0, 0, 0])))
    try_only_tohpe: DepthSchedule[bool] = field(default_factory=lambda: DepthSchedule.constant(True))
    max_from_single_ns: DepthSchedule[int] = field(default_factory=lambda: DepthSchedule.constant(5))
    min_reduction: DepthSchedule[int] = field(default_factory=lambda: DepthSchedule.constant(0))
    max_reduction: DepthSchedule[int] = field(default_factory=lambda: DepthSchedule.constant(100))
    min_pool_size: DepthSchedule[int] = field(default_factory=lambda: DepthSchedule.constant(0))
    non_improving_prob: DepthSchedule[float] = field(default_factory=lambda: DepthSchedule.constant(0.0))
    num_tohpe_sample: DepthSchedule[int] = field(default_factory=lambda: DepthSchedule.constant(1))
    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "ModeDao":
        mz = d.get("max_z_to_research", None)
        if mz is None:
            mz = d.get("gen_part", 1.0)
        return ModeDao(
            num_samples=DepthSchedule.from_any(d.get("num_samples", 64)),
            top_pool=DepthSchedule.from_any(d.get("top_pool", 96)),
            selection=DepthSchedule.from_any(d.get("selection", "best")),
            temperature=DepthSchedule.from_any(d.get("temperature", 0.0)),
            max_tohpe=DepthSchedule.from_any(d.get("max_tohpe", 5)),
            max_z_to_research=DepthSchedule.from_any(mz),
            pool_scores=DepthSchedule.from_any(d.get("weights", ExplorationScore(0.5, 0.5, 0.0,  0.0))),
            final_scores=DepthSchedule.from_any(d.get("weights", FinalizationScore(0.5, 0.5, 0.0,  0.0, 1.0))),
            try_only_tohpe=DepthSchedule.from_any(d.get("try_only_tohpe", True)),
            max_from_single_ns=DepthSchedule.from_any(d.get("max_from_single_ns", 100)),
            max_reduction=DepthSchedule.from_any(d.get("max_reduction", 100)),
            min_reduction=DepthSchedule.from_any(d.get("min_reduction", 0)),
            non_improving_prob=DepthSchedule.from_any(d.get("non_improving_prob", 0.0)),
        )

    def policy_kwargs(self, *, depth: int, num_candidates: int,) -> Dict[str, Any]:
        return {
            "num_samples": _as_int(self.num_samples.at(depth)),
            "num_candidates": _as_int(num_candidates),
            "top_pool": _as_int(self.top_pool.at(depth)),
            "selection": str(self.selection.at(depth)),
            "temperature": _as_float(self.temperature.at(depth)),
            "max_tohpe": _as_int(self.max_tohpe.at(depth)),
            "gen_part": _as_float(self.gen_part.at(depth)),
            "max_z_to_research": _as_int(self.max_z_to_research.at(depth)),
            "min_pool_size": _as_int(self.min_pool_size.at(depth)),
            "ExplorationScore": self.pool_scores.at(depth),
            "FinalizationScore": self.final_scores.at(depth),
            "try_only_tohpe": _as_bool(self.try_only_tohpe.at(depth)),
            "non_improving_prob": _as_float(self.non_improving_prob.at(depth)),
            "max_from_single_ns": _as_int(self.max_from_single_ns.at(depth)),
            "min_reduction": _as_int(self.min_reduction.at(depth)),
            "max_reduction": _as_int(self.max_reduction.at(depth)),
            "tohpe_sample": _as_int(self.num_tohpe_sample.at(depth)),
        }


@dataclass(slots=True)
class Dao:
    iterations: int = 3600
    discount: float = 0.995
    max_depth: int = 32
    threads: int = 4

    branching: DepthSchedule[int] = field(default_factory=lambda: DepthSchedule.constant(4))
    rollout_depth: DepthSchedule[int] = field(default_factory=lambda: DepthSchedule.constant(10))

    uct: UctDao = field(default_factory=UctDao)
    tree: TreeDao = field(default_factory=TreeDao)

    modes: Dict[str, ModeDao] = field(default_factory=dict)

    unfrozen_top: int = 0
    freeze_spread: int = 0
    def __post_init__(self):
        if "default" not in self.modes:
            self.modes["default"] = ModeDao()
    @staticmethod
    def from_dict(cfg: Mapping[str, Any]) -> "Dao":
        modes_in = cfg.get("modes", {}) or {}
        modes: Dict[str, ModeDao] = {}
        if isinstance(modes_in, Mapping):
            for k, v in modes_in.items():
                if isinstance(v, Mapping):
                    modes[str(k)] = ModeDao.from_dict(v)

        freeze = cfg.get("freeze", {}) or {}
        return Dao(
            iterations=_as_int(cfg.get("iterations", 3600)),
            discount=_as_float(cfg.get("discount", 0.995)),
            max_depth=_as_int(cfg.get("max_depth", 32)),
            threads=_as_int(cfg.get("threads", 4)),
            branching=DepthSchedule.from_any(cfg.get("branching", 1)),
            rollout_depth=DepthSchedule.from_any(cfg.get("rollout_depth", 0)),
            uct=UctDao.from_dict(cfg.get("uct", {}) or {}),
            tree=TreeDao.from_dict(cfg.get("tree", {}) or {}),
            modes=modes,
            unfrozen_top=_as_int(freeze.get("unfrozen_top", cfg.get("unfrozen_top", 0))),
            freeze_spread=_as_int(freeze.get("freeze_spread", cfg.get("freeze_spread", 0)))
        )
    @property
    def mode(self):
        return self.modes["default"]
    
    def branching_at(self, depth: int) -> int:
        return _as_int(self.branching.at(depth))

    def rollout_depth_at(self, depth: int) -> int:
        return _as_int(self.rollout_depth.at(depth))

    def policy_config_at(self, depth: int, mode: str="default", num_candidates: int = 1) -> PolicyConfig:
        m = self.modes.get(mode)
        if m is None:
            raise KeyError(f"unknown mode: {mode}")
        # num_candidates = self.branching_at(depth) if mode == "expand" else 1
        # num
        kwargs = m.policy_kwargs(depth=depth, num_candidates=num_candidates)
        kwargs["threads"] = self.threads
        return PolicyConfig(**kwargs)
