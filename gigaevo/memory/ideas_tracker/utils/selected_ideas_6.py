"""
Idea ORIGIN analysis with quartile rows (Q1..Q4 + ALL) in ONE summary table.

IMPORTANT ASSUMPTION:
- banks.json `active_bank[*].programs` means: the idea ORIGINATED in those programs
  (NOT that the idea is present/used in those programs).

Therefore this script DOES NOT compute "idea presence" or with/without metrics.
Instead it computes selection-friendly ORIGIN metrics per quartile and overall (ALL),
based on intro/origin events and lineage outcomes.

Output (single CSV):
- idea_origin_quartile_summary.csv
  One row per (idea, quartile) for quartiles Q1..Q4 plus ALL => 5 rows per idea.

Run:
  python selected_ideas_6.py --ideas banks.json --programs programs.json --output_dir out

Key args:
  --quartile_mode generation_range | generation_quantiles
  --elite_pct 0.05
  --desc_k 10 (descendant window in generations; -1 means up to final generation)
  --sibling_mode best_parent | parent_set
  --sibling_gen_window 0 (0 = exact generation; >0 buckets generations for sibling comparison)

NEW (this patch):
- Adds an additional sibling comparison that ignores generation entirely, i.e.
  compares the intro child against ALL sibling programs across the whole evolution.

New event-level fields:
  SiblingWin_allgens
  SiblingPercentile_allgens
  SiblingDelta_allgens

New summary columns:
  SiblingWinRate_allgens
  SiblingPercentile_allgens_median
  SiblingDelta_allgens_median
"""

from __future__ import annotations

import argparse
import bisect
from collections import defaultdict, deque
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

from loguru import logger
import pandas as pd


# -----------------------------
# Loading
# -----------------------------
def load_ideas(path: str) -> tuple[dict[str, set[str]], dict[str, str]]:
    """
    Returns:
      idea_to_origin_programs: idea_id -> set(program_id) where idea originated
      idea_desc: idea_id -> description
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    bank = None
    if isinstance(data, list) and data and isinstance(data[0], dict):
        if "active_bank" in data[0]:
            bank = data[0]["active_bank"]
        elif "ideas" in data[0]:
            bank = data[0]["ideas"]
    elif isinstance(data, dict) and "active_bank" in data:
        bank = data["active_bank"]

    if bank is None:
        raise ValueError("Could not find 'active_bank' list in banks JSON.")

    idea_to_origin_programs: dict[str, set[str]] = {}
    idea_desc: dict[str, str] = {}

    for idea in bank:
        if not isinstance(idea, dict):
            continue
        idea_id = idea.get("id") or idea.get("short_id") or idea.get("name")
        if not idea_id:
            continue
        # New format uses `programs`; old format used `linked_programs`.
        origin_programs = idea.get("programs")
        if origin_programs is None:
            origin_programs = idea.get("linked_programs", [])
        origin_programs = origin_programs or []

        idea_to_origin_programs[str(idea_id)] = set(str(x) for x in origin_programs)
        idea_desc[str(idea_id)] = str(idea.get("description", "") or "")

    return idea_to_origin_programs, idea_desc


def load_programs(path: str) -> dict[str, dict]:
    """
    Supports:
    - list of snapshots with {"programs":[...]}
    - list of program dicts
    - dict with {"programs":[...]}
    Keeps the best fitness version if duplicates appear.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    programs: dict[str, dict] = {}

    def fit_of(p: dict) -> float:
        try:
            return float(p.get("fitness", float("-inf")))
        except Exception:
            return float("-inf")

    if isinstance(data, list):
        for snap in data:
            if (
                isinstance(snap, dict)
                and "programs" in snap
                and isinstance(snap["programs"], list)
            ):
                for p in snap["programs"]:
                    if not isinstance(p, dict) or "id" not in p:
                        continue
                    pid = str(p["id"])
                    if pid not in programs or fit_of(p) > fit_of(programs[pid]):
                        programs[pid] = p
            elif isinstance(snap, dict) and "id" in snap:
                pid = str(snap["id"])
                if pid not in programs or fit_of(snap) > fit_of(programs[pid]):
                    programs[pid] = snap
    elif (
        isinstance(data, dict)
        and "programs" in data
        and isinstance(data["programs"], list)
    ):
        for p in data["programs"]:
            if not isinstance(p, dict) or "id" not in p:
                continue
            programs[str(p["id"])] = p
    else:
        raise ValueError("Unexpected programs JSON format.")

    return programs


def invert_idea_to_programs(
    idea_to_programs: dict[str, set[str]],
) -> dict[str, set[str]]:
    prog_to_ideas: dict[str, set[str]] = defaultdict(set)
    for idea, pids in idea_to_programs.items():
        for pid in pids:
            prog_to_ideas[str(pid)].add(idea)
    return prog_to_ideas


def build_parents(programs: dict[str, dict]) -> dict[str, list[str]]:
    parents_of: dict[str, list[str]] = {}
    for pid, p in programs.items():
        parents = p.get("parents", []) or []
        if isinstance(parents, str):
            try:
                parents = json.loads(parents)
            except (json.JSONDecodeError, TypeError):
                parents = []
        parents = [str(x) for x in parents if str(x) in programs]
        parents_of[str(pid)] = parents
    return parents_of


def build_children(parents_of: dict[str, list[str]]) -> dict[str, list[str]]:
    children_of: dict[str, list[str]] = defaultdict(list)
    for child, pars in parents_of.items():
        for par in pars:
            children_of[par].append(child)
    return children_of


# -----------------------------
# Quartiles
# -----------------------------
def generation_quantile_bounds(
    gens: list[int], qs=(0.25, 0.50, 0.75)
) -> tuple[float, float, float]:
    gs = sorted(gens)
    if not gs:
        raise ValueError("No generations available.")

    def qval(q: float) -> float:
        idx = int(round(q * (len(gs) - 1)))
        idx = max(0, min(len(gs) - 1, idx))
        return float(gs[idx])

    return qval(qs[0]), qval(qs[1]), qval(qs[2])


def generation_range_bounds(gens: list[int]) -> tuple[float, float, float]:
    gmin, gmax = min(gens), max(gens)
    span = (gmax - gmin) + 1
    b1 = gmin + 0.25 * span
    b2 = gmin + 0.50 * span
    b3 = gmin + 0.75 * span
    return b1, b2, b3


def generation_to_quartile(gen: int, b1: float, b2: float, b3: float) -> str:
    if gen < b1:
        return "Q1"
    if gen < b2:
        return "Q2"
    if gen < b3:
        return "Q3"
    return "Q4"


# -----------------------------
# Robust stats
# -----------------------------
def robust_median(xs: list[float]) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    n = len(ys)
    m = n // 2
    return ys[m] if n % 2 == 1 else 0.5 * (ys[m - 1] + ys[m])


def robust_quantile(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    q = max(0.0, min(1.0, float(q)))
    idx = int(round(q * (len(ys) - 1)))
    idx = max(0, min(len(ys) - 1, idx))
    return ys[idx]


def mad(xs: list[float]) -> float:
    if not xs:
        return float("nan")
    med = robust_median(xs)
    devs = [abs(x - med) for x in xs]
    return robust_median(devs)


def percentile_rank(sorted_vals: list[float], x: float) -> float:
    if not sorted_vals:
        return float("nan")
    k = bisect.bisect_right(sorted_vals, x)
    return k / len(sorted_vals)


# -----------------------------
# Elite threshold
# -----------------------------
def elite_threshold_by_top_k(
    fitness_vals: list[float], elite_pct: float
) -> tuple[float, int]:
    xs = [float(x) for x in fitness_vals if math.isfinite(float(x))]
    if not xs:
        return float("nan"), 0
    xs.sort()
    n = len(xs)
    elite_pct = float(elite_pct)
    elite_pct = max(0.000001, min(1.0, elite_pct))
    elite_k = int(math.ceil(elite_pct * n))
    elite_k = max(1, min(n, elite_k))
    threshold = xs[-elite_k]
    return threshold, elite_k


# -----------------------------
# Baselines
# -----------------------------
def pick_best_parent(
    parents: list[str], programs: dict[str, dict]
) -> tuple[str, float] | None:
    best_pid = None
    best_fit = float("-inf")
    for par in parents:
        p = programs.get(par)
        if not p:
            continue
        try:
            f = float(p.get("fitness", float("nan")))
        except Exception:
            continue
        if math.isfinite(f) and f > best_fit:
            best_fit = f
            best_pid = par
    if best_pid is None:
        return None
    return best_pid, best_fit


def mean_parent_fitness(parents: list[str], programs: dict[str, dict]) -> float | None:
    fits = []
    for par in parents:
        p = programs.get(par)
        if not p:
            continue
        try:
            f = float(p.get("fitness", float("nan")))
        except Exception:
            continue
        if math.isfinite(f):
            fits.append(f)
    if not fits:
        return None
    return sum(fits) / len(fits)


# -----------------------------
# Intro events (origin-only)
# -----------------------------
@dataclass
class IntroEvent:
    idea_id: str
    child_id: str
    child_gen: int
    child_fit: float
    parents: list[str]
    best_parent_id: str
    best_parent_fit: float
    mean_parent_fit: float
    quartile: str


def compute_intro_events_origin_only(
    programs: dict[str, dict],
    prog_to_origin_ideas: dict[str, set[str]],
    parents_of: dict[str, list[str]],
    b1: float,
    b2: float,
    b3: float,
) -> list[IntroEvent]:
    """
    Intro/origin event definition:
    - The idea originated in `child` (child listed in linked_programs)
    - The idea did NOT originate in any of `child`'s parents
      (supports independent rediscovery / reintroduction)
    """
    events: list[IntroEvent] = []

    for child_id, parents in parents_of.items():
        if not parents:
            continue

        child_origin_ideas = prog_to_origin_ideas.get(child_id, set())
        if not child_origin_ideas:
            continue

        parent_union: set[str] = set()
        for par in parents:
            parent_union |= prog_to_origin_ideas.get(par, set())

        introduced = child_origin_ideas - parent_union
        if not introduced:
            continue

        pchild = programs.get(child_id, {})
        gen_child = pchild.get("generation", None)
        try:
            f_child = float(pchild.get("fitness", float("nan")))
        except Exception:
            f_child = float("nan")
        if not (isinstance(gen_child, int) and math.isfinite(f_child)):
            continue

        best = pick_best_parent(parents, programs)
        mfit = mean_parent_fitness(parents, programs)
        if best is None or mfit is None:
            continue
        best_pid, best_fit = best
        q = generation_to_quartile(int(gen_child), b1, b2, b3)

        for idea in introduced:
            events.append(
                IntroEvent(
                    idea_id=idea,
                    child_id=child_id,
                    child_gen=int(gen_child),
                    child_fit=f_child,
                    parents=parents,
                    best_parent_id=best_pid,
                    best_parent_fit=float(best_fit),
                    mean_parent_fit=float(mfit),
                    quartile=q,
                )
            )
    return events


# -----------------------------
# Sibling groups (per generation bucket)
# -----------------------------
def build_sibling_groups(
    programs: dict[str, dict],
    parents_of: dict[str, list[str]],
    mode: str,
    gen_window: int,
) -> dict[tuple, list[str]]:
    """
    Returns mapping sibling_key -> list of program_ids in that sibling group.

    mode:
      - "best_parent": group by (best_parent_id, gen_bucket)
      - "parent_set":  group by (sorted(parents), gen_bucket)

    gen_bucket:
      - if gen_window == 0 => exact generation
      - else bucket = gen // (gen_window+1)
    """
    groups: dict[tuple, list[str]] = defaultdict(list)

    def bucket(gen: int) -> int:
        if gen_window <= 0:
            return gen
        return gen // (gen_window + 1)

    for pid, pars in parents_of.items():
        if not pars:
            continue
        p = programs.get(pid, {})
        gen = p.get("generation", None)
        fit = p.get("fitness", None)
        if not (isinstance(gen, int) and fit is not None):
            continue
        try:
            f = float(fit)
        except Exception:
            continue
        if not math.isfinite(f):
            continue

        if mode == "best_parent":
            best = pick_best_parent(pars, programs)
            if best is None:
                continue
            best_pid, _ = best
            key: tuple[Any, ...] = ("best_parent", best_pid, bucket(gen))
        else:
            key = ("parent_set", tuple(sorted(pars)), bucket(gen))

        groups[key].append(pid)

    return groups


# -----------------------------
# Sibling groups (ALL generations)
# -----------------------------
def build_sibling_groups_allgens(
    programs: dict[str, dict],
    parents_of: dict[str, list[str]],
    mode: str,
) -> dict[tuple, list[str]]:
    """
    Returns mapping sibling_key -> list of program_ids in that sibling group,
    ignoring generation completely.

    mode:
      - "best_parent": group by (best_parent_id)
      - "parent_set":  group by (sorted(parents))
    """
    groups: dict[tuple, list[str]] = defaultdict(list)

    for pid, pars in parents_of.items():
        if not pars:
            continue
        p = programs.get(pid, {})
        fit = p.get("fitness", None)
        if fit is None:
            continue
        try:
            f = float(fit)
        except Exception:
            continue
        if not math.isfinite(f):
            continue

        if mode == "best_parent":
            best = pick_best_parent(pars, programs)
            if best is None:
                continue
            best_pid, _ = best
            key_ag: tuple[Any, ...] = ("best_parent_allgens", best_pid)
        else:
            key_ag = ("parent_set_allgens", tuple(sorted(pars)))

        groups[key_ag].append(pid)

    return groups


# -----------------------------
# Descendant metrics (cached per child)
# -----------------------------
@dataclass
class DescMetrics:
    desc_max_fit_k: float
    time_to_peak_k: float
    desc_count_k: int
    reaches_elite_k: float
    time_to_elite_k: float
    lineage_reaches_final: float
    branching_factor: int


def compute_descendant_metrics_for_child(
    child_id: str,
    child_gen: int,
    programs: dict[str, dict],
    children_of: dict[str, list[str]],
    elite_pids: set[str],
    gmax: int,
    k: int,
) -> DescMetrics:
    """
    BFS forward from child_id within <= k generations (or up to gmax if k=-1).
    Includes child itself in max fitness.
    """
    branching = len(children_of.get(child_id, []))
    max_gen = gmax if k < 0 else child_gen + k

    best_fit = float("-inf")
    best_gen: int | None = None
    reaches_elite = False
    best_time_to_elite: int | None = None
    reaches_final = False
    desc_count = 0

    visited: set[str] = set([child_id])
    dq = deque([child_id])

    while dq:
        node = dq.popleft()
        p = programs.get(node)
        if not p:
            continue
        gen = p.get("generation", None)
        fit = p.get("fitness", None)
        if not (isinstance(gen, int) and fit is not None):
            continue
        try:
            f = float(fit)
        except Exception:
            continue
        if not math.isfinite(f):
            continue
        if gen > max_gen:
            continue

        if f > best_fit:
            best_fit = f
            best_gen = gen

        if node in elite_pids:
            reaches_elite = True
            dt = gen - child_gen
            if best_time_to_elite is None or dt < best_time_to_elite:
                best_time_to_elite = dt

        if gen == gmax:
            reaches_final = True

        for ch in children_of.get(node, []):
            if ch in visited:
                continue
            pc = programs.get(ch, {})
            gch = pc.get("generation", None)
            if isinstance(gch, int) and gch > max_gen:
                continue
            visited.add(ch)
            dq.append(ch)
            if ch != child_id:
                desc_count += 1

    time_to_peak = float(best_gen - child_gen) if best_gen is not None else float("nan")
    time_to_elite = (
        float(best_time_to_elite) if best_time_to_elite is not None else float("nan")
    )

    return DescMetrics(
        desc_max_fit_k=float(best_fit) if best_fit != float("-inf") else float("nan"),
        time_to_peak_k=time_to_peak,
        desc_count_k=int(desc_count),
        reaches_elite_k=1.0 if reaches_elite else 0.0,
        time_to_elite_k=time_to_elite,
        lineage_reaches_final=1.0 if reaches_final else 0.0,
        branching_factor=int(branching),
    )


# -----------------------------
# Roots for lineage diversity
# -----------------------------
def compute_roots_memoized(parents_of: dict[str, list[str]]) -> dict[str, set[str]]:
    memo: dict[str, set[str]] = {}

    def roots(pid: str) -> set[str]:
        if pid in memo:
            return memo[pid]
        pars = parents_of.get(pid, [])
        if not pars:
            memo[pid] = {pid}
            return memo[pid]
        out: set[str] = set()
        for par in pars:
            out |= roots(par)
        memo[pid] = out
        return out

    for pid in parents_of.keys():
        roots(pid)
    return memo


# -----------------------------
# Aggregation helpers
# -----------------------------
def nanmedian(vals: list[float]) -> float:
    xs = [float(x) for x in vals if math.isfinite(float(x))]
    return robust_median(xs) if xs else float("nan")


def nanquantile(vals: list[float], q: float) -> float:
    xs = [float(x) for x in vals if math.isfinite(float(x))]
    return robust_quantile(xs, q) if xs else float("nan")


def nanrate_bool(vals: list[float]) -> float:
    xs = [float(x) for x in vals if math.isfinite(float(x))]
    if not xs:
        return float("nan")
    return sum(1 for x in xs if x > 0.5) / len(xs)


def nancount(vals: list[float]) -> int:
    return sum(1 for x in vals if math.isfinite(float(x)))


# -----------------------------
# Core computation (reusable from other modules)
# -----------------------------
def compute_origin_analysis(
    banks_path: str,
    programs_path: str,
    quartile_mode: str = "generation_range",
    elite_pct: float = 0.05,
    desc_k: int = 10,
    sibling_mode: str = "best_parent",
    sibling_gen_window: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run origin-based evolutionary statistics analysis.

    Args:
        banks_path: Path to banks.json with idea-to-program mappings.
        programs_path: Path to programs.json with program records.
        quartile_mode: "generation_range" or "generation_quantiles".
        elite_pct: Top fraction considered elite (default 0.05).
        desc_k: Descendant window in generations (-1 = up to final).
        sibling_mode: "best_parent" or "parent_set".
        sibling_gen_window: Generation window for sibling grouping (0 = exact gen).

    Returns:
        Tuple of (summary_df, best_ideas_df).
    """
    idea_to_origin_programs, idea_desc = load_ideas(banks_path)
    programs = load_programs(programs_path)
    parents_of = build_parents(programs)
    children_of = build_children(parents_of)
    roots_memo = compute_roots_memoized(parents_of)

    # Valid programs
    valid_pids: list[str] = []
    gens: list[int] = []
    fits_all: list[float] = []
    for pid, p in programs.items():
        gen = p.get("generation", None)
        fit = p.get("fitness", None)
        if isinstance(gen, int) and fit is not None:
            try:
                f = float(fit)
            except Exception:
                continue
            if math.isfinite(f):
                valid_pids.append(pid)
                gens.append(int(gen))
                fits_all.append(f)

    if not valid_pids:
        raise RuntimeError(
            "No valid programs with numeric generation and fitness found."
        )

    gmax = max(gens)
    distinct_gens = sorted(set(gens))
    total_distinct_gens = len(distinct_gens)

    # Quartile bounds
    if quartile_mode == "generation_quantiles":
        b1, b2, b3 = generation_quantile_bounds(gens)
    else:
        b1, b2, b3 = generation_range_bounds(gens)

    # Precompute which generations belong to which quartile (for denom in reinvention rates)
    gens_by_quartile: dict[str, set[int]] = {q: set() for q in ["Q1", "Q2", "Q3", "Q4"]}
    for g in distinct_gens:
        q = generation_to_quartile(g, b1, b2, b3)
        gens_by_quartile[q].add(g)

    # Elite set
    elite_threshold, elite_k = elite_threshold_by_top_k(fits_all, elite_pct)
    elite_pids: set[str] = set()
    if math.isfinite(elite_threshold):
        for pid in valid_pids:
            f = float(programs[pid]["fitness"])
            if f >= elite_threshold:
                elite_pids.add(pid)

    # Program->origin ideas
    prog_to_origin_ideas = invert_idea_to_programs(idea_to_origin_programs)

    # Fitness distributions per generation for parent fitness percentiles
    gen_to_sorted_fits: dict[int, list[float]] = defaultdict(list)
    for pid in valid_pids:
        gen = int(programs[pid]["generation"])
        gen_to_sorted_fits[gen].append(float(programs[pid]["fitness"]))
    for gen, xs in gen_to_sorted_fits.items():
        xs.sort()

    def parent_fitness_percentile_within_gen(pid: str) -> float:
        p = programs.get(pid, {})
        gen = p.get("generation", None)
        fit = p.get("fitness", None)
        if not (isinstance(gen, int) and fit is not None):
            return float("nan")
        try:
            f = float(fit)
        except Exception:
            return float("nan")
        xs = gen_to_sorted_fits.get(int(gen), [])
        return percentile_rank(xs, f)

    # Sibling groups (per generation bucket) + (all generations)
    sibling_groups = build_sibling_groups(
        programs, parents_of, sibling_mode, sibling_gen_window
    )
    sibling_groups_allgens = build_sibling_groups_allgens(
        programs, parents_of, sibling_mode
    )

    def sibling_key(best_parent_id: str, parents: list[str], child_gen: int) -> tuple:
        def bucket(gen: int) -> int:
            if sibling_gen_window <= 0:
                return gen
            return gen // (sibling_gen_window + 1)

        if sibling_mode == "best_parent":
            return ("best_parent", best_parent_id, bucket(child_gen))
        return ("parent_set", tuple(sorted(parents)), bucket(child_gen))

    def sibling_key_allgens(best_parent_id: str, parents: list[str]) -> tuple:
        if sibling_mode == "best_parent":
            return ("best_parent_allgens", best_parent_id)
        return ("parent_set_allgens", tuple(sorted(parents)))

    # Intro/origin events
    intro_events = compute_intro_events_origin_only(
        programs=programs,
        prog_to_origin_ideas=prog_to_origin_ideas,
        parents_of=parents_of,
        b1=b1,
        b2=b2,
        b3=b3,
    )

    # Build event-level rows
    # (We’ll aggregate by (idea, quartile) and also compute ALL.)
    desc_cache: dict[str, DescMetrics] = {}

    event_rows = []
    eps = 1e-12

    # First pass: compute gains for global & quartile distributions
    gains_all: list[float] = []
    gains_by_q: dict[str, list[float]] = defaultdict(list)

    for ev in intro_events:
        gain_best = ev.child_fit - ev.best_parent_fit
        if math.isfinite(gain_best):
            gains_all.append(gain_best)
            gains_by_q[ev.quartile].append(gain_best)

    gains_all_sorted = sorted(gains_all)
    gains_by_q_sorted = {q: sorted(xs) for q, xs in gains_by_q.items()}

    # Robust z params
    overall_med = robust_median(gains_all) if gains_all else float("nan")
    overall_mad = mad(gains_all) if gains_all else float("nan")
    q_med: dict[str, float] = {q: robust_median(xs) for q, xs in gains_by_q.items()}
    q_mad: dict[str, float] = {q: mad(xs) for q, xs in gains_by_q.items()}

    for ev in intro_events:
        gain_best = ev.child_fit - ev.best_parent_fit
        gain_mean = ev.child_fit - ev.mean_parent_fit

        gain_best_rel = (
            gain_best / (abs(ev.best_parent_fit) + eps)
            if math.isfinite(gain_best)
            else float("nan")
        )

        # Percentiles: within quartile distribution + overall distribution
        dist_q = gains_by_q_sorted.get(ev.quartile, [])
        gain_pct_in_q = (
            percentile_rank(dist_q, gain_best)
            if math.isfinite(gain_best)
            else float("nan")
        )
        gain_pct_overall = (
            percentile_rank(gains_all_sorted, gain_best)
            if math.isfinite(gain_best)
            else float("nan")
        )

        # Robust z: within quartile and overall (MAD scaled)
        med_q = q_med.get(ev.quartile, float("nan"))
        mad_qv = q_mad.get(ev.quartile, float("nan"))
        denom_q = (
            (1.4826 * mad_qv + eps)
            if (math.isfinite(mad_qv) and mad_qv > 0)
            else float("nan")
        )
        z_in_q = (
            (gain_best - med_q) / denom_q
            if (math.isfinite(gain_best) and math.isfinite(denom_q))
            else float("nan")
        )

        denom_all = (
            (1.4826 * overall_mad + eps)
            if (math.isfinite(overall_mad) and overall_mad > 0)
            else float("nan")
        )
        z_overall = (
            (gain_best - overall_med) / denom_all
            if (math.isfinite(gain_best) and math.isfinite(denom_all))
            else float("nan")
        )

        # -------------------------
        # Sibling-controlled (same gen bucket)
        # -------------------------
        skey = sibling_key(ev.best_parent_id, ev.parents, ev.child_gen)
        sib_percentile = float("nan")
        sib_delta = float("nan")
        sib_win = float("nan")
        sibs = sibling_groups.get(skey, [])
        sib_fits = []
        for pid in sibs:
            if pid == ev.child_id:
                continue
            try:
                f = float(programs[pid].get("fitness", float("nan")))
            except Exception:
                continue
            if math.isfinite(f):
                sib_fits.append(f)
        if sib_fits:
            sib_sorted = sorted(sib_fits)
            sib_percentile = percentile_rank(sib_sorted, ev.child_fit)
            sib_med = robust_median(sib_fits)
            sib_delta = ev.child_fit - sib_med
            sib_win = 1.0 if ev.child_fit > sib_med else 0.0

        # -------------------------
        # Sibling-controlled (ALL generations)
        # -------------------------
        skey_all = sibling_key_allgens(ev.best_parent_id, ev.parents)
        sib_percentile_all = float("nan")
        sib_delta_all = float("nan")
        sib_win_all = float("nan")
        sibs_all = sibling_groups_allgens.get(skey_all, [])
        sib_fits_all = []
        for pid in sibs_all:
            if pid == ev.child_id:
                continue
            try:
                f = float(programs[pid].get("fitness", float("nan")))
            except Exception:
                continue
            if math.isfinite(f):
                sib_fits_all.append(f)
        if sib_fits_all:
            sib_sorted_all = sorted(sib_fits_all)
            sib_percentile_all = percentile_rank(sib_sorted_all, ev.child_fit)
            sib_med_all = robust_median(sib_fits_all)
            sib_delta_all = ev.child_fit - sib_med_all
            sib_win_all = 1.0 if ev.child_fit > sib_med_all else 0.0

        # Descendant metrics (cached per child)
        if ev.child_id not in desc_cache:
            desc_cache[ev.child_id] = compute_descendant_metrics_for_child(
                child_id=ev.child_id,
                child_gen=ev.child_gen,
                programs=programs,
                children_of=children_of,
                elite_pids=elite_pids,
                gmax=gmax,
                k=desc_k,
            )
        dm = desc_cache[ev.child_id]
        desc_max_lift_k_best = (
            (dm.desc_max_fit_k - ev.best_parent_fit)
            if math.isfinite(dm.desc_max_fit_k)
            else float("nan")
        )

        # Context
        parent_pct = parent_fitness_percentile_within_gen(ev.best_parent_id)
        born_elite = 1.0 if ev.child_id in elite_pids else 0.0

        event_rows.append(
            {
                "idea_id": ev.idea_id,
                "quartile": ev.quartile,
                "child_id": ev.child_id,
                "IntroGain_best": gain_best,
                "IntroGain_mean": gain_mean,
                "IntroGain_best_rel": gain_best_rel,
                "IntroGain_percentile_in_quartile": gain_pct_in_q,
                "IntroGain_percentile_overall": gain_pct_overall,
                "IntroGain_z_in_quartile": z_in_q,
                "IntroGain_z_overall": z_overall,
                # sibling (same gen bucket)
                "SiblingWin": sib_win,
                "SiblingPercentile": sib_percentile,
                "SiblingDelta": sib_delta,
                # sibling (all generations)
                "SiblingWin_allgens": sib_win_all,
                "SiblingPercentile_allgens": sib_percentile_all,
                "SiblingDelta_allgens": sib_delta_all,
                "ParentFitnessPercentile_within_gen": parent_pct,
                "BornInElite": born_elite,
                "DescMaxLift_k_best": desc_max_lift_k_best,
                "ReachesElite_k": dm.reaches_elite_k,
                "TimeToElite_k": dm.time_to_elite_k,
                "LineageReachesFinal": dm.lineage_reaches_final,
                "DescendantCount_k": dm.desc_count_k,
                "BranchingFactor": dm.branching_factor,
                "TimeToPeak_k": dm.time_to_peak_k,
            }
        )

    df_events = pd.DataFrame(event_rows)
    # When there are no intro/origin events (e.g. very small runs, no detected idea introductions),
    # pandas builds an empty frame with no columns, which breaks downstream column access.
    if df_events.empty and "idea_id" not in df_events.columns:
        df_events = pd.DataFrame(
            columns=[
                "idea_id",
                "quartile",
                "child_id",
                "child_gen",
                "child_fit",
                "best_parent_fit",
                "mean_parent_fit",
                "IntroGain_best",
                "IntroGain_best_rel",
                "IntroGain_mean",
                "IntroGain_percentile_in_quartile",
                "IntroGain_percentile_overall",
                "IntroGain_z_in_quartile",
                "IntroGain_z_overall",
                "SiblingWin",
                "SiblingPercentile",
                "SiblingDelta",
                "SiblingWin_allgens",
                "SiblingPercentile_allgens",
                "SiblingDelta_allgens",
                "DescMaxLift_k_best",
                "ReachesElite_k",
                "TimeToElite_k",
                "LineageReachesFinal",
                "DescendantCount_k",
                "BranchingFactor",
                "TimeToPeak_k",
                "ParentFitnessPercentile_within_gen",
                "BornInElite",
            ]
        )

    # -----------------------------
    # Build the ONE output table with 5 rows per idea: Q1..Q4 + ALL
    # -----------------------------
    quartile_order = ["Q1", "Q2", "Q3", "Q4", "ALL"]
    out_rows = []

    for idea_id, origin_pids in idea_to_origin_programs.items():
        # origin stats come from idea_to_origin_programs (not only intro events)
        origin_pids_valid = [
            pid
            for pid in origin_pids
            if pid in programs
            and isinstance(programs[pid].get("generation", None), int)
        ]

        # Precompute origin info per quartile (and ALL)
        origin_by_q: dict[str, list[str]] = {q: [] for q in ["Q1", "Q2", "Q3", "Q4"]}
        for pid in origin_pids_valid:
            gen = int(programs[pid]["generation"])
            q = generation_to_quartile(gen, b1, b2, b3)
            origin_by_q[q].append(pid)

        # Helper to compute origin-derived metrics for a set of origin pids
        def origin_metrics(pids: list[str], q_label: str) -> dict[str, float]:
            if not pids:
                denom_gens = (
                    len(gens_by_quartile[q_label]) if q_label in gens_by_quartile else 0
                )
                return {
                    "origin_programs": 0,
                    "origin_in_elite_rate": float("nan"),
                    "origin_generation_span": 0.0,
                    "origin_root_diversity": 0.0,
                    "reinvention_rate_origins_per_distinct_gen": (0.0 / denom_gens)
                    if denom_gens > 0
                    else float("nan"),
                }

            gens_local = sorted(int(programs[pid]["generation"]) for pid in pids)
            span = (
                float(gens_local[-1] - gens_local[0]) if len(gens_local) >= 2 else 0.0
            )

            root_set: set[str] = set()
            for pid in pids:
                root_set |= roots_memo.get(pid, {pid})
            root_div = float(len(root_set))

            elite_rate = sum(1 for pid in pids if pid in elite_pids) / len(pids)

            if q_label == "ALL":
                denom_gens = total_distinct_gens
            else:
                denom_gens = len(gens_by_quartile.get(q_label, set()))
            reinvention_rate = (
                (len(pids) / denom_gens) if denom_gens > 0 else float("nan")
            )

            return {
                "origin_programs": float(len(pids)),
                "origin_in_elite_rate": float(elite_rate),
                "origin_generation_span": float(span),
                "origin_root_diversity": float(root_div),
                "reinvention_rate_origins_per_distinct_gen": float(reinvention_rate),
            }

        # Prepare per-quartile and ALL subsets of events for this idea
        sub_all = df_events[df_events["idea_id"] == idea_id].copy()
        sub_by_q = {
            q: sub_all[sub_all["quartile"] == q].copy()
            for q in ["Q1", "Q2", "Q3", "Q4"]
        }

        for q in quartile_order:
            if q == "ALL":
                sub = sub_all
                om = origin_metrics(origin_pids_valid, "ALL")
            else:
                sub = sub_by_q[q]
                om = origin_metrics(origin_by_q[q], q)

            gains = [
                float(x)
                for x in sub["IntroGain_best"].tolist()
                if math.isfinite(float(x))
            ]
            intro_events_ct = len(gains)

            downside_rate = (
                (sum(1 for x in gains if x < 0) / len(gains)) if gains else float("nan")
            )
            tail_risk = (
                robust_median([min(x, 0.0) for x in gains]) if gains else float("nan")
            )

            # Percentiles: median of per-event ranks
            pct_in_q = (
                nanmedian(sub["IntroGain_percentile_in_quartile"].tolist())
                if q != "ALL"
                else float("nan")
            )
            pct_overall = nanmedian(sub["IntroGain_percentile_overall"].tolist())

            # Z: quartile-z only meaningful for quartile rows; overall-z meaningful everywhere
            z_in_q = (
                nanmedian(sub["IntroGain_z_in_quartile"].tolist())
                if q != "ALL"
                else float("nan")
            )
            z_overall = nanmedian(sub["IntroGain_z_overall"].tolist())

            out_rows.append(
                {
                    "idea_id": idea_id,
                    "quartile": q,
                    # Evidence
                    "intro_events": int(intro_events_ct),
                    # Intro effect (best baseline)
                    "IntroGain_best_p10": nanquantile(gains, 0.10),
                    "IntroGain_best_median": nanquantile(gains, 0.50),
                    "IntroGain_best_rel_median": nanmedian(
                        sub["IntroGain_best_rel"].tolist()
                    ),
                    "IntroGain_best_p90": nanquantile(gains, 0.90),
                    "DownsideRate_best": downside_rate,
                    "TailRisk_best_median(min(gain,0))": tail_risk,
                    # Normalizations
                    "IntroGain_percentile_median_in_quartile": pct_in_q,  # NaN for ALL
                    "IntroGain_percentile_median_overall": pct_overall,  # always defined (if any events)
                    "IntroGain_z_median_in_quartile": z_in_q,  # NaN for ALL
                    "IntroGain_z_median_overall": z_overall,  # always defined (if any events)
                    # Sibling-controlled (same gen bucket)
                    "SiblingWinRate": nanrate_bool(sub["SiblingWin"].tolist()),
                    "SiblingPercentile_median": nanmedian(
                        sub["SiblingPercentile"].tolist()
                    ),
                    "SiblingDelta_median": nanmedian(sub["SiblingDelta"].tolist()),
                    # Sibling-controlled (ALL generations)
                    "SiblingWinRate_allgens": nanrate_bool(
                        sub["SiblingWin_allgens"].tolist()
                    ),
                    "SiblingPercentile_allgens_median": nanmedian(
                        sub["SiblingPercentile_allgens"].tolist()
                    ),
                    "SiblingDelta_allgens_median": nanmedian(
                        sub["SiblingDelta_allgens"].tolist()
                    ),
                    # Downstream lineage (window k)
                    "DescMaxLift_k_best_median": nanmedian(
                        sub["DescMaxLift_k_best"].tolist()
                    ),
                    "ReachesElite_k_rate": nanrate_bool(sub["ReachesElite_k"].tolist()),
                    "TimeToElite_k_median": nanmedian(sub["TimeToElite_k"].tolist()),
                    "LineageReachesFinal_rate": nanrate_bool(
                        sub["LineageReachesFinal"].tolist()
                    ),
                    "DescendantCount_k_median": nanmedian(
                        sub["DescendantCount_k"].tolist()
                    ),
                    "BranchingFactor_median": nanmedian(
                        sub["BranchingFactor"].tolist()
                    ),
                    "TimeToPeak_k_median": nanmedian(sub["TimeToPeak_k"].tolist()),
                    # Context at birth
                    "ParentFitnessPercentile_within_gen_median": nanmedian(
                        sub["ParentFitnessPercentile_within_gen"].tolist()
                    ),
                    "BornInElite_rate": nanrate_bool(sub["BornInElite"].tolist()),
                    # Reinvention / origin-only (from active_bank[*].programs)
                    "origin_programs": int(om["origin_programs"]),
                    "origin_in_elite_rate": om["origin_in_elite_rate"],
                    "origin_generation_span": om["origin_generation_span"],
                    "origin_root_diversity": om["origin_root_diversity"],
                    "reinvention_rate_origins_per_distinct_gen": om[
                        "reinvention_rate_origins_per_distinct_gen"
                    ],
                    "description": idea_desc.get(idea_id, ""),
                }
            )

    df_out = pd.DataFrame(out_rows)

    # Order rows
    q_rank = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4, "ALL": 5}
    df_out["_qrank"] = df_out["quartile"].map(q_rank).fillna(99).astype(int)
    df_out = df_out.sort_values(["idea_id", "_qrank"]).drop(columns=["_qrank"])

    # Second table: filter "best ideas" then dedupe per idea
    df_sel = df_out.copy()

    df_sel["DescCount_rank_in_quartile"] = df_sel.groupby("quartile")[
        "DescendantCount_k_median"
    ].rank(method="min", ascending=False)
    top50_desc_mask = df_sel["DescCount_rank_in_quartile"] <= 50

    eps = 1e-12

    base_ok = (
        (df_sel["intro_events"] >= 1)
        & (pd.to_numeric(df_sel["IntroGain_best_rel_median"], errors="coerce") > 0.01)
        & (pd.to_numeric(df_sel["DownsideRate_best"], errors="coerce") < 0.4)
    )

    sib_win_all = pd.to_numeric(df_sel["SiblingWinRate_allgens"], errors="coerce")
    p10 = pd.to_numeric(df_sel["IntroGain_best_p10"], errors="coerce")
    born_rate = pd.to_numeric(df_sel["BornInElite_rate"], errors="coerce")
    reaches_elite_rate = pd.to_numeric(df_sel["ReachesElite_k_rate"], errors="coerce")

    cond_ge3 = (df_sel["intro_events"] >= 3) & (sib_win_all >= 0.5)

    cond_eq2 = (df_sel["intro_events"] == 2) & (p10 > 0) & (sib_win_all >= 1.0 - eps)

    cond_eq1 = (df_sel["intro_events"] == 1) & (
        (born_rate >= 1.0 - eps) | (top50_desc_mask & (reaches_elite_rate >= 1.0 - eps))
    )

    keep_mask = base_ok & (cond_ge3 | cond_eq2 | cond_eq1)
    df_filtered_rows = df_sel[keep_mask].copy()

    pref_rank = {"ALL": 0, "Q4": 1, "Q3": 2, "Q2": 3, "Q1": 4}
    df_filtered_rows["_pref"] = (
        df_filtered_rows["quartile"].map(pref_rank).fillna(99).astype(int)
    )

    score = pd.to_numeric(df_filtered_rows["IntroGain_best_median"], errors="coerce")
    df_filtered_rows["_score"] = score.fillna(-1e18)

    df_best_ideas = (
        df_filtered_rows.sort_values(
            ["idea_id", "_pref", "_score"], ascending=[True, True, False]
        )
        .drop_duplicates(subset=["idea_id"], keep="first")
        .drop(columns=["_pref", "_score", "DescCount_rank_in_quartile"])
        .reset_index(drop=True)
    )

    return df_out, df_best_ideas


# -----------------------------
# CLI entry point
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ideas",
        default="gigaevo/memory/ideas_tracker/logs/2026-02-17_13-46-22/banks.json",
    )
    ap.add_argument(
        "--programs",
        default="gigaevo/memory/ideas_tracker/logs/2026-02-17_13-46-22/programs.json",
    )
    ap.add_argument("--output_dir", default="selected_ideas/idea_origin_analysis_out")
    ap.add_argument("--output_name", default="idea_origin_quartile_summary6.csv")

    ap.add_argument(
        "--quartile_mode",
        choices=["generation_range", "generation_quantiles"],
        default="generation_range",
    )
    ap.add_argument("--elite_pct", type=float, default=0.05)
    ap.add_argument("--desc_k", type=int, default=10)

    ap.add_argument(
        "--sibling_mode", choices=["best_parent", "parent_set"], default="best_parent"
    )
    ap.add_argument("--sibling_gen_window", type=int, default=0)

    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / args.output_name

    df_out, df_best_ideas = compute_origin_analysis(
        banks_path=args.ideas,
        programs_path=args.programs,
        quartile_mode=args.quartile_mode,
        elite_pct=args.elite_pct,
        desc_k=args.desc_k,
        sibling_mode=args.sibling_mode,
        sibling_gen_window=args.sibling_gen_window,
    )

    df_out.to_csv(out_csv, index=False)

    best_csv = out_dir / (Path(args.output_name).stem + "_best_ideas.csv")
    df_best_ideas.to_csv(best_csv, index=False)

    logger.info(f"Wrote (filtered best ideas, deduped per idea_id): {best_csv}")
    logger.debug(
        f"Sanity check (first 15 rows of best ideas):\n{df_best_ideas.head(15).to_string(index=False)}"
    )

    logger.info(f"Wrote: {out_csv}")
    logger.debug(
        f"Sanity check (first 15 rows):\n{df_out.head(15).to_string(index=False)}"
    )


if __name__ == "__main__":
    main()
