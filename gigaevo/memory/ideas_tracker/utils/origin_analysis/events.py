"""Intro event detection and descendant metric computation."""

from __future__ import annotations

from collections import deque
import math

from gigaevo.memory.ideas_tracker.utils.origin_analysis.quartiles import (
    generation_to_quartile,
)
from gigaevo.memory.ideas_tracker.utils.origin_analysis.types import (
    DescMetrics,
    IntroEvent,
)


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
        except (TypeError, ValueError):
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
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            fits.append(f)
    if not fits:
        return None
    return sum(fits) / len(fits)


def compute_intro_events(
    programs: dict[str, dict],
    prog_to_origin_ideas: dict[str, set[str]],
    parents_of: dict[str, list[str]],
    b1: float,
    b2: float,
    b3: float,
) -> list[IntroEvent]:
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
        except (TypeError, ValueError):
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


def compute_descendant_metrics(
    child_id: str,
    child_gen: int,
    programs: dict[str, dict],
    children_of: dict[str, list[str]],
    elite_pids: set[str],
    gmax: int,
    k: int,
) -> DescMetrics:
    branching = len(children_of.get(child_id, []))
    max_gen = gmax if k < 0 else child_gen + k

    best_fit = float("-inf")
    best_gen: int | None = None
    reaches_elite = False
    best_time_to_elite: int | None = None
    reaches_final = False
    desc_count = 0

    visited: set[str] = {child_id}
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
        except (TypeError, ValueError):
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
