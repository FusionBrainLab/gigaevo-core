"""
analyse() — orchestrates the full origin analysis pipeline.

Phases:
  1. load_ideas + load_programs + build graph
  2. compute quartile bounds + elite set
  3. build inverted indices + per-gen fitness distributions
  4. build sibling groups
  5. detect intro events + compute gain distributions + event-level metrics
  6. aggregate per-idea summary rows + filter best ideas
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import math
from pathlib import Path

from loguru import logger
import pandas as pd

from gigaevo.memory.ideas_tracker.utils.origin_analysis.aggregation import (
    _EMPTY_EVENTS_COLUMNS,
    aggregate_idea_rows,
    filter_best_ideas,
)
from gigaevo.memory.ideas_tracker.utils.origin_analysis.events import (
    compute_descendant_metrics,
    compute_intro_events,
)
from gigaevo.memory.ideas_tracker.utils.origin_analysis.loader import (
    build_children,
    build_parents,
    compute_roots_memoized,
    invert_idea_to_programs,
    load_ideas,
    load_programs,
)
from gigaevo.memory.ideas_tracker.utils.origin_analysis.quartiles import (
    generation_quantile_bounds,
    generation_range_bounds,
    generation_to_quartile,
)
from gigaevo.memory.ideas_tracker.utils.origin_analysis.siblings import (
    build_sibling_groups,
    build_sibling_groups_allgens,
)
from gigaevo.memory.ideas_tracker.utils.origin_analysis.statistics import (
    elite_threshold_by_top_k,
    mad,
    percentile_rank,
    robust_median,
)
from gigaevo.memory.ideas_tracker.utils.origin_analysis.types import (
    AnalysisResult,
    DescMetrics,
)


def analyse(
    banks_path: str,
    programs_path: str,
    quartile_mode: str = "generation_range",
    elite_pct: float = 0.05,
    desc_k: int = 10,
    sibling_mode: str = "best_parent",
    sibling_gen_window: int = 0,
) -> AnalysisResult:
    """Run origin-based evolutionary statistics analysis."""
    idea_to_origin_programs, idea_desc = load_ideas(banks_path)
    programs = load_programs(programs_path)
    parents_of = build_parents(programs)
    children_of = build_children(parents_of)
    roots_memo = compute_roots_memoized(parents_of)

    valid_pids: list[str] = []
    gens: list[int] = []
    fits_all: list[float] = []
    for pid, p in programs.items():
        gen = p.get("generation", None)
        fit = p.get("fitness", None)
        if isinstance(gen, int) and fit is not None:
            try:
                f = float(fit)
            except (TypeError, ValueError):
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

    if quartile_mode == "generation_quantiles":
        b1, b2, b3 = generation_quantile_bounds(gens)
    else:
        b1, b2, b3 = generation_range_bounds(gens)

    gens_by_quartile: dict[str, set[int]] = {q: set() for q in ["Q1", "Q2", "Q3", "Q4"]}
    for g in distinct_gens:
        q = generation_to_quartile(g, b1, b2, b3)
        gens_by_quartile[q].add(g)

    elite_threshold, _ = elite_threshold_by_top_k(fits_all, elite_pct)
    elite_pids: set[str] = set()
    if math.isfinite(elite_threshold):
        for pid in valid_pids:
            if float(programs[pid]["fitness"]) >= elite_threshold:
                elite_pids.add(pid)

    prog_to_origin_ideas = invert_idea_to_programs(idea_to_origin_programs)

    gen_to_sorted_fits: dict[int, list[float]] = defaultdict(list)
    for pid in valid_pids:
        gen = int(programs[pid]["generation"])
        gen_to_sorted_fits[gen].append(float(programs[pid]["fitness"]))
    for xs in gen_to_sorted_fits.values():
        xs.sort()

    def parent_fitness_percentile_within_gen(pid: str) -> float:
        p = programs.get(pid, {})
        gen = p.get("generation", None)
        fit = p.get("fitness", None)
        if not (isinstance(gen, int) and fit is not None):
            return float("nan")
        try:
            f = float(fit)
        except (TypeError, ValueError):
            return float("nan")
        xs = gen_to_sorted_fits.get(int(gen), [])
        return percentile_rank(xs, f)

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

    intro_events = compute_intro_events(
        programs=programs,
        prog_to_origin_ideas=prog_to_origin_ideas,
        parents_of=parents_of,
        b1=b1,
        b2=b2,
        b3=b3,
    )

    desc_cache: dict[str, DescMetrics] = {}
    event_rows = []
    eps = 1e-12

    gains_all: list[float] = []
    gains_by_q: dict[str, list[float]] = defaultdict(list)
    for ev in intro_events:
        gain_best = ev.child_fit - ev.best_parent_fit
        if math.isfinite(gain_best):
            gains_all.append(gain_best)
            gains_by_q[ev.quartile].append(gain_best)

    gains_all_sorted = sorted(gains_all)
    gains_by_q_sorted = {q: sorted(xs) for q, xs in gains_by_q.items()}
    overall_med = robust_median(gains_all) if gains_all else float("nan")
    overall_mad = mad(gains_all) if gains_all else float("nan")
    q_med = {q: robust_median(xs) for q, xs in gains_by_q.items()}
    q_mad = {q: mad(xs) for q, xs in gains_by_q.items()}

    for ev in intro_events:
        gain_best = ev.child_fit - ev.best_parent_fit
        gain_mean = ev.child_fit - ev.mean_parent_fit
        gain_best_rel = (
            gain_best / (abs(ev.best_parent_fit) + eps)
            if math.isfinite(gain_best)
            else float("nan")
        )

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

        skey = sibling_key(ev.best_parent_id, ev.parents, ev.child_gen)
        sib_percentile = sib_delta = sib_win = float("nan")
        sibs = sibling_groups.get(skey, [])
        sib_fits = [
            float(programs[pid].get("fitness", float("nan")))
            for pid in sibs
            if pid != ev.child_id
        ]
        sib_fits = [f for f in sib_fits if math.isfinite(f)]
        if sib_fits:
            sib_sorted = sorted(sib_fits)
            sib_percentile = percentile_rank(sib_sorted, ev.child_fit)
            sib_med = robust_median(sib_fits)
            sib_delta = ev.child_fit - sib_med
            sib_win = 1.0 if ev.child_fit > sib_med else 0.0

        skey_all = sibling_key_allgens(ev.best_parent_id, ev.parents)
        sib_percentile_all = sib_delta_all = sib_win_all = float("nan")
        sibs_all = sibling_groups_allgens.get(skey_all, [])
        sib_fits_all = [
            float(programs[pid].get("fitness", float("nan")))
            for pid in sibs_all
            if pid != ev.child_id
        ]
        sib_fits_all = [f for f in sib_fits_all if math.isfinite(f)]
        if sib_fits_all:
            sib_sorted_all = sorted(sib_fits_all)
            sib_percentile_all = percentile_rank(sib_sorted_all, ev.child_fit)
            sib_med_all = robust_median(sib_fits_all)
            sib_delta_all = ev.child_fit - sib_med_all
            sib_win_all = 1.0 if ev.child_fit > sib_med_all else 0.0

        if ev.child_id not in desc_cache:
            desc_cache[ev.child_id] = compute_descendant_metrics(
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
                "SiblingWin": sib_win,
                "SiblingPercentile": sib_percentile,
                "SiblingDelta": sib_delta,
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
    if df_events.empty and "idea_id" not in df_events.columns:
        df_events = pd.DataFrame(columns=_EMPTY_EVENTS_COLUMNS)

    df_out = aggregate_idea_rows(
        df_events=df_events,
        idea_to_origin_programs=idea_to_origin_programs,
        idea_desc=idea_desc,
        programs=programs,
        elite_pids=elite_pids,
        roots_memo=roots_memo,
        b1=b1,
        b2=b2,
        b3=b3,
        gens_by_quartile=gens_by_quartile,
        total_distinct_gens=total_distinct_gens,
    )
    df_best = filter_best_ideas(df_out)
    return AnalysisResult(summary_df=df_out, best_ideas_df=df_best)


def main() -> None:
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

    result = analyse(
        banks_path=args.ideas,
        programs_path=args.programs,
        quartile_mode=args.quartile_mode,
        elite_pct=args.elite_pct,
        desc_k=args.desc_k,
        sibling_mode=args.sibling_mode,
        sibling_gen_window=args.sibling_gen_window,
    )

    out_csv = out_dir / args.output_name
    result.summary_df.to_csv(out_csv, index=False)
    best_csv = out_dir / (Path(args.output_name).stem + "_best_ideas.csv")
    result.best_ideas_df.to_csv(best_csv, index=False)
    logger.info("Wrote: {}", out_csv)
    logger.info("Wrote (best ideas): {}", best_csv)


if __name__ == "__main__":
    main()
