"""Aggregation: per-idea summary rows and best-ideas filter."""

from __future__ import annotations

import math

import pandas as pd

from gigaevo.memory.ideas_tracker.utils.origin_analysis.quartiles import (
    generation_to_quartile,
)
from gigaevo.memory.ideas_tracker.utils.origin_analysis.statistics import (
    nanmedian,
    nanquantile,
    nanrate_bool,
)

_EMPTY_EVENTS_COLUMNS = [
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


def aggregate_idea_rows(
    df_events: pd.DataFrame,
    idea_to_origin_programs: dict[str, set[str]],
    idea_desc: dict[str, str],
    programs: dict[str, dict],
    elite_pids: set[str],
    roots_memo: dict[str, set[str]],
    b1: float,
    b2: float,
    b3: float,
    gens_by_quartile: dict[str, set[int]],
    total_distinct_gens: int,
) -> pd.DataFrame:
    quartile_order = ["Q1", "Q2", "Q3", "Q4", "ALL"]
    out_rows = []

    for idea_id, origin_pids in idea_to_origin_programs.items():
        origin_pids_valid = [
            pid
            for pid in origin_pids
            if pid in programs
            and isinstance(programs[pid].get("generation", None), int)
        ]

        origin_by_q: dict[str, list[str]] = {q: [] for q in ["Q1", "Q2", "Q3", "Q4"]}
        for pid in origin_pids_valid:
            gen = int(programs[pid]["generation"])
            q = generation_to_quartile(gen, b1, b2, b3)
            origin_by_q[q].append(pid)

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
            elite_rate = sum(1 for pid in pids if pid in elite_pids) / len(pids)
            denom_gens = (
                total_distinct_gens
                if q_label == "ALL"
                else len(gens_by_quartile.get(q_label, set()))
            )
            reinvention_rate = (
                (len(pids) / denom_gens) if denom_gens > 0 else float("nan")
            )
            return {
                "origin_programs": float(len(pids)),
                "origin_in_elite_rate": float(elite_rate),
                "origin_generation_span": float(span),
                "origin_root_diversity": float(len(root_set)),
                "reinvention_rate_origins_per_distinct_gen": float(reinvention_rate),
            }

        sub_all = df_events[df_events["idea_id"] == idea_id].copy()
        sub_by_q = {
            q: sub_all[sub_all["quartile"] == q].copy()
            for q in ["Q1", "Q2", "Q3", "Q4"]
        }

        for q in quartile_order:
            sub = sub_all if q == "ALL" else sub_by_q[q]
            om = origin_metrics(origin_pids_valid if q == "ALL" else origin_by_q[q], q)

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
                nanmedian([min(x, 0.0) for x in gains]) if gains else float("nan")
            )

            pct_in_q = (
                nanmedian(sub["IntroGain_percentile_in_quartile"].tolist())
                if q != "ALL"
                else float("nan")
            )
            pct_overall = nanmedian(sub["IntroGain_percentile_overall"].tolist())
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
                    "intro_events": int(intro_events_ct),
                    "IntroGain_best_p10": nanquantile(gains, 0.10),
                    "IntroGain_best_median": nanquantile(gains, 0.50),
                    "IntroGain_best_rel_median": nanmedian(
                        sub["IntroGain_best_rel"].tolist()
                    ),
                    "IntroGain_best_p90": nanquantile(gains, 0.90),
                    "DownsideRate_best": downside_rate,
                    "TailRisk_best_median(min(gain,0))": tail_risk,
                    "IntroGain_percentile_median_in_quartile": pct_in_q,
                    "IntroGain_percentile_median_overall": pct_overall,
                    "IntroGain_z_median_in_quartile": z_in_q,
                    "IntroGain_z_median_overall": z_overall,
                    "SiblingWinRate": nanrate_bool(sub["SiblingWin"].tolist()),
                    "SiblingPercentile_median": nanmedian(
                        sub["SiblingPercentile"].tolist()
                    ),
                    "SiblingDelta_median": nanmedian(sub["SiblingDelta"].tolist()),
                    "SiblingWinRate_allgens": nanrate_bool(
                        sub["SiblingWin_allgens"].tolist()
                    ),
                    "SiblingPercentile_allgens_median": nanmedian(
                        sub["SiblingPercentile_allgens"].tolist()
                    ),
                    "SiblingDelta_allgens_median": nanmedian(
                        sub["SiblingDelta_allgens"].tolist()
                    ),
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
                    "ParentFitnessPercentile_within_gen_median": nanmedian(
                        sub["ParentFitnessPercentile_within_gen"].tolist()
                    ),
                    "BornInElite_rate": nanrate_bool(sub["BornInElite"].tolist()),
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
    q_rank = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4, "ALL": 5}
    df_out["_qrank"] = df_out["quartile"].map(q_rank).fillna(99).astype(int)
    df_out = df_out.sort_values(["idea_id", "_qrank"]).drop(columns=["_qrank"])
    return df_out


def filter_best_ideas(df_out: pd.DataFrame) -> pd.DataFrame:
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
    df_filtered = df_sel[keep_mask].copy()

    pref_rank = {"ALL": 0, "Q4": 1, "Q3": 2, "Q2": 3, "Q1": 4}
    df_filtered["_pref"] = df_filtered["quartile"].map(pref_rank).fillna(99).astype(int)
    score = pd.to_numeric(df_filtered["IntroGain_best_median"], errors="coerce")
    df_filtered["_score"] = score.fillna(-1e18)

    return (
        df_filtered.sort_values(
            ["idea_id", "_pref", "_score"], ascending=[True, True, False]
        )
        .drop_duplicates(subset=["idea_id"], keep="first")
        .drop(columns=["_pref", "_score", "DescCount_rank_in_quartile"])
        .reset_index(drop=True)
    )
