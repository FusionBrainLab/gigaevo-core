"""Aggregation: per-idea summary rows."""

from __future__ import annotations

import math
from typing import Any

from gigaevo.memory.core.idea_stats import IdeaStats
from gigaevo.memory.ideas_tracker.utils.origin_analysis.quartiles import (
    generation_to_quartile,
)
from gigaevo.memory.ideas_tracker.utils.origin_analysis.statistics import (
    nanmedian,
    nanquantile,
    nanrate_bool,
)
from gigaevo.memory.shared_memory.injection_posterior import (
    beta_binomial_posterior,
    noise_band,
    parent_local_baseline,
)
from gigaevo.memory.shared_memory.models import Quartile

_NAN = float("nan")
_QUARTILE_SORT_RANK = {q: rank for rank, q in enumerate(Quartile)}


def _column(events: list[dict[str, Any]], key: str) -> list[float]:
    return [e.get(key, _NAN) for e in events]


def aggregate_idea_rows(
    events: list[dict[str, Any]],
    idea_to_origin_programs: dict[str, set[str]],
    idea_desc: dict[str, str],
    programs: dict[str, dict],
    elite_pids: set[str],
    roots_memo: dict[str, set[str]],
    b1: float,
    b2: float,
    b3: float,
    gens_by_quartile: dict[Quartile, set[int]],
    total_distinct_gens: int,
) -> list[IdeaStats]:
    out_rows: list[IdeaStats] = []

    population: list[tuple[float, float]] = []
    for ev in events:
        ref = ev.get("best_parent_fit", _NAN)
        gain = ev.get("IntroGain_best", _NAN)
        if math.isfinite(float(ref)) and math.isfinite(float(gain)):
            population.append((float(ref), float(gain)))
    baseline = parent_local_baseline(population)
    epsilon = noise_band([g - baseline(ref) for ref, g in population])

    for idea_id, origin_pids in idea_to_origin_programs.items():
        origin_pids_valid = [
            pid
            for pid in origin_pids
            if pid in programs
            and isinstance(programs[pid].get("generation", None), int)
        ]

        origin_by_q: dict[Quartile, list[str]] = {q: [] for q in Quartile.quarters()}
        for pid in origin_pids_valid:
            gen = int(programs[pid]["generation"])
            q = generation_to_quartile(gen, b1, b2, b3)
            origin_by_q[q].append(pid)

        def origin_metrics(pids: list[str], q_label: Quartile) -> dict[str, float]:
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
                if q_label is Quartile.ALL
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

        sub_all = [ev for ev in events if ev.get("idea_id") == idea_id]
        sub_by_q = {
            q: [ev for ev in sub_all if ev.get("quartile") == q]
            for q in Quartile.quarters()
        }

        for q in Quartile:
            sub = sub_all if q is Quartile.ALL else sub_by_q[q]
            om = origin_metrics(
                origin_pids_valid if q is Quartile.ALL else origin_by_q[q], q
            )

            paired = [
                (float(ref), float(gain))
                for ref, gain in zip(
                    _column(sub, "best_parent_fit"),
                    _column(sub, "IntroGain_best"),
                )
                if math.isfinite(float(ref)) and math.isfinite(float(gain))
            ]
            gains = [g for _, g in paired]
            intro_events_ct = len(gains)
            tail_risk = (
                nanmedian([min(g, 0.0) for g in gains]) if gains else float("nan")
            )

            adj_gains = [g - baseline(ref) for ref, g in paired]
            post = beta_binomial_posterior(adj_gains, threshold=-epsilon)
            k_harm = post.k_harm or 0
            posterior_a = post.posterior_a
            posterior_b = post.posterior_b
            p_help_mean = post.p_help_mean
            p_help_lo20 = post.p_help_lo20
            efficacy_confident = post.efficacy_confident
            downside_rate = (
                (k_harm / intro_events_ct) if intro_events_ct else float("nan")
            )

            pct_in_q = (
                nanmedian(_column(sub, "IntroGain_percentile_in_quartile"))
                if q is not Quartile.ALL
                else float("nan")
            )
            pct_overall = nanmedian(_column(sub, "IntroGain_percentile_overall"))
            z_in_q = (
                nanmedian(_column(sub, "IntroGain_z_in_quartile"))
                if q is not Quartile.ALL
                else float("nan")
            )
            z_overall = nanmedian(_column(sub, "IntroGain_z_overall"))

            out_rows.append(
                IdeaStats.model_validate(
                    {
                        "idea_id": idea_id,
                        "quartile": q,
                        "intro_events": int(intro_events_ct),
                        "IntroGain_best_p10": nanquantile(gains, 0.10),
                        "IntroGain_best_median": nanquantile(gains, 0.50),
                        "IntroGain_best_adj_median": nanquantile(adj_gains, 0.50),
                        "IntroGain_best_rel_median": nanmedian(
                            _column(sub, "IntroGain_best_rel")
                        ),
                        "IntroGain_best_p90": nanquantile(gains, 0.90),
                        "DownsideRate_best": downside_rate,
                        "TailRisk_best_median": tail_risk,
                        "posterior_a": posterior_a,
                        "posterior_b": posterior_b,
                        "p_help_mean": p_help_mean,
                        "p_help_lo20": p_help_lo20,
                        "efficacy_confident": efficacy_confident,
                        "IntroGain_percentile_median_in_quartile": pct_in_q,
                        "IntroGain_percentile_median_overall": pct_overall,
                        "IntroGain_z_median_in_quartile": z_in_q,
                        "IntroGain_z_median_overall": z_overall,
                        "SiblingWinRate": nanrate_bool(_column(sub, "SiblingWin")),
                        "SiblingPercentile_median": nanmedian(
                            _column(sub, "SiblingPercentile")
                        ),
                        "SiblingDelta_median": nanmedian(_column(sub, "SiblingDelta")),
                        "SiblingWinRate_allgens": nanrate_bool(
                            _column(sub, "SiblingWin_allgens")
                        ),
                        "SiblingPercentile_allgens_median": nanmedian(
                            _column(sub, "SiblingPercentile_allgens")
                        ),
                        "SiblingDelta_allgens_median": nanmedian(
                            _column(sub, "SiblingDelta_allgens")
                        ),
                        "DescMaxLift_k_best_median": nanmedian(
                            _column(sub, "DescMaxLift_k_best")
                        ),
                        "ReachesElite_k_rate": nanrate_bool(
                            _column(sub, "ReachesElite_k")
                        ),
                        "TimeToElite_k_median": nanmedian(
                            _column(sub, "TimeToElite_k")
                        ),
                        "LineageReachesFinal_rate": nanrate_bool(
                            _column(sub, "LineageReachesFinal")
                        ),
                        "DescendantCount_k_median": nanmedian(
                            _column(sub, "DescendantCount_k")
                        ),
                        "BranchingFactor_median": nanmedian(
                            _column(sub, "BranchingFactor")
                        ),
                        "TimeToPeak_k_median": nanmedian(_column(sub, "TimeToPeak_k")),
                        "ParentFitnessPercentile_within_gen_median": nanmedian(
                            _column(sub, "ParentFitnessPercentile_within_gen")
                        ),
                        "BornInElite_rate": nanrate_bool(_column(sub, "BornInElite")),
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
            )

    out_rows.sort(key=lambda s: (s.idea_id, _QUARTILE_SORT_RANK[s.quartile]))
    return out_rows
