"""Aggregation: per-idea summary rows."""

from __future__ import annotations

import math

from gigaevo.memory.core.idea_stats import IdeaStats
from gigaevo.memory.efficacy import (
    EfficacyEvent,
    EfficacyScorer,
    GainObservation,
    GenerationBucketer,
)
from gigaevo.memory.ideas_tracker.utils.origin_analysis.statistics import (
    nanmedian,
    nanquantile,
    nanrate_bool,
)
from gigaevo.memory.shared_memory.models import Quartile

_QUARTILE_SORT_RANK = {q: rank for rank, q in enumerate(Quartile)}


def aggregate_idea_rows(
    events: list[EfficacyEvent],
    idea_to_origin_programs: dict[str, set[str]],
    idea_desc: dict[str, str],
    programs: dict[str, dict],
    elite_pids: set[str],
    roots_memo: dict[str, set[str]],
    bucketer: GenerationBucketer,
    gens_by_quartile: dict[Quartile, set[int]],
    total_distinct_gens: int,
    *,
    scorer: EfficacyScorer,
) -> list[IdeaStats]:
    out_rows: list[IdeaStats] = []

    def to_observations(rows: list[EfficacyEvent]) -> list[GainObservation]:
        return [
            GainObservation(
                child_id=ev.child_id,
                parent_fitness=ev.best_parent_fit,
                gain=ev.IntroGain_best,
            )
            for ev in rows
            if math.isfinite(ev.best_parent_fit) and math.isfinite(ev.IntroGain_best)
        ]

    fitted = scorer.fit(to_observations(events))

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
            origin_by_q[bucketer.bucket(gen)].append(pid)

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

        sub_all = [ev for ev in events if ev.idea_id == idea_id]
        sub_by_q = {
            q: [ev for ev in sub_all if ev.quartile is q] for q in Quartile.quarters()
        }

        for q in Quartile:
            sub = sub_all if q is Quartile.ALL else sub_by_q[q]
            om = origin_metrics(
                origin_pids_valid if q is Quartile.ALL else origin_by_q[q], q
            )

            scorable = to_observations(sub)
            gains = [o.gain for o in scorable]
            intro_events_ct = len(gains)
            tail_risk = (
                nanmedian([min(g, 0.0) for g in gains]) if gains else float("nan")
            )

            adj_gains = fitted.adjusted_gains(scorable)
            post = fitted.posterior(scorable)
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
                nanmedian([e.IntroGain_percentile_in_quartile for e in sub])
                if q is not Quartile.ALL
                else float("nan")
            )
            pct_overall = nanmedian([e.IntroGain_percentile_overall for e in sub])
            z_in_q = (
                nanmedian([e.IntroGain_z_in_quartile for e in sub])
                if q is not Quartile.ALL
                else float("nan")
            )
            z_overall = nanmedian([e.IntroGain_z_overall for e in sub])

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
                            [e.IntroGain_best_rel for e in sub]
                        ),
                        "IntroGain_best_p90": nanquantile(gains, 0.90),
                        "DownsideRate_best": downside_rate,
                        "TailRisk_best_median": tail_risk,
                        "k_harm": int(k_harm),
                        "posterior_a": posterior_a,
                        "posterior_b": posterior_b,
                        "p_help_mean": p_help_mean,
                        "p_help_lo20": p_help_lo20,
                        "efficacy_confident": efficacy_confident,
                        "IntroGain_percentile_median_in_quartile": pct_in_q,
                        "IntroGain_percentile_median_overall": pct_overall,
                        "IntroGain_z_median_in_quartile": z_in_q,
                        "IntroGain_z_median_overall": z_overall,
                        "SiblingWinRate": nanrate_bool([e.SiblingWin for e in sub]),
                        "SiblingPercentile_median": nanmedian(
                            [e.SiblingPercentile for e in sub]
                        ),
                        "SiblingDelta_median": nanmedian([e.SiblingDelta for e in sub]),
                        "SiblingWinRate_allgens": nanrate_bool(
                            [e.SiblingWin_allgens for e in sub]
                        ),
                        "SiblingPercentile_allgens_median": nanmedian(
                            [e.SiblingPercentile_allgens for e in sub]
                        ),
                        "SiblingDelta_allgens_median": nanmedian(
                            [e.SiblingDelta_allgens for e in sub]
                        ),
                        "DescMaxLift_k_best_median": nanmedian(
                            [e.DescMaxLift_k_best for e in sub]
                        ),
                        "ReachesElite_k_rate": nanrate_bool(
                            [e.ReachesElite_k for e in sub]
                        ),
                        "TimeToElite_k_median": nanmedian(
                            [e.TimeToElite_k for e in sub]
                        ),
                        "LineageReachesFinal_rate": nanrate_bool(
                            [e.LineageReachesFinal for e in sub]
                        ),
                        "DescendantCount_k_median": nanmedian(
                            [e.DescendantCount_k for e in sub]
                        ),
                        "BranchingFactor_median": nanmedian(
                            [e.BranchingFactor for e in sub]
                        ),
                        "TimeToPeak_k_median": nanmedian([e.TimeToPeak_k for e in sub]),
                        "ParentFitnessPercentile_within_gen_median": nanmedian(
                            [e.ParentFitnessPercentile_within_gen for e in sub]
                        ),
                        "BornInElite_rate": nanrate_bool([e.BornInElite for e in sub]),
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
