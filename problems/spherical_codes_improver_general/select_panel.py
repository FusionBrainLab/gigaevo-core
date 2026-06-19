"""Finalize the evolution panel from the full90 headroom maps.

Reads the E7/E8 full90 validations, tabulates per-config headroom (the best gain
over Cohn that either paper baseline achieves), and proposes a large-N-biased
evolution panel: the most-improvable configs, stratified to keep every dimension
d in [8, 16] represented so the evolved improver must generalise across d.

The headline metric stays full90; this only chooses WHERE evolution spends its
per-mutant budget. Small-N Cohn codes are near-optimal -> structural zeros that
give the search no gradient (and are floored at Cohn by monotone acceptance on
the headline too, so excluding them from the panel changes no claim). Panel
selection uses the *paper baselines* as the headroom proxy, never the improver
being evolved -> no leakage from the optimisation target.

Run after both maps land:
    python select_panel.py [--per-dim 2] [--rounds 3] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import os

_PROB = os.path.dirname(os.path.abspath(__file__))


def _load(label: str, rounds: int, seed: int):
    path = os.path.join(_PROB, f"validation_{label}_full90_R{rounds}_seed{seed}.json")
    if not os.path.exists(path):
        return None, path
    with open(path) as f:
        return json.load(f), path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-dim", type=int, default=2)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    e7, p7 = _load("E7", args.rounds, args.seed)
    e8, p8 = _load("E8", args.rounds, args.seed)
    if e7 is None or e8 is None:
        print("WAITING for full90 maps; missing:")
        for src, p in ((e7, p7), (e8, p8)):
            if src is None:
                print("  ", p)
        return

    rec: dict[tuple[int, int], dict] = {}
    for src, tag in ((e7, "e7"), (e8, "e8")):
        for r in src["results"]:
            key = (r["d"], r["n"])
            v = rec.setdefault(key, {"d": r["d"], "n": r["n"], "secs": 0.0})
            v[tag] = r["gain"]
            v["secs"] = max(v["secs"], r.get("secs", 0.0))
    for v in rec.values():
        v["headroom"] = max(v.get("e7", 0.0), v.get("e8", 0.0))

    by_d: dict[int, list] = {}
    for v in rec.values():
        by_d.setdefault(v["d"], []).append(v)

    print(f"full90 headline:  E7 {e7['fitness']:.4f}%   E8 {e8['fitness']:.4f}%")
    print(f"  E7 improved {e7['improved']}/90 valid {e7['valid']}/90 wall {e7['elapsed_min']:.1f}m;"
          f"  E8 improved {e8['improved']}/90 valid {e8['valid']}/90 wall {e8['elapsed_min']:.1f}m")
    zero = sum(1 for v in rec.values() if v["headroom"] <= 1e-9)
    print(f"  zero-headroom configs (neither baseline beats Cohn): {zero}/90\n")

    print("per-dimension  headroom vs N  (gain% over Cohn; * = panel pick):")
    panel: list[tuple[int, int]] = []
    for d in sorted(by_d):
        rows = sorted(by_d[d], key=lambda v: v["n"])
        ranked = sorted(by_d[d], key=lambda v: v["headroom"], reverse=True)
        picks = {(v["d"], v["n"]) for v in ranked[: args.per_dim]}
        for v in rows:
            mark = "*" if (v["d"], v["n"]) in picks else " "
            print(f"  d={d:2d} N={v['n']:4d} {mark} headroom {100 * v['headroom']:7.4f}%"
                  f"  (E7 {100 * v.get('e7', 0.0):6.4f}  E8 {100 * v.get('e8', 0.0):6.4f})"
                  f"  {v['secs']:6.1f}s")
        panel.extend(sorted(picks))

    panel = sorted(panel)
    pset = set(panel)

    def panel_mean(src) -> float:
        gs = [r["gain"] for r in src["results"] if (r["d"], r["n"]) in pset]
        return 100.0 * sum(gs) / len(gs) if gs else 0.0

    pm7, pm8 = panel_mean(e7), panel_mean(e8)
    cost = sum(rec[k]["secs"] for k in panel)
    best = max(rec.values(), key=lambda v: v["headroom"])
    print(f"\nproposed panel ({len(panel)} configs, top-{args.per_dim}/dim by headroom):")
    print("PANEL =", tuple(panel))
    print(f"  panel-restricted baseline mean gain:  E7 {pm7:.4f}%   E8 {pm8:.4f}%")
    print(f"  max single-config headroom: (d={best['d']},N={best['n']}) {100 * best['headroom']:.4f}%")
    print(f"  panel grading cost (sum of max per-config secs, R={args.rounds}): "
          f"{cost:.0f}s (~{cost / 60:.1f} min/mutant single-thread)")
    print(f"  suggested metrics.yaml upper_bound >= {max(1.5, 3 * max(pm7, pm8)):.2f} "
          f"(headroom above the baseline panel mean for the evolved improver)")

    big: list[tuple[int, int]] = []
    for d in sorted(by_d):
        ns = sorted(v["n"] for v in by_d[d])
        big.extend((d, n) for n in ns[-args.per_dim:])
    print(f"\n  cf. structural 'largest-{args.per_dim}-N/dim': {tuple(sorted(big))}")


if __name__ == "__main__":
    main()
