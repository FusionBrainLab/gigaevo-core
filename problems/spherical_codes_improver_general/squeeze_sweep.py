"""Hyperparameter sweep of the Stage-A/Stage-B grader protocol on the VAL set.

Scientific protocol-tuning: rather than running one blind heavy budget, we sweep
the basin-hopping knobs (noising steps M, sigma schedule sigma_max->sigma_min,
restarts R, fresh-restart period) on the validation PANEL (the high-headroom
configs used to steer evolution), pick the protocol that maximises mean panel
gain over Cohn, then apply that single protocol unchanged to the full90 TEST set
for every program (a fair head-to-head).

FIXED-WORK by design: each cell runs a prescribed R x M search (wall is only a
safety backstop), so contention on the shared box slows the run but never
corrupts the fairness of the comparison. Replicate base-seeds give genuine
independent restart streams (see squeeze_eval base_seed offset).

Usage (stage 1, shape):
    python squeeze_sweep.py \
        --program initial_programs/champion_gen11_199c6110.py \
        --eval-set panel --workers 40 \
        --restarts-grid 20 --bsteps-grid 8,16,24,36 \
        --sigma-grid 1.0:1e-4,0.5:1e-5,1.0:1e-6 \
        --fresh-grid 5 --seeds 0,1 --wall 3600 --dry-patience 8 \
        --tag shape --out-dir sweep_results
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

import numpy as np

_PROB = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROB)

from squeeze_eval import _cost, _init_worker, _squeeze_one  # noqa: E402


def _parse_sigma_grid(s: str) -> list[tuple[float, float]]:
    out = []
    for tok in s.split(","):
        hi, _, lo = tok.partition(":")
        out.append((float(hi), float(lo)))
    return out


def _ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x != ""]


def _cell_label(r, m, hi, lo, fe) -> str:
    return f"R{r}_M{m}_hi{hi:g}_lo{lo:g}_fe{fe}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--program",
        required=True,
        help="path to champion .py (the program we tune for)",
    )
    ap.add_argument("--eval-set", default="panel")
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--restarts-grid", default="20")
    ap.add_argument("--bsteps-grid", default="8,16,24,36")
    ap.add_argument("--sigma-grid", default="1.0:1e-4,0.5:1e-5,1.0:1e-6")
    ap.add_argument("--fresh-grid", default="5")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument(
        "--wall", type=float, default=3600.0, help="per-config safety backstop (s)"
    )
    ap.add_argument("--dry-patience", type=int, default=8)
    ap.add_argument("--norm-tol", type=float, default=1e-9)
    ap.add_argument("--tag", default="sweep")
    ap.add_argument("--out-dir", default="sweep_results")
    args = ap.parse_args()

    import cohn_catalogue as cc

    configs = cc.eval_configs(args.eval_set)
    Rs = _ints(args.restarts_grid)
    Ms = _ints(args.bsteps_grid)
    sigmas = _parse_sigma_grid(args.sigma_grid)
    fresh = _ints(args.fresh_grid)
    seeds = _ints(args.seeds)
    prog_path = os.path.abspath(args.program)

    cells = [
        (r, m, hi, lo, fe)
        for r in Rs
        for m in Ms
        for (hi, lo) in sigmas
        for fe in fresh
    ]

    tasks = []
    for r, m, hi, lo, fe in cells:
        label = _cell_label(r, m, hi, lo, fe)
        for d, n in configs:
            for bs in seeds:
                tasks.append(
                    (
                        label,
                        prog_path,
                        d,
                        n,
                        r,
                        m,
                        hi,
                        lo,
                        args.wall,
                        args.norm_tol,
                        fe,
                        args.dry_patience,
                        bs,
                    )
                )
    # long poles first: heaviest config x largest R*M
    tasks.sort(key=lambda t: _cost(t[2], t[3]) * t[4] * t[5], reverse=True)

    print(
        f"sweep[{args.tag}]: {len(cells)} cells x {len(configs)} configs x {len(seeds)} seeds "
        f"= {len(tasks)} tasks | {args.workers}w x {args.threads}t | eval_set={args.eval_set} "
        f"| wall<= {args.wall:.0f}s (backstop) dry_patience={args.dry_patience}",
        flush=True,
    )
    print(f"  cells: {[_cell_label(*c) for c in cells]}", flush=True)

    t0 = time.perf_counter()
    ctx = mp.get_context("spawn")
    results = []
    done = 0
    with ctx.Pool(
        args.workers, initializer=_init_worker, initargs=(args.threads,)
    ) as pool:
        for res in pool.imap_unordered(_squeeze_one, tasks):
            res.pop("best_X", None)  # don't keep packings during sweep
            results.append(res)
            done += 1
            if done % 20 == 0 or done == len(tasks):
                print(
                    f"  [{done}/{len(tasks)}] {(time.perf_counter() - t0) / 60:.1f} min",
                    flush=True,
                )
    elapsed = time.perf_counter() - t0

    # aggregate: per cell -> per config mean gain over seeds -> mean over configs
    by_cell: dict[str, list[dict]] = {}
    for r in results:
        by_cell.setdefault(r["name"], []).append(r)

    cell_stats = []
    for label, rs in by_cell.items():
        by_cfg: dict[tuple, list[float]] = {}
        for r in rs:
            by_cfg.setdefault((r["d"], r["n"]), []).append(r["gain"])
        per_cfg_mean = {k: float(np.mean(v)) for k, v in by_cfg.items()}
        per_cfg_best = {k: float(np.max(v)) for k, v in by_cfg.items()}
        vals = np.array(list(per_cfg_mean.values()))
        best_vals = np.array(list(per_cfg_best.values()))
        n_cfg = len(vals)
        mean_pct = 100.0 * float(vals.mean())
        se_pct = 100.0 * float(vals.std(ddof=1) / np.sqrt(n_cfg)) if n_cfg > 1 else 0.0
        best_mean_pct = 100.0 * float(best_vals.mean())  # if we kept best-over-seeds
        improved = sum(1 for v in per_cfg_best.values() if v > 1e-12)
        cell_stats.append(
            dict(
                cell=label,
                mean_gain_pct=mean_pct,
                se_pct=se_pct,
                best_over_seeds_mean_pct=best_mean_pct,
                improved_configs=improved,
                n_configs=n_cfg,
                per_cfg_mean={f"{d}_{n}": g for (d, n), g in per_cfg_mean.items()},
                per_cfg_best={f"{d}_{n}": g for (d, n), g in per_cfg_best.items()},
            )
        )
    cell_stats.sort(key=lambda c: c["best_over_seeds_mean_pct"], reverse=True)

    out_dir = Path(_PROB) / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(
        tag=args.tag,
        program=prog_path,
        eval_set=args.eval_set,
        grid=dict(
            restarts=Rs,
            bsteps=Ms,
            sigma=sigmas,
            fresh=fresh,
            seeds=seeds,
            wall=args.wall,
            dry_patience=args.dry_patience,
        ),
        elapsed_min=elapsed / 60,
        cells=cell_stats,
        raw=results,
    )
    out_path = out_dir / f"sweep_{args.tag}.json"
    out_path.write_text(json.dumps(payload, indent=2))

    print(
        f"\n========  SWEEP[{args.tag}] RANKING ({args.eval_set}, {elapsed / 60:.1f} min)  ========"
    )
    print(f"{'cell':>26} | {'mean±SE %':>16} | {'best/seed %':>11} | improved")
    for c in cell_stats:
        print(
            f"{c['cell']:>26} | {c['mean_gain_pct']:8.4f} ± {c['se_pct']:5.4f} | "
            f"{c['best_over_seeds_mean_pct']:11.4f} | {c['improved_configs']}/{c['n_configs']}"
        )
    best = cell_stats[0]
    print(
        f"\nBEST cell (by best-over-seeds mean): {best['cell']}  -> {best['best_over_seeds_mean_pct']:.4f}%"
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
