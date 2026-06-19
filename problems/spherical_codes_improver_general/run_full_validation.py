"""Standalone full-catalogue validation harness (the headline / head-to-head number).

The evolution pipeline's per-stage timeout (40 min) cannot host a multi-hour grade, so
the paper-matched validation runs here instead. It reuses the *exact* grader primitive
(`validate._eval_config`) on every config, so the number is identical to what evolution
would compute — only parallelised across the 90 independent configs.

Usage:
    python run_full_validation.py initial_programs/paper_evolved.py [--label E7] \
        [--eval-set full90] [--rounds 3] [--steps 10] [--workers 8] [--threads 4] \
        [--seed 42] [--out results_E7.json]

Run the evolved champion and the paper baselines (paper_evolved.py / paper_evolved_plus.py)
identically, then compare the `fitness` line — that is the head-to-head.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing as mp
import os
import sys
import time
import types

_PROB = os.path.dirname(os.path.abspath(__file__))

_IMP = None  # per-worker improver class, set by _init_worker


def _load_improver(path: str):
    spec = importlib.util.spec_from_file_location("sc_improver_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.entrypoint()


def _init_worker(path: str, threads: int) -> None:
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = str(threads)
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    sys.path.insert(0, _PROB)
    global _IMP
    _IMP = _load_improver(path)


def _grade_one(task: tuple) -> dict:
    d, n, r_rounds, b_steps, hi, lo, dry, config_timeout, norm_tol, seed = task
    import cohn_catalogue as cc
    import validate as V

    X_cohn, mu_cohn = cc.load_frozen(d, n)
    cfg = types.SimpleNamespace(
        r_rounds=r_rounds, b_steps=b_steps, intensity_hi=hi, intensity_lo=lo,
        dry_patience=dry, config_timeout=config_timeout, norm_tol=norm_tol, seed=seed,
    )
    deadline = time.monotonic() + config_timeout
    t0 = time.perf_counter()
    res = V._eval_config(_IMP, d, n, X_cohn, mu_cohn, cfg, deadline)
    res["secs"] = time.perf_counter() - t0
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("program", help="path to a .py with entrypoint() -> Improver")
    ap.add_argument("--label", default=None)
    ap.add_argument("--eval-set", default="full90")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--intensity-hi", type=float, default=1.0)
    ap.add_argument("--intensity-lo", type=float, default=1e-4)
    ap.add_argument("--dry-patience", type=int, default=3)
    ap.add_argument("--config-timeout", type=float, default=100000.0)  # full effort
    ap.add_argument("--norm-tol", type=float, default=1e-12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    label = args.label or os.path.basename(args.program)
    sys.path.insert(0, _PROB)
    import cohn_catalogue as cc

    configs = cc.eval_configs(args.eval_set)
    panel = set(cc.eval_configs("panel"))
    tasks = [
        (d, n, args.rounds, args.steps, args.intensity_hi, args.intensity_lo,
         args.dry_patience, args.config_timeout, args.norm_tol, args.seed)
        for (d, n) in configs
    ]

    t0 = time.perf_counter()
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=_init_worker,
                  initargs=(os.path.abspath(args.program), args.threads)) as pool:
        results = pool.map(_grade_one, tasks)
    elapsed = time.perf_counter() - t0

    results.sort(key=lambda r: (r["d"], r["n"]))
    total = len(results)
    fitness = 100.0 * sum(r["gain"] for r in results) / total
    improved = sum(1 for r in results if r["improved"])
    valid = sum(1 for r in results if r["produced_valid"])
    abs_red = sum(r["mu_cohn"] - r["mu_best"] for r in results) / total

    in_panel = [r for r in results if (r["d"], r["n"]) in panel]
    out_panel = [r for r in results if (r["d"], r["n"]) not in panel]

    def mean_gain_pct(rs):
        return 100.0 * sum(r["gain"] for r in rs) / len(rs) if rs else float("nan")

    print(f"\n================  {label}  ({args.eval_set}, R={args.rounds}, B={args.steps}, "
          f"seed={args.seed}, {args.workers}w×{args.threads}t)  ================")
    print(f"FITNESS (mean rel. improvement over Cohn): {fitness:.4f}%")
    print(f"  improved {improved}/{total}   valid {valid}/{total}   "
          f"mean abs μ reduction {abs_red:.6f}   wall {elapsed/60:.1f} min")
    print(f"  in-panel({len(in_panel)})  mean gain {mean_gain_pct(in_panel):.4f}%   "
          f"out-of-panel({len(out_panel)}) mean gain {mean_gain_pct(out_panel):.4f}%   "
          f"(panel→rest gap shows generalization)")
    by_d: dict[int, list] = {}
    for r in results:
        by_d.setdefault(r["d"], []).append(r)
    print("  per-dimension  mean-gain% / improved / slowest-config-secs:")
    for d in sorted(by_d):
        rs = by_d[d]
        mg = 100.0 * sum(x["gain"] for x in rs) / len(rs)
        sc = sum(1 for x in rs if x["improved"])
        slow = max(x.get("secs", 0.0) for x in rs)
        print(f"    d={d:2d}:  {mg:8.4f}%   {sc}/{len(rs)}   {slow:6.1f}s")
    top = sorted(results, key=lambda r: r["gain"], reverse=True)[:5]
    print("  largest gains: " + ", ".join(
        f"(d={r['d']},N={r['n']}) {r['gain_pct']:.3f}%" for r in top if r["gain"] > 0))
    errs = [r for r in results if r.get("error")]
    if errs:
        print(f"  {len(errs)} configs reported an error; e.g. "
              f"(d={errs[0]['d']},N={errs[0]['n']}): {errs[0]['error']}")

    out = args.out or f"validation_{label}_{args.eval_set}_R{args.rounds}_seed{args.seed}.json"
    with open(os.path.join(_PROB, out), "w") as f:
        json.dump({"label": label, "eval_set": args.eval_set, "rounds": args.rounds,
                   "steps": args.steps, "seed": args.seed, "fitness": fitness,
                   "improved": improved, "valid": valid, "elapsed_min": elapsed / 60,
                   "results": results}, f, indent=2)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
