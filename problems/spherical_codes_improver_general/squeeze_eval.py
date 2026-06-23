"""Max-effort head-to-head evaluation on the full Cohn catalogue.

Pushes each improver as hard as practical per config (the "squeeze"): Stage A
warm-start improve of the Cohn config, then a deep Stage-B basin-hopping search
with many restart chains, distinct per-restart seed streams, occasional fresh
restarts (escape the warm-start basin), a hard per-config wall budget, and
anytime tracking of the best TRUE mu = max_{i<j}<x_i,x_j> ever seen.

This is intentionally heavier than the evolution grader (validate._eval_config),
which caps Stage B at R rounds x B steps with seeds fixed by (r,step). Here the
seed streams diversify per restart, so repeated chains give a genuine spread
(mean +/- sigma) on top of the best-achieved number.

The score is identical in spirit to the grader's fitness
(100 * mean_config max(0,(mu_cohn-mu_best)/|mu_cohn|)), only computed under a
deeper search, so champion vs E7 vs E8 stay directly comparable when all three
run through THIS harness.

Usage:
    python squeeze_eval.py \
        --program champion=initial_programs/champion_gen11_199c6110.py \
        --program E7=initial_programs/paper_evolved.py \
        --program E8=initial_programs/paper_evolved_plus.py \
        --eval-set full90 --workers 64 --restarts 60 --b-steps 24 \
        --wall 1200 --threads 1 --out-dir squeeze_results --save-packings champion
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import signal
import sys
import time

import numpy as np

_PROB = os.path.dirname(os.path.abspath(__file__))
_PROGRAMS: dict[str, object] = {}  # path -> Improver class (per-worker cache)


class _Timeout(Exception):
    pass


def _on_alarm(signum, frame):
    raise _Timeout()


def _load_improver(path: str):
    if path in _PROGRAMS:
        return _PROGRAMS[path]
    spec = importlib.util.spec_from_file_location(
        f"sc_under_test_{abs(hash(path))}", path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    imp = mod.entrypoint()
    _PROGRAMS[path] = imp
    return imp


def _init_worker(threads: int) -> None:
    for v in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[v] = str(threads)
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    os.environ.setdefault(
        "XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
    )
    sys.path.insert(0, _PROB)


def _true_mu(X: np.ndarray) -> float:
    G = X @ X.T
    np.fill_diagonal(G, -np.inf)
    return float(G.max())


def _check(X, n: int, d: int, tol: float) -> tuple[bool, float]:
    A = np.asarray(X, dtype=np.float64)
    if A.shape != (n, d) or not np.all(np.isfinite(A)):
        return False, math.inf
    norms = np.linalg.norm(A, axis=1)
    if np.max(np.abs(norms - 1.0)) > tol:
        return False, math.inf
    return True, _true_mu(A)


def _call(fn, call_seed: int, deadline: float, *args, **kwargs):
    """Run an improver call with the global RNG seeded and a hard wall deadline."""
    np.random.seed(call_seed & 0x7FFFFFFF)
    remaining = deadline - time.monotonic()
    if remaining <= 0.0:
        raise _Timeout()
    prev = signal.signal(signal.SIGALRM, _on_alarm)
    signal.setitimer(signal.ITIMER_REAL, remaining)
    try:
        return fn(*args, **kwargs)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, prev)


def _squeeze_one(task: tuple) -> dict:
    (
        name,
        path,
        d,
        n,
        restarts,
        b_steps,
        hi,
        lo,
        wall,
        norm_tol,
        fresh_every,
        dry_patience,
        base_seed,
    ) = task
    bo = (
        base_seed * 100_000_000
    )  # base-seed offset: independent restart streams per replicate
    import cohn_catalogue as cc

    Improver = _load_improver(path)
    X_cohn, mu_cohn = cc.load_frozen(d, n)
    t0 = time.perf_counter()
    deadline = time.monotonic() + wall

    out = {
        "name": name,
        "d": d,
        "n": n,
        "mu_cohn": mu_cohn,
        "mu_best": mu_cohn,
        "gain": 0.0,
        "gain_pct": 0.0,
        "improved": False,
        "produced_valid": False,
        "n_improve": 0,
        "n_accept": 0,
        "n_restarts": 0,
        "chain_bests": [],
        "secs": 0.0,
        "error": None,
        "best_X": None,
    }

    try:
        imp = Improver(n=n, d=d, seed=base_seed)
    except Exception as e:  # noqa: BLE001
        out["error"] = f"__init__ {type(e).__name__}: {e}"
        out["secs"] = time.perf_counter() - t0
        return out

    best_mu = mu_cohn
    best_X = X_cohn.copy()
    n_improve = n_accept = 0

    # STAGE A: warm-start improve of the Cohn config.
    if time.monotonic() < deadline:
        try:
            cand = _call(imp.improve, bo + 1, deadline, X_cohn.copy(), seed=base_seed)
            n_improve += 1
            ok, m = _check(cand, n, d, norm_tol)
            if ok:
                out["produced_valid"] = True
                if m < best_mu:
                    best_mu, best_X = m, np.asarray(cand, dtype=np.float64)
                    n_accept += 1
        except _Timeout:
            pass
        except Exception as e:  # noqa: BLE001
            out["error"] = f"improve(A) {type(e).__name__}: {e}"

    # STAGE B: deep basin-hopping with restart chains + distinct seed streams.
    intensities = np.geomspace(hi, lo, num=max(1, b_steps))
    chain_bests: list[float] = []
    r = 0
    dry = 0
    while r < restarts and time.monotonic() < deadline:
        improved_global = False
        # Most chains hop from the global best (exploit); periodically start from a
        # fresh random config (explore) to escape the warm-start basin.
        if fresh_every > 0 and r % fresh_every == (fresh_every - 1):
            try:
                gseed = bo + 7_000_000 + r
                start = _call(imp.generate_config, gseed, deadline, seed=gseed)
                start = np.asarray(start, dtype=np.float64)
            except Exception:  # noqa: BLE001
                start = best_X.copy()
        else:
            start = best_X.copy()

        cur, cur_mu = start, _true_mu(start)
        local_best = cur_mu
        for step, inten in enumerate(intensities):
            if time.monotonic() > deadline:
                break
            s = bo + r * 100_000 + step
            try:
                pert = _call(
                    imp.perturb, s, deadline, cur.copy(), intensity=float(inten), seed=s
                )
                cand = _call(imp.improve, s + 50_000, deadline, pert, seed=s + 50_000)
                n_improve += 1
            except _Timeout:
                break
            except Exception as e:  # noqa: BLE001
                out["error"] = f"stageB {type(e).__name__}: {e}"
                continue
            ok, m = _check(cand, n, d, norm_tol)
            if not ok:
                continue
            out["produced_valid"] = True
            if m < cur_mu:  # local monotone accept (basin walk)
                cur, cur_mu = np.asarray(cand, dtype=np.float64), m
            if m < local_best:
                local_best = m
            if m < best_mu:  # anytime global best
                best_X, best_mu = np.asarray(cand, dtype=np.float64), m
                n_accept += 1
                improved_global = True
        chain_bests.append(local_best)
        r += 1
        dry = 0 if improved_global else dry + 1
        if dry_patience > 0 and dry >= dry_patience:
            break  # no global gain for dry_patience consecutive restarts

    denom = abs(mu_cohn) if abs(mu_cohn) > 1e-12 else 1.0
    gain = max(0.0, (mu_cohn - best_mu) / denom)
    out.update(
        mu_best=best_mu,
        gain=gain,
        gain_pct=100.0 * gain,
        improved=best_mu < mu_cohn - 1e-15,
        n_improve=n_improve,
        n_accept=n_accept,
        n_restarts=r,
        chain_bests=chain_bests,
        secs=time.perf_counter() - t0,
        best_X=best_X.tolist() if (best_mu < mu_cohn - 1e-15) else None,
    )
    return out


def _cost(d: int, n: int) -> float:
    return float(n) * n + float(n) * d  # ~ Gram + variable count


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--program",
        action="append",
        dest="programs",
        required=True,
        help="name=path (repeatable)",
    )
    ap.add_argument("--eval-set", default="full90")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--restarts", type=int, default=60)
    ap.add_argument("--b-steps", type=int, default=24)
    ap.add_argument("--intensity-hi", type=float, default=1.0)
    ap.add_argument("--intensity-lo", type=float, default=1e-4)
    ap.add_argument(
        "--wall", type=float, default=1200.0, help="per-config wall budget (s)"
    )
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument(
        "--fresh-every",
        type=int,
        default=6,
        help="every k-th restart starts fresh (0=never)",
    )
    ap.add_argument(
        "--dry-patience",
        type=int,
        default=6,
        help="stop a config after k restarts with no global gain (0=never)",
    )
    ap.add_argument(
        "--base-seed",
        type=int,
        default=0,
        help="replicate offset: independent restart streams",
    )
    ap.add_argument("--norm-tol", type=float, default=1e-9)
    ap.add_argument("--out-dir", default="squeeze_results")
    ap.add_argument(
        "--save-packings",
        default="",
        help="comma-sep program names whose best packings to save",
    )
    args = ap.parse_args()

    sys.path.insert(0, _PROB)
    import cohn_catalogue as cc

    progs = []
    for spec in args.programs:
        name, _, path = spec.partition("=")
        progs.append((name, os.path.abspath(path)))
    configs = cc.eval_configs(args.eval_set)

    tasks = [
        (
            name,
            path,
            d,
            n,
            args.restarts,
            args.b_steps,
            args.intensity_hi,
            args.intensity_lo,
            args.wall,
            args.norm_tol,
            args.fresh_every,
            args.dry_patience,
            args.base_seed,
        )
        for (name, path) in progs
        for (d, n) in configs
    ]
    tasks.sort(key=lambda t: _cost(t[2], t[3]), reverse=True)  # long poles first

    print(
        f"squeeze: {len(progs)} programs x {len(configs)} configs = {len(tasks)} tasks | "
        f"{args.workers}w x {args.threads}t | restarts<={args.restarts} b_steps={args.b_steps} "
        f"wall<={args.wall:.0f}s | eval_set={args.eval_set}",
        flush=True,
    )

    t0 = time.perf_counter()
    ctx = mp.get_context("spawn")
    results: list[dict] = []
    done = 0
    with ctx.Pool(
        args.workers, initializer=_init_worker, initargs=(args.threads,)
    ) as pool:
        for res in pool.imap_unordered(_squeeze_one, tasks):
            results.append(res)
            done += 1
            if done % 10 == 0 or done == len(tasks):
                el = (time.perf_counter() - t0) / 60
                print(f"  [{done}/{len(tasks)}] {el:.1f} min elapsed", flush=True)
    elapsed = time.perf_counter() - t0

    out_dir = Path(_PROB) / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    save_names = {s for s in args.save_packings.split(",") if s}

    by_prog: dict[str, list[dict]] = {}
    for r in results:
        by_prog.setdefault(r["name"], []).append(r)

    summary = {}
    for name, rs in sorted(by_prog.items()):
        rs.sort(key=lambda r: (r["d"], r["n"]))
        total = len(rs)
        fitness = 100.0 * sum(r["gain"] for r in rs) / total
        improved = sum(1 for r in rs if r["improved"])
        valid = sum(1 for r in rs if r["produced_valid"])
        # per-config sigma over restart-chain bests -> mean gain sigma signal
        chain_sigmas = []
        for r in rs:
            cb = r.get("chain_bests") or []
            if len(cb) >= 2:
                gains = [
                    max(0.0, (r["mu_cohn"] - m) / (abs(r["mu_cohn"]) or 1.0))
                    for m in cb
                ]
                chain_sigmas.append(float(np.std(gains)) * 100.0)
        mean_chain_sigma = float(np.mean(chain_sigmas)) if chain_sigmas else 0.0

        # strip best_X out of the per-config json unless saving packings
        slim = []
        packings = {}
        for r in rs:
            rr = dict(r)
            bx = rr.pop("best_X", None)
            slim.append(rr)
            if name in save_names and bx is not None:
                packings[f"{r['d']}_{r['n']}"] = bx
        payload = {
            "name": name,
            "eval_set": args.eval_set,
            "restarts": args.restarts,
            "b_steps": args.b_steps,
            "wall": args.wall,
            "fitness": fitness,
            "improved": improved,
            "valid": valid,
            "mean_chain_sigma_pct": mean_chain_sigma,
            "elapsed_min": elapsed / 60,
            "results": slim,
        }
        (out_dir / f"squeeze_{name}.json").write_text(json.dumps(payload, indent=2))
        if packings:
            np.savez_compressed(
                out_dir / f"packings_{name}.npz",
                **{k: np.asarray(v) for k, v in packings.items()},
            )

        by_d: dict[int, list] = {}
        for r in rs:
            by_d.setdefault(r["d"], []).append(r)
        summary[name] = dict(
            fitness=fitness,
            improved=improved,
            valid=valid,
            sigma=mean_chain_sigma,
            by_d=by_d,
            rows=rs,
        )

    print(
        f"\n================  SQUEEZE RESULTS  ({args.eval_set}, wall<={args.wall:.0f}s, {elapsed / 60:.1f} min)  ================"
    )
    print(
        f"{'program':>10} | {'fitness%':>9} | {'improved':>9} | {'valid':>6} | {'mean σ%':>8}"
    )
    for name in sorted(summary):
        s = summary[name]
        print(
            f"{name:>10} | {s['fitness']:9.4f} | {s['improved']:>4}/{len(s['rows'])} | {s['valid']:>4}/{len(s['rows'])} | {s['sigma']:8.4f}"
        )

    # head-to-head per dimension for the first listed program vs others
    print("\nper-dimension fitness% (mean gain):")
    dims = sorted({r["d"] for r in results})
    hdr = "  d  | " + " | ".join(f"{name:>9}" for name in sorted(summary))
    print(hdr)
    for d in dims:
        cells = []
        for name in sorted(summary):
            rows = [r for r in summary[name]["rows"] if r["d"] == d]
            mg = (
                100.0 * sum(r["gain"] for r in rows) / len(rows)
                if rows
                else float("nan")
            )
            cells.append(f"{mg:9.4f}")
        print(f" {d:2d}  | " + " | ".join(cells))

    print(f"\nwrote per-program json + packings to {out_dir}")


if __name__ == "__main__":
    main()
