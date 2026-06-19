"""Multi-config warm-start grader for the general spherical-codes improver.

Mirrors the ImprovEvolve scan protocol (Appendix E.3). For each (d, N) in the
eval set:

  * current = the frozen Cohn config; current_mu = mu(current).
  * STAGE A: cand = improve(Cohn); accept if mu(cand) <= current_mu.
  * STAGE B: R rounds x B steps of perturb -> improve, intensities along
    geomspace(hi, lo, B); monotone acceptance (only accept mu(cand) <= current_mu).
    A fully-dry round (no acceptance) advances an early-stop counter.
  * gain = max(0, (mu_Cohn - current_mu) / |mu_Cohn|)   (>= 0 by construction).

fitness = 100 * mean_i gain_i   (mean relative improvement over Cohn, in %).

Configs are evaluated CONCURRENTLY across forked worker processes (one per
config, capped at SPHERICAL_CONFIG_WORKERS); per-mutant wall-clock collapses to
~one config's budget instead of the sum. Results are gathered as workers finish;
any config still running at the global eval deadline is killed and scored as
skipped (gain 0), so a slow program yields partial results rather than nothing.

The improver is floored at the Cohn baseline (can never score below 0). All
behaviour is env-switchable so evolution and paper-matched validation share
one grader. NO prints; feedback is returned via artifact["feedback_preview"].
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import signal
import time

import cohn_catalogue as cc
from loguru import logger
import numpy as np


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw not in (None, "") else default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw not in (None, "") else default


class _Cfg:
    __slots__ = (
        "eval_set",
        "r_rounds",
        "b_steps",
        "intensity_hi",
        "intensity_lo",
        "dry_patience",
        "config_timeout",
        "eval_timeout",
        "config_workers",
        "norm_tol",
        "seed",
    )

    def __init__(self) -> None:
        # Default = panel: the only consumer of validate()/_Cfg is the live evolution
        # pipeline. Full90 validation runs through run_full_validation.py with an explicit
        # --eval-set, so a "full90" default here would only ever mean "someone forgot to set
        # panel" -> every mutant silently graded for ~40-100 min.
        self.eval_set = os.environ.get("SPHERICAL_EVAL_SET", "panel")
        self.r_rounds = _env_int("SPHERICAL_R_ROUNDS", 1)
        self.b_steps = _env_int("SPHERICAL_B_STEPS", 10)
        # geomspace endpoints must be strictly positive (0 -> ValueError); clamp the footgun.
        self.intensity_hi = max(1e-12, _env_float("SPHERICAL_INTENSITY_HI", 1.0))
        self.intensity_lo = max(1e-12, _env_float("SPHERICAL_INTENSITY_LO", 1e-4))
        self.dry_patience = _env_int("SPHERICAL_DRY_PATIENCE", 1)
        self.config_timeout = _env_float("SPHERICAL_CONFIG_TIMEOUT", 25.0)
        self.eval_timeout = _env_float("SPHERICAL_EVAL_TIMEOUT", 1800.0)
        # Concurrency: one forked worker per config, capped here. Each worker is single-
        # threaded (EVO_EXEC_THREADS=1 in the launch), so parallelism is across cores.
        self.config_workers = _env_int("SPHERICAL_CONFIG_WORKERS", 8)
        self.norm_tol = _env_float("SPHERICAL_NORM_TOL", 1e-12)
        self.seed = _env_int("SPHERICAL_SEED", 42)


def _check(arr, n: int, d: int, norm_tol: float) -> tuple[bool, float | None]:
    """Validate an improver output; return (ok, mu). Never raises."""
    try:
        a = np.asarray(arr, dtype=np.float64)
    except Exception:
        return False, None
    if a.shape != (n, d):
        return False, None
    if not np.all(np.isfinite(a)):
        return False, None
    norms = np.linalg.norm(a, axis=1)
    if float(np.max(np.abs(norms - 1.0))) > norm_tol:
        return False, None
    G = a @ a.T
    np.fill_diagonal(G, -np.inf)
    return True, float(G.max())


class _CallTimeout(Exception):
    """An improver call exceeded its per-config wall-clock deadline."""


def _on_alarm(signum, frame):
    raise _CallTimeout()


def _call(fn, call_seed: int, deadline: float, *args, **kwargs):
    """Run an untrusted improver call with the global RNG seeded (reproducibility) and a hard
    wall-clock deadline. Each config runs in a forked worker's main thread, so SIGALRM fires
    even mid-call; the timer and prior handler are always restored on exit."""
    np.random.seed(call_seed & 0x7FFFFFFF)
    remaining = deadline - time.monotonic()
    if remaining <= 0.0:
        raise _CallTimeout()
    prev = signal.signal(signal.SIGALRM, _on_alarm)
    signal.setitimer(signal.ITIMER_REAL, remaining)
    try:
        return fn(*args, **kwargs)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, prev)


def _eval_config(
    Improver,
    d: int,
    n: int,
    X_cohn: np.ndarray,
    mu_cohn: float,
    cfg: _Cfg,
    deadline: float,
) -> dict:
    assert n > d, (
        f"eval-set invariant violated for (d={d}, n={n}): n must exceed d. The grader's "
        "integrity guarantee (Welch bound forces mu>0; output shape (n,d) != (d,d)) needs n > d."
    )
    current = X_cohn.copy()
    current_mu = mu_cohn
    produced_valid = False
    n_improve = 0
    n_accept = 0
    n_timeout = 0
    last_err: str | None = None

    try:
        imp = Improver(n=n, d=d, seed=cfg.seed)
    except Exception as e:
        return {
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
            "n_timeout": 0,
            "error": f"__init__ {type(e).__name__}: {e}",
        }

    # STAGE A: single warm-start improve of the Cohn config.
    if time.monotonic() < deadline:
        try:
            cand = _call(imp.improve, cfg.seed, deadline, X_cohn.copy(), seed=cfg.seed)
            n_improve += 1
            ok, m = _check(cand, n, d, cfg.norm_tol)
            if ok:
                produced_valid = True
                if m <= current_mu:
                    current, current_mu = np.array(cand, dtype=np.float64), m
                    n_accept += 1
        except _CallTimeout:
            n_timeout += 1
        except Exception as e:
            last_err = f"improve(A) {type(e).__name__}: {e}"

    # STAGE B: R rounds x B steps perturb -> improve, monotone accept.
    intensities = np.geomspace(
        cfg.intensity_hi, cfg.intensity_lo, num=max(1, cfg.b_steps)
    )
    dry_rounds = 0
    for r in range(cfg.r_rounds):
        if time.monotonic() > deadline:
            break
        accepts = 0
        for step, inten in enumerate(intensities):
            if time.monotonic() > deadline:
                break
            try:
                pert = _call(
                    imp.perturb,
                    r * 1000 + step,
                    deadline,
                    current.copy(),
                    intensity=float(inten),
                    seed=r * 1000 + step,
                )
                cand = _call(
                    imp.improve,
                    10000 + r * 1000 + step,
                    deadline,
                    pert,
                    seed=10000 + r * 1000 + step,
                )
                n_improve += 1
            except _CallTimeout:
                n_timeout += 1
                continue
            except Exception as e:
                last_err = f"stageB {type(e).__name__}: {e}"
                continue
            ok, m = _check(cand, n, d, cfg.norm_tol)
            if not ok:
                continue
            produced_valid = True
            if m <= current_mu:
                current, current_mu = np.array(cand, dtype=np.float64), m
                accepts += 1
                n_accept += 1
        if accepts == 0:
            dry_rounds += 1
            if dry_rounds >= cfg.dry_patience:
                break
        else:
            dry_rounds = 0

    denom = abs(mu_cohn) if abs(mu_cohn) > 1e-12 else 1.0
    gain = max(0.0, (mu_cohn - current_mu) / denom)
    return {
        "d": d,
        "n": n,
        "mu_cohn": mu_cohn,
        "mu_best": current_mu,
        "gain": gain,
        "gain_pct": 100.0 * gain,
        "improved": current_mu < mu_cohn - 1e-12,
        "produced_valid": produced_valid,
        "n_improve": n_improve,
        "n_accept": n_accept,
        "n_timeout": n_timeout,
        "error": last_err,
    }


def _skipped(d: int, n: int, mu_cohn: float, reason: str) -> dict:
    return {
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
        "n_timeout": 0,
        "error": reason,
    }


def _worker(
    Improver, d: int, n: int, X_cohn: np.ndarray, mu_cohn: float, cfg: _Cfg, out_q
) -> None:
    """Forked child: evaluate ONE config and put its result dict on out_q.

    Runs in the child's main thread, so _call's SIGALRM per-call deadline fires here
    exactly as in a sequential run. Each child self-bounds to cfg.config_timeout; the
    parent's eval deadline (and a hard terminate) is the backstop for a child that
    swallows the alarm. Only the result dict crosses the queue -- the Improver class
    and X_cohn are inherited via fork, never re-pickled (the class lives in a throwaway
    user_code module and would not survive plain pickling)."""
    try:
        deadline = time.monotonic() + cfg.config_timeout
        res = _eval_config(Improver, d, n, X_cohn, mu_cohn, cfg, deadline)
    except Exception as e:
        res = _skipped(d, n, mu_cohn, f"worker {type(e).__name__}: {e}")
    out_q.put(res)


def _run_parallel(Improver, configs_mu: list, cfg: _Cfg, eval_deadline: float) -> dict:
    """Evaluate configs across forked workers in waves of cfg.config_workers.

    Returns {(d, n): result_dict} for every config that finished before eval_deadline.
    Workers still running at the deadline are terminated; the caller marks them skipped."""
    ctx = mp.get_context("fork")
    out: dict = {}
    workers = max(1, cfg.config_workers)
    idx = 0
    while idx < len(configs_mu):
        if time.monotonic() >= eval_deadline:
            break
        batch = configs_mu[idx : idx + workers]
        idx += len(batch)
        q = ctx.Queue()
        procs = []
        for d, n, X_cohn, mu_cohn in batch:
            p = ctx.Process(
                target=_worker, args=(Improver, d, n, X_cohn, mu_cohn, cfg, q)
            )
            p.start()
            procs.append(p)
        try:
            got = 0
            while got < len(batch):
                remaining = eval_deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                try:
                    res = q.get(timeout=min(remaining, 2.0))
                    out[(res["d"], res["n"])] = res
                    got += 1
                except queue.Empty:
                    if not any(p.is_alive() for p in procs):
                        while True:  # drain results put just before the children exited
                            try:
                                res = q.get_nowait()
                                out[(res["d"], res["n"])] = res
                                got += 1
                            except queue.Empty:
                                break
                        break
        finally:
            for p in procs:
                if p.is_alive():
                    p.terminate()
                p.join(timeout=2.0)
                if p.is_alive():
                    p.kill()
                    p.join(timeout=1.0)
    return out


def _feedback(results: list[dict], cfg: _Cfg, fitness: float) -> str:
    total = len(results)
    improved = sum(1 for r in results if r["improved"])
    valid = sum(1 for r in results if r["produced_valid"])
    lines = [
        f"Mean relative improvement over Cohn: {fitness:.4f}%  "
        f"(eval_set={cfg.eval_set}, R={cfg.r_rounds}, B={cfg.b_steps}, configs={total})",
        f"Improved on Cohn: {improved}/{total} ({100.0 * improved / total:.0f}%); "
        f"valid array on {valid}/{total}.",
        "Per-dimension  mean-gain% / success:",
    ]
    by_d: dict[int, list[dict]] = {}
    for r in results:
        by_d.setdefault(r["d"], []).append(r)
    for d in sorted(by_d):
        rs = by_d[d]
        mg = 100.0 * sum(x["gain"] for x in rs) / len(rs)
        sc = sum(1 for x in rs if x["improved"])
        lines.append(f"  d={d:2d}:  {mg:7.4f}%   {sc}/{len(rs)}")
    top = sorted(results, key=lambda r: r["gain"], reverse=True)[:3]
    if top and top[0]["gain"] > 0:
        lines.append(
            "Largest gains: "
            + ", ".join(
                f"(d={r['d']},N={r['n']}) {r['gain_pct']:.3f}%"
                for r in top
                if r["gain"] > 0
            )
        )
    stuck = [r for r in results if r["produced_valid"] and not r["improved"]]
    if stuck:
        ex = ", ".join(f"(d={r['d']},N={r['n']})" for r in stuck[:6])
        lines.append(
            f"No improvement on {len(stuck)} configs (these cap the headroom), e.g. {ex}."
        )
    errs = [r for r in results if r["error"]]
    if errs:
        lines.append(
            f"{len(errs)} configs reported an error; e.g. (d={errs[0]['d']},N={errs[0]['n']}): {errs[0]['error']}"
        )
    n_budget = sum(r.get("n_timeout", 0) for r in results)
    if n_budget:
        lines.append(
            f"{n_budget} calls reached the per-config time budget (normal for deep search; the best result so far was kept)."
        )
    return "\n".join(lines)


def validate(Improver_class):
    cfg = _Cfg()
    logger.info(
        "[spherical-cfg] eval_set={} R={} B={} workers={} config_timeout={}s eval_timeout={}s seed={}",
        cfg.eval_set,
        cfg.r_rounds,
        cfg.b_steps,
        cfg.config_workers,
        cfg.config_timeout,
        cfg.eval_timeout,
        cfg.seed,
    )
    configs = cc.eval_configs(cfg.eval_set)
    # Preload each frozen Cohn array in the PARENT so forked workers inherit it
    # copy-on-write (no per-worker disk read, no array pickling across the queue).
    configs_mu = [(d, n, *cc.load_frozen(d, n)) for d, n in configs]

    eval_deadline = time.monotonic() + cfg.eval_timeout
    done = _run_parallel(Improver_class, configs_mu, cfg, eval_deadline)

    results: list[dict] = []
    for d, n, _X, mu_cohn in configs_mu:
        if (d, n) in done:
            results.append(done[(d, n)])
        else:
            results.append(
                _skipped(d, n, mu_cohn, "skipped: still running at eval timeout")
            )

    if not results:
        raise ValueError(
            f"eval set {cfg.eval_set!r} produced no configurations to score."
        )

    any_ran = any(r["n_improve"] > 0 for r in results)
    if not any_ran:
        raise ValueError(
            "No improve() call completed within the eval-time budget. "
            "Increase SPHERICAL_EVAL_TIMEOUT / SPHERICAL_CONFIG_TIMEOUT, or speed up improve()."
        )
    if not any(r["produced_valid"] for r in results):
        errs = [r["error"] for r in results if r["error"]]
        raise ValueError(
            "Improver produced NO valid (n, d) unit-norm array on ANY config. "
            "Required: output shape exactly (n, d); row norms within 1e-12 of 1 (use float64); "
            f"no NaN/Inf. First error: {errs[0] if errs else 'none recorded'}"
        )

    fitness = 100.0 * sum(r["gain"] for r in results) / len(results)
    artifact = {"feedback_preview": _feedback(results, cfg, fitness)}
    return {"fitness": fitness, "is_valid": 1.0}, artifact
